"""
Optional AsyncSSH listener: password auth via Bastion /api/auth/login, same BBSSession as telnet.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

import asyncssh

from backend_client import BackendClient
from config.settings import settings
from connection_budget import ConnectionBudget

if TYPE_CHECKING:
    from session import BBSSession

logger = logging.getLogger(__name__)


class _ChannelWriter:
    """Minimal asyncio.StreamWriter-like adapter for SSHServerChannel."""

    __slots__ = ("_chan",)

    def __init__(self, chan: Any) -> None:
        self._chan = chan

    def write(self, data: Any) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8", errors="replace")
        elif not isinstance(data, (bytes, bytearray)):
            data = bytes(data)
        self._chan.write(data)

    async def drain(self) -> None:
        dr = getattr(self._chan, "drain", None)
        if callable(dr):
            await dr()

    def close(self) -> None:
        self._chan.close()

    async def wait_closed(self) -> None:
        wc = getattr(self._chan, "wait_closed", None)
        if callable(wc):
            await wc()

    def is_closing(self) -> bool:
        ic = getattr(self._chan, "is_closing", None)
        if callable(ic):
            return bool(ic())
        return False

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        ge = getattr(self._chan, "get_extra_info", None)
        if callable(ge):
            try:
                return ge(name, default)
            except Exception:
                return default
        return default


class BastionSSHSession(asyncssh.SSHServerSession):
    """PTY + shell session driving BBSSession after password auth."""

    def __init__(self, budget: ConnectionBudget, login_result: Dict[str, Any]) -> None:
        super().__init__()
        self._budget = budget
        self._login_result = dict(login_result)
        self._reader = asyncio.StreamReader()
        self._chan: Any = None
        self._tw = 80
        self._th = 24
        self._task: Optional[asyncio.Task] = None
        self._bbs: Optional["BBSSession"] = None

    def connection_made(self, chan: Any) -> None:
        self._chan = chan

    def pty_requested(self, term_type: str, term_size: Any, term_modes: Any) -> bool:
        if term_size and len(term_size) >= 2:
            self._tw = max(20, min(int(term_size[0]), 500))
            self._th = max(5, min(int(term_size[1]), 200))
        return True

    def shell_requested(self) -> bool:
        return True

    def exec_requested(self, command: str) -> bool:
        """Accept non-interactive exec the same as shell (BBS is line-oriented)."""
        return True

    def terminal_size_changed(self, width: int, height: int, pixwidth: int, pixheight: int) -> None:
        self._tw = max(20, min(width, 500))
        self._th = max(5, min(height, 200))
        if self._bbs is not None:
            self._bbs.term_width = self._tw
            self._bbs.term_height = self._th

    def data_received(self, data: Any, datatype: Optional[int] = None) -> None:
        # With server encoding=None, stdin arrives as bytes; only skip real stderr.
        if datatype == asyncssh.EXTENDED_DATA_STDERR:
            return
        if isinstance(data, str):
            data = data.encode("utf-8", errors="replace")
        elif not isinstance(data, (bytes, bytearray)):
            return
        self._reader.feed_data(data)

    def eof_received(self) -> bool:
        self._reader.feed_eof()
        return False

    def session_started(self) -> None:
        self._task = asyncio.create_task(self._run_bbs(), name="bbs-ssh")

    def connection_lost(self, exc: Optional[Exception]) -> None:
        t = self._task
        if t is not None and not t.done():
            t.cancel()
        self._task = None
        self._bbs = None

    async def _run_bbs(self) -> None:
        from session import BBSSession

        if self._chan is None:
            return
        if not await self._budget.acquire():
            try:
                self._chan.write(b"Too many connections. Try again later.\r\n")
                await _ChannelWriter(self._chan).drain()
            except Exception:
                pass
            self._chan.close()
            return

        writer = _ChannelWriter(self._chan)
        try:
            sid = str(uuid.uuid4())[:8]
            session = BBSSession(
                self._reader,
                writer,
                session_id=sid,
                connected_count=self._budget.count,
                telnet_mode=False,
            )
            session.term_width = self._tw
            session.term_height = self._th
            self._bbs = session
            session.apply_login_result(self._login_result)
            await session.run_after_authenticated()
        except asyncio.CancelledError:
            raise
        except ConnectionError:
            pass
        except Exception as e:
            logger.exception("SSH BBS session error: %s", e)
            try:
                self._chan.write(f"\r\nError: {str(e)[:200]}\r\n".encode("utf-8", errors="replace"))
                await writer.drain()
            except Exception:
                pass
        finally:
            self._bbs = None
            await self._budget.release()
            try:
                self._chan.close()
            except Exception:
                pass


def _build_ssh_server_class(budget: ConnectionBudget) -> type:
    class BastionSSHServer(asyncssh.SSHServer):
        def __init__(self) -> None:
            super().__init__()
            self._login_result: Optional[Dict[str, Any]] = None
            self._conn: Optional[asyncssh.SSHServerConnection] = None

        def connection_made(self, conn: asyncssh.SSHServerConnection) -> None:
            self._conn = conn

        def connection_lost(self, exc: Optional[Exception]) -> None:
            self._conn = None

        def begin_auth(self, username: str) -> bool:
            self._login_result = None
            c = self._conn
            if c is not None:
                try:
                    c.send_auth_banner(
                        "Bastion BBS - use your Bastion username and password.\r\n"
                    )
                except Exception:
                    pass
            return True

        def password_auth_supported(self) -> bool:
            return True

        async def validate_password(self, username: str, password: str) -> bool:
            self._login_result = None
            client = BackendClient()
            result = await client.login(username, password)
            if result.get("error"):
                return False
            token = result.get("access_token") or result.get("accessToken")
            u = result.get("user") or {}
            if not token or not u.get("user_id"):
                return False
            self._login_result = result
            return True

        def session_requested(self) -> Any:
            if not self._login_result:
                return False
            return BastionSSHSession(budget, self._login_result)

    return BastionSSHServer


async def start_ssh_listener(budget: ConnectionBudget) -> asyncssh.SSHAcceptor:
    """Start SSH listener; caller must await acceptor.wait_closed() or serve_forever()."""
    cls = _build_ssh_server_class(budget)
    acceptor = await asyncssh.create_server(
        cls,
        "0.0.0.0",
        settings.BBS_SSH_PORT,
        server_host_keys=[settings.BBS_SSH_HOST_KEY],
        # Default AsyncSSH server encoding is utf-8, which makes channel.write()
        # expect str; BBSSession writes bytes. Raw bytes mode matches telnet StreamWriter.
        encoding=None,
        # With encoding=None, stdin is bytes. AsyncSSH's built-in line editor wrapper
        # (SSHLineEditorSession) expects str and breaks on the first key — disable it;
        # BBSSession uses its own LineEditor in read_line().
        line_editor=False,
        # AsyncSSH defaults gss_host to the container hostname and enables GSS KEX/auth;
        # clients may then negotiate GSS and hang after the version banner. Bastion BBS
        # uses password auth only — disable GSS and host-based auth for predictable KEX.
        gss_kex=False,
        gss_auth=False,
        host_based_auth=False,
        public_key_auth=False,
    )
    logger.info(
        "BBS SSH listening on 0.0.0.0:%s (host key %s)",
        settings.BBS_SSH_PORT,
        settings.BBS_SSH_HOST_KEY,
    )
    return acceptor
