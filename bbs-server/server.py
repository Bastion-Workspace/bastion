"""
Asyncio TCP telnet listener: connection limits, minimal telnet negotiation, client dispatch.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from config.settings import settings
from connection_budget import get_connection_budget

logger = logging.getLogger(__name__)

# Telnet IAC
IAC = bytes([255])
DO = bytes([253])
WILL = bytes([251])
SB = bytes([250])
SE = bytes([240])
NAWS = bytes([31])
SGA = bytes([3])


def _telnet_init_bytes() -> bytes:
    """Request NAWS; offer suppress-go-ahead."""
    return IAC + DO + NAWS + IAC + WILL + SGA


class BBSTelnetServer:
    def __init__(self) -> None:
        self.server: Optional[asyncio.Server] = None

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._handle_client,
            "0.0.0.0",
            settings.BBS_TELNET_PORT,
        )

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        budget = get_connection_budget()
        if not await budget.acquire():
            try:
                writer.write(b"Too many connections. Try again later.\r\n")
                await writer.drain()
            except Exception:
                pass
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            logger.warning("Rejected client (max connections): %s", peer)
            return

        session_id = str(uuid.uuid4())[:8]
        try:
            writer.write(_telnet_init_bytes())
            await writer.drain()

            await asyncio.sleep(0.2)
            naws_buf = b""
            while not reader.at_eof():
                try:
                    chunk = await asyncio.wait_for(reader.read(128), timeout=0.05)
                    naws_buf += chunk
                except asyncio.TimeoutError:
                    break

            from session import BBSSession
            from telnet_input import strip_telnet_from_buffer

            init_width, init_height = 80, 24
            def _capture_naws(w: int, h: int) -> None:
                nonlocal init_width, init_height
                init_width, init_height = w, h

            if naws_buf:
                strip_telnet_from_buffer(naws_buf, on_naws=_capture_naws)

            session = BBSSession(
                reader,
                writer,
                session_id=session_id,
                connected_count=budget.count,
                telnet_mode=True,
            )
            session.term_width = init_width
            session.term_height = init_height
            await session.run()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError) as e:
            logger.debug("Client closed connection %s: %s", peer, e)
        except Exception as e:
            logger.exception("Client handler error %s: %s", peer, e)
        finally:
            await budget.release()
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug("Client disconnected %s", peer)
