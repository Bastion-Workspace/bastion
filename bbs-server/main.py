"""
BBS server entry point: optional Telnet and/or SSH listeners sharing one connection budget.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Any, Optional

from config.settings import settings

try:
    _root_log_level = getattr(logging, (settings.LOG_LEVEL or "INFO").upper())
except AttributeError:
    _root_log_level = logging.INFO

logging.basicConfig(
    level=_root_log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
if _root_log_level > logging.DEBUG:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def _async_main() -> None:
    from connection_budget import ConnectionBudget, set_connection_budget
    from server import BBSTelnetServer

    settings.validate()

    budget = ConnectionBudget(settings.BBS_MAX_CONNECTIONS)
    set_connection_budget(budget)

    srv: Optional[BBSTelnetServer] = None
    ssh_listener: Any = None

    if settings.BBS_ENABLE_TELNET:
        srv = BBSTelnetServer()
        await srv.start()
        logger.info(
            "%s telnet listening on 0.0.0.0:%s",
            settings.SERVICE_NAME,
            settings.BBS_TELNET_PORT,
        )

    if settings.BBS_ENABLE_SSH:
        from ssh_server import start_ssh_listener

        ssh_listener = await start_ssh_listener(budget)

    stopped = asyncio.Event()

    def _shutdown() -> None:
        logger.info("Shutdown requested")
        stopped.set()
        if srv and srv.server is not None:
            srv.server.close()
        if ssh_listener is not None:
            try:
                ssh_listener.close()
            except Exception:
                pass

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass

    try:
        tasks: list[Any] = []
        if srv and srv.server is not None:
            tasks.append(srv.server.serve_forever())
        if ssh_listener is not None:
            serve = getattr(ssh_listener, "serve_forever", None)
            if callable(serve):
                tasks.append(serve())
        tasks.append(stopped.wait())
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        stopped.set()
        if srv and srv.server is not None:
            srv.server.close()
            try:
                await srv.server.wait_closed()
            except Exception:
                pass
        if ssh_listener is not None:
            try:
                ssh_listener.close()
            except Exception:
                pass
            try:
                await ssh_listener.wait_closed()
            except Exception:
                pass
        logger.info("BBS server stopped")


def main() -> None:
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.exception("Fatal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
