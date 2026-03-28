"""
Connections Service - Main Entry Point
"""

import asyncio
import logging
import signal
import sys

import grpc
from concurrent import futures

sys.path.insert(0, "/app")

from config.settings import settings

try:
    _root_log_level = getattr(logging, (settings.LOG_LEVEL or "INFO").upper())
except AttributeError:
    _root_log_level = logging.INFO

logging.basicConfig(
    level=_root_log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
from utils.log_redaction import RedactTelegramSecretsFilter

_telegram_log_redact = RedactTelegramSecretsFilter()
for _handler in logging.root.handlers:
    _handler.addFilter(_telegram_log_redact)

# httpx logs every request at INFO; keep that noise (and token risk) for DEBUG-only sessions
if _root_log_level > logging.DEBUG:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
from service.channel_listener_manager import ChannelListenerManager
from service.grpc_service import ConnectionsServiceImplementation
from connections_service_pb2_grpc import add_ConnectionsServiceServicer_to_server


class GracefulShutdown:
    def __init__(self, server):
        self.server = server
        self.shutdown_event = asyncio.Event()

    def signal_handler(self, signum, frame):
        logger.info("Received signal %s, initiating graceful shutdown...", signum)
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        logger.info("Stopping server...")
        await self.server.stop(grace=5)
        logger.info("Server shutdown complete")
        self.shutdown_event.set()


async def _restore_bots(listener_manager: ChannelListenerManager) -> None:
    """Background task: fetch active chat bots from backend and register them (self-restore on startup)."""
    await asyncio.sleep(3)
    max_attempts = 10
    delay = 5.0
    for attempt in range(max_attempts):
        try:
            result = await listener_manager._backend_client.get_active_chat_bots()
            if result.get("error"):
                raise RuntimeError(result["error"])
            bots = result.get("bots") or []
            if not bots:
                logger.info("No active chat bots to restore")
                return
            restored = 0
            for bot in bots:
                conn_id = bot.get("connection_id", "")
                user_id = bot.get("user_id", "")
                provider = bot.get("provider", "")
                token = bot.get("bot_token", "")
                display_name = bot.get("display_name", "")
                config = bot.get("config") or {}
                if not conn_id or not provider or not token:
                    continue
                reg = await listener_manager.register_bot(
                    connection_id=conn_id,
                    user_id=user_id,
                    provider=provider,
                    bot_token=token,
                    display_name=display_name,
                    config=config,
                )
                if reg.get("success"):
                    restored += 1
                else:
                    logger.warning("Failed to restore bot %s: %s", conn_id, reg.get("error", "unknown"))
            if restored:
                logger.info("Restored %s of %s chat bot connection(s) on startup", restored, len(bots))
            return
        except Exception as e:
            logger.warning("Bot restore attempt %s/%s failed: %s", attempt + 1, max_attempts, e)
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)


async def serve():
    try:
        settings.validate()
        logger.info("Starting %s on port %s", settings.SERVICE_NAME, settings.GRPC_PORT)

        listener_manager = ChannelListenerManager()
        service_impl = ConnectionsServiceImplementation(listener_manager=listener_manager)
        await service_impl.initialize()

        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=settings.PARALLEL_WORKERS),
            options=options,
        )
        add_ConnectionsServiceServicer_to_server(service_impl, server)
        server.add_insecure_port(f"[::]:{settings.GRPC_PORT}")

        shutdown_handler = GracefulShutdown(server)
        signal.signal(signal.SIGINT, shutdown_handler.signal_handler)
        signal.signal(signal.SIGTERM, shutdown_handler.signal_handler)

        await server.start()
        logger.info("Connections Service ready on port %s", settings.GRPC_PORT)
        asyncio.create_task(_restore_bots(listener_manager))
        await shutdown_handler.shutdown_event.wait()
    except Exception as e:
        logger.exception("Failed to start server: %s", e)
        raise


def main():
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Fatal error: %s", e)
        raise


if __name__ == "__main__":
    main()
