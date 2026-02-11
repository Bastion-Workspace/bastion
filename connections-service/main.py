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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from config.settings import settings
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
