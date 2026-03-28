"""
Voice Service - Main Entry Point
"""

import asyncio
import logging
import signal
import sys

import grpc
from concurrent import futures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, "/app")

from config.settings import settings
from service.grpc_service import VoiceServiceImplementation
from service.piper_bootstrap import ensure_piper_bootstrap_voice

from protos import voice_service_pb2_grpc


class GracefulShutdown:
    """Handle graceful shutdown."""

    def __init__(self, server, service_impl):
        self.server = server
        self.service_impl = service_impl
        self.shutdown_event = asyncio.Event()

    def signal_handler(self, signum, frame):
        logger.info("Received signal %s, initiating graceful shutdown...", signum)
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        logger.info("Stopping server...")
        await self.service_impl.cleanup()
        await self.server.stop(grace=5)
        logger.info("Server shutdown complete")
        self.shutdown_event.set()


async def serve():
    """Start the gRPC server."""
    try:
        settings.validate()
        logger.info("Starting %s on port %s", settings.SERVICE_NAME, settings.GRPC_PORT)

        await ensure_piper_bootstrap_voice()

        impl = VoiceServiceImplementation()
        logger.info("Initializing service components...")
        await impl.initialize()

        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=4),
            options=options,
        )

        voice_service_pb2_grpc.add_VoiceServiceServicer_to_server(impl, server)
        server.add_insecure_port(f"[::]:{settings.GRPC_PORT}")

        shutdown_handler = GracefulShutdown(server, impl)
        signal.signal(signal.SIGINT, shutdown_handler.signal_handler)
        signal.signal(signal.SIGTERM, shutdown_handler.signal_handler)

        await server.start()
        logger.info("Voice Service ready on port %s", settings.GRPC_PORT)

        await shutdown_handler.shutdown_event.wait()
    except Exception as e:
        logger.error("Failed to start server: %s", e)
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
