"""
Document Service - Main Entry Point
"""

import asyncio
import logging
import signal
import grpc
from concurrent import futures

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from config.settings import settings
from service.grpc_service import DocumentServiceImplementation

import sys

sys.path.insert(0, "/app")
from protos import document_service_pb2_grpc


class GracefulShutdown:
    """Handle graceful shutdown"""

    def __init__(self, server):
        self.server = server
        self.shutdown_event = asyncio.Event()

    def signal_handler(self, signum, frame):
        """Handle shutdown signal"""
        logger.info("Received signal %s, initiating graceful shutdown...", signum)
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        """Shutdown server gracefully"""
        logger.info("Stopping server...")
        await self.server.stop(grace=5)
        logger.info("Server shutdown complete")
        self.shutdown_event.set()


async def serve():
    """Start the gRPC server"""
    try:
        settings.validate()
        logger.info("Starting %s on port %s", settings.SERVICE_NAME, settings.GRPC_PORT)

        service_impl = DocumentServiceImplementation()

        logger.info("Initializing service components...")
        await service_impl.initialize()

        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=settings.PARALLEL_WORKERS),
            options=options,
        )

        document_service_pb2_grpc.add_DocumentServiceServicer_to_server(
            service_impl, server
        )

        server.add_insecure_port(f"[::]:{settings.GRPC_PORT}")

        shutdown_handler = GracefulShutdown(server)
        signal.signal(signal.SIGINT, shutdown_handler.signal_handler)
        signal.signal(signal.SIGTERM, shutdown_handler.signal_handler)

        await server.start()
        logger.info("Document Service ready on port %s", settings.GRPC_PORT)
        logger.info("spaCy model: %s", settings.SPACY_MODEL)

        await shutdown_handler.shutdown_event.wait()

    except Exception as e:
        logger.error("Failed to start server: %s", e)
        raise


def main():
    """Main entry point"""
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Fatal error: %s", e)
        raise


if __name__ == "__main__":
    main()
