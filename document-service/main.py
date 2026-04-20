"""
Document Service - Main Entry Point (standalone: vendored backend pipeline + NER).
"""

import asyncio
import logging
import os
import signal
import sys
from concurrent import futures

import grpc

# Shims must precede ds_* so `services.*` and `utils.*` resolve to stubs.
sys.path.insert(0, "/app/ds/shims")
sys.path.insert(0, "/app/ds")
sys.path.insert(0, "/app")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from ds_config import settings, validate_runtime
from service.grpc_service import DocumentServiceImplementation

from protos import document_service_pb2_grpc


class GracefulShutdown:
    def __init__(self, server, file_watcher=None, service_impl=None):
        self.server = server
        self.file_watcher = file_watcher
        self.service_impl = service_impl
        self.shutdown_event = asyncio.Event()

    def signal_handler(self, signum, frame):
        logger.info("Received signal %s, initiating graceful shutdown...", signum)
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        logger.info("Stopping server...")
        if self.service_impl:
            try:
                await self.service_impl.shutdown_neo4j_maintenance()
            except Exception as e:
                logger.warning("Neo4j maintenance shutdown: %s", e)
        if self.file_watcher:
            try:
                await self.file_watcher.stop()
            except Exception as e:
                logger.warning("File watcher stop: %s", e)
            try:
                from ds_services.file_watcher_service import reset_file_watcher

                reset_file_watcher()
            except Exception:
                pass
        await self.server.stop(grace=5)
        logger.info("Server shutdown complete")
        self.shutdown_event.set()


async def serve():
    file_watcher = None
    try:
        validate_runtime()
        logger.info("Starting %s on port %s", settings.SERVICE_NAME, settings.GRPC_PORT)

        service_impl = DocumentServiceImplementation()
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

        if os.getenv("DATABASE_URL") and os.getenv("DS_ENABLE_FILE_WATCHER", "true").lower() in (
            "1",
            "true",
            "yes",
        ):
            try:
                from ds_services.file_watcher_service import get_file_watcher

                file_watcher = await get_file_watcher()
                await file_watcher.start()
                logger.info("File system watcher started")
            except Exception as e:
                logger.warning("File watcher not started: %s", e)
                file_watcher = None

        shutdown_handler = GracefulShutdown(
            server, file_watcher=file_watcher, service_impl=service_impl
        )
        signal.signal(signal.SIGINT, shutdown_handler.signal_handler)
        signal.signal(signal.SIGTERM, shutdown_handler.signal_handler)

        await server.start()
        logger.info("Document Service ready on port %s", settings.GRPC_PORT)

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
