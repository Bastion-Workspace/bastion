"""
Crawl4AI Service - Main Entry Point
"""

import asyncio
import logging
import signal
import grpc
from concurrent import futures

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config.settings import settings
from service.grpc_service import CrawlServiceGRPCImplementation, BrowserSessionServiceGRPCImplementation

# Import proto after adding path
import sys
sys.path.insert(0, '/app')
from protos import crawl_service_pb2_grpc


class GracefulShutdown:
    """Handle graceful shutdown"""

    def __init__(self, server, service_impls):
        self.server = server
        self.service_impls = service_impls if isinstance(service_impls, list) else [service_impls]
        self.shutdown_event = asyncio.Event()

    def signal_handler(self, signum, frame):
        """Handle shutdown signal"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        """Shutdown server gracefully"""
        logger.info("Stopping server...")
        for impl in self.service_impls:
            await impl.cleanup()
        await self.server.stop(grace=5)
        logger.info("Server shutdown complete")
        self.shutdown_event.set()


async def serve():
    """Start the gRPC server"""
    try:
        # Validate settings
        settings.validate()
        logger.info(f"Starting {settings.SERVICE_NAME} on port {settings.GRPC_PORT}")
        
        # Create service implementations
        crawl_impl = CrawlServiceGRPCImplementation()
        browser_impl = BrowserSessionServiceGRPCImplementation()

        # Initialize services
        logger.info("Initializing service components...")
        await crawl_impl.initialize()
        await browser_impl.initialize()

        # Create gRPC server with increased message size limits
        # Default is 4MB, increase to 100MB for large crawl responses and file downloads
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
        ]
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_CRAWLS),
            options=options
        )

        # Add services
        crawl_service_pb2_grpc.add_CrawlServiceServicer_to_server(crawl_impl, server)
        crawl_service_pb2_grpc.add_BrowserSessionServiceServicer_to_server(browser_impl, server)

        # Bind to port
        server.add_insecure_port(f'[::]:{settings.GRPC_PORT}')

        # Setup graceful shutdown
        shutdown_handler = GracefulShutdown(server, [crawl_impl, browser_impl])
        signal.signal(signal.SIGINT, shutdown_handler.signal_handler)
        signal.signal(signal.SIGTERM, shutdown_handler.signal_handler)
        
        # Start server
        await server.start()
        logger.info(f"Crawl4AI Service ready on port {settings.GRPC_PORT}")
        logger.info(f"Max concurrent crawls: {settings.MAX_CONCURRENT_CRAWLS}")
        logger.info(f"Headless mode: {settings.HEADLESS}")
        
        # Wait for shutdown
        await shutdown_handler.shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


def main():
    """Main entry point"""
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()








