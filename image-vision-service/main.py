"""
Image Vision Service - Main Entry Point
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
from service.grpc_service import ImageVisionServiceImplementation

# Import proto after adding path
import sys
sys.path.insert(0, '/app')
from protos import image_vision_pb2_grpc


class GracefulShutdown:
    """Handle graceful shutdown"""
    
    def __init__(self, server, service_impl):
        self.server = server
        self.service_impl = service_impl
        self.shutdown_event = asyncio.Event()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signal"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
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
        # Validate settings
        settings.validate()
        logger.info(f"Starting {settings.SERVICE_NAME} on port {settings.GRPC_PORT}")
        
        # Create service implementation
        service_impl = ImageVisionServiceImplementation()
        
        # Initialize service
        logger.info("Initializing service components...")
        await service_impl.initialize()
        
        # Create gRPC server with increased message size limits
        # Face encodings are 128 floats = ~512 bytes per face, but allow for large images
        options = [
            ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10 MB
            ('grpc.max_receive_message_length', 10 * 1024 * 1024),  # 10 MB
        ]
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=settings.MAX_WORKERS),
            options=options
        )
        
        # Add service
        image_vision_pb2_grpc.add_ImageVisionServiceServicer_to_server(
            service_impl,
            server
        )
        
        # Bind to port
        server.add_insecure_port(f'[::]:{settings.GRPC_PORT}')
        
        # Setup graceful shutdown
        shutdown_handler = GracefulShutdown(server, service_impl)
        signal.signal(signal.SIGINT, shutdown_handler.signal_handler)
        signal.signal(signal.SIGTERM, shutdown_handler.signal_handler)
        
        # Start server
        await server.start()
        logger.info(f"Image Vision Service ready on port {settings.GRPC_PORT}")
        logger.info(f"Device: {settings.get_device()}")
        logger.info(f"Max workers: {settings.MAX_WORKERS}")
        
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
