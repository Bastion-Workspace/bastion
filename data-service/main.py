import asyncio
import logging
import signal
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from config.settings import settings
from grpc_service import DataServiceImplementation
from db.connection_manager import get_db_manager, close_db_manager

# Import generated protobuf code
try:
    import data_service_pb2
    import data_service_pb2_grpc
except ImportError:
    logging.error("Protobuf code not generated")
    raise

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handle graceful shutdown of the gRPC server"""
    
    def __init__(self, server):
        self.server = server
        self.shutdown_event = asyncio.Event()
        self.is_shutting_down = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signal - sets event for async shutdown"""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # Signal the async shutdown to proceed
        self.shutdown_event.set()
    
    async def shutdown(self):
        """Shutdown server gracefully - called from async context"""
        logger.info("Stopping server...")
        await self.server.stop(grace=5)
        await close_db_manager()
        logger.info("Server shutdown complete")


async def serve():
    """Start the gRPC server"""
    try:
        logger.info(f"Starting {settings.SERVICE_NAME} on port {settings.GRPC_PORT}...")
        
        # Initialize database connection
        logger.info("Initializing database connection...")
        try:
            await get_db_manager()
            logger.info("Database connection initialized")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
        
        # Create gRPC server
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10)
        )
        
        # Register data service
        data_service = DataServiceImplementation()
        await data_service.initialize()
        data_service_pb2_grpc.add_DataServiceServicer_to_server(
            data_service, server
        )
        
        # Register health checking service
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        health_servicer.set(
            "dataservice.DataService",
            health_pb2.HealthCheckResponse.SERVING
        )
        
        # Enable server reflection for debugging
        service_names = (
            data_service_pb2.DESCRIPTOR.services_by_name['DataService'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)
        
        # Bind to port
        server.add_insecure_port(f'[::]:{settings.GRPC_PORT}')
        
        # Setup graceful shutdown
        shutdown_handler = GracefulShutdown(server)
        
        # Start server
        await server.start()
        logger.info(f"{settings.SERVICE_NAME} started successfully on port {settings.GRPC_PORT}")
        
        # Wait for shutdown signal
        await shutdown_handler.shutdown_event.wait()
        
        # Perform graceful shutdown
        await shutdown_handler.shutdown()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}")
        exit(1)









