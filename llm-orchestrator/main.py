"""
LLM Orchestrator Service - Main Entry Point
Runs gRPC server for LangGraph agent orchestration
"""

import asyncio
import logging
import signal
from concurrent import futures

import grpc

from config.settings import settings
from protos import orchestrator_pb2_grpc
from orchestrator.grpc_service import OrchestratorGRPCService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handle graceful shutdown of gRPC server"""
    
    def __init__(self, server):
        self.server = server
        self.is_shutting_down = False
    
    def shutdown(self, signum, frame):
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logger.info(f"Received shutdown signal {signum}, gracefully shutting down...")
        
        # Give server 10 seconds to finish in-flight requests
        self.server.stop(grace=10)
        logger.info("Server shutdown complete")


async def serve():
    """Start the gRPC server"""
    try:
        # Import gRPC health checking and reflection inside function to avoid module-level issues
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc
        from grpc_reflection.v1alpha import reflection
        from orchestrator.backend_tool_client import get_backend_tool_client, close_backend_tool_client
        
        logger.info(f"Starting {settings.SERVICE_NAME} on port {settings.GRPC_PORT}...")
        
        # Initialize backend tool client connection
        logger.info("Initializing backend tool client...")
        try:
            await get_backend_tool_client()
            logger.info("✅ Backend tool client connected")
        except Exception as e:
            logger.warning(f"⚠️  Backend tool client initialization failed (will retry on demand): {e}")
        
        # Create gRPC server with thread pool
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_REQUESTS)
        )
        
        # Register orchestrator service
        orchestrator_service = OrchestratorGRPCService()
        orchestrator_pb2_grpc.add_OrchestratorServiceServicer_to_server(
            orchestrator_service, server
        )
        
        # Register health checking service
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        health_servicer.set(
            "orchestrator.OrchestratorService",
            health_pb2.HealthCheckResponse.SERVING
        )
        
        # Enable server reflection for debugging
        from protos import orchestrator_pb2
        service_names = (
            orchestrator_pb2.DESCRIPTOR.services_by_name['OrchestratorService'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, server)
        
        # Bind to port
        server.add_insecure_port(f'[::]:{settings.GRPC_PORT}')
        
        # Setup graceful shutdown
        shutdown_handler = GracefulShutdown(server)
        signal.signal(signal.SIGTERM, shutdown_handler.shutdown)
        signal.signal(signal.SIGINT, shutdown_handler.shutdown)
        
        # Start server
        await server.start()
        logger.info(f"✅ {settings.SERVICE_NAME} listening on port {settings.GRPC_PORT}")
        logger.info(f"✅ Health check available at localhost:{settings.GRPC_PORT}")
        
        # Wait for termination
        await server.wait_for_termination()
        
        # Cleanup backend tool client
        logger.info("Closing backend tool client...")
        await close_backend_tool_client()
        logger.info("✅ Backend tool client closed")
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        # Cleanup on error
        try:
            from orchestrator.backend_tool_client import close_backend_tool_client
            await close_backend_tool_client()
        except:
            pass
        raise


if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        exit(1)

