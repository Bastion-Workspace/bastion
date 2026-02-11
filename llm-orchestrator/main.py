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


class CleanFormatter(logging.Formatter):
    """Formatter that strips trailing newlines from log messages"""
    
    def format(self, record):
        # Get the formatted message
        message = super().format(record)
        # Strip trailing whitespace and newlines, then ensure single newline
        message = message.rstrip()
        return message


# Configure logging with clean formatter
formatter = CleanFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[console_handler],
    force=True  # Override any existing configuration
)

# Set specific log levels for noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("grpc._cython.cygrpc").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handle graceful shutdown of gRPC server"""
    
    def __init__(self, server):
        self.server = server
        self.shutdown_event = asyncio.Event()
        self.is_shutting_down = False
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signal - sets event for async shutdown"""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logger.debug(f"Received shutdown signal {signum}, gracefully shutting down...")
        # Signal the async shutdown to proceed
        self.shutdown_event.set()
    
    async def shutdown(self):
        """Shutdown server gracefully - called from async context"""
        logger.debug("Stopping server...")
        await self.server.stop(grace=10)
        logger.debug("Server shutdown complete")


async def serve():
    """Start the gRPC server"""
    try:
        # Import gRPC health checking and reflection inside function to avoid module-level issues
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc
        from grpc_reflection.v1alpha import reflection
        from orchestrator.backend_tool_client import get_backend_tool_client, close_backend_tool_client
        
        logger.info(f"Starting {settings.SERVICE_NAME} on port {settings.GRPC_PORT}...")
        
        # Initialize backend tool client connection
        logger.debug("Initializing backend tool client...")
        try:
            await get_backend_tool_client()
            logger.debug("✅ Backend tool client connected")
        except Exception as e:
            logger.warning(f"⚠️  Backend tool client initialization failed (will retry on demand): {e}")
        
        # Create gRPC server with thread pool and increased message size limits
        # Default is 4MB, increase to 100MB for large responses
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
        ]
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_REQUESTS),
            options=options
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
        signal.signal(signal.SIGTERM, shutdown_handler.signal_handler)
        signal.signal(signal.SIGINT, shutdown_handler.signal_handler)
        
        # Start server
        await server.start()
        logger.info(f"✅ {settings.SERVICE_NAME} listening on port {settings.GRPC_PORT}")
        logger.info(f"✅ Health check available at localhost:{settings.GRPC_PORT}")
        
        # Wait for shutdown signal
        await shutdown_handler.shutdown_event.wait()
        
        # Perform graceful shutdown
        await shutdown_handler.shutdown()
        
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

