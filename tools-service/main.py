"""
Tools Service - Dedicated gRPC service for tool operations
Provides document, RSS, entity, weather, and org-mode data access
"""

import asyncio
import logging
import os
import sys

# Add paths for shared code access
# We add /app/backend so that imports like 'from services.xxx' work 
# (since backend code uses relative-style imports)
sys.path.insert(0, '/app/backend')
# We add /app so that 'from backend.services.xxx' also works
sys.path.insert(0, '/app')

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for tools service"""
    try:
        logger.info("Starting Tools Service...")
        
        # Import the gRPC server function from backend
        # Use explicit backend.services path to avoid shadowing from tools-service/services/
        from backend.services.grpc_tool_service import serve_tool_service
        
        # Get port from environment
        port = int(os.getenv('GRPC_TOOL_SERVICE_PORT', '50052'))
        
        # Start the gRPC server
        await serve_tool_service(port)
        
    except KeyboardInterrupt:
        logger.info("Tools Service stopped by user")
    except Exception as e:
        logger.error(f"Tools Service failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    asyncio.run(main())
