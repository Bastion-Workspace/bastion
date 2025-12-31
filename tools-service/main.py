"""
Tools Service - Dedicated gRPC service for tool operations
Provides document, RSS, entity, weather, and org-mode data access
"""

import asyncio
import logging
import os
import sys

# CRITICAL: Set up Python path BEFORE any other imports
# Python adds the script's directory (/app/tools-service) to sys.path automatically
# We need to ensure /app/backend is FIRST so 'from models.xxx' imports work
# Backend code uses relative imports like 'from models.api_models'

# Ensure /app/backend is FIRST in sys.path (most important for 'from models.xxx')
if '/app/backend' in sys.path:
    sys.path.remove('/app/backend')
sys.path.insert(0, '/app/backend')

# Ensure /app is in path (for 'from backend.services.xxx')
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

# Keep /app/tools_service in path (for tools_service local imports)
# It's already there from Python's automatic addition, but ensure it's after /app/backend
if '/app/tools_service' in sys.path and sys.path.index('/app/tools_service') < sys.path.index('/app/backend'):
    sys.path.remove('/app/tools_service')
    sys.path.append('/app/tools_service')

# Verify critical paths are accessible
if not os.path.exists('/app/backend/models'):
    raise ImportError("Critical: /app/backend/models directory not found! Cannot import models.api_models")
if not os.path.exists('/app/backend/models/api_models.py'):
    raise ImportError("Critical: /app/backend/models/api_models.py not found!")

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log path setup for debugging
logger.info(f"Python sys.path (first 5): {sys.path[:5]}")
logger.info(f"PYTHONPATH env: {os.getenv('PYTHONPATH', 'NOT SET')}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"/app/backend/models exists: {os.path.exists('/app/backend/models')}")
logger.info(f"/app/backend/models/api_models.py exists: {os.path.exists('/app/backend/models/api_models.py')}")

# Note: We don't test import here because Python's import system needs the package structure
# The actual import will happen when grpc_tool_service imports document_repository
# If that fails, we'll get a clearer error message


async def main():
    """Main entry point for tools service"""
    try:
        logger.info("Starting Tools Service...")
        
        # Ensure we're in the right directory context for imports
        # Change to /app/backend so relative imports like 'from models.xxx' work
        original_cwd = os.getcwd()
        try:
            os.chdir('/app/backend')
            logger.debug(f"Changed working directory to /app/backend for imports")
        except Exception as e:
            logger.warning(f"Could not change to /app/backend: {e}, continuing with current directory")
        
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
