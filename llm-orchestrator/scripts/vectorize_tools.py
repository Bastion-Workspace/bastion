"""
Bootstrap Script - Vectorize all tools into Qdrant

This script should be run on deployment to populate the 'tools' collection
in Qdrant with embeddings for all available tools.

Usage:
    python scripts/vectorize_tools.py
"""

import asyncio
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.utils.tool_vector_store import get_tool_vector_store

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point"""
    try:
        logger.info("Starting tool vectorization bootstrap...")
        
        # Get tool vector store
        vector_store = await get_tool_vector_store()
        
        # Vectorize all tools
        result = await vector_store.vectorize_all_tools()
        
        # Report results
        logger.info("=" * 60)
        logger.info("Vectorization Results:")
        logger.info(f"  Total tools: {result['total']}")
        logger.info(f"  Successfully vectorized: {result['success']}")
        if result['failures']:
            logger.warning(f"  Failed: {len(result['failures'])}")
            logger.warning(f"  Failed tools: {', '.join(result['failures'])}")
        logger.info("=" * 60)
        
        # Close connections
        await vector_store.close()
        
        if result['success'] == result['total']:
            logger.info("✅ All tools successfully vectorized!")
            return 0
        else:
            logger.warning(f"⚠️  {len(result['failures'])} tools failed to vectorize")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Vectorization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
