#!/usr/bin/env python3
"""
Test script for shared database pool mechanism
"""

import asyncio
import logging
from utils.shared_db_pool import get_shared_db_pool, close_shared_db_pool
from services.category_service import CategoryService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_shared_pool():
    """Test the shared database pool mechanism"""
    logger.info("🧪 Testing shared database pool mechanism...")
    
    try:
        # Get shared pool
        pool1 = await get_shared_db_pool()
        logger.info(f"✅ Got first pool: {pool1}")
        
        # Get shared pool again (should be the same instance)
        pool2 = await get_shared_db_pool()
        logger.info(f"✅ Got second pool: {pool2}")
        
        # Verify they're the same instance
        if pool1 is pool2:
            logger.info("✅ Shared pool mechanism working correctly - same instance returned")
        else:
            logger.error("❌ Shared pool mechanism failed - different instances returned")
            return False
        
        # Test CategoryService with shared pool
        category_service = CategoryService()
        await category_service.initialize(shared_db_pool=pool1)
        logger.info("✅ CategoryService initialized with shared pool")
        
        # Test basic database operation
        async with pool1.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            logger.info(f"✅ Database connection test successful: {result}")
        
        # Close shared pool
        await close_shared_db_pool()
        logger.info("✅ Shared pool closed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_shared_pool())
    if success:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("💥 Tests failed!")
        exit(1)
