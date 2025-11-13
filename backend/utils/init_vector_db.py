"""
Vector Database Initialization
Creates Qdrant collection if it doesn't exist
"""

import asyncio
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CreateCollection

from config import settings

logger = logging.getLogger(__name__)


async def initialize_qdrant_collection():
    """Initialize Qdrant collection for document embeddings"""
    try:
        logger.info("🔧 Initializing Qdrant collection...")
        
        client = QdrantClient(url=settings.QDRANT_URL)
        
        # Check if collection exists
        try:
            collection_info = client.get_collection(settings.VECTOR_COLLECTION_NAME)
            logger.info(f"✅ Collection '{settings.VECTOR_COLLECTION_NAME}' already exists")
            return True
        except Exception:
            # Collection doesn't exist, create it
            logger.info(f"🆕 Creating collection '{settings.VECTOR_COLLECTION_NAME}'...")
            
            client.create_collection(
                collection_name=settings.VECTOR_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI text-embedding-3-small dimensions
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"✅ Collection '{settings.VECTOR_COLLECTION_NAME}' created successfully")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize Qdrant collection: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(initialize_qdrant_collection())
