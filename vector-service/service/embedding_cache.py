"""
Embedding Cache - Hash-based caching with TTL
"""

import hashlib
import time
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Hash-based embedding cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 10800):  # 3 hours default
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple[List[float], float]] = {}  # hash -> (embedding, timestamp)
        self.hits = 0
        self.misses = 0
        self._enabled = True
    
    async def initialize(self):
        """Initialize cache"""
        logger.info(f"Embedding cache initialized with {self.ttl_seconds}s TTL")
    
    def hash_text(self, text: str) -> str:
        """Generate stable hash for text content"""
        # Use SHA256 for consistent hashing
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    async def get(self, content_hash: str) -> Optional[List[float]]:
        """Get embedding from cache if not expired"""
        if not self._enabled:
            self.misses += 1
            return None
            
        if content_hash in self.cache:
            embedding, timestamp = self.cache[content_hash]
            age = time.time() - timestamp
            
            if age < self.ttl_seconds:
                self.hits += 1
                logger.debug(f"Cache hit: {content_hash[:16]}... (age: {age:.1f}s)")
                return embedding
            else:
                # Expired - remove from cache
                del self.cache[content_hash]
                logger.debug(f"Cache expired: {content_hash[:16]}... (age: {age:.1f}s)")
        
        self.misses += 1
        return None
    
    async def set(self, content_hash: str, embedding: List[float]):
        """Store embedding in cache"""
        if self._enabled:
            self.cache[content_hash] = (embedding, time.time())
            logger.debug(f"Cached embedding: {content_hash[:16]}...")
    
    async def clear(self, content_hash: Optional[str] = None) -> int:
        """Clear cache (all or specific hash)"""
        if content_hash:
            if content_hash in self.cache:
                del self.cache[content_hash]
                logger.info(f"Cleared cache entry: {content_hash[:16]}...")
                return 1
            return 0
        else:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared entire cache ({count} entries)")
            return count
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds,
            'enabled': self._enabled
        }
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries (periodic cleanup)"""
        now = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if (now - timestamp) >= self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def disable(self):
        """Disable cache"""
        self._enabled = False
        logger.info("Embedding cache disabled")
    
    def enable(self):
        """Enable cache"""
        self._enabled = True
        logger.info("Embedding cache enabled")

