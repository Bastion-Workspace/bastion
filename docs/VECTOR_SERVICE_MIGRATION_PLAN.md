# Vector Service Migration Plan

**Date:** November 10, 2025  
**Status:** Planning Phase  
**Proposed Port:** 50053

## Executive Summary

This document outlines the plan to extract embedding generation and vector search functionality from the backend into a dedicated **Vector Service** microservice. This architectural change will:

- **Isolate vector operations** - Separate embedding/search from main backend logic
- **Enable independent scaling** - Scale vector operations based on load
- **Improve reusability** - Multiple services can use vector capabilities
- **Simplify maintenance** - Focused service with single responsibility
- **Support future growth** - Foundation for advanced vector features

## Current Architecture Analysis

### EmbeddingManager Responsibilities

The current `EmbeddingManager` (and `ParallelEmbeddingManager`) handle:

1. **Embedding Generation:**
   - Generate embeddings via OpenAI API
   - Batch processing with parallel workers
   - Token management and text truncation
   - Error handling and retry logic

2. **Vector Storage (Qdrant):**
   - Store document chunk embeddings
   - Manage user-specific and global collections
   - Update/delete embeddings
   - Collection lifecycle management

3. **Vector Search:**
   - Semantic similarity search
   - Score threshold filtering
   - User isolation (per-user collections)
   - Category/tag filtering
   - Query expansion (LLM-based)

4. **Cache Management:**
   - Query expansion caching
   - Result caching (time-based TTL)

### Current Usage Points

**Services that use EmbeddingManager:**

| Service | Usage | Location |
|---------|-------|----------|
| `DocumentService` | Generate/store embeddings during upload | `backend/services/document_service_v2.py` |
| `UserDocumentService` | User-isolated embedding storage | `backend/services/user_document_service.py` |
| `ZipProcessorService` | Batch document embedding | `backend/services/zip_processor_service.py` |
| `DirectSearchService` | Semantic search | `backend/services/direct_search_service.py` |
| `ChatService` | Context retrieval for LLM | `backend/services/chat_service.py` |
| `UnifiedSearchTools` | LangGraph agent search | `backend/services/langgraph_tools/unified_search_tools.py` |
| `VideoSearchService` | Video transcript search | `backend/services/video_search_service.py` |
| `gRPC Tool Service` | Search for orchestrator | `backend/services/grpc_tool_service.py` |
| `ServiceContainer` | Global singleton initialization | `backend/services/service_container.py` |

### Current Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Document Upload                       │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              DocumentProcessor                           │
│              (Extract chunks)                            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│          EmbeddingManager (Backend)                      │
│          - Generate embeddings (OpenAI)                  │
│          - Store in Qdrant                               │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  Qdrant Vector DB                        │
└─────────────────────────────────────────────────────────┘
```

**Search Flow:**

```
┌─────────────────────────────────────────────────────────┐
│              User Search Query                           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│          EmbeddingManager (Backend)                      │
│          - Generate query embedding                      │
│          - Search Qdrant                                 │
│          - Filter results                                │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Format and Return Results                   │
└─────────────────────────────────────────────────────────┘
```

## Proposed Architecture

### New Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Document Upload                       │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              DocumentProcessor (Backend)                 │
│              (Extract chunks)                            │
└───────────────────────┬─────────────────────────────────┘
                        │ gRPC (port 50053)
                        ▼
┌─────────────────────────────────────────────────────────┐
│          Vector Service (Microservice)                   │
│          - Generate embeddings (OpenAI)                  │
│          - Manage Qdrant collections                     │
│          - Store/update/delete vectors                   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Qdrant Vector DB (in vector-service)        │
└─────────────────────────────────────────────────────────┘
```

**Search Flow:**

```
┌─────────────────────────────────────────────────────────┐
│         User Search Query (via Backend or Orchestrator)  │
└───────────────────────┬─────────────────────────────────┘
                        │ gRPC (port 50053)
                        ▼
┌─────────────────────────────────────────────────────────┐
│               Vector Service                             │
│               - Generate query embedding                 │
│               - Search Qdrant                            │
│               - Apply filters                            │
│               - Return ranked results                    │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Format and Return Results                   │
└─────────────────────────────────────────────────────────┘
```

### Service Communication Pattern

```
┌──────────────┐           ┌──────────────┐
│   Backend    │──gRPC────▶│Vector Service│
│   (50052)    │           │   (50053)    │
└──────────────┘           └──────┬───────┘
                                  │
                                  ├─▶ Qdrant
                                  ├─▶ OpenAI API
                                  └─▶ Cache

┌──────────────┐
│LLM Orchestr. │──gRPC────▶│Vector Service│
│   (50051)    │           │   (50053)    │
└──────────────┘           └──────────────┘
```

## Proto File Design

Create `/opt/bastion/protos/vector_service.proto`:

```protobuf
syntax = "proto3";

package vector_service;

// Vector Service - Handles embedding generation and vector search
service VectorService {
  // Embedding Generation
  rpc GenerateEmbedding(EmbeddingRequest) returns (EmbeddingResponse);
  rpc GenerateBatchEmbeddings(BatchEmbeddingRequest) returns (BatchEmbeddingResponse);
  
  // Document Storage
  rpc StoreDocumentChunks(StoreChunksRequest) returns (StoreChunksResponse);
  rpc UpdateDocumentChunks(UpdateChunksRequest) returns (UpdateChunksResponse);
  rpc DeleteDocumentChunks(DeleteChunksRequest) returns (DeleteChunksResponse);
  
  // Search Operations
  rpc SearchSimilar(SearchRequest) returns (SearchResponse);
  
  // Collection Management
  rpc EnsureUserCollection(EnsureCollectionRequest) returns (EnsureCollectionResponse);
  rpc DeleteUserCollection(DeleteCollectionRequest) returns (DeleteCollectionResponse);
  rpc GetCollectionInfo(CollectionInfoRequest) returns (CollectionInfoResponse);
  
  // Cache Management
  rpc ClearEmbeddingCache(ClearCacheRequest) returns (ClearCacheResponse);
  rpc GetCacheStats(CacheStatsRequest) returns (CacheStatsResponse);
  
  // Health Check
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}

// ============================================================================
// Embedding Generation Messages
// ============================================================================

message EmbeddingRequest {
  string text = 1;
  string model = 2;  // Optional, defaults to service config
}

message EmbeddingResponse {
  repeated float embedding = 1;
  int32 token_count = 2;
  string model = 3;
}

message BatchEmbeddingRequest {
  repeated string texts = 1;
  string model = 2;  // Optional
  int32 batch_size = 3;  // Optional, for parallel processing
}

message BatchEmbeddingResponse {
  repeated EmbeddingVector embeddings = 1;
  int32 total_tokens = 2;
  string model = 3;
}

message EmbeddingVector {
  repeated float vector = 1;
  int32 index = 2;
  int32 token_count = 3;
}

// ============================================================================
// Document Storage Messages
// ============================================================================

message StoreChunksRequest {
  repeated DocumentChunk chunks = 1;
  string user_id = 2;  // Empty = global collection
  string document_id = 3;
  map<string, string> document_metadata = 4;  // category, tags, title, author, etc.
}

message DocumentChunk {
  string chunk_id = 1;
  string content = 2;
  int32 chunk_index = 3;
  float quality_score = 4;
  string method = 5;  // chunking method used
  map<string, string> metadata = 6;
}

message StoreChunksResponse {
  bool success = 1;
  int32 chunks_stored = 2;
  repeated string chunk_ids = 3;
  string collection_name = 4;
  string error = 5;
}

message UpdateChunksRequest {
  string document_id = 1;
  repeated DocumentChunk updated_chunks = 2;
  string user_id = 3;
}

message UpdateChunksResponse {
  bool success = 1;
  int32 chunks_updated = 2;
  string error = 3;
}

message DeleteChunksRequest {
  string document_id = 1;
  string user_id = 2;  // Empty = global collection
}

message DeleteChunksResponse {
  bool success = 1;
  int32 chunks_deleted = 2;
  string error = 3;
}

// ============================================================================
// Search Messages
// ============================================================================

message SearchRequest {
  string query = 1;
  int32 limit = 2;
  float score_threshold = 3;
  string user_id = 4;  // Empty = global only, "*" = all collections
  
  // Filters
  string filter_category = 5;
  repeated string filter_tags = 6;
  string document_type = 7;
  int64 date_from = 8;  // Unix timestamp
  int64 date_to = 9;    // Unix timestamp
  
  // Collection targeting
  string collection_name = 10;  // Optional: target specific collection
}

message SearchResponse {
  bool success = 1;
  repeated SearchResult results = 2;
  int32 total_results = 3;
  SearchMetadata metadata = 4;
  string error = 5;
}

message SearchResult {
  string chunk_id = 1;
  string document_id = 2;
  string content = 3;
  float score = 4;
  int32 chunk_index = 5;
  map<string, string> metadata = 6;
  string collection_name = 7;
}

message SearchMetadata {
  int32 queries_executed = 1;
  int32 total_candidates = 2;
  float search_duration_ms = 3;
  bool embedding_from_cache = 4;
  int32 vectors_searched = 5;
}

// ============================================================================
// Collection Management Messages
// ============================================================================

message EnsureCollectionRequest {
  string user_id = 1;
  string collection_type = 2;  // "documents", "videos", etc.
}

message EnsureCollectionResponse {
  bool success = 1;
  string collection_name = 2;
  bool already_existed = 3;
  string error = 4;
}

message DeleteCollectionRequest {
  string user_id = 1;
  string collection_type = 2;
}

message DeleteCollectionResponse {
  bool success = 1;
  string collection_name = 2;
  string error = 3;
}

message CollectionInfoRequest {
  string collection_name = 1;
}

message CollectionInfoResponse {
  bool exists = 1;
  int64 vector_count = 2;
  int32 dimension = 3;
  string distance_metric = 4;
  map<string, string> metadata = 5;
}

// ============================================================================
// Cache Management Messages
// ============================================================================

message ClearCacheRequest {
  bool clear_embeddings = 1;  // Clear embedding cache
  string content_hash = 2;     // Optional: clear specific hash
}

message ClearCacheResponse {
  bool success = 1;
  int32 entries_cleared = 2;
  string error = 3;
}

message CacheStatsRequest {}

message CacheStatsResponse {
  int64 embedding_cache_size = 1;
  int64 embedding_cache_hits = 2;
  int64 embedding_cache_misses = 3;
  float cache_hit_rate = 4;
  int32 ttl_seconds = 5;
}

// ============================================================================
// Health Check Messages
// ============================================================================

message HealthCheckRequest {}

message HealthCheckResponse {
  string status = 1;  // "healthy", "degraded", "unhealthy"
  bool openai_available = 2;
  bool qdrant_available = 3;
  int32 active_collections = 4;
  string service_version = 5;
  map<string, string> details = 6;
}
```

## Service Implementation Structure

```
vector-service/
├── service/
│   ├── __init__.py
│   ├── grpc_service.py          # gRPC service implementation
│   ├── embedding_engine.py      # OpenAI embedding generation
│   ├── vector_store.py          # Qdrant operations
│   ├── embedding_cache.py       # Hash-based embedding cache (3-hour TTL)
│   └── parallel_processor.py   # Concurrent request handling
├── config/
│   └── settings.py              # Service configuration
├── models/
│   └── models.py                # Pydantic models
├── utils/
│   ├── token_counter.py         # Token management
│   └── text_processor.py       # Text truncation/preprocessing
├── Dockerfile
├── requirements.txt
├── main.py
└── README.md
```

### Embedding Cache Design

The embedding cache is a critical performance optimization:

```python
# vector-service/service/embedding_cache.py
import hashlib
import time
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Hash-based embedding cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 10800):  # 3 hours
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple[List[float], float]] = {}  # hash -> (embedding, timestamp)
        self.hits = 0
        self.misses = 0
    
    async def initialize(self):
        """Initialize cache"""
        logger.info(f"Embedding cache initialized with {self.ttl_seconds}s TTL")
    
    def hash_text(self, text: str) -> str:
        """Generate stable hash for text content"""
        # Use SHA256 for consistent hashing
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    async def get(self, content_hash: str) -> Optional[List[float]]:
        """Get embedding from cache if not expired"""
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
        self.cache[content_hash] = (embedding, time.time())
        logger.debug(f"Cached embedding: {content_hash[:16]}...")
    
    async def clear(self, content_hash: Optional[str] = None) -> int:
        """Clear cache (all or specific hash)"""
        if content_hash:
            if content_hash in self.cache:
                del self.cache[content_hash]
                return 1
            return 0
        else:
            count = len(self.cache)
            self.cache.clear()
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
            'ttl_seconds': self.ttl_seconds
        }
    
    async def cleanup_expired(self):
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
```

**Why this design:**
- **Content-based hashing:** Same text always produces same hash, maximizing cache hits
- **In-memory storage:** Fast access (microseconds), no external dependencies
- **TTL-based expiration:** Prevents stale embeddings from persisting
- **Simple eviction:** Lazy cleanup on access + periodic background cleanup
- **Observable:** Hit/miss metrics for monitoring cache effectiveness

### Key Implementation Files

#### `vector-service/service/grpc_service.py`

```python
import grpc
import logging
from concurrent import futures
from protos import vector_service_pb2, vector_service_pb2_grpc
from service.embedding_engine import EmbeddingEngine
from service.vector_store import VectorStore
from service.query_expander import QueryExpander

logger = logging.getLogger(__name__)

class VectorServiceImplementation(vector_service_pb2_grpc.VectorServiceServicer):
    """Vector service gRPC implementation"""
    
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.embedding_cache = EmbeddingCache(ttl_seconds=10800)  # 3-hour TTL
    
    async def initialize(self):
        """Initialize all components"""
        await self.embedding_engine.initialize()
        await self.vector_store.initialize()
        await self.embedding_cache.initialize()
        logger.info("Vector Service initialized")
    
    async def GenerateEmbedding(self, request, context):
        """Generate single embedding with cache lookup"""
        try:
            # Check cache first
            content_hash = self.embedding_cache.hash_text(request.text)
            cached_embedding = await self.embedding_cache.get(content_hash)
            
            if cached_embedding:
                logger.debug(f"Cache hit for embedding: {content_hash}")
                return vector_service_pb2.EmbeddingResponse(
                    embedding=cached_embedding,
                    token_count=len(request.text.split()),
                    model=request.model or "text-embedding-3-small"
                )
            
            # Cache miss - generate embedding
            embedding = await self.embedding_engine.generate_embedding(request.text)
            
            # Store in cache
            await self.embedding_cache.set(content_hash, embedding)
            
            return vector_service_pb2.EmbeddingResponse(
                embedding=embedding,
                token_count=len(request.text.split()),
                model=request.model or "text-embedding-3-small"
            )
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vector_service_pb2.EmbeddingResponse()
    
    async def GenerateBatchEmbeddings(self, request, context):
        """Generate batch embeddings with parallel processing and cache lookup"""
        try:
            # Check cache for each text
            texts = list(request.texts)
            embeddings = []
            texts_to_generate = []
            text_indices = []
            
            for idx, text in enumerate(texts):
                content_hash = self.embedding_cache.hash_text(text)
                cached_embedding = await self.embedding_cache.get(content_hash)
                
                if cached_embedding:
                    embeddings.append((idx, cached_embedding))
                else:
                    texts_to_generate.append(text)
                    text_indices.append(idx)
            
            # Generate embeddings for cache misses
            if texts_to_generate:
                new_embeddings = await self.embedding_engine.generate_batch_embeddings(
                    texts=texts_to_generate,
                    batch_size=request.batch_size or 100
                )
                
                # Cache new embeddings and add to results
                for text, embedding, idx in zip(texts_to_generate, new_embeddings, text_indices):
                    content_hash = self.embedding_cache.hash_text(text)
                    await self.embedding_cache.set(content_hash, embedding)
                    embeddings.append((idx, embedding))
            
            # Sort by original index
            embeddings.sort(key=lambda x: x[0])
            sorted_embeddings = [emb for _, emb in embeddings]
            
            embedding_vectors = []
            for idx, embedding in enumerate(embeddings):
                embedding_vectors.append(
                    vector_service_pb2.EmbeddingVector(
                        vector=embedding,
                        index=idx,
                        token_count=len(request.texts[idx].split())
                    )
                )
            
            return vector_service_pb2.BatchEmbeddingResponse(
                embeddings=embedding_vectors,
                total_tokens=sum(len(t.split()) for t in request.texts),
                model=request.model or "text-embedding-3-small"
            )
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return vector_service_pb2.BatchEmbeddingResponse()
    
    async def StoreDocumentChunks(self, request, context):
        """Store document chunks with embeddings"""
        try:
            # Generate embeddings for all chunks
            texts = [chunk.content for chunk in request.chunks]
            embeddings = await self.embedding_engine.generate_batch_embeddings(texts)
            
            # Store in vector database
            chunk_ids = await self.vector_store.store_chunks(
                chunks=request.chunks,
                embeddings=embeddings,
                user_id=request.user_id,
                document_id=request.document_id,
                document_metadata=dict(request.document_metadata)
            )
            
            collection_name = self.vector_store.get_collection_name(request.user_id)
            
            return vector_service_pb2.StoreChunksResponse(
                success=True,
                chunks_stored=len(chunk_ids),
                chunk_ids=chunk_ids,
                collection_name=collection_name
            )
        except Exception as e:
            logger.error(f"Store chunks failed: {e}")
            return vector_service_pb2.StoreChunksResponse(
                success=False,
                error=str(e)
            )
    
    async def SearchSimilar(self, request, context):
        """Semantic similarity search"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_engine.generate_embedding(request.query)
            
            # Perform vector search
            results = await self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=request.limit,
                score_threshold=request.score_threshold,
                user_id=request.user_id,
                filter_category=request.filter_category or None,
                filter_tags=list(request.filter_tags) if request.filter_tags else None,
                collection_name=request.collection_name or None
            )
            
            # Convert to proto format
            search_results = []
            for result in results:
                search_results.append(
                    vector_service_pb2.SearchResult(
                        chunk_id=result['chunk_id'],
                        document_id=result['document_id'],
                        content=result['content'],
                        score=result['score'],
                        chunk_index=result.get('chunk_index', 0),
                        metadata=result.get('metadata', {}),
                        collection_name=result.get('collection_name', '')
                    )
                )
            
            metadata = vector_service_pb2.SearchMetadata(
                queries_executed=1,
                query_expansion_used=False,
                total_candidates=len(results)
            )
            
            return vector_service_pb2.SearchResponse(
                success=True,
                results=search_results,
                total_results=len(search_results),
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return vector_service_pb2.SearchResponse(
                success=False,
                error=str(e)
            )
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        try:
            openai_ok = await self.embedding_engine.health_check()
            qdrant_ok = await self.vector_store.health_check()
            collections = await self.vector_store.get_collection_count()
            
            status = "healthy" if (openai_ok and qdrant_ok) else "degraded"
            
            return vector_service_pb2.HealthCheckResponse(
                status=status,
                openai_available=openai_ok,
                qdrant_available=qdrant_ok,
                active_collections=collections,
                service_version="1.0.0"
            )
        except:
            return vector_service_pb2.HealthCheckResponse(
                status="unhealthy"
            )
```

#### `vector-service/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY vector-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared protos from root (CRITICAL)
COPY protos /app/protos

# Copy service code
COPY vector-service /app

# Generate gRPC code from protos
RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/protos/vector_service.proto

# Expose gRPC port
EXPOSE 50053

# Run service
CMD ["python", "main.py"]
```

#### `vector-service/requirements.txt`

```
grpcio==1.60.0
grpcio-tools==1.60.0
openai==1.6.1
qdrant-client==1.7.0
pydantic==2.5.0
python-dotenv==1.0.0
```

## Docker Compose Configuration

Add to `docker-compose.yml`:

```yaml
services:
  # ... existing services ...

  vector-service:
    build:
      context: .  # ROOT CONTEXT (access shared protos)
      dockerfile: ./vector-service/Dockerfile
    container_name: ${COMPOSE_PROJECT_NAME:-bastion}-vector-service
    depends_on:
      qdrant:
        condition: service_started
    environment:
      # Service Configuration
      - SERVICE_NAME=vector-service
      - GRPC_PORT=50053
      - LOG_LEVEL=INFO
      
      # OpenAI Configuration (for embeddings)
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_EMBEDDING_MODEL=text-embedding-3-small
      - OPENAI_MAX_RETRIES=3
      - OPENAI_TIMEOUT=30
      
      # Future: OpenRouter embedding support
      # - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      # - OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
      
      # Qdrant Configuration
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - QDRANT_URL=http://qdrant:6333
      - VECTOR_COLLECTION_NAME=documents
      - EMBEDDING_DIMENSIONS=1536
      
      # Performance Tuning
      - PARALLEL_WORKERS=4
      - BATCH_SIZE=100
      - MAX_TEXT_LENGTH=8000
      
      # Cache Configuration
      - EMBEDDING_CACHE_ENABLED=true
      - EMBEDDING_CACHE_TTL=10800  # 3 hours
      - CACHE_CLEANUP_INTERVAL=3600  # Cleanup every hour
    ports:
      - "50053:50053"  # gRPC port
    networks:
      - default
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "python -c 'import socket; s = socket.socket(); s.settimeout(5); s.connect((\"localhost\", 50053)); s.close()' || exit 1"]
      interval: 30s
      timeout: 10s
      start_period: 40s
      retries: 3

  # Update qdrant service to expose ports for vector-service
  qdrant:
    # ... existing config ...
    # Ensure it's accessible from vector-service container
```

## Client Implementation

### Backend Client

Create `backend/clients/vector_service_client.py`:

```python
import grpc
import os
import logging
from typing import List, Dict, Any, Optional
from protos import vector_service_pb2, vector_service_pb2_grpc

logger = logging.getLogger(__name__)

class VectorServiceClient:
    """Client for calling vector service gRPC API"""
    
    def __init__(self):
        host = os.getenv('VECTOR_SERVICE_HOST', 'vector-service')
        port = os.getenv('VECTOR_SERVICE_PORT', '50053')
        self.channel = grpc.aio.insecure_channel(f'{host}:{port}')
        self.stub = vector_service_pb2_grpc.VectorServiceStub(self.channel)
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate single embedding"""
        try:
            request = vector_service_pb2.EmbeddingRequest(text=text)
            response = await self.stub.GenerateEmbedding(request)
            return list(response.embedding)
        except grpc.RpcError as e:
            logger.error(f"Generate embedding failed: {e.code()} - {e.details()}")
            raise
    
    async def generate_batch_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate batch embeddings"""
        try:
            request = vector_service_pb2.BatchEmbeddingRequest(
                texts=texts,
                batch_size=batch_size
            )
            response = await self.stub.GenerateBatchEmbeddings(request)
            return [list(emb.vector) for emb in response.embeddings]
        except grpc.RpcError as e:
            logger.error(f"Batch embeddings failed: {e.code()} - {e.details()}")
            raise
    
    async def store_document_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        document_id: str,
        user_id: Optional[str] = None,
        document_metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Store document chunks with embeddings"""
        try:
            # Convert chunks to proto format
            proto_chunks = []
            for chunk in chunks:
                proto_chunks.append(
                    vector_service_pb2.DocumentChunk(
                        chunk_id=chunk['chunk_id'],
                        content=chunk['content'],
                        chunk_index=chunk.get('chunk_index', 0),
                        quality_score=chunk.get('quality_score', 1.0),
                        method=chunk.get('method', 'unknown'),
                        metadata=chunk.get('metadata', {})
                    )
                )
            
            request = vector_service_pb2.StoreChunksRequest(
                chunks=proto_chunks,
                user_id=user_id or '',
                document_id=document_id,
                document_metadata=document_metadata or {}
            )
            
            response = await self.stub.StoreDocumentChunks(request)
            
            return {
                'success': response.success,
                'chunks_stored': response.chunks_stored,
                'chunk_ids': list(response.chunk_ids),
                'collection_name': response.collection_name,
                'error': response.error if response.error else None
            }
        except grpc.RpcError as e:
            logger.error(f"Store chunks failed: {e.code()} - {e.details()}")
            raise
    
    async def search_similar(
        self,
        query: str,
        limit: int = 20,
        score_threshold: float = 0.3,
        user_id: Optional[str] = None,
        filter_category: Optional[str] = None,
        filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            request = vector_service_pb2.SearchRequest(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                user_id=user_id or '',
                filter_category=filter_category or '',
                filter_tags=filter_tags or []
            )
            
            response = await self.stub.SearchSimilar(request)
            
            if not response.success:
                raise Exception(response.error)
            
            # Convert to dict format
            results = []
            for result in response.results:
                results.append({
                    'chunk_id': result.chunk_id,
                    'document_id': result.document_id,
                    'content': result.content,
                    'score': result.score,
                    'chunk_index': result.chunk_index,
                    'metadata': dict(result.metadata),
                    'collection_name': result.collection_name
                })
            
            return results
        except grpc.RpcError as e:
            logger.error(f"Search failed: {e.code()} - {e.details()}")
            raise
    
    async def delete_document_chunks(self, document_id: str, user_id: Optional[str] = None):
        """Delete document embeddings"""
        try:
            request = vector_service_pb2.DeleteChunksRequest(
                document_id=document_id,
                user_id=user_id or ''
            )
            response = await self.stub.DeleteDocumentChunks(request)
            return response.success
        except grpc.RpcError as e:
            logger.error(f"Delete chunks failed: {e.code()} - {e.details()}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check vector service health"""
        try:
            request = vector_service_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(request)
            return {
                'status': response.status,
                'openai_available': response.openai_available,
                'qdrant_available': response.qdrant_available,
                'active_collections': response.active_collections,
                'service_version': response.service_version
            }
        except:
            return {'status': 'unhealthy'}
```

### Update Backend Services

Replace `EmbeddingManager` calls with `VectorServiceClient`:

```python
# OLD: backend/services/document_service_v2.py
self.embedding_manager = ParallelEmbeddingManager()
await self.embedding_manager.initialize()
await self.embedding_manager.embed_and_store_chunks(result.chunks, user_id=user_id)

# NEW: backend/services/document_service_v2.py
from clients.vector_service_client import VectorServiceClient

self.vector_client = VectorServiceClient()
await self.vector_client.store_document_chunks(
    chunks=result.chunks,
    document_id=document_id,
    user_id=user_id,
    document_metadata={
        'category': document_category,
        'tags': ','.join(document_tags) if document_tags else '',
        'title': document_title,
        'author': document_author
    }
)
```

## Migration Strategy

### Phase 1: Build and Test Vector Service (Week 1)

1. **Create vector-service directory structure**
2. **Implement proto file** (`protos/vector_service.proto`)
3. **Build core service** (embedding engine, vector store)
4. **Create Dockerfile and docker-compose entry**
5. **Test in isolation** (unit tests, health checks)

**Success Criteria:**
- Vector service starts successfully
- Health check returns healthy status
- Can generate embeddings via gRPC
- Can store and search vectors

### Phase 2: Parallel Deployment (Week 2)

1. **Deploy vector service alongside backend** (both systems running)
2. **Add vector service client to backend**
3. **Create feature flag** for vector service usage
4. **No changes to existing flows** (still using EmbeddingManager)

**Success Criteria:**
- Vector service runs in production
- Backend can connect to vector service
- No disruption to existing functionality

### Phase 3: Gradual Migration (Week 3-4)

**Migrate one service at a time:**

1. **Start with DirectSearchService** (simplest, read-only)
   - Update to use `VectorServiceClient`
   - Feature flag: `USE_VECTOR_SERVICE_FOR_SEARCH=true`
   - Monitor performance and accuracy
   - Rollback capability via feature flag

2. **Migrate DocumentService** (write operations)
   - Update upload/edit flows
   - Feature flag: `USE_VECTOR_SERVICE_FOR_STORAGE=true`
   - Dual-write initially (both systems)
   - Verify data consistency

3. **Migrate remaining services:**
   - ChatService
   - UnifiedSearchTools
   - VideoSearchService
   - gRPC Tool Service

**Success Criteria:**
- Each service migrated individually
- Feature flags allow rollback
- Performance metrics meet or exceed baseline
- No data loss or corruption

### Phase 4: Full Cutover (Week 5)

1. **All services using vector service**
2. **Remove EmbeddingManager from backend**
3. **Clean up old code**
4. **Update documentation**

**Success Criteria:**
- 100% of vector operations via vector service
- No EmbeddingManager code in backend
- Documentation updated

### Phase 5: Optimization (Week 6)

1. **Performance tuning**
2. **Add monitoring dashboards**
3. **Implement advanced features** (batch operations, caching)
4. **Scale testing**

## Rollback Plan

At any point during migration:

1. **Set feature flags to `false`:**
   ```yaml
   - USE_VECTOR_SERVICE_FOR_SEARCH=false
   - USE_VECTOR_SERVICE_FOR_STORAGE=false
   ```

2. **Restart backend:**
   ```bash
   docker compose restart backend
   ```

3. **System reverts to EmbeddingManager**

4. **Vector service can be stopped** (no impact):
   ```bash
   docker compose stop vector-service
   ```

## Testing Strategy

### Unit Tests

```python
# test_vector_service.py
async def test_generate_embedding():
    client = VectorServiceClient()
    embedding = await client.generate_embedding("test text")
    assert len(embedding) == 1536
    assert all(isinstance(x, float) for x in embedding)

async def test_store_and_search():
    client = VectorServiceClient()
    
    # Store test chunks
    chunks = [{'chunk_id': 'test1', 'content': 'test content'}]
    result = await client.store_document_chunks(
        chunks=chunks,
        document_id='test_doc'
    )
    assert result['success']
    
    # Search for chunks
    results = await client.search_similar('test content', limit=1)
    assert len(results) > 0
    assert results[0]['document_id'] == 'test_doc'
```

### Integration Tests

```bash
# Test full document upload flow
curl -X POST http://localhost:8081/api/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test.pdf"

# Verify vectors stored in vector service
docker compose logs vector-service | grep "Stored.*chunks"

# Test search flow
curl -X POST http://localhost:8081/api/search/direct \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "test content"}'
```

### Performance Benchmarks

Compare before/after metrics:

| Metric | EmbeddingManager (Baseline) | Vector Service (Target) |
|--------|---------------------------|-------------------------|
| Upload time (10 docs) | X seconds | ≤ X seconds |
| Search latency (p50) | Y ms | ≤ Y ms |
| Search latency (p95) | Z ms | ≤ Z * 1.1 ms |
| Memory usage (backend) | A MB | ≤ A * 0.8 MB |
| Concurrent uploads | N | ≥ N |

## Monitoring and Observability

### Vector Service Metrics

Add Prometheus metrics:

```python
# vector-service/service/metrics.py
from prometheus_client import Counter, Histogram, Gauge

embeddings_generated = Counter('embeddings_generated_total', 'Total embeddings generated')
embedding_duration = Histogram('embedding_duration_seconds', 'Embedding generation duration')
search_duration = Histogram('search_duration_seconds', 'Search duration')
active_collections = Gauge('active_collections', 'Number of active collections')
```

### Logging

Structured logging for key operations:

```python
logger.info("Embedding generated", extra={
    'text_length': len(text),
    'token_count': token_count,
    'duration_ms': duration
})

logger.info("Search completed", extra={
    'query': query,
    'results_count': len(results),
    'score_threshold': threshold,
    'duration_ms': duration
})
```

### Alerts

Set up alerts for:

1. **Service health degradation** (OpenAI or Qdrant unavailable)
2. **High latency** (p95 > threshold)
3. **Error rate** (> 1% failures)
4. **Resource exhaustion** (memory/CPU > 80%)

## Benefits Summary

### Architectural Benefits

1. **Separation of Concerns** - Vector operations isolated from business logic
2. **Reusability** - Multiple services can use vector capabilities
3. **Independent Scaling** - Scale vector service based on embedding load
4. **Technology Flexibility** - Can swap embedding models/providers without touching backend
5. **Simplified Testing** - Mock vector service for backend tests

### Performance Benefits

1. **Resource Isolation** - Embedding operations don't block backend
2. **Parallel Processing** - Dedicated workers for embedding generation
3. **Reduced Memory** - Backend doesn't load embedding models
4. **Better Caching** - Centralized cache for embeddings/searches

### Operational Benefits

1. **Clear Monitoring** - Dedicated metrics for vector operations
2. **Independent Deployment** - Update vector service without backend restart
3. **Failure Isolation** - Vector service issues don't crash backend
4. **Easier Debugging** - Centralized logs for all vector operations

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Network latency (gRPC calls) | Medium | Use batch operations, local caching |
| Service unavailability | High | Implement circuit breakers, fallback logic |
| Data inconsistency | High | Dual-write during migration, verification scripts |
| Performance degradation | Medium | Extensive benchmarking before cutover |
| Migration complexity | Medium | Phased approach with feature flags |

## Success Metrics

Migration is successful when:

1. ✅ Vector service handles 100% of embedding/search operations
2. ✅ Performance meets or exceeds baseline (p95 latency)
3. ✅ Zero data loss during migration
4. ✅ Backend memory usage reduced by 20%+
5. ✅ All services migrated and feature flags removed
6. ✅ Monitoring dashboards show healthy status
7. ✅ Documentation updated and team trained

## Timeline

| Week | Phase | Activities |
|------|-------|-----------|
| 1 | Build | Create vector service, implement core functionality |
| 2 | Deploy | Parallel deployment, integration testing |
| 3-4 | Migrate | Gradual migration service by service |
| 5 | Cutover | Full cutover, remove old code |
| 6 | Optimize | Performance tuning, monitoring setup |

**Total estimated time:** 6 weeks

## Next Steps

1. **Review this plan** with team
2. **Create vector-service directory** with initial structure
3. **Implement proto file** and get feedback
4. **Build Phase 1** (isolated service)
5. **Test Phase 1** before proceeding to Phase 2

## Decisions Made

### Architecture Decisions

1. **Video Capabilities:** ✅ REMOVE - All video embedding/search code will be removed. Can be re-added post-refactor if needed.

2. **Query Expansion:** ✅ NOT IN VECTOR SERVICE - Query expansion is an LLM tool function and stays in backend/orchestrator as a tool.

3. **Embedding Models:** ✅ OpenAI ONLY - Start with OpenAI. OpenRouter embedding support will be added when available.

4. **Qdrant Connection:** ✅ SHARED CONNECTION - Vector service uses same Qdrant connection string as backend (external infrastructure).

5. **Collection Management:** ✅ KEEP CURRENT PATTERN - Continue with user-specific (`user_{user_id}_documents`) and global (`documents`) collections.

6. **Caching Strategy:** ✅ ADD EMBEDDING CACHE - Hash-based embedding cache with 3-hour TTL for performance.

7. **Data Migration:** ✅ NOT NEEDED - Vectors stay in Qdrant. We're migrating the *mechanism*, not the data.

8. **Parallel Processing:** ✅ REQUIRED - Vector service must handle concurrent embedding requests efficiently.

### Scope Clarification

**What Vector Service Does:**
- Generate embeddings (single & batch)
- Store embeddings in Qdrant
- Search Qdrant for similar vectors
- Manage collections (create, ensure exists)
- Delete embeddings
- Cache embeddings by content hash (3-hour TTL)
- Handle concurrent requests efficiently

**What Vector Service Does NOT Do:**
- Query expansion (that's an LLM tool in backend)
- Video embeddings (removed from system)
- LLM reasoning or chat
- Document processing or chunking
- Business logic or filtering

### Updated Service Interface

Vector Service is a **pure vector operations service** - embeddings in, embeddings out.

---

**Document prepared by:** Claude (AI Assistant)  
**Review Status:** ✅ Approved with clarifications  
**Next Steps:** Remove video code, update proto file, begin Phase 1 implementation

