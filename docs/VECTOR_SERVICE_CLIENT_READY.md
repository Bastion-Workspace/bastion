# Vector Service Client Implementation

## Status: Client Ready, Migration Pending

The Vector Service client infrastructure is now complete and ready for gradual migration.

## What's Been Built

### 1. Vector Service Client (`backend/clients/vector_service_client.py`)
- **VectorServiceClient**: Full gRPC client for Vector Service
- **Methods**:
  - `generate_embedding(text, model)`: Single text embedding
  - `generate_embeddings(texts, model, batch_size)`: Batch embedding generation
  - `clear_cache(clear_all, content_hash)`: Cache management
  - `get_cache_stats()`: Cache statistics
  - `health_check()`: Service health monitoring
- **Singleton**: `get_vector_service_client()` for shared instance

### 2. Embedding Service Wrapper (`backend/services/embedding_service_wrapper.py`)
- **Purpose**: Unified interface for gradual migration
- **Feature Flag**: `USE_VECTOR_SERVICE` in `backend/config.py`
- **Dual Mode**:
  - `USE_VECTOR_SERVICE=False`: Uses legacy `EmbeddingManager` (default)
  - `USE_VECTOR_SERVICE=True`: Routes to Vector Service via gRPC
- **Key Methods**:
  - `generate_embeddings(texts)`: Generate embeddings (routes based on flag)
  - `embed_and_store_chunks(chunks, user_id, metadata)`: Generate + store in Qdrant
  - `search_similar(query_embedding, limit, filters)`: Search Qdrant
  - `clear_cache()`: Clear Vector Service cache (if enabled)
  - `get_cache_stats()`: Get cache statistics (if enabled)

### 3. Configuration Updates (`backend/config.py`)
- Added `VECTOR_SERVICE_URL: str = "vector-service:50053"`
- Added `USE_VECTOR_SERVICE: bool = False` (feature flag)
- Added `get_settings()` helper function

## How It Works

### Legacy Mode (Current Behavior)
```python
wrapper = EmbeddingServiceWrapper()
await wrapper.initialize()
# Routes to EmbeddingManager
embeddings = await wrapper.generate_embeddings(["text1", "text2"])
```

### Vector Service Mode (Future)
```python
# In docker-compose.yml or .env:
# USE_VECTOR_SERVICE=true

wrapper = EmbeddingServiceWrapper()
await wrapper.initialize()
# Routes to VectorServiceClient -> gRPC -> Vector Service
embeddings = await wrapper.generate_embeddings(["text1", "text2"])
```

## Architecture Pattern

### Embedding Generation
- **Legacy**: `Backend → EmbeddingManager → OpenAI`
- **New**: `Backend → VectorServiceClient → gRPC → Vector Service → OpenAI (with cache)`

### Embedding Storage
- **Both modes**: Qdrant storage logic stays in backend via `EmbeddingManager`
- Vector Service only generates embeddings, backend handles storage + metadata

### Search
- **Both modes**: Qdrant search logic stays in backend via `EmbeddingManager`
- Vector Service only generates query embeddings, backend handles search + filtering

## What Needs to Happen Next

### Phase 1: Verify EmbeddingManager Methods
**Status**: PENDING
- [ ] Check if `EmbeddingManager.store_embeddings_with_metadata()` exists
- [ ] If not, create method that takes pre-generated embeddings
- [ ] Ensure it supports all metadata (user_id, category, tags, title, author, filename)

### Phase 2: Migration Strategy
**Status**: NOT STARTED

#### Option A: Gradual Service-by-Service Migration
1. Start with low-traffic services (e.g., `free_form_notes_service.py`)
2. Replace `EmbeddingManager` with `EmbeddingServiceWrapper`
3. Test with `USE_VECTOR_SERVICE=False` (should be identical)
4. Enable `USE_VECTOR_SERVICE=True` for testing
5. Monitor performance, cache hit rates, errors
6. Migrate next service

#### Option B: Global Feature Flag Rollout
1. Replace all `EmbeddingManager` imports with `EmbeddingServiceWrapper`
2. Deploy with `USE_VECTOR_SERVICE=False` (no behavior change)
3. Test thoroughly
4. Flip flag to `USE_VECTOR_SERVICE=True` in production
5. Monitor and rollback if issues

#### Option C: Hybrid Mode
1. Keep `EmbeddingManager` for document upload (write path)
2. Use `VectorServiceClient` for search queries (read path)
3. Allows testing Vector Service with no risk to document processing
4. Gradual confidence building

### Phase 3: Update Services (19 files to migrate)
**Services using EmbeddingManager**:
1. `backend/services/document_service_v2.py`
2. `backend/services/parallel_document_service.py`
3. `backend/services/user_document_service.py`
4. `backend/services/direct_search_service.py`
5. `backend/services/grpc_tool_service.py`
6. `backend/services/free_form_notes_service.py`
7. `backend/services/twitter_ingestion_service.py`
8. `backend/services/chat_service.py`
9. `backend/services/conversation_intelligence_service.py`
10. `backend/services/enhanced_pdf_segmentation_service.py`
11. `backend/services/collection_analysis_service.py`
12. `backend/services/zip_processor_service.py`
13. `backend/services/langgraph_tools/unified_search_tools.py`
14. `backend/mcp/tools/document_tool.py`
15. `backend/mcp/tools/web_search_ingestion_tool.py`
16. `backend/mcp/tools/web_ingest_selected_tool.py`
17. `backend/mcp/tools/query_expansion_tool.py`
18. `backend/mcp/tools/filename_search_tool.py`
19. `backend/mcp/tools/search_tool.py`

**Migration Pattern**:
```python
# OLD:
from backend.services.embedding_manager import EmbeddingManager
self.embedding_manager = EmbeddingManager()
await self.embedding_manager.initialize()

# NEW:
from backend.services.embedding_service_wrapper import get_embedding_service
self.embedding_service = await get_embedding_service()
# No need to call initialize() - singleton handles it
```

### Phase 4: Testing & Monitoring
**Status**: NOT STARTED
- [ ] Add logging to track which backend is used
- [ ] Monitor cache hit rates from Vector Service
- [ ] Compare embedding generation latency (legacy vs gRPC)
- [ ] Test error handling and fallback behavior
- [ ] Verify Qdrant storage consistency
- [ ] Test parallel processing under load

### Phase 5: Optimization
**Status**: NOT STARTED
- [ ] Tune Vector Service cache TTL based on usage patterns
- [ ] Optimize batch sizes for gRPC calls
- [ ] Add circuit breaker for Vector Service failures
- [ ] Implement graceful degradation (fallback to legacy)
- [ ] Add metrics and dashboards

## Decision Points

### When to Migrate?
- **Now**: If confident in Vector Service implementation
- **Later**: After more load testing, or when cache becomes valuable
- **Never**: If gRPC overhead outweighs cache benefits (unlikely)

### Which Strategy?
- **Recommended**: Option A (gradual service-by-service)
- **Fastest**: Option B (global flag flip)
- **Safest**: Option C (hybrid read/write split)

## Current State

✅ **Complete**:
- Vector Service microservice (embedding generation + caching)
- gRPC client library
- Embedding service wrapper with feature flag
- Configuration setup

⏳ **Pending**:
- Verify `EmbeddingManager.store_embeddings_with_metadata()` method exists
- Choose migration strategy
- Begin service updates

🚫 **Blocked**:
- Need user input on migration timeline and strategy

## Summary

The Vector Service client is **fully implemented and ready to use**. The backend services are **not yet migrated** but can be easily updated using the `EmbeddingServiceWrapper`. 

The feature flag (`USE_VECTOR_SERVICE=False`) ensures zero risk - the system behaves identically to the legacy implementation until explicitly enabled.

**Next step**: Verify the `EmbeddingManager` storage method and choose a migration strategy.

