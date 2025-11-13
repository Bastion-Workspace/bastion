# Vector Service - Implementation Ready Summary

**Status:** ✅ Plan Approved & Updated  
**Date:** November 10, 2025

## What's Been Decided

### Core Architecture

**Vector Service (Port 50053):**
- Pure embedding operations service
- No LLM reasoning, no query expansion, no video
- Handles concurrent requests efficiently
- Hash-based embedding cache (3-hour TTL)
- Connects to existing Qdrant infrastructure

### Scope - What It Does

✅ **Generate embeddings** (single & batch, with OpenAI)  
✅ **Store embeddings** in Qdrant collections  
✅ **Search** Qdrant for similar vectors  
✅ **Manage collections** (user-specific + global)  
✅ **Delete embeddings** by document ID  
✅ **Cache embeddings** by content hash (3-hour TTL)  
✅ **Handle concurrent requests** efficiently  

### Scope - What It Doesn't Do

❌ Query expansion (that's an LLM tool in backend)  
❌ Video embeddings (removed from system)  
❌ LLM reasoning or chat  
❌ Document processing or chunking  
❌ Business logic or filtering  

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Video capabilities** | Remove entirely | Can re-add post-refactor if needed |
| **Query expansion** | NOT in vector service | It's an LLM tool function |
| **Embedding models** | OpenAI only (for now) | OpenRouter embeddings when available |
| **Qdrant connection** | Shared connection string | No isolation needed |
| **Collections** | Keep user/global pattern | Current system depends on it |
| **Caching** | Hash-based, 3-hour TTL | Performance optimization |
| **Data migration** | NOT needed | Migrating mechanism, not data |
| **Parallel processing** | Required | Vector service handles concurrent requests |

## What's Ready

### 1. Complete Proto File ✅

`/opt/bastion/protos/vector_service.proto` is fully specified:

- **Embedding Generation:** Single & batch with cache support
- **Document Storage:** Store/update/delete chunks
- **Search:** Semantic similarity search
- **Collection Management:** User/global collections
- **Cache Management:** Clear cache, get stats
- **Health Checks:** OpenAI & Qdrant status

### 2. Implementation Plan ✅

`/opt/bastion/docs/VECTOR_SERVICE_MIGRATION_PLAN.md` includes:

- Current architecture analysis
- Proposed microservice design
- Service structure and file layout
- Complete implementation examples
- Docker configuration
- Migration strategy (6 phases)
- Testing strategy
- Monitoring & observability
- Rollback plan

### 3. Embedding Cache Design ✅

Full implementation with:
- SHA256 content hashing
- In-memory storage (fast access)
- 3-hour TTL with lazy + periodic cleanup
- Hit/miss metrics
- Observable performance

### 4. Docker Integration ✅

Ready for `docker-compose.yml`:
- Port 50053 allocated
- Environment variables defined
- Health check configured
- Depends on Qdrant

### 5. Client Library Spec ✅

Backend client implementation pattern:
- `VectorServiceClient` class
- Async gRPC calls
- Retry logic
- Error handling

## Services to Migrate

**9 services** currently use `EmbeddingManager`:

| Priority | Service | Complexity | Notes |
|----------|---------|------------|-------|
| 1 | `DirectSearchService` | Low | Read-only, simple migration |
| 2 | `DocumentService` | Medium | Write operations, critical path |
| 3 | `UserDocumentService` | Medium | User isolation logic |
| 4 | `ZipProcessorService` | Medium | Batch processing |
| 5 | `UnifiedSearchTools` | Medium | LangGraph agent integration |
| 6 | `ChatService` | High | Complex retrieval logic |
| 7 | `VideoSearchService` | N/A | **REMOVING** |
| 8 | `gRPC Tool Service` | Medium | Orchestrator integration |
| 9 | `ServiceContainer` | Low | Global singleton |

## Migration Strategy

### Phased Approach (Feature Flag-Driven)

**Phase 1:** Build isolated service (no backend changes)  
**Phase 2:** Parallel deployment (both systems running)  
**Phase 3:** Gradual migration (one service at a time)  
**Phase 4:** Full cutover (remove old code)  
**Phase 5:** Optimization & monitoring  

### Feature Flags

```yaml
# Enable vector service for specific operations
USE_VECTOR_SERVICE_FOR_SEARCH: true/false
USE_VECTOR_SERVICE_FOR_STORAGE: true/false
```

### Rollback

At any point, set flags to `false` and restart backend → instant rollback.

## Next Steps

### Before Implementation

**Must complete:**
1. ✅ Migration plan approved (DONE)
2. ⚠️ **Remove video code** from codebase
   - Delete `backend/services/video_search_service.py`
   - Delete `backend/services/video_service.py`
   - Remove video-related API endpoints
   - Remove video collection code from `EmbeddingManager`

### Phase 1: Build Service (Week 1)

**When ready to start:**
1. Create `vector-service/` directory structure
2. Implement proto file in `/opt/bastion/protos/`
3. Build core service (embedding engine, vector store, cache)
4. Create Dockerfile and docker-compose entry
5. Unit tests for embedding cache
6. Integration tests for gRPC endpoints
7. Test in isolation (no backend connection)

**Success criteria:**
- ✅ Service starts successfully
- ✅ Health check returns healthy
- ✅ Can generate embeddings via gRPC
- ✅ Can store vectors in Qdrant
- ✅ Can search vectors
- ✅ Cache hit rate > 0% on repeated requests

### Phase 2: Parallel Deployment (Week 2)

1. Deploy vector service to production
2. Add `VectorServiceClient` to backend
3. Create feature flags
4. No functional changes yet
5. Monitor service health

### Phase 3-5: Migration & Optimization (Weeks 3-6)

Follow plan in `/opt/bastion/docs/VECTOR_SERVICE_MIGRATION_PLAN.md`

## Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Upload time (10 docs) | Baseline | ≤ Baseline |
| Search latency (p50) | Baseline | ≤ Baseline |
| Search latency (p95) | Baseline | ≤ Baseline * 1.1 |
| Backend memory | Baseline | ≤ Baseline * 0.8 |
| Cache hit rate | N/A | > 40% |

## Benefits

### Architectural
- Separation of concerns (vector ops isolated)
- Reusable by backend, orchestrator, future services
- Independent scaling
- Technology flexibility

### Performance
- Resource isolation
- Embedding cache (estimated 40%+ hit rate)
- Reduced backend memory (20%+ target)
- Parallel request handling

### Operational
- Clear monitoring (cache stats, embedding metrics)
- Failure isolation
- Independent deployment
- Easier debugging

## Questions Answered

**Q: Should we migrate video embeddings?**  
A: No, remove all video capabilities. Can re-add post-refactor.

**Q: Where does query expansion live?**  
A: NOT in vector service. It's an LLM tool function.

**Q: Do we need parallel processing?**  
A: Yes, vector service must handle concurrent requests efficiently.

**Q: Same Qdrant connection?**  
A: Yes, use shared connection string (external infrastructure).

**Q: What collection pattern?**  
A: Keep current user/global pattern.

**Q: What about caching?**  
A: Full embedding cache, 3-hour TTL, hash-based.

**Q: Data migration needed?**  
A: No, vectors stay in Qdrant. Migrating mechanism only.

## Ready to Begin

The plan is **implementation-ready**. When you're ready to start:

1. Begin with video code removal (clean up existing code)
2. Create vector-service directory structure
3. Implement Phase 1 (isolated service)

All architectural decisions are made, proto file is designed, and migration strategy is clear.

---

**Prepared by:** Claude (AI Assistant)  
**Status:** ✅ Ready for Implementation  
**Last Updated:** November 10, 2025

