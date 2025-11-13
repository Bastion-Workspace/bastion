# Vector Service (Embedding Generation)

Dedicated microservice for embedding generation with intelligent caching.

## Overview

The Vector Service is a **pure embedding generation service** that:
- **Generates embeddings** via OpenAI API
- **Caches embeddings** (3-hour TTL, content-hash based)
- **Returns embeddings** to caller

**What it does NOT do:**
- Does NOT store vectors in Qdrant (caller's responsibility)
- Does NOT search Qdrant (caller's responsibility)
- Does NOT handle metadata (caller's responsibility)

This keeps the service simple and focused on one task: fast, cached embedding generation.

## Architecture

```
Vector Service (Port 50053)
├── Embedding Engine (OpenAI API)
│   ├── Single embedding generation
│   ├── Batch embedding generation (parallel)
│   └── Text truncation (8000 chars)
├── Embedding Cache (3-hour TTL)
│   ├── SHA256 content hashing
│   ├── Hit/miss tracking
│   └── Automatic expiration
└── gRPC Interface (5 RPCs)
    ├── GenerateEmbedding
    ├── GenerateBatchEmbeddings
    ├── ClearEmbeddingCache
    ├── GetCacheStats
    └── HealthCheck
```

## Use Cases

### 1. Document Upload (Backend)
```
1. Backend chunks document
2. Backend calls GenerateBatchEmbeddings(chunk_texts[])
3. Vector service returns embeddings[]
4. Backend stores embeddings + metadata in Qdrant
```

### 2. Search Query (LangGraph Agent)
```
1. Agent receives search query
2. Agent calls GenerateEmbedding(query_text)
3. Vector service returns query_embedding
4. Agent searches Qdrant with embedding + filters
```

### 3. Future Document Processor (Separate Container)
```
1. Document processor extracts chunks
2. Processor calls GenerateBatchEmbeddings(chunks[])
3. Vector service returns embeddings[]
4. Processor sends embeddings + metadata to backend/Qdrant
```

## gRPC Interface

### GenerateEmbedding
```protobuf
rpc GenerateEmbedding(EmbeddingRequest) returns (EmbeddingResponse);

message EmbeddingRequest {
  string text = 1;
  string model = 2;  // Optional
}

message EmbeddingResponse {
  repeated float embedding = 1;  // 1536 dimensions
  int32 token_count = 2;
  string model = 3;
  bool from_cache = 4;
}
```

### GenerateBatchEmbeddings
```protobuf
rpc GenerateBatchEmbeddings(BatchEmbeddingRequest) returns (BatchEmbeddingResponse);

message BatchEmbeddingRequest {
  repeated string texts = 1;
  string model = 2;       // Optional
  int32 batch_size = 3;   // Optional, default 100
}

message BatchEmbeddingResponse {
  repeated EmbeddingVector embeddings = 1;
  int32 total_tokens = 2;
  string model = 3;
  int32 cache_hits = 4;
  int32 cache_misses = 5;
}
```

## Configuration

Environment variables:

```bash
# Service
SERVICE_NAME=vector-service
GRPC_PORT=50053
LOG_LEVEL=INFO

# OpenAI
OPENAI_API_KEY=<your-key>
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MAX_RETRIES=3
OPENAI_TIMEOUT=30

# Performance
PARALLEL_WORKERS=4
BATCH_SIZE=100
MAX_TEXT_LENGTH=8000

# Cache
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_TTL=10800  # 3 hours
CACHE_CLEANUP_INTERVAL=3600
```

## Performance

**Expected metrics:**
- Single embedding: ~100ms (OpenAI API latency)
- Batch (100 texts): ~1-2s (parallel processing)
- Cache hit: <1ms (in-memory lookup)
- Cache hit rate: 40-60% (after warmup)

**Scaling:**
- Horizontal: Run multiple instances with load balancer
- Vertical: Increase PARALLEL_WORKERS for more concurrency

## Development

### Build and Run

```bash
# Build
docker compose build vector-service

# Run
docker compose up vector-service

# Check logs
docker compose logs -f vector-service
```

### Health Check

```bash
# Internal health check
grpcurl -plaintext localhost:50053 vector_service.VectorService/HealthCheck
```

### Cache Statistics

```bash
# Get cache stats
grpcurl -plaintext localhost:50053 vector_service.VectorService/GetCacheStats
```

## Caller Responsibilities

**Backend must:**
1. Call vector service for embeddings
2. Store embeddings in Qdrant with metadata
3. Search Qdrant with filters
4. Handle collection management

**LangGraph agents must:**
1. Call vector service for query embeddings
2. Search Qdrant with embeddings
3. Apply filters (category, tags, etc.)

**Future document processor must:**
1. Extract/chunk documents
2. Call vector service for embeddings
3. Send embeddings + metadata to backend
4. Handle error cases

## Benefits of Simplified Design

✅ **Single Responsibility** - Just embedding generation  
✅ **No Duplicate Logic** - Backend keeps Qdrant operations  
✅ **Less Network Traffic** - No metadata over gRPC  
✅ **Easier Migration** - Just swap OpenAI calls  
✅ **Future-Proof** - Works with external document processors  
✅ **Simpler Testing** - Fewer moving parts  

## Production Considerations

- **API Keys**: Secure OpenAI API key management
- **Rate Limits**: OpenAI API rate limiting (3,500 RPM for tier 1)
- **Scaling**: Deploy multiple instances for high throughput
- **Monitoring**: Cache hit rate, latency, error rate
- **Security**: TLS/SSL for gRPC (future enhancement)

## Version

**Version:** 1.0.0  
**Mode:** Embedding Generation + Caching Only  
**Status:** Production Ready
