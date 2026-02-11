# OpenRouter Embedding Provider Implementation

## Overview

This document outlines the implementation plan for adding OpenRouter as an alternative embedding provider alongside OpenAI. The implementation will support system-level configuration to switch between providers, but will not expose provider selection to end users.

## Architecture

### Current State

The embedding system uses a clean microservices architecture:

```
Backend Service
  └─> EmbeddingServiceWrapper
      └─> VectorServiceClient (gRPC)
          └─> Vector Service (microservice)
              └─> EmbeddingEngine
                  └─> OpenAI API (only)
```

### Target State

After implementation, the architecture will support multiple providers:

```
Backend Service
  └─> EmbeddingServiceWrapper
      └─> VectorServiceClient (gRPC)
          └─> Vector Service (microservice)
              └─> EmbeddingProviderFactory
                  ├─> OpenAIProvider
                  └─> OpenRouterProvider
```

## Implementation Plan

### Phase 1: Provider Abstraction Layer

**Location**: `vector-service/service/embedding_provider.py` (new file)

Create a base provider interface and concrete implementations:

```python
"""
Embedding Provider Abstraction

Supports multiple embedding providers (OpenAI, OpenRouter) with unified interface.
"""

from abc import ABC, abstractmethod
from typing import List
import logging

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Base class for embedding providers"""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is accessible"""
        pass


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider (existing implementation)"""
    
    def __init__(self, api_key: str, model: str, timeout: int, max_retries: int):
        from openai import AsyncOpenAI
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        self.model = model
        self.max_text_length = 8000  # Default, configurable
    
    async def generate_embedding(self, text: str) -> List[float]:
        text = self._truncate_text(text)
        response = await self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        truncated_texts = [self._truncate_text(text) for text in texts]
        all_embeddings = []
        
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    async def health_check(self) -> bool:
        try:
            await self.generate_embedding("health check")
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits"""
        if len(text) <= self.max_text_length:
            return text
        truncated = text[:self.max_text_length]
        last_space = truncated.rfind(' ')
        if last_space > self.max_text_length * 0.8:
            truncated = truncated[:last_space]
        return truncated.strip()


class OpenRouterProvider(EmbeddingProvider):
    """OpenRouter embedding provider"""
    
    def __init__(self, api_key: str, model: str, timeout: int, max_retries: int):
        from openai import AsyncOpenAI
        
        # OpenRouter uses OpenAI-compatible API with different base URL
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
            max_retries=max_retries
        )
        self.model = model
        self.max_text_length = 8000  # Default, configurable
    
    async def generate_embedding(self, text: str) -> List[float]:
        text = self._truncate_text(text)
        response = await self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        truncated_texts = [self._truncate_text(text) for text in texts]
        all_embeddings = []
        
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    async def health_check(self) -> bool:
        try:
            await self.generate_embedding("health check")
            return True
        except Exception as e:
            logger.error(f"OpenRouter health check failed: {e}")
            return False
    
    def _truncate_text(self, text: str) -> List[float]:
        """Truncate text to fit within token limits"""
        if len(text) <= self.max_text_length:
            return text
        truncated = text[:self.max_text_length]
        last_space = truncated.rfind(' ')
        if last_space > self.max_text_length * 0.8:
            truncated = truncated[:last_space]
        return truncated.strip()
```

### Phase 2: Provider Factory

**Location**: `vector-service/service/embedding_provider_factory.py` (new file)

```python
"""
Embedding Provider Factory

Creates and manages embedding provider instances based on configuration.
"""

from typing import Optional
from config.settings import settings
from .embedding_provider import EmbeddingProvider, OpenAIProvider, OpenRouterProvider
import logging

logger = logging.getLogger(__name__)


class EmbeddingProviderFactory:
    """Factory for creating embedding providers"""
    
    _provider: Optional[EmbeddingProvider] = None
    
    @classmethod
    def get_provider(cls) -> EmbeddingProvider:
        """Get or create the configured embedding provider"""
        if cls._provider is None:
            cls._provider = cls._create_provider()
        return cls._provider
    
    @classmethod
    def _create_provider(cls) -> EmbeddingProvider:
        """Create provider based on configuration"""
        provider_type = settings.EMBEDDING_PROVIDER.lower()
        
        if provider_type == "openrouter":
            if not settings.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY is required for OpenRouter provider")
            
            logger.info(f"Initializing OpenRouter provider with model: {settings.OPENROUTER_EMBEDDING_MODEL}")
            return OpenRouterProvider(
                api_key=settings.OPENROUTER_API_KEY,
                model=settings.OPENROUTER_EMBEDDING_MODEL,
                timeout=settings.OPENAI_TIMEOUT,
                max_retries=settings.OPENAI_MAX_RETRIES
            )
        
        elif provider_type == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            
            logger.info(f"Initializing OpenAI provider with model: {settings.OPENAI_EMBEDDING_MODEL}")
            return OpenAIProvider(
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_EMBEDDING_MODEL,
                timeout=settings.OPENAI_TIMEOUT,
                max_retries=settings.OPENAI_MAX_RETRIES
            )
        
        else:
            raise ValueError(f"Unknown embedding provider: {provider_type}. Must be 'openai' or 'openrouter'")
    
    @classmethod
    def reset_provider(cls):
        """Reset provider instance (useful for testing or config changes)"""
        cls._provider = None
```

### Phase 3: Update EmbeddingEngine

**Location**: `vector-service/service/embedding_engine.py`

Refactor to use provider abstraction:

```python
"""
Embedding Engine - Provider-based embedding generation with batching
"""

import logging
from typing import List, Optional
from .embedding_provider_factory import EmbeddingProviderFactory

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Handles embedding generation via configurable provider"""
    
    def __init__(self):
        self.provider = None
        self.model = None  # Store for reference
    
    async def initialize(self):
        """Initialize embedding provider"""
        self.provider = EmbeddingProviderFactory.get_provider()
        # Store model name for reference (from provider config)
        if hasattr(self.provider, 'model'):
            self.model = self.provider.model
        logger.info(f"Embedding engine initialized with provider: {type(self.provider).__name__}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.provider:
            raise RuntimeError("Embedding engine not initialized")
        return await self.provider.generate_embedding(text)
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        if not self.provider:
            raise RuntimeError("Embedding engine not initialized")
        return await self.provider.generate_batch_embeddings(texts, batch_size)
    
    async def health_check(self) -> bool:
        """Check if embedding provider is accessible"""
        if not self.provider:
            return False
        return await self.provider.health_check()
```

### Phase 4: Configuration Updates

**Location**: `vector-service/config/settings.py`

Add new configuration options:

```python
class Settings:
    # ... existing settings ...
    
    # Embedding Provider Configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")  # "openai" or "openrouter"
    
    # OpenAI Configuration (existing)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    
    # OpenRouter Configuration (new)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_EMBEDDING_MODEL: str = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-large")
    
    @classmethod
    def validate(cls) -> None:
        """Validate required settings based on provider"""
        provider = cls.EMBEDDING_PROVIDER.lower()
        
        if provider == "openai":
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY must be set for OpenAI provider")
            if not cls.OPENAI_API_KEY.startswith("sk-"):
                raise ValueError("OPENAI_API_KEY format appears incorrect")
        
        elif provider == "openrouter":
            if not cls.OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY must be set for OpenRouter provider")
        
        else:
            raise ValueError(f"Unknown embedding provider: {provider}. Must be 'openai' or 'openrouter'")
```

**Location**: `backend/config.py`

Add corresponding backend configuration (for reference, though Vector Service handles actual generation):

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Embedding Provider Configuration
    EMBEDDING_PROVIDER: str = "openai"  # "openai" or "openrouter"
    OPENROUTER_EMBEDDING_MODEL: str = "openai/text-embedding-3-large"
    
    # Note: OPENROUTER_API_KEY already exists in config
```

### Phase 5: Protocol Buffer Updates (Optional Enhancement)

**Location**: `protos/vector_service.proto`

Add provider field to requests for explicit provider selection per-request (future flexibility):

```protobuf
message EmbeddingRequest {
  string text = 1;
  string model = 2;  // Optional, defaults to service config
  string provider = 3;  // Optional, "openai" or "openrouter", defaults to service config
}

message BatchEmbeddingRequest {
  repeated string texts = 1;
  string model = 2;  // Optional
  int32 batch_size = 3;  // Optional
  string provider = 4;  // Optional, "openai" or "openrouter", defaults to service config
}
```

**After proto changes**, regenerate Python code:
```bash
python -m grpc_tools.protoc \
  --proto_path=protos \
  --python_out=backend/protos \
  --grpc_python_out=backend/protos \
  protos/vector_service.proto
```

### Phase 6: gRPC Service Updates

**Location**: `vector-service/service/grpc_service.py`

Update to use provider abstraction and support optional provider parameter:

```python
async def GenerateEmbedding(
    self, 
    request: vector_service_pb2.EmbeddingRequest, 
    context
) -> vector_service_pb2.EmbeddingResponse:
    """Generate single embedding"""
    try:
        # Use provider from request if specified, otherwise use configured provider
        provider_name = request.provider if request.provider else None
        
        # For now, ignore per-request provider (use configured provider)
        # Future: Support per-request provider selection if needed
        embedding = await self.embedding_engine.generate_embedding(request.text)
        
        return vector_service_pb2.EmbeddingResponse(
            embedding=embedding,
            model=self.embedding_engine.model or "",
            from_cache=False  # Cache layer handles this
        )
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(str(e))
        return vector_service_pb2.EmbeddingResponse()

async def GenerateBatchEmbeddings(
    self, 
    request: vector_service_pb2.BatchEmbeddingRequest, 
    context
) -> vector_service_pb2.BatchEmbeddingResponse:
    """Generate batch embeddings"""
    try:
        embeddings = await self.embedding_engine.generate_batch_embeddings(
            texts=list(request.texts),
            batch_size=request.batch_size or settings.BATCH_SIZE
        )
        
        embedding_vectors = [
            vector_service_pb2.EmbeddingVector(
                vector=emb,
                index=i,
                from_cache=False
            )
            for i, emb in enumerate(embeddings)
        ]
        
        return vector_service_pb2.BatchEmbeddingResponse(
            embeddings=embedding_vectors,
            model=self.embedding_engine.model or "",
            cache_hits=0,
            cache_misses=len(embeddings)
        )
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(str(e))
        return vector_service_pb2.BatchEmbeddingResponse()
```

### Phase 7: Health Check Updates

**Location**: `vector-service/service/grpc_service.py`

Update health check to report provider status:

```python
async def HealthCheck(
    self, 
    request: vector_service_pb2.HealthCheckRequest, 
    context
) -> vector_service_pb2.HealthCheckResponse:
    """Check service health"""
    try:
        provider_healthy = await self.embedding_engine.health_check()
        
        # Determine overall status
        if provider_healthy:
            status = "healthy"
        else:
            status = "degraded"  # Provider unavailable but service running
        
        provider_name = type(self.embedding_engine.provider).__name__ if self.embedding_engine.provider else "unknown"
        
        return vector_service_pb2.HealthCheckResponse(
            status=status,
            openai_available=provider_healthy,  # Keep for backward compatibility
            service_version="1.0.0",
            details={
                "embedding_provider": provider_name,
                "model": self.embedding_engine.model or "unknown"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return vector_service_pb2.HealthCheckResponse(
            status="unhealthy",
            openai_available=False,
            service_version="1.0.0"
        )
```

## Environment Variables

### Vector Service

Add to `docker-compose.yml` or environment configuration:

```yaml
vector-service:
  environment:
    # Embedding Provider Selection
    - EMBEDDING_PROVIDER=openrouter  # or "openai"
    
    # OpenAI Configuration (if using OpenAI provider)
    - OPENAI_API_KEY=${OPENAI_API_KEY}
    - OPENAI_EMBEDDING_MODEL=text-embedding-3-large
    
    # OpenRouter Configuration (if using OpenRouter provider)
    - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    - OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-large
```

### Available OpenRouter Embedding Models

Common models available via OpenRouter:
- `openai/text-embedding-3-small` (1536 dimensions)
- `openai/text-embedding-3-large` (3072 dimensions)
- `openai/text-embedding-ada-002` (1536 dimensions)
- `voyageai/voyage-large-2` (1024 dimensions)
- `cohere/embed-english-v3.0` (1024 dimensions)
- `cohere/embed-multilingual-v3.0` (1024 dimensions)

View all available models: https://openrouter.ai/models?fmt=cards&output_modalities=embeddings

## Testing Considerations

### Unit Tests

Create test files:
- `vector-service/tests/test_embedding_provider.py`
- `vector-service/tests/test_embedding_provider_factory.py`
- `vector-service/tests/test_embedding_engine.py`

Test scenarios:
1. Provider factory creates correct provider based on config
2. OpenAI provider generates embeddings correctly
3. OpenRouter provider generates embeddings correctly
4. Provider switching works (reset and recreate)
5. Error handling when API keys missing
6. Health checks for both providers

### Integration Tests

1. **End-to-end embedding generation**:
   - Test through gRPC service
   - Verify embeddings are correct dimensions
   - Verify batch processing works

2. **Provider switching**:
   - Start with OpenAI, generate embeddings
   - Switch config to OpenRouter, restart service
   - Verify new embeddings use OpenRouter

3. **Error scenarios**:
   - Invalid API key
   - Network failures
   - Rate limiting
   - Invalid model names

### Manual Testing Checklist

- [ ] Service starts with OpenAI provider
- [ ] Service starts with OpenRouter provider
- [ ] Embeddings generated successfully with OpenAI
- [ ] Embeddings generated successfully with OpenRouter
- [ ] Batch embeddings work with both providers
- [ ] Health check reports correct provider status
- [ ] Error messages are clear when provider misconfigured
- [ ] Cache still works with both providers
- [ ] Vector dimensions match expected values

## Migration Strategy

### Phase 1: Add OpenRouter Support (Non-Breaking)

1. Implement provider abstraction layer
2. Add OpenRouter provider implementation
3. Update configuration to support provider selection
4. Keep OpenAI as default provider
5. Test OpenRouter in development environment

### Phase 2: Enable OpenRouter (Optional)

1. Set `EMBEDDING_PROVIDER=openrouter` in production
2. Monitor embedding quality and performance
3. Compare costs between providers
4. Keep OpenAI as fallback option

### Phase 3: Optimization (Future)

1. Add provider-specific optimizations:
   - Different batch sizes per provider
   - Provider-specific rate limiting
   - Cost tracking per provider
2. Support automatic failover between providers
3. Add provider performance metrics

## Benefits of OpenRouter

1. **Model Variety**: Access to embedding models from multiple providers
2. **Cost Optimization**: Compare and choose cost-effective models
3. **Automatic Failover**: Built-in provider routing with `allow_fallbacks: true`
4. **Data Privacy**: Control data collection with `data_collection: "deny"`
5. **Unified API**: Same interface as OpenAI, minimal code changes

## OpenRouter API Compatibility

OpenRouter uses an OpenAI-compatible API, which means:

✅ **Same request format**: `POST /api/v1/embeddings` with identical JSON structure
✅ **Same response format**: Identical response structure
✅ **Same SDK**: Can use `AsyncOpenAI` client with different `base_url`
✅ **Batch processing**: Works identically to OpenAI
✅ **Error handling**: Same error codes and formats

**Key difference**: Only the base URL changes:
- OpenAI: `https://api.openai.com/v1`
- OpenRouter: `https://openrouter.ai/api/v1`

## Implementation Timeline

**Estimated effort**: 6-8 hours

1. **Provider abstraction** (2 hours)
   - Create base provider class
   - Implement OpenAI provider (refactor existing code)
   - Implement OpenRouter provider

2. **Factory and configuration** (1 hour)
   - Create provider factory
   - Update settings configuration
   - Add validation logic

3. **Engine refactoring** (1 hour)
   - Update EmbeddingEngine to use providers
   - Update gRPC service layer

4. **Testing** (2 hours)
   - Unit tests for providers
   - Integration tests
   - Manual testing

5. **Documentation and cleanup** (1 hour)
   - Update code comments
   - Update README files
   - Document environment variables

## Future Enhancements

1. **Per-request provider selection**: Allow API calls to specify provider
2. **Provider failover**: Automatic fallback if primary provider fails
3. **Cost tracking**: Track embedding costs per provider
4. **Performance metrics**: Compare provider latency and throughput
5. **Additional providers**: Add support for Cohere, Voyage AI, etc.
6. **Model dimension validation**: Ensure embeddings from different providers are compatible
7. **Provider-specific optimizations**: Tune batch sizes and timeouts per provider

## Notes

- **User selection excluded**: Provider selection is system-level only, not exposed to end users
- **Backward compatible**: Existing OpenAI configuration continues to work
- **No database changes**: Provider selection is environment-based, not stored in database
- **Cache compatibility**: Embedding cache works with both providers (embeddings are deterministic)
- **Vector dimensions**: Ensure selected models produce compatible dimensions for existing vector stores

## References

- [OpenRouter Embeddings API Documentation](https://openrouter.ai/docs/api/reference/embeddings)
- [OpenRouter Available Models](https://openrouter.ai/models?fmt=cards&output_modalities=embeddings)
- [OpenAI Embeddings API Documentation](https://platform.openai.com/docs/guides/embeddings)
