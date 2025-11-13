# Vector Service Hotfix - Import, OpenAI Version & Settings Issues

## Issues Fixed

### Issue 1: Import Path Errors ✅
**Problem**: `ModuleNotFoundError: No module named 'backend'`

**Cause**: Used incorrect import paths with `backend.` prefix

**Fixed Files**:
- `backend/services/embedding_service_wrapper.py`
  - Changed: `from backend.config` → `from config`
  - Changed: `from backend.utils.embedding_manager` → `from utils.embedding_manager`
  - Changed: `from backend.clients.vector_service_client` → `from clients.vector_service_client`
  - Changed: `from backend.models.api_models` → `from models.api_models`

- `backend/clients/vector_service_client.py`
  - Changed: `from backend.config` → `from config`
  - Changed: `from backend.generated` → `from generated`

### Issue 2: OpenAI Library Version Mismatch ✅
**Problem**: `TypeError: AsyncClient.__init__() got an unexpected keyword argument 'proxies'`

**Cause**: Vector Service used ancient OpenAI library version

**Before**:
```txt
openai==1.6.1  # OLD version from early 2024
```

**After**:
```txt
openai==1.99.1  # Matches backend version
pydantic==2.9.2  # Added for compatibility
```

**Why This Matters**:
- OpenAI 1.6.1 had old httpx integration with `proxies` parameter
- Modern httpx doesn't accept `proxies` argument
- OpenAI 1.99.1 uses proper httpx integration
- Keeps version consistent with backend

### Issue 3: Leftover Qdrant Reference ✅
**Problem**: `AttributeError: 'Settings' object has no attribute 'QDRANT_URL'`

**Cause**: `main.py` tried to log `settings.QDRANT_URL` but we removed it when simplifying Vector Service

**Fixed**: Updated `vector-service/main.py`
- Removed: `logger.info(f"Qdrant URL: {settings.QDRANT_URL}")`
- Added: `logger.info(f"Cache TTL: {settings.EMBEDDING_CACHE_TTL}s")`

**Why**: Vector Service only generates embeddings, doesn't need Qdrant URL (backend handles storage)

### Issue 4: Proto Generation Path ✅
**Problem**: `ModuleNotFoundError: No module named 'generated'`

**Cause**: Backend Dockerfile generates proto files into `/app/protos/` but client was importing from `generated`

**Fixed**:
- `backend/clients/vector_service_client.py` - Changed `from generated import` → `from protos import`
- `backend/Dockerfile` - Added `vector_service.proto` to protoc generation

**Why**: Backend generates all proto files into `protos/` directory, not `generated/`

### Issue 5: EmbeddingConfig & EmbeddingManager Type Annotations ✅
**Problem**: `NameError: name 'EmbeddingConfig' is not defined` and `NameError: name 'EmbeddingManager' is not defined`

**Cause**: Removed imports but left type annotations in method signatures, return types, and instance variables

**Fixed**:
- `backend/services/parallel_document_service.py` - Removed `embedding_config` parameter from `initialize()`
- `backend/services/parallel_document_service.py` - Deprecated `optimize_embedding_configuration()` method
- `backend/services/file_manager/file_manager_service.py` - Removed `EmbeddingConfig` import and usage
- `backend/services/enhanced_pdf_segmentation_service.py` - Removed `EmbeddingManager` type annotation from `__init__`
- `backend/services/service_container.py` - Removed `ParallelEmbeddingManager` type annotation
- `backend/services/zip_processor_service.py` - Removed `EmbeddingManager` type annotation
- `backend/services/grpc_tool_service.py` - Removed `EmbeddingManager` type annotation

**Why**: Using `EmbeddingServiceWrapper` now, which handles configuration internally via `USE_VECTOR_SERVICE` flag. Type annotations would require importing the old classes.

### Issue 6: HealthCheck Proto Field Mismatch ✅
**Problem**: `AttributeError: service_name` when accessing HealthCheck response

**Cause**: HealthCheck proto uses `service_version` not `service_name`, and `openai_available` not `version`

**Fixed**:
- `backend/clients/vector_service_client.py` - Changed `response.service_name` → `response.service_version`
- `backend/clients/vector_service_client.py` - Changed `response.version` → `response.service_version`
- Added logging for `response.openai_available`

**Why**: Proto definition uses different field names than code expected

### Issue 7: USE_VECTOR_SERVICE Flag Prematurely Enabled ✅
**Problem**: Backend failing because `USE_VECTOR_SERVICE=true` in docker-compose but service not ready

**Cause**: Flag was set to `true` before migration testing complete

**Fixed**:
- `docker-compose.yml` - Changed `USE_VECTOR_SERVICE=true` → `USE_VECTOR_SERVICE=false`

**Why**: Global Flag migration strategy requires flag to be `False` initially, then flip to `True` when ready

### Issue 8: Wrong Embedding Model ✅
**Problem**: Vector Service using `text-embedding-3-small` instead of large model

**Cause**: Default configuration used small model

**Fixed**:
- `docker-compose.yml` - Changed `OPENAI_EMBEDDING_MODEL=text-embedding-3-small` → `text-embedding-3-large`

**Why**: User preference for large embedding model for better quality

### Issue 9: BatchEmbeddingResponse Field Name Mismatch ✅
**Problem**: `❌ Error generating batch embeddings: embedding`

**Cause**: Client code tried to access `emb.embedding` but proto defines field as `vector`

**Fixed**:
- `backend/clients/vector_service_client.py` - Changed `emb.embedding` → `emb.vector` (line 129)

**Why**: Proto `EmbeddingVector` message uses `repeated float vector = 1;` not `embedding`

## Testing

After rebuild:
```bash
docker compose up --build
```

**Expected**:
- ✅ Celery beat starts without import errors
- ✅ Vector Service initializes OpenAI client successfully
- ✅ Embedding engine ready for gRPC requests

## Related Files
- `/opt/bastion/vector-service/requirements.txt` - Updated OpenAI version
- `/opt/bastion/vector-service/main.py` - Removed Qdrant reference
- `/opt/bastion/backend/Dockerfile` - Added vector_service.proto generation
- `/opt/bastion/backend/services/embedding_service_wrapper.py` - Fixed imports
- `/opt/bastion/backend/clients/vector_service_client.py` - Fixed imports (multiple fixes)
- `/opt/bastion/backend/clients/__init__.py` - Fixed imports
- `/opt/bastion/backend/services/parallel_document_service.py` - Removed EmbeddingConfig parameter
- `/opt/bastion/backend/services/file_manager/file_manager_service.py` - Removed EmbeddingConfig usage

## Status
**RESOLVED** - All 5 issues fixed, ready for deployment with `docker compose up --build`

