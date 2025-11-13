# gRPC Quick Start Guide

**5-Minute Guide to Adding New gRPC Methods**

---

## Quick Reference: Add a New RPC Method

### 1. Update Proto File (2 minutes)

**File:** `protos/tool_service.proto` (shared by both services)

```protobuf
service ToolService {
  // ... existing methods ...
  
  // Add your new method
  rpc YourNewMethod(YourRequest) returns (YourResponse);
}

// Add request message
message YourRequest {
  string user_id = 1;
  string query = 2;
  int32 limit = 3;
}

// Add response message
message YourResponse {
  repeated YourResult results = 1;
  int32 total_count = 2;
}

message YourResult {
  string id = 1;
  string title = 2;
  float score = 3;
}
```

**Important:** Edit once in the shared `protos/` directory - both services use the same file!

---

### 2. Implement Server (5 minutes)

**File:** `backend/services/grpc_tool_service.py`

```python
async def YourNewMethod(
    self,
    request: tool_service_pb2.YourRequest,
    context: grpc.aio.ServicerContext
) -> tool_service_pb2.YourResponse:
    """Your method description"""
    try:
        logger.info(f"YourNewMethod: user={request.user_id}, query={request.query}")
        
        # Your implementation here
        # results = await your_service.do_something(request.query)
        
        # Convert to proto response
        response = tool_service_pb2.YourResponse(
            total_count=len(results)
        )
        
        for item in results:
            result = tool_service_pb2.YourResult(
                id=item['id'],
                title=item['title'],
                score=item['score']
            )
            response.results.append(result)
        
        return response
        
    except Exception as e:
        logger.error(f"YourNewMethod error: {e}")
        await context.abort(grpc.StatusCode.INTERNAL, f"Error: {str(e)}")
```

---

### 3. Implement Client (5 minutes)

**File:** `llm-orchestrator/orchestrator/backend_tool_client.py`

```python
async def your_new_method(
    self,
    query: str,
    limit: int = 10,
    user_id: str = "system"
) -> List[Dict[str, Any]]:
    """Call your new method"""
    try:
        await self._ensure_connected()
        
        request = tool_service_pb2.YourRequest(
            user_id=user_id,
            query=query,
            limit=limit
        )
        
        response = await self._stub.YourNewMethod(request)
        
        # Convert to Python dict
        results = []
        for item in response.results:
            results.append({
                'id': item.id,
                'title': item.title,
                'score': item.score
            })
        
        return results
        
    except grpc.RpcError as e:
        logger.error(f"Method failed: {e.code()} - {e.details()}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []
```

---

### 4. Create LangGraph Tool (3 minutes)

**File:** `llm-orchestrator/orchestrator/tools/your_tools.py`

```python
from langchain_core.tools import tool
from orchestrator.backend_tool_client import backend_tool_client

@tool
async def your_new_tool(query: str, limit: int = 10) -> str:
    """
    Tool description for LLM.
    
    Args:
        query: What to search for
        limit: Max results
        
    Returns:
        Formatted results
    """
    try:
        results = await backend_tool_client.your_new_method(
            query=query,
            limit=limit
        )
        
        if not results:
            return f"No results found for: {query}"
        
        output = f"Found {len(results)} results:\n\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']} (score: {result['score']:.2f})\n"
        
        return output
        
    except Exception as e:
        return f"Error: {str(e)}"
```

---

### 5. Rebuild and Test (2 minutes)

```bash
# Rebuild containers (regenerates proto code)
docker compose build backend llm-orchestrator

# Restart services
docker compose up -d

# Watch logs
docker logs -f codex-dev-llm-orchestrator
docker logs -f codex-dev-backend
```

---

## Common Proto Patterns

### Simple Query

```protobuf
message SearchRequest {
  string query = 1;
  int32 limit = 2;
}

message SearchResponse {
  repeated Result results = 1;
}
```

### With Filters

```protobuf
message SearchRequest {
  string query = 1;
  repeated string filters = 2;
  map<string, string> metadata = 3;
}
```

### Pagination

```protobuf
message SearchRequest {
  string query = 1;
  int32 page_size = 2;
  string page_token = 3;
}

message SearchResponse {
  repeated Result results = 1;
  string next_page_token = 2;
}
```

### Batch Operations

```protobuf
message BatchRequest {
  repeated SingleRequest requests = 1;
}

message BatchResponse {
  repeated SingleResponse responses = 1;
}
```

### Streaming Response

```protobuf
service YourService {
  // Returns stream of chunks
  rpc StreamingMethod(Request) returns (stream ResponseChunk);
}

message ResponseChunk {
  string type = 1;  // "status", "data", "complete"
  string content = 2;
}
```

---

## Error Handling Template

```python
# Server side
try:
    # Validate input
    if not request.query:
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            "Query is required"
        )
    
    # Do work
    result = await do_work(request)
    return result
    
except ValueError as e:
    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
except PermissionError as e:
    await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(e))
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    await context.abort(grpc.StatusCode.INTERNAL, "Internal error")

# Client side
try:
    response = await stub.YourMethod(request)
    return process_response(response)
    
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
        logger.warning(f"Invalid input: {e.details()}")
        return None
    elif e.code() == grpc.StatusCode.UNAVAILABLE:
        logger.error(f"Service down: {e.details()}")
        return None
    else:
        logger.error(f"gRPC error: {e.code()} - {e.details()}")
        return None
```

---

## Testing Template

```python
# test_your_method.py

import asyncio
import grpc
from protos import tool_service_pb2, tool_service_pb2_grpc

async def test_your_method():
    """Test your new gRPC method"""
    
    print("🧪 Testing YourNewMethod")
    
    async with grpc.aio.insecure_channel('localhost:50052') as channel:
        stub = tool_service_pb2_grpc.ToolServiceStub(channel)
        
        try:
            request = tool_service_pb2.YourRequest(
                user_id="test_user",
                query="test query",
                limit=5
            )
            
            response = await stub.YourNewMethod(request)
            
            print(f"✅ Success: {response.total_count} results")
            for result in response.results:
                print(f"   - {result.title} (score: {result.score})")
            
        except grpc.RpcError as e:
            print(f"❌ Error: {e.code()} - {e.details()}")

if __name__ == "__main__":
    asyncio.run(test_your_method())
```

Run test:
```bash
docker exec codex-dev-backend python test_your_method.py
```

---

## Troubleshooting Checklist

### Build Errors

- [ ] Proto file valid? (check `protos/` directory)
- [ ] All message types defined?
- [ ] Field numbers unique and sequential?
- [ ] RPC signature correct?

### Runtime Errors

- [ ] Services running? (`docker ps`)
- [ ] Proto code regenerated? (rebuild containers)
- [ ] Method implemented on server?
- [ ] Client calling correct service?
- [ ] Error handling in place?

### Connection Issues

```bash
# Check service is accessible
docker exec codex-dev-backend python -c "
import socket
s = socket.socket()
s.connect(('llm-orchestrator', 50051))
print('✅ Connection successful')
s.close()
"
```

---

## Field Number Rules

**Remember:**
- Field numbers 1-15: Use for frequent fields (1 byte)
- Field numbers 16+: Use for optional fields (2+ bytes)
- **NEVER** reuse deleted field numbers
- **NEVER** change existing field numbers

**Example:**
```protobuf
message Request {
  string query = 1;        // Frequent, small number
  string user_id = 2;      // Frequent, small number
  int32 limit = 3;         // Frequent, small number
  string rare_field = 20;  // Rare, higher number OK
}
```

---

## Common Mistakes

### ❌ Wrong: Creating channel per request
```python
async def search(query):
    channel = grpc.aio.insecure_channel('backend:50052')
    stub = ToolServiceStub(channel)
    result = await stub.Search(request)
    await channel.close()
    return result
```

### ✅ Right: Reuse channel
```python
class Client:
    def __init__(self):
        self._channel = grpc.aio.insecure_channel('backend:50052')
        self._stub = ToolServiceStub(self._channel)
    
    async def search(self, query):
        return await self._stub.Search(request)
```

---

### ❌ Wrong: No error handling
```python
response = await stub.Search(request)
return response.results
```

### ✅ Right: Handle errors
```python
try:
    response = await stub.Search(request)
    return response.results
except grpc.RpcError as e:
    logger.error(f"Search failed: {e.code()}")
    return []
```

---

## Next Steps

1. **Read full architecture:** `docs/GRPC_MICROSERVICES_ARCHITECTURE.md`
2. **Explore shared protos:** `protos/` directory (single source of truth)
   - `protos/orchestrator.proto` - LLM orchestrator service API
   - `protos/tool_service.proto` - Backend tool service API
   - `protos/vector_service.proto` - Vector service API
3. **See working examples:** 
   - `backend/services/grpc_tool_service.py` - Server implementation
   - `llm-orchestrator/orchestrator/backend_tool_client.py` - Client implementation
4. **Test existing methods:**
   - `llm-orchestrator/test_phase3.py`
   - `backend/test_tool_service.py`

---

**Ready to build? Follow the 5-step process above!** 🚀



