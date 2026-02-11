# LLM Orchestrator Service

## Overview

This is a **Phase 1 implementation** of the LLM Orchestrator microservice. It runs **in parallel** with the existing backend without affecting current functionality.

## Phase 1 Status: Skeleton/Testing Only

**What's Implemented:**
- ✅ gRPC server infrastructure
- ✅ Protocol buffer definitions
- ✅ Basic service skeleton
- ✅ Health checks
- ✅ Docker containerization
- ✅ Echo/test endpoints

**Not Yet Implemented:**
- ⏳ Full LangGraph orchestration (Phase 2)
- ⏳ Agent implementations (Phase 2)
- ⏳ Tool callbacks to backend (Phase 2)
- ⏳ State management with checkpointer (Phase 2)
- ⏳ Production routing from main API (Phase 3)

## Architecture

```
┌─────────────────┐
│   Frontend      │
└────────┬────────┘
         │
         │ REST API (unchanged)
         ▼
┌─────────────────┐
│   Backend       │  ← Current production system
│   (Existing)    │  ← All requests still go here
└────────┬────────┘
         │
         │ gRPC (Phase 1: Test only)
         ▼
┌─────────────────┐
│ LLM Orchestrator│  ← New service (skeleton)
│ (Parallel)      │  ← Not in production path yet
└─────────────────┘
```

## Running the Service

The service automatically starts with `docker compose up --build`:

```bash
# Start all services (including new orchestrator)
docker compose up --build

# View orchestrator logs
docker compose logs -f llm-orchestrator

# Restart just orchestrator (test independent restart)
docker compose restart llm-orchestrator

# Health check
curl -X POST http://localhost:50051  # gRPC port (requires grpcurl)
```

## Testing the gRPC Interface

Install `grpcurl` for testing:

```bash
# Install grpcurl
brew install grpcurl  # macOS
# or
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# Test health check (using proto files)
grpcurl -plaintext -import-path llm-orchestrator/protos -proto orchestrator.proto localhost:50051 orchestrator.OrchestratorService/HealthCheck

# Test streaming chat (Phase 1: echo response)
grpcurl -plaintext -import-path llm-orchestrator/protos -proto orchestrator.proto -d '{
  "user_id": "test",
  "conversation_id": "test-123",
  "query": "Hello from Phase 1",
  "session_id": "session-1"
}' localhost:50051 orchestrator.OrchestratorService/StreamChat
```

## Configuration

Environment variables are set in `docker-compose.yml`:

- `GRPC_PORT`: 50051 (gRPC service port)
- `POSTGRES_*`: Database connection for checkpointer
- `OPENROUTER_API_KEY`: LLM provider API key
- `ENABLE_TOOL_CALLBACKS`: false (Phase 1)

## Next Steps

### Phase 2: Tool Integration
1. Implement tool callback gRPC service in backend
2. Wire up tool registry
3. Test tool calls end-to-end

### Phase 3: Agent Migration
1. Copy agents from backend
2. Implement full orchestration logic
3. Wire up state management

### Phase 4: Production Cutover
1. Add feature flag in backend API
2. Proxy requests to gRPC orchestrator
3. Gradual rollout (10% → 50% → 100%)
4. Remove legacy orchestrator

## Safety Features

- **Non-destructive**: Runs alongside existing backend
- **Independent**: Can be stopped without affecting main app
- **Testable**: Isolated testing environment
- **Rollback-ready**: Easy to disable if issues arise

## Development Workflow

```bash
# Make changes to orchestrator code
vim llm-orchestrator/orchestrator/grpc_service.py

# Restart only orchestrator (15 seconds)
docker compose restart llm-orchestrator

# Test changes
# Main backend/frontend still running normally
```

## Adding New Agents

When implementing new agents in the orchestrator:

### 1. Agent Returns gRPC ChatChunks

Your agent should yield `orchestrator_pb2.ChatChunk` messages:

```python
yield orchestrator_pb2.ChatChunk(
    type="status",  # or "content", "complete", "error"
    message="Your message here",
    timestamp=datetime.now().isoformat(),
    agent_name="your_agent_name"
)
```

### 2. Streaming Format is Handled Automatically

**IMPORTANT:** Do NOT manually format JSON in your agent code!

The backend proxy (`backend/api/grpc_orchestrator_proxy.py`) automatically converts your gRPC chunks to properly formatted JSON for the frontend using the centralized `format_sse_message()` function.

**Your agent code:** Just yield gRPC protobuf messages  
**Proxy handles:** Converting to valid JSON with proper escaping  
**Frontend receives:** Valid JSON that JavaScript can parse

See project documentation for complete streaming format documentation.

### 3. Standard Message Types

Use these standard types:
- `"status"` - Progress updates, intermediate steps
- `"content"` - Main response content
- `"complete"` - Task completion signal
- `"error"` - Error notifications

## Monitoring

Check orchestrator status:

```bash
# Container status
docker ps | grep llm-orchestrator

# Recent logs
docker compose logs --tail=50 llm-orchestrator

# Follow logs
docker compose logs -f llm-orchestrator

# Health check
docker compose exec llm-orchestrator python -c "import grpc; print('OK')"
```

## Troubleshooting

**Container won't start:**
- Check logs: `docker compose logs llm-orchestrator`
- Verify database is healthy: `docker compose ps postgres`
- Check port 50051 isn't in use: `lsof -i :50051`

**gRPC connection fails:**
- Ensure container is running: `docker ps`
- Check network: `docker network inspect bastion-network`
- Verify health check: see Monitoring section above

## Important Notes

⚠️ **Phase 1 is for testing infrastructure only**
- Does NOT handle production traffic
- Echo responses only
- No agent logic yet
- Main backend unchanged

✅ **Safe to run in parallel**
- No impact on existing functionality
- Can be stopped anytime
- Independent restart for testing

