# Microservices Documentation Index

## Overview

This directory contains comprehensive documentation for our gRPC-based microservices architecture. Start here to understand how our AI orchestration services communicate and scale independently.

---

## Documentation Structure

### 1. **Quick Start** - Get up and running fast
📄 **[GRPC_QUICK_START.md](./GRPC_QUICK_START.md)**

**For:** Developers who need to add a new gRPC method quickly

**Contains:**
- 5-minute guide to adding RPC methods
- Code templates and patterns
- Common mistakes to avoid
- Quick troubleshooting checklist

**Use when:** You need to add a new tool, service method, or data access endpoint

---

### 2. **Architecture Guide** - Understand the system
📄 **[GRPC_MICROSERVICES_ARCHITECTURE.md](./GRPC_MICROSERVICES_ARCHITECTURE.md)**

**For:** Developers who need to understand the full architecture

**Contains:**
- Complete architecture topology
- Service communication patterns
- Protocol Buffers deep dive
- Connection management
- Error handling strategies
- Docker integration
- Feature flags and rollout
- Testing strategies
- Performance optimization
- Monitoring and observability

**Use when:** You're new to the project, planning major changes, or need architectural guidance

---

### 3. **Phase Documentation** - Implementation details
📄 **[llm-orchestrator/PHASE5_COMPLETE.md](../llm-orchestrator/PHASE5_COMPLETE.md)**  
📄 **[llm-orchestrator/PHASE6_INTEGRATION.md](../llm-orchestrator/PHASE6_INTEGRATION.md)**

**For:** Understanding the migration journey

**Contains:**
- Step-by-step implementation phases
- Research agent migration details
- Integration points with backend
- Testing procedures for each phase

**Use when:** You need historical context or are planning similar migrations

---

## Quick Reference

### Service Ports

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Backend FastAPI | 8081 | HTTP | REST API, WebSocket |
| Backend Tool Service | 50052 | gRPC | Data access for orchestrator |
| LLM Orchestrator | 50051 | gRPC | AI orchestration service |

### Key Files

| File | Purpose |
|------|---------|
| `backend/protos/tool_service.proto` | Backend tool service contract |
| `backend/protos/orchestrator.proto` | Orchestrator service contract |
| `backend/services/grpc_tool_service.py` | Tool service implementation |
| `backend/api/grpc_orchestrator_proxy.py` | Proxy for routing to orchestrator |
| `backend/utils/feature_flags.py` | Feature flag routing logic |
| `llm-orchestrator/orchestrator/grpc_service.py` | Orchestrator service implementation |
| `llm-orchestrator/orchestrator/backend_tool_client.py` | Client for backend tools |
| `llm-orchestrator/orchestrator/agents/full_research_agent.py` | Research agent implementation |

---

## Common Tasks

### Task: Add a New RPC Method

1. Read: [GRPC_QUICK_START.md](./GRPC_QUICK_START.md) - 5-minute guide
2. Update: Proto files in both `backend/` and `llm-orchestrator/`
3. Implement: Server in `grpc_tool_service.py`
4. Implement: Client in `backend_tool_client.py`
5. Create: LangGraph tool wrapper
6. Test: Rebuild and verify

### Task: Understand Service Communication

1. Read: [GRPC_MICROSERVICES_ARCHITECTURE.md](./GRPC_MICROSERVICES_ARCHITECTURE.md) - Service Topology section
2. Review: Docker Compose service definitions
3. Check: Connection code in client implementations
4. Explore: Example RPC calls in test files

### Task: Debug gRPC Issues

1. Check: [GRPC_QUICK_START.md](./GRPC_QUICK_START.md) - Troubleshooting Checklist
2. Verify: Services running (`docker ps`)
3. Check: Logs (`docker logs codex-dev-backend`, `docker logs codex-dev-llm-orchestrator`)
4. Test: Connection from within containers
5. Review: [GRPC_MICROSERVICES_ARCHITECTURE.md](./GRPC_MICROSERVICES_ARCHITECTURE.md) - Error Handling section

### Task: Migrate a New Component to Microservice

1. Read: [GRPC_MICROSERVICES_ARCHITECTURE.md](./GRPC_MICROSERVICES_ARCHITECTURE.md) - Migration Strategy section
2. Review: Phase documentation for examples
3. Implement: Feature flag for gradual rollout
4. Create: Parallel implementation
5. Test: Canary deployment (5%)
6. Roll out: Gradual increase to 100%
7. Clean up: Remove legacy code

### Task: Performance Optimization

1. Read: [GRPC_MICROSERVICES_ARCHITECTURE.md](./GRPC_MICROSERVICES_ARCHITECTURE.md) - Performance Optimization section
2. Profile: Identify bottlenecks
3. Optimize: Connection pooling, message sizes, compression
4. Test: Load testing with concurrent requests
5. Monitor: Latency, throughput, error rates

---

## Architecture Overview

### High-Level Flow

```
User → Frontend → Backend → Feature Flag Router
                              ↓
                    ┌─────────┴────────┐
                    │                  │
                    ↓                  ↓
            Local Orchestrator    gRPC Proxy
                                      ↓
                               LLM Orchestrator (50051)
                                      ↓
                               Backend Tool Client
                                      ↓
                               Backend Tool Service (50052)
                                      ↓
                               Data Access Layer
```

### Communication Pattern

1. **Frontend** sends user query to backend (HTTP/REST)
2. **Backend** checks feature flag for routing decision
3. If gRPC enabled:
   - **Backend** forwards to orchestrator via gRPC (port 50051)
   - **Orchestrator** processes with research agent
   - **Orchestrator** calls backend tools via gRPC (port 50052)
   - **Backend tools** access data (database, search, etc.)
   - Response streams back through the chain
4. If gRPC disabled:
   - **Backend** uses local LangGraph orchestrator
   - Direct access to services and data

---

## Testing

### Unit Tests

```bash
# Test backend tool service
docker exec codex-dev-backend python test_tool_service.py

# Test orchestrator
docker exec codex-dev-llm-orchestrator python test_phase3.py

# Test full research agent
docker exec codex-dev-llm-orchestrator python test_phase5.py
```

### Integration Tests

```bash
# Test end-to-end flow through frontend
# Open browser to http://localhost:3001
# Ask a research question
# Watch logs to verify gRPC routing
docker logs -f codex-dev-backend 2>&1 | grep "ROUTING"
docker logs -f codex-dev-llm-orchestrator
```

### Health Checks

```bash
# Backend health
curl http://localhost:8081/health

# Orchestrator health (via proxy)
curl http://localhost:8081/api/async/orchestrator/grpc/health
```

---

## Development Workflow

### Making Changes

1. **Update proto files** (if changing contracts)
   - Edit `backend/protos/*.proto`
   - Copy to `llm-orchestrator/protos/*.proto`
   - Keep files synchronized

2. **Implement changes**
   - Backend: `backend/services/grpc_tool_service.py`
   - Orchestrator: `llm-orchestrator/orchestrator/`

3. **Rebuild and test**
   ```bash
   docker compose build backend llm-orchestrator
   docker compose up -d
   ```

4. **Verify**
   - Check logs for errors
   - Run test scripts
   - Test through frontend

### Debugging

```bash
# View live logs
docker logs -f codex-dev-backend
docker logs -f codex-dev-llm-orchestrator

# Enter container for debugging
docker exec -it codex-dev-backend bash
docker exec -it codex-dev-llm-orchestrator bash

# Check gRPC connectivity
docker exec codex-dev-backend python -c "
import grpc
channel = grpc.insecure_channel('llm-orchestrator:50051')
print('Connected!' if channel else 'Failed')
"
```

---

## Feature Flags

Control routing between local and gRPC orchestrator:

```yaml
# docker-compose.yml
environment:
  - USE_GRPC_ORCHESTRATOR=true   # Enable/disable gRPC routing
  - GRPC_ORCHESTRATOR_PERCENTAGE=100  # 0-100% traffic to gRPC
```

### Rollout Stages

- **0%**: Testing only, no production traffic
- **5%**: Canary deployment, monitor closely
- **50%**: Half traffic, validate stability
- **100%**: Full rollout, production ready

---

## Best Practices

### ✅ DO:

- Keep proto files in sync across services
- Reuse gRPC channels (expensive to create)
- Implement comprehensive error handling
- Use feature flags for gradual rollout
- Add health checks to all services
- Log structured data for monitoring
- Test services independently
- Set appropriate timeouts
- Handle connection failures gracefully
- Version your proto APIs

### ❌ DON'T:

- Change field numbers in proto files
- Create new channel for each request
- Ignore error handling
- Deploy without testing
- Skip health checks
- Hardcode service URLs
- Return huge responses without streaming
- Mix local and microservice calls without flags
- Log sensitive data
- Make breaking proto changes without versioning

---

## Resources

### Internal Documentation

- [GRPC_QUICK_START.md](./GRPC_QUICK_START.md) - Quick guide for adding RPC methods
- [GRPC_MICROSERVICES_ARCHITECTURE.md](./GRPC_MICROSERVICES_ARCHITECTURE.md) - Complete architecture guide
- [llm-orchestrator/PHASE5_COMPLETE.md](../llm-orchestrator/PHASE5_COMPLETE.md) - Research agent migration
- [llm-orchestrator/PHASE6_INTEGRATION.md](../llm-orchestrator/PHASE6_INTEGRATION.md) - Backend integration

### External Resources

- [gRPC Python Documentation](https://grpc.io/docs/languages/python/)
- [Protocol Buffers Guide](https://protobuf.dev/)
- [gRPC Best Practices](https://grpc.io/docs/guides/performance/)
- [gRPC Error Handling](https://grpc.io/docs/guides/error/)

### Example Code

- Backend tool service: `backend/services/grpc_tool_service.py`
- Orchestrator service: `llm-orchestrator/orchestrator/grpc_service.py`
- Tool client: `llm-orchestrator/orchestrator/backend_tool_client.py`
- Research agent: `llm-orchestrator/orchestrator/agents/full_research_agent.py`

---

## Questions?

1. **Quick task?** → Start with [GRPC_QUICK_START.md](./GRPC_QUICK_START.md)
2. **Need context?** → Read [GRPC_MICROSERVICES_ARCHITECTURE.md](./GRPC_MICROSERVICES_ARCHITECTURE.md)
3. **Debugging?** → Check troubleshooting sections in both docs
4. **Planning migration?** → Review Phase documentation and migration strategy

---

**Last Updated:** November 2025 (Phase 6)  
**Architecture Status:** Production Ready  
**Feature Flag:** Enabled (100% traffic to gRPC orchestrator)







