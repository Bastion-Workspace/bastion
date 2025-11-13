# Shared Protocol Buffers

This directory contains the gRPC protocol buffer definitions shared between the backend and llm-orchestrator services.

## Files

- **tool_service.proto** - Backend tool service API for document search, web operations, and query enhancement
- **orchestrator.proto** - Orchestrator service API for LLM-based conversation management

## Architecture

Both `backend` and `llm-orchestrator` services generate gRPC code from these shared proto files during their Docker build process.

### Build Process

1. During `docker compose build`, the shared `protos/` directory is copied into each container
2. The `protoc` compiler generates Python code (`*_pb2.py` and `*_pb2_grpc.py`) inside each container
3. The generated code is placed in `/app/protos/` within the container

### Why Shared Protos?

- **Single source of truth** - Proto definitions exist in one place
- **Consistency** - Both services use identical protocol definitions
- **Maintainability** - Changes only need to be made once
- **Version control** - Easy to track API changes over time

## Making Changes

When you modify a proto file:

1. Edit the `.proto` file in this directory
2. Rebuild affected containers: `docker compose build backend llm-orchestrator`
3. Restart services: `docker compose up -d backend llm-orchestrator`

## Generated Files

Generated `*_pb2.py` and `*_pb2_grpc.py` files are created during the Docker build process and should **NOT** be committed to version control. They are automatically regenerated on each build.

