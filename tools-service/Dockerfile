FROM python:3.11-slim

# Install system dependencies for document processing and web operations
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared proto files
COPY protos /app/protos

# Copy VERSION file for version management
COPY VERSION /app/VERSION

# Copy backend code (needed for repositories, config, and services)
COPY backend /app/backend

# Copy tools service code
COPY tools-service /app/tools-service

# Generate gRPC code from shared proto files
RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    --pyi_out=/app \
    /app/protos/tool_service.proto \
    /app/protos/orchestrator.proto \
    /app/protos/vector_service.proto \
    /app/protos/data_service.proto \
    /app/protos/crawl_service.proto

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs

# Set PYTHONPATH to include /app and /app/backend for imports
ENV PYTHONPATH=/app:/app/backend

# Expose gRPC port
EXPOSE 50052

# Change to backend directory so relative imports work
# Then run the tools service script
WORKDIR /app/backend
CMD ["python", "/app/tools-service/main.py"]

