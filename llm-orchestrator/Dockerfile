FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including PostgreSQL client libraries for psycopg)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY llm-orchestrator/requirements.txt .

# Install gRPC packages first to ensure proper dependency resolution
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir grpcio==1.76.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy shared proto files first
COPY protos /app/protos

# Copy application code
COPY llm-orchestrator /app

# Generate gRPC code from shared proto files with proper imports
RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/protos/orchestrator.proto \
    /app/protos/tool_service.proto \
    /app/protos/vector_service.proto \
    /app/protos/data_service.proto \
    /app/protos/crawl_service.proto

# Expose gRPC port
EXPOSE 50051

# Health check - simple port check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('localhost', 50051)); s.close()" || exit 1

# Run the orchestrator service
CMD ["python", "main.py"]

