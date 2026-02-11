FROM python:3.11-slim

# Install system dependencies for document processing and web operations
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    build-essential \
    python3-dev \
    chromium \
    libnss3 \
    libatk-bridge2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
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
# NOTE: Directory is tools-service but we copy as tools_service so Python can import it
COPY tools-service /app/tools_service

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
    /app/protos/crawl_service.proto \
    /app/protos/image_vision.proto

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs

# Set PYTHONPATH to include /app and /app/backend for imports
# CRITICAL: /app/backend must be first so 'from models.xxx' imports work for backend code
# Then /app/tools_service for tools-service imports, then /app for general imports
ENV PYTHONPATH=/app/backend:/app:/app/tools_service

# Expose gRPC port
EXPOSE 50052

# Run from /app so paths are consistent
# The tools_service/main.py will handle path setup
WORKDIR /app
CMD ["python", "/app/tools_service/main.py"]

