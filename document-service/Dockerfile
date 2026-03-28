FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and spaCy model
COPY document-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_lg

# Copy shared protos from root
COPY protos /app/protos

# Copy service code
COPY document-service /app

# Generate gRPC code from protos
RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/protos/document_service.proto

# Expose gRPC port
EXPOSE 50058

# Run service
CMD ["python", "main.py"]
