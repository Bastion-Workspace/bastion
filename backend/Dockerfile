FROM python:3.11-slim

# Install system dependencies for OCR, document processing, video processing, and web crawling
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    poppler-utils \
    ghostscript \
    qpdf \
    unpaper \
    curl \
    postgresql-client \
    build-essential \
    python3-dev \
    wget \
    gnupg \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy verification script first
COPY backend/verify_spacy.py .

# Download spaCy model with explicit verification
RUN python -m spacy download en_core_web_lg && \
    python verify_spacy.py

# Copy shared proto files first
COPY protos /app/protos

# Copy application code
COPY backend /app

# Generate gRPC code from shared proto files (after dependencies installed)
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

# Make startup script executable
RUN chmod +x docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/uploads /app/processed /app/logs

# Expose ports (8000 for FastAPI, 50052 for gRPC tool service)
EXPOSE 8000 50052

# Run the application (V2 with PostgreSQL migration)
CMD ["./docker-entrypoint.sh"]
