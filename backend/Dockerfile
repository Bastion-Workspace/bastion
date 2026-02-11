FROM python:3.11-slim

# Install system dependencies for OCR, document processing, and web crawling
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

# Copy VERSION file for version management
COPY VERSION /app/VERSION

# Copy application code
COPY backend /app

# Copy tools-service code (needed for imports like tools_service.models.rss_models)
# NOTE: Directory is tools-service but we copy as tools_service so Python can import it
COPY tools-service /app/tools_service

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
    /app/protos/crawl_service.proto \
    /app/protos/image_vision.proto \
    /app/protos/connections_service.proto

# Make startup script executable
RUN chmod +x docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/uploads /app/processed /app/logs

# Set PYTHONPATH to include tools_service for imports
ENV PYTHONPATH=/app:/app/tools_service

# Expose ports (8000 for FastAPI)
EXPOSE 8000

# Run the application (V2 with PostgreSQL migration)
CMD ["./docker-entrypoint.sh"]
