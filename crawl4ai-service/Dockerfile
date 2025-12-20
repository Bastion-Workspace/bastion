FROM python:3.11-slim

# Install system dependencies for Crawl4AI and browser
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for Crawl4AI browser dependencies
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY crawl4ai-service/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers for Crawl4AI
RUN pip install playwright \
    && playwright install chromium

# Install Crawl4AI and run setup
RUN pip install crawl4ai>=0.7.2 \
    && crawl4ai-setup

# Copy shared proto files first
COPY protos /app/protos

# Copy application code
COPY crawl4ai-service /app

# Generate gRPC code from shared proto files
RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    --pyi_out=/app \
    /app/protos/crawl_service.proto

# Expose gRPC port
EXPOSE 50055

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('localhost', 50055)); s.close()" || exit 1

# Run the service
CMD ["python", "main.py"]







