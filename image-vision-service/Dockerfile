FROM python:3.11-slim

# Install system dependencies for dlib/face_recognition and OpenCV (for YOLO)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY image-vision-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared proto files from root
COPY protos /app/protos

# Ensure protos is a Python package
RUN touch /app/protos/__init__.py

# Copy service code
COPY image-vision-service /app

# Generate gRPC code from protos (FORCE REBUILD 2026-01-22-1)
RUN echo "=== Proto files before generation ===" && \
    ls -la /app/protos/ && \
    echo "=== Running protoc ===" && \
    python -m grpc_tools.protoc \
        -I/app \
        --python_out=/app \
        --grpc_python_out=/app \
        /app/protos/image_vision.proto && \
    echo "=== Proto files after generation ===" && \
    ls -la /app/protos/ && \
    echo "=== Verifying generated files ===" && \
    test -f /app/protos/image_vision_pb2.py || (echo "ERROR: image_vision_pb2.py not generated" && exit 1) && \
    test -f /app/protos/image_vision_pb2_grpc.py || (echo "ERROR: image_vision_pb2_grpc.py not generated" && exit 1) && \
    echo "=== SUCCESS: Proto generation verified ==="

# Expose gRPC port
EXPOSE 50056

CMD ["python", "main.py"]
