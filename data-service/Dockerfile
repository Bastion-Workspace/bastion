FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY data-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY data-service/ .
COPY protos/ ./protos/

# Generate gRPC code from proto files
RUN python -m grpc_tools.protoc \
    -I./grpc \
    --python_out=. \
    --grpc_python_out=. \
    --pyi_out=. \
    ./grpc/data_service.proto

# Expose gRPC port
EXPOSE 50054

# Run the service
CMD ["python", "main.py"]





