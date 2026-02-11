FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY connections-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY protos /app/protos
COPY connections-service /app

RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/protos/connections_service.proto

# Generated _pb2 modules are written under the proto path (e.g. /app/protos/)
# so Python must have /app/protos on the path to find connections_service_pb2
ENV PYTHONPATH=/app:/app/protos

EXPOSE 50057

CMD ["python", "main.py"]
