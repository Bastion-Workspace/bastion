FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY vector-service/requirements.txt vector-service/requirements-build.txt ./
RUN pip install --no-cache-dir -r requirements-build.txt && \
    pip install --no-cache-dir --prefix=/install --ignore-installed -r requirements.txt

COPY protos /app/protos

RUN mkdir -p /app/_proto_out && \
    python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app/_proto_out \
    --grpc_python_out=/app/_proto_out \
    /app/protos/vector_service.proto && \
    cp /app/_proto_out/protos/vector_service_pb2.py /app/protos/ && \
    cp /app/_proto_out/protos/vector_service_pb2_grpc.py /app/protos/ && \
    rm -rf /app/_proto_out

FROM python:3.11-slim

# Must match docker-compose `user:` (BASTION_RUNTIME_* / PUID/PGID from .env). Pass as build-args
# so /app and /data ownership matches the runtime user (SQLite cache on vector-embedding-cache).
ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /app/protos /app/protos

COPY vector-service /app

RUN groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash vectoruser && \
    mkdir -p /data /app/.cache && \
    chown -R "${BASTION_RUNTIME_UID}:${BASTION_RUNTIME_GID}" /app /data
USER vectoruser

EXPOSE 50053

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('localhost', 50053)); s.close()" || exit 1

CMD ["python", "main.py"]
