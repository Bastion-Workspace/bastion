FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY data-service/requirements.txt data-service/requirements-build.txt ./
RUN pip install --no-cache-dir -r requirements-build.txt && \
    pip install --no-cache-dir --prefix=/install --ignore-installed -r requirements.txt

COPY data-service/ .
COPY protos/ ./protos/

RUN python3 -m grpc_tools.protoc \
    -I./grpc \
    --python_out=. \
    --grpc_python_out=. \
    --pyi_out=. \
    ./grpc/data_service.proto

FROM python:3.11-slim

ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

WORKDIR /app

COPY --from=builder /install /usr/local
COPY data-service/ .
COPY protos/ ./protos/
COPY --from=builder /app/data_service_pb2.py /app/data_service_pb2.py
COPY --from=builder /app/data_service_pb2_grpc.py /app/data_service_pb2_grpc.py

RUN groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash datauser && \
    chown -R datauser:appgroup /app
USER datauser

EXPOSE 50054

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('localhost', 50054)); s.close()" || exit 1

CMD ["python", "main.py"]
