# Builder: venv, deps, gRPC codegen; strip grpcio-tools (no compilers in runtime).
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY connections-service/requirements-build.txt connections-service/requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements-build.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY protos /app/protos

RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/protos/connections_service.proto \
    /app/protos/voice_service.proto

RUN pip uninstall -y grpcio-tools && pip cache purge

# Runtime: wheels-only venv; no apt / no grpcio-tools.
FROM python:3.11-slim

ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

RUN groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash appuser

WORKDIR /app

ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONPATH=/app:/app/protos
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/protos /app/protos
COPY connections-service /app

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 50057

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('127.0.0.1', 50057)); s.close()" || exit 1

CMD ["python", "main.py"]
