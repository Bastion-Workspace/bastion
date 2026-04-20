# Builder: compile wheels, gRPC codegen; strip grpcio-tools (no Chromium/Node in builder).
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY backend/requirements-build.txt backend/requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements-build.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY protos /app/protos

RUN mkdir -p /app/_proto_out && \
    python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app/_proto_out \
    --grpc_python_out=/app/_proto_out \
    --pyi_out=/app/_proto_out \
    /app/protos/tool_service.proto \
    /app/protos/orchestrator.proto \
    /app/protos/vector_service.proto \
    /app/protos/data_service.proto \
    /app/protos/crawl_service.proto \
    /app/protos/image_vision.proto \
    /app/protos/connections_service.proto && \
    cp /app/_proto_out/protos/*_pb2.py /app/protos/ && \
    cp /app/_proto_out/protos/*_pb2_grpc.py /app/protos/ && \
    (cp /app/_proto_out/protos/*_pb2.pyi /app/protos/ 2>/dev/null || true) && \
    rm -rf /app/_proto_out

RUN pip uninstall -y grpcio-tools && pip cache purge

# Runtime: Chromium stack + Node (npx MCP) + uv/uvx in /usr/local; no compilers or grpcio-tools.
FROM python:3.11-slim

ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    chromium \
    libnss3 \
    libatk-bridge2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# System-wide uv/uvx (not under /root/.local) so non-root can invoke.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local sh

RUN groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash appuser

WORKDIR /app

ENV PYTHONPATH=/app/backend:/app:/app/tools_service
# uv install.sh places binaries in UV_INSTALL_DIR root (/usr/local/uv, not bin/)
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/local:/usr/bin:/bin
ENV UV_CACHE_DIR=/app/.cache/uv
ENV npm_config_cache=/app/.cache/npm
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/protos /app/protos
COPY VERSION /app/VERSION
COPY backend /app/backend
COPY tools-service /app/tools_service

RUN mkdir -p /app/uploads /app/logs /app/.cache/uv /app/.cache/npm && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 50052

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('127.0.0.1', 50052)); s.close()" || exit 1

CMD ["python", "/app/tools_service/main.py"]
