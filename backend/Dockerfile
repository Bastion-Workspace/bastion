# Builder: compile wheels, install Python deps, run gRPC codegen; strip grpcio-tools before final copy.
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY backend/requirements-build.txt backend/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-build.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY protos /app/protos
COPY VERSION /app/VERSION
COPY vector-service/data/bm25_default_idf.json /app/data/bm25_default_idf.json
COPY bastion_indexing /app/bastion_indexing
COPY backend /app
COPY tools-service /app/tools_service

# Nested --python_out avoids grpc_tools writing over large proto input paths.
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
    /app/protos/connections_service.proto \
    /app/protos/document_service.proto \
    /app/protos/voice_service.proto && \
    cp /app/_proto_out/protos/*_pb2.py /app/protos/ && \
    cp /app/_proto_out/protos/*_pb2_grpc.py /app/protos/ && \
    (cp /app/_proto_out/protos/*_pb2.pyi /app/protos/ 2>/dev/null || true) && \
    rm -rf /app/_proto_out

RUN pip uninstall -y grpcio-tools && pip cache purge

# Runtime: OCR/PDF/WeasyPrint native libs only; no compilers or grpcio-tools.
FROM python:3.11-slim

ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    poppler-utils \
    ghostscript \
    qpdf \
    unpaper \
    curl \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi8 \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash appuser

WORKDIR /app
ENV PYTHONPATH=/app:/app/tools_service
ENV PATH=/opt/venv/bin:$PATH
ENV UV_CACHE_DIR=/app/.cache/uv
ENV npm_config_cache=/app/.cache/npm

COPY --from=builder /opt/venv /opt/venv
COPY bastion_indexing /app/bastion_indexing
COPY backend /app
COPY tools-service /app/tools_service
COPY --from=builder /app/protos /app/protos
COPY VERSION /app/VERSION
COPY vector-service/data/bm25_default_idf.json /app/data/bm25_default_idf.json

RUN chmod +x /app/docker-entrypoint.sh && \
    mkdir -p /app/uploads /app/processed /app/logs /app/.cache/uv /app/.cache/npm && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000 50052

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health >/dev/null || exit 1

CMD ["./docker-entrypoint.sh"]
