# Builder: compile wheels, install deps, gRPC codegen, spaCy model; strip grpcio-tools before final copy.
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY document-service/requirements-build.txt document-service/requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements-build.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -m spacy download en_core_web_lg

COPY protos /app/protos
COPY bastion_indexing /app/bastion_indexing

RUN mkdir -p /app/_proto_out && \
    python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app/_proto_out \
    --grpc_python_out=/app/_proto_out \
    --pyi_out=/app/_proto_out \
    /app/protos/tool_service.proto \
    /app/protos/document_service.proto \
    /app/protos/vector_service.proto && \
    cp /app/_proto_out/protos/*_pb2.py /app/protos/ && \
    cp /app/_proto_out/protos/*_pb2_grpc.py /app/protos/ && \
    (cp /app/_proto_out/protos/*_pb2.pyi /app/protos/ 2>/dev/null || true) && \
    rm -rf /app/_proto_out

RUN pip uninstall -y grpcio-tools && pip cache purge

# Runtime: OCR/PDF pipeline native libs only; no compilers or grpcio-tools.
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
ENV PYTHONPATH=/app/ds/shims:/app/ds:/app
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/protos /app/protos
COPY --from=builder /app/bastion_indexing /app/bastion_indexing
COPY VERSION /app/VERSION
COPY vector-service/data/bm25_default_idf.json /app/data/bm25_default_idf.json
COPY document-service /app/ds

RUN mkdir -p /app/uploads /app/logs /app/data && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 50058

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import socket; s = socket.socket(); s.settimeout(10); s.connect(('127.0.0.1', 50058)); s.close()" || exit 1

CMD ["python", "/app/ds/main.py"]
