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

COPY llm-orchestrator/requirements-build.txt llm-orchestrator/requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements-build.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY protos /app/protos

# Nested out dir avoids grpc_tools truncating large tool_service.proto when output overwrites input.
RUN mkdir -p /app/_proto_out && \
    python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app/_proto_out \
    --grpc_python_out=/app/_proto_out \
    /app/protos/orchestrator.proto \
    /app/protos/tool_service.proto \
    /app/protos/document_service.proto \
    /app/protos/cli_service.proto \
    /app/protos/vector_service.proto \
    /app/protos/data_service.proto \
    /app/protos/crawl_service.proto \
    /app/protos/voice_service.proto && \
    cp /app/_proto_out/protos/*_pb2.py /app/protos/ && \
    cp /app/_proto_out/protos/*_pb2_grpc.py /app/protos/ && \
    rm -rf /app/_proto_out

RUN pip uninstall -y grpcio-tools && pip cache purge

# Runtime: pure Python gRPC + LangGraph; no system libs required for wheels.
FROM python:3.11-slim

ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

RUN groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash appuser

WORKDIR /app
ENV PYTHONPATH=/app
ENV PATH=/opt/venv/bin:$PATH

COPY --from=builder /opt/venv /opt/venv
COPY llm-orchestrator /app
COPY --from=builder /app/protos /app/protos

RUN mkdir -p /app/logs && chown -R appuser:appgroup /app

USER appuser

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('127.0.0.1', 50051)); s.close()" || exit 1

CMD ["python", "main.py"]
