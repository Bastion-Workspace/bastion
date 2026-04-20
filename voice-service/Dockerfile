FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY voice-service/requirements.txt voice-service/requirements-build.txt ./
RUN pip install --no-cache-dir -r requirements-build.txt && \
    pip install --no-cache-dir --prefix=/install --ignore-installed -r requirements.txt

COPY protos /app/protos
RUN touch /app/protos/__init__.py

RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/protos/voice_service.proto

FROM python:3.11-slim

ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /app/protos /app/protos

COPY voice-service /app

RUN mkdir -p /app/models && \
    groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash voiceuser && \
    chown -R voiceuser:appgroup /app
USER voiceuser

EXPOSE 50059

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('localhost', 50059)); s.close()" || exit 1

CMD ["python", "main.py"]
