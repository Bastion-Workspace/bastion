FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY image-vision-service/requirements.txt image-vision-service/requirements-build.txt ./
RUN pip install --no-cache-dir -r requirements-build.txt && \
    pip install --no-cache-dir --prefix=/install --ignore-installed -r requirements.txt

COPY protos /app/protos
RUN touch /app/protos/__init__.py

COPY image-vision-service /app

RUN python3 -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/protos/image_vision.proto && \
    test -f /app/protos/image_vision_pb2.py && \
    test -f /app/protos/image_vision_pb2_grpc.py

FROM python:3.11-slim

ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /app/protos /app/protos

COPY image-vision-service /app

RUN groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash visionuser && \
    chown -R visionuser:appgroup /app
USER visionuser

EXPOSE 50056

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('localhost', 50056)); s.close()" || exit 1

CMD ["python", "main.py"]
