# Stage 1: gRPC codegen only (grpcio-tools stays out of the runtime image)
FROM python:3.11-slim AS builder

WORKDIR /app

COPY cli-worker/requirements-build.txt .
RUN pip install --no-cache-dir -r requirements-build.txt

COPY protos /app/protos

RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    --pyi_out=/app \
    /app/protos/cli_service.proto

# Stage 2: runtime — CLI tools + Python deps + generated protos
FROM python:3.11-slim

ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ghostscript \
    imagemagick \
    jpegoptim \
    librsvg2-bin \
    optipng \
    pandoc \
    poppler-utils \
    qrencode \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    graphviz \
    libimage-exiftool-perl \
    texlive-latex-base \
    texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY cli-worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app/protos /app/protos

COPY cli-worker /app/cli_worker

RUN groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash cliworker && \
    mkdir -p /scratch /tmp/cli-worker && \
    chown -R cliworker:appgroup /app /scratch /tmp/cli-worker
USER cliworker

ENV PYTHONPATH=/app:/app/protos
ENV CLI_SCRATCH_DIR=/scratch

EXPOSE 50060

WORKDIR /app
CMD ["python", "-m", "cli_worker.main"]
