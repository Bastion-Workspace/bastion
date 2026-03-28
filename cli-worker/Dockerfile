FROM python:3.11-slim

# Install CLI tools and runtime deps
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

# Python deps only (no backend)
COPY cli-worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proto and cli-worker code only
COPY protos /app/protos
COPY cli-worker /app/cli_worker

# Generate gRPC code from cli_service.proto
RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    --pyi_out=/app \
    /app/protos/cli_service.proto

# Unprivileged user
RUN useradd --create-home --shell /bin/bash cliworker \
    && mkdir -p /scratch /tmp/cli-worker \
    && chown -R cliworker:cliworker /app /scratch /tmp/cli-worker
USER cliworker

# Generated _pb2 modules are written under the proto path (e.g. /app/protos/)
# so Python must have /app/protos on the path to find cli_service_pb2
ENV PYTHONPATH=/app:/app/protos
ENV CLI_SCRATCH_DIR=/scratch

EXPOSE 50060

WORKDIR /app
CMD ["python", "-m", "cli_worker.main"]
