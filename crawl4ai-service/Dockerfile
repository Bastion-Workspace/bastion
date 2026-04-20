FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY crawl4ai-service/requirements.txt crawl4ai-service/requirements-build.txt ./
# Install codegen to default site-packages (protoc). Runtime deps go to --prefix only.
# Use --ignore-installed so deps are not skipped when already satisfied in /usr/local
# (e.g. after setuptools); the final stage copies only /install.
RUN pip install --no-cache-dir -r requirements-build.txt && \
    pip install --no-cache-dir --prefix=/install --ignore-installed -r requirements.txt

ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright

RUN mkdir -p /opt/ms-playwright && \
    PYTHONPATH=/install/lib/python3.11/site-packages \
    PATH=/install/bin:${PATH} \
    python3 -m playwright install chromium

RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    PATH=/install/bin:${PATH} \
    crawl4ai-setup

COPY protos /app/protos
COPY crawl4ai-service /app

RUN python3 -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    --pyi_out=/app \
    /app/protos/crawl_service.proto

FROM python:3.11-slim

ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 \
    libatk-bridge2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright

COPY --from=builder /install /usr/local
COPY --from=builder /opt/ms-playwright /opt/ms-playwright
COPY --from=builder /app/protos /app/protos
COPY crawl4ai-service /app

RUN groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash crawluser && \
    chown -R crawluser:appgroup /app /opt/ms-playwright
USER crawluser

EXPOSE 50055

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('localhost', 50055)); s.close()" || exit 1

CMD ["python", "main.py"]
