FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY voice-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY protos /app/protos
RUN touch /app/protos/__init__.py

COPY voice-service /app

RUN python -m grpc_tools.protoc \
    -I/app \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/protos/voice_service.proto

EXPOSE 50059

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import socket; s = socket.socket(); s.settimeout(5); s.connect(('localhost', 50059)); s.close()" || exit 1

CMD ["python", "main.py"]
