FROM python:3.11-slim

ARG BASTION_RUNTIME_UID=10001
ARG BASTION_RUNTIME_GID=10001

WORKDIR /app

COPY bbs-server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bbs-server /app

COPY bbs-server/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod 755 /docker-entrypoint.sh /app/scripts/ensure_ssh_host_key.py

RUN groupadd --gid "${BASTION_RUNTIME_GID}" appgroup && \
    useradd --uid "${BASTION_RUNTIME_UID}" --gid appgroup --create-home --shell /bin/bash bbsuser && \
    chown -R bbsuser:appgroup /app

ENV PYTHONPATH=/app

EXPOSE 2323 2222

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import os,socket;t=lambda v:str(v or '').strip().lower() in('1','true','yes','on');p=(int(os.getenv('BBS_TELNET_PORT','2323'))if t(os.getenv('BBS_ENABLE_TELNET','0'))else int(os.getenv('BBS_SSH_PORT','2222')));s=socket.socket();s.settimeout(5);s.connect(('127.0.0.1',p));s.close()" || exit 1

USER root
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "main.py"]
