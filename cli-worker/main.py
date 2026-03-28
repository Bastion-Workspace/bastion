"""
CLI Worker - gRPC service for sandboxed CLI tool execution.
"""
import asyncio
import logging
import os

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    from cli_worker.service.grpc_service import serve
    port = int(os.getenv("CLI_WORKER_PORT", "50060"))
    logger.info("Starting CLI Worker on port %s", port)
    await serve(port)


if __name__ == "__main__":
    asyncio.run(main())
