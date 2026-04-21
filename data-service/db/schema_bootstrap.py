"""
Apply bundled data-workspace DDL when postgres-data was created without running Docker init.

If the named volume already contained PGDATA (e.g. from an older image), the entrypoint skips
init scripts and public.data_workspaces never gets created. We detect that and run sql/01_init.sql
via psql (same file shipped in the data-service image under /app/sql/).
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path

import asyncpg

from config.settings import settings

logger = logging.getLogger(__name__)

_INIT_SQL = Path(__file__).resolve().parent.parent / "sql" / "01_init.sql"


async def ensure_data_workspace_schema(conn: asyncpg.Connection) -> None:
    exists = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'data_workspaces'
        )
        """
    )
    if exists:
        return

    if not _INIT_SQL.is_file():
        raise RuntimeError(f"Bundled data workspace schema not found at {_INIT_SQL}")

    logger.warning(
        "public.data_workspaces is missing (postgres-data init was likely skipped on an "
        "existing volume). Applying bundled sql/01_init.sql via psql."
    )

    def _run_psql() -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PGPASSWORD"] = settings.POSTGRES_PASSWORD
        cmd = [
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-h",
            settings.POSTGRES_HOST,
            "-p",
            str(settings.POSTGRES_PORT),
            "-U",
            settings.POSTGRES_USER,
            "-d",
            settings.POSTGRES_DB,
            "-f",
            str(_INIT_SQL),
        ]
        return subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

    result = await asyncio.to_thread(_run_psql)
    if result.returncode != 0:
        logger.error("psql schema bootstrap stderr: %s", result.stderr)
        logger.error("psql schema bootstrap stdout: %s", result.stdout)
        raise RuntimeError(
            f"Failed to apply data workspace schema (psql exit {result.returncode})"
        )

    ok = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'data_workspaces'
        )
        """
    )
    if not ok:
        raise RuntimeError("Schema bootstrap ran but public.data_workspaces is still missing")

    logger.info("Data workspace schema bootstrap completed successfully")
