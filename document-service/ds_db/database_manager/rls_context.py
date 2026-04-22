"""Async context manager for PostgreSQL RLS session variables.

Unsafe with DatabaseManager pooling: each internal execute() may use a different
connection than the following fetch_one/execute. Prefer passing rls_context=dict
into database_helpers.execute/fetch_one (see document_repository._db_rls_context).
"""

from contextlib import asynccontextmanager
from typing import Optional

from ds_db.database_manager.database_helpers import execute


@asynccontextmanager
async def rls_context(user_id: Optional[str]):
    """Set app.current_user_id and app.current_user_role for the current connection."""
    if user_id:
        await execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
        await execute("SELECT set_config('app.current_user_role', 'user', false)")
    else:
        await execute("SELECT set_config('app.current_user_id', '', false)")
        await execute("SELECT set_config('app.current_user_role', 'admin', false)")
    yield
