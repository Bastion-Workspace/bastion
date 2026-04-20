"""
Append-only brief snapshots for agent lines (heartbeat output history and diff context).
"""

import difflib
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SNAPSHOT_CONTENT_MAX = 500_000
_SNAPSHOT_INJECT_PRIOR_MAX = 12_000
_DIFF_MAX_LINES = 120


async def insert_snapshot(
    line_id: str,
    user_id: str,
    content: str,
    source: str = "heartbeat_report",
) -> None:
    from services.database_manager.database_helpers import execute
    from utils.grpc_rls import grpc_user_rls as _rls

    text = (content or "")[:_SNAPSHOT_CONTENT_MAX]
    if not text.strip():
        return
    src = (source or "heartbeat_report")[:200]
    await execute(
        """
        INSERT INTO agent_line_brief_snapshots (line_id, user_id, content, source)
        VALUES ($1::uuid, $2, $3, $4)
        """,
        line_id,
        user_id,
        text,
        src,
        rls_context=_rls(user_id),
    )


async def prune_snapshots(line_id: str, user_id: str, keep: int = 100) -> None:
    """Keep the newest `keep` rows per line; delete older rows."""
    from services.database_manager.database_helpers import execute, fetch_all
    from utils.grpc_rls import grpc_user_rls as _rls

    rows = await fetch_all(
        """
        SELECT id FROM agent_line_brief_snapshots
        WHERE line_id = $1::uuid AND user_id = $2
        ORDER BY created_at DESC
        LIMIT $3
        """,
        line_id,
        user_id,
        keep,
        rls_context=_rls(user_id),
    )
    if not rows:
        return
    ids = [r["id"] for r in rows]
    await execute(
        """
        DELETE FROM agent_line_brief_snapshots
        WHERE line_id = $1::uuid AND user_id = $2
          AND NOT (id = ANY($3::uuid[]))
        """,
        line_id,
        user_id,
        ids,
        rls_context=_rls(user_id),
    )


async def list_snapshots(line_id: str, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    from services.database_manager.database_helpers import fetch_all
    from utils.grpc_rls import grpc_user_rls as _rls

    rows = await fetch_all(
        """
        SELECT id, content, source, created_at
        FROM agent_line_brief_snapshots
        WHERE line_id = $1::uuid AND user_id = $2
        ORDER BY created_at DESC
        LIMIT $3
        """,
        line_id,
        user_id,
        limit,
        rls_context=_rls(user_id),
    )
    out = []
    for r in rows or []:
        out.append(
            {
                "id": str(r["id"]),
                "source": r.get("source"),
                "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                "preview": ((r.get("content") or "")[:400] + "…") if len(r.get("content") or "") > 400 else (r.get("content") or ""),
            }
        )
    return out


async def get_snapshot_detail(line_id: str, user_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
    from services.database_manager.database_helpers import fetch_one
    from utils.grpc_rls import grpc_user_rls as _rls

    row = await fetch_one(
        """
        SELECT id, content, source, created_at
        FROM agent_line_brief_snapshots
        WHERE id = $1::uuid AND line_id = $2::uuid AND user_id = $3
        """,
        snapshot_id,
        line_id,
        user_id,
        rls_context=_rls(user_id),
    )
    if not row:
        return None
    return {
        "id": str(row["id"]),
        "content": row.get("content") or "",
        "source": row.get("source"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


async def _fetch_recent_snapshots(line_id: str, user_id: str, limit: int = 2) -> List[Dict[str, Any]]:
    from services.database_manager.database_helpers import fetch_all
    from utils.grpc_rls import grpc_user_rls as _rls

    return await fetch_all(
        """
        SELECT content, source, created_at
        FROM agent_line_brief_snapshots
        WHERE line_id = $1::uuid AND user_id = $2
        ORDER BY created_at DESC
        LIMIT $3
        """,
        line_id,
        user_id,
        limit,
        rls_context=_rls(user_id),
    )


def build_prior_brief_context_lines(rows: List[Dict[str, Any]]) -> List[str]:
    """Build injectable lines: most recent snapshot as PRIOR; unified diff vs previous snapshot."""
    if not rows:
        return []
    newest = (rows[0].get("content") or "").strip()
    if not newest:
        return []
    lines: List[str] = [
        "PRIOR PUBLISHED BRIEF (snapshot from last successful heartbeat):",
        "",
        newest[:_SNAPSHOT_INJECT_PRIOR_MAX] + ("…" if len(newest) > _SNAPSHOT_INJECT_PRIOR_MAX else ""),
        "",
    ]
    if len(rows) < 2:
        return lines
    older = (rows[1].get("content") or "").strip()
    if not older:
        return lines
    diff_lines = list(
        difflib.unified_diff(
            older.splitlines(),
            newest.splitlines(),
            fromfile="prior_run",
            tofile="last_run",
            lineterm="",
            n=2,
        )
    )
    if not diff_lines:
        lines.append("CHANGES (last two snapshots): (no textual diff)")
        lines.append("")
        return lines
    capped = diff_lines[:_DIFF_MAX_LINES]
    lines.append("CHANGES (unified diff between the two most recent snapshots):")
    lines.append("")
    lines.extend(capped)
    if len(diff_lines) > _DIFF_MAX_LINES:
        lines.append(f"... ({len(diff_lines) - _DIFF_MAX_LINES} more diff lines omitted)")
    lines.append("")
    return lines


async def resolve_canonical_brief_text(
    line_id: str,
    user_id: str,
    heartbeat_config: Optional[Dict[str, Any]],
    full_response: str,
) -> Tuple[str, str]:
    """Return (text, source) for snapshot storage; prefer workspace key from delivery config when set."""
    cfg = heartbeat_config if isinstance(heartbeat_config, dict) else {}
    delivery = cfg.get("delivery") if isinstance(cfg.get("delivery"), dict) else {}
    key = (delivery.get("canonical_snapshot_key") or delivery.get("publish_workspace_key") or "").strip()
    text = (full_response or "").strip()
    src = "heartbeat_report"
    if key:
        from services import agent_workspace_service

        ent = await agent_workspace_service.get_workspace_entry(line_id, key, user_id)
        val = (ent or {}).get("value") if isinstance(ent, dict) else None
        if val and str(val).strip():
            return str(val).strip(), f"workspace_key:{key}"
    return text, src


async def record_heartbeat_brief_snapshot(
    line_id: str,
    user_id: str,
    full_response: str,
    heartbeat_config: Optional[Dict[str, Any]],
) -> None:
    """Persist canonical brief text after a successful heartbeat."""
    try:
        text, src = await resolve_canonical_brief_text(line_id, user_id, heartbeat_config, full_response)
        if not text:
            return
        await insert_snapshot(line_id, user_id, text, src)
        await prune_snapshots(line_id, user_id, keep=100)
    except Exception as e:
        logger.warning("record_heartbeat_brief_snapshot failed: %s", e)


async def build_snapshot_context_lines(line_id: str, user_id: str) -> List[str]:
    """Context lines for next heartbeat: prior brief + diff vs previous snapshot."""
    try:
        rows = await _fetch_recent_snapshots(line_id, user_id, limit=2)
        return build_prior_brief_context_lines(rows or [])
    except Exception as e:
        logger.warning("build_snapshot_context_lines failed: %s", e)
        return []
