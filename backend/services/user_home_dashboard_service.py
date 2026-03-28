"""
User home dashboards stored in PostgreSQL (`user_home_dashboards`).

Migrates legacy `user_settings` keys `home_dashboards_v2` / `home_dashboard_v1` on first access
when the user has no rows, then removes those keys.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, List, Optional

from pydantic import ValidationError

from models.home_dashboard_models import (
    HOME_DASHBOARD_SETTING_KEY,
    HOME_DASHBOARDS_SETTING_KEY_V2,
    HomeDashboardLayout,
    UserDashboardCreateRequest,
    UserDashboardPatchRequest,
    UserDashboardsEnvelope,
    UserDashboardsListResponse,
    UserDashboardSummary,
    default_home_dashboard_layout,
    default_user_dashboards_envelope,
    envelope_from_legacy_v1_layout,
    MAX_USER_DASHBOARDS,
)
from services.database_manager.database_helpers import (
    execute,
    execute_transaction,
    fetch_all,
    fetch_one,
    fetch_value,
)
from services.user_settings_kv_service import delete_user_setting, get_user_setting

logger = logging.getLogger(__name__)


def _parse_envelope_v2(raw: str) -> Optional[UserDashboardsEnvelope]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return UserDashboardsEnvelope.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        logger.warning("Invalid home_dashboards_v2 envelope: %s", e)
        return None


def _normalize_dashboard_uuid(dashboard_id: str) -> str:
    try:
        return str(uuid.UUID(dashboard_id))
    except (ValueError, TypeError):
        return str(uuid.uuid4())


def _layout_from_db(value: Any) -> HomeDashboardLayout:
    if isinstance(value, str):
        data = json.loads(value)
    else:
        data = value
    return HomeDashboardLayout.model_validate(data)


async def _row_count(user_id: str) -> int:
    n = await fetch_value(
        "SELECT COUNT(*)::int FROM user_home_dashboards WHERE user_id = $1",
        user_id,
    )
    return int(n or 0)


async def _insert_envelope_rows(user_id: str, env: UserDashboardsEnvelope) -> None:
    async def _op(conn: Any) -> None:
        for i, d in enumerate(env.dashboards):
            did = _normalize_dashboard_uuid(d.id)
            layout_js = json.dumps(d.layout.model_dump(mode="json"))
            await conn.execute(
                """
                INSERT INTO user_home_dashboards (id, user_id, name, is_default, layout_json, created_at, updated_at)
                VALUES (
                    $1::uuid, $2, $3, $4, $5::jsonb,
                    NOW() + ($6::bigint * interval '1 microsecond'),
                    NOW() + ($6::bigint * interval '1 microsecond')
                )
                """,
                did,
                user_id,
                d.name[:100],
                d.is_default,
                layout_js,
                i,
            )

    await execute_transaction([_op])


async def ensure_user_dashboard_rows(user_id: str) -> None:
    if await _row_count(user_id) > 0:
        return

    raw_v2 = await get_user_setting(user_id, HOME_DASHBOARDS_SETTING_KEY_V2)
    if raw_v2:
        env = _parse_envelope_v2(raw_v2)
        if env:
            await _insert_envelope_rows(user_id, env)
            await delete_user_setting(user_id, HOME_DASHBOARDS_SETTING_KEY_V2)
            return

    raw_v1 = await get_user_setting(user_id, HOME_DASHBOARD_SETTING_KEY)
    if raw_v1:
        try:
            data = json.loads(raw_v1) if isinstance(raw_v1, str) else raw_v1
            layout = HomeDashboardLayout.model_validate(data)
            env = envelope_from_legacy_v1_layout(layout)
            await _insert_envelope_rows(user_id, env)
            await delete_user_setting(user_id, HOME_DASHBOARD_SETTING_KEY)
            return
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            logger.warning("Invalid legacy home_dashboard_v1 for user %s: %s", user_id, e)

    env = default_user_dashboards_envelope()
    await _insert_envelope_rows(user_id, env)


def _summaries_from_rows(rows: List[dict]) -> UserDashboardsListResponse:
    return UserDashboardsListResponse(
        dashboards=[
            UserDashboardSummary(
                id=str(r["id"]),
                name=r["name"],
                is_default=bool(r["is_default"]),
            )
            for r in rows
        ]
    )


async def list_dashboards(user_id: str) -> UserDashboardsListResponse:
    await ensure_user_dashboard_rows(user_id)
    rows = await fetch_all(
        """
        SELECT id, name, is_default
        FROM user_home_dashboards
        WHERE user_id = $1
        ORDER BY is_default DESC, created_at ASC, id ASC
        """,
        user_id,
    )
    return _summaries_from_rows(rows)


async def create_dashboard(
    user_id: str,
    body: UserDashboardCreateRequest,
) -> UserDashboardsListResponse:
    await ensure_user_dashboard_rows(user_id)
    n = await _row_count(user_id)
    if n >= MAX_USER_DASHBOARDS:
        raise ValueError(f"At most {MAX_USER_DASHBOARDS} dashboards allowed")

    layout: HomeDashboardLayout
    if body.duplicate_from_id:
        row = await fetch_one(
            """
            SELECT layout_json FROM user_home_dashboards
            WHERE id = $1::uuid AND user_id = $2
            """,
            body.duplicate_from_id,
            user_id,
        )
        if not row:
            raise LookupError("Source dashboard not found")
        layout = _layout_from_db(row["layout_json"])
    else:
        layout = default_home_dashboard_layout()

    name = (body.name or "").strip() or "New dashboard"
    new_id = str(uuid.uuid4())
    layout_js = json.dumps(layout.model_dump(mode="json"))
    await execute(
        """
        INSERT INTO user_home_dashboards (id, user_id, name, is_default, layout_json, created_at, updated_at)
        VALUES ($1::uuid, $2, $3, false, $4::jsonb, NOW(), NOW())
        """,
        new_id,
        user_id,
        name[:100],
        layout_js,
    )
    return await list_dashboards(user_id)


async def patch_dashboard(
    user_id: str,
    dashboard_id: str,
    body: UserDashboardPatchRequest,
) -> UserDashboardsListResponse:
    await ensure_user_dashboard_rows(user_id)
    exists = await fetch_one(
        "SELECT 1 FROM user_home_dashboards WHERE id = $1::uuid AND user_id = $2",
        dashboard_id,
        user_id,
    )
    if not exists:
        raise LookupError("Dashboard not found")

    if body.name is not None:
        await execute(
            """
            UPDATE user_home_dashboards
            SET name = $1, updated_at = NOW()
            WHERE id = $2::uuid AND user_id = $3
            """,
            body.name.strip()[:100],
            dashboard_id,
            user_id,
        )

    if body.is_default is True:
        async def _clear(conn: Any) -> None:
            await conn.execute(
                """
                UPDATE user_home_dashboards
                SET is_default = false, updated_at = NOW()
                WHERE user_id = $1
                """,
                user_id,
            )

        async def _set(conn: Any) -> None:
            await conn.execute(
                """
                UPDATE user_home_dashboards
                SET is_default = true, updated_at = NOW()
                WHERE id = $1::uuid AND user_id = $2
                """,
                dashboard_id,
                user_id,
            )

        await execute_transaction([_clear, _set])

    return await list_dashboards(user_id)


async def delete_dashboard(user_id: str, dashboard_id: str) -> UserDashboardsListResponse:
    await ensure_user_dashboard_rows(user_id)

    async def _tx(conn: Any) -> None:
        row = await conn.fetchrow(
            """
            SELECT is_default FROM user_home_dashboards
            WHERE id = $1::uuid AND user_id = $2
            FOR UPDATE
            """,
            dashboard_id,
            user_id,
        )
        if not row:
            raise LookupError("Dashboard not found")
        cnt = await conn.fetchval(
            "SELECT COUNT(*)::int FROM user_home_dashboards WHERE user_id = $1",
            user_id,
        )
        if int(cnt or 0) <= 1:
            raise ValueError("Cannot delete the last dashboard")

        if row["is_default"]:
            other = await conn.fetchrow(
                """
                SELECT id FROM user_home_dashboards
                WHERE user_id = $1 AND id != $2::uuid
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """,
                user_id,
                dashboard_id,
            )
            if not other:
                raise ValueError("Cannot delete the last dashboard")
            await conn.execute(
                """
                UPDATE user_home_dashboards
                SET is_default = false, updated_at = NOW()
                WHERE user_id = $1
                """,
                user_id,
            )
            await conn.execute(
                """
                UPDATE user_home_dashboards
                SET is_default = true, updated_at = NOW()
                WHERE id = $1::uuid AND user_id = $2
                """,
                other["id"],
                user_id,
            )

        await conn.execute(
            """
            DELETE FROM user_home_dashboards
            WHERE id = $1::uuid AND user_id = $2
            """,
            dashboard_id,
            user_id,
        )

    await execute_transaction([_tx])
    return await list_dashboards(user_id)


async def get_layout(user_id: str, dashboard_id: str) -> HomeDashboardLayout:
    await ensure_user_dashboard_rows(user_id)
    row = await fetch_one(
        """
        SELECT layout_json FROM user_home_dashboards
        WHERE id = $1::uuid AND user_id = $2
        """,
        dashboard_id,
        user_id,
    )
    if not row:
        raise LookupError("Dashboard not found")
    return _layout_from_db(row["layout_json"])


async def put_layout(
    user_id: str,
    dashboard_id: str,
    layout: HomeDashboardLayout,
) -> HomeDashboardLayout:
    await ensure_user_dashboard_rows(user_id)
    row = await fetch_one(
        """
        UPDATE user_home_dashboards
        SET layout_json = $1::jsonb, updated_at = NOW()
        WHERE id = $2::uuid AND user_id = $3
        RETURNING id
        """,
        json.dumps(layout.model_dump(mode="json")),
        dashboard_id,
        user_id,
    )
    if not row:
        raise LookupError("Dashboard not found")
    return layout


async def get_default_layout(user_id: str) -> HomeDashboardLayout:
    await ensure_user_dashboard_rows(user_id)
    row = await fetch_one(
        """
        SELECT layout_json FROM user_home_dashboards
        WHERE user_id = $1 AND is_default = true
        LIMIT 1
        """,
        user_id,
    )
    if not row:
        row = await fetch_one(
            """
            SELECT layout_json FROM user_home_dashboards
            WHERE user_id = $1
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            user_id,
        )
    if not row:
        raise LookupError("No dashboard found")
    return _layout_from_db(row["layout_json"])


async def put_default_layout(user_id: str, layout: HomeDashboardLayout) -> HomeDashboardLayout:
    await ensure_user_dashboard_rows(user_id)
    row = await fetch_one(
        """
        SELECT id FROM user_home_dashboards
        WHERE user_id = $1 AND is_default = true
        LIMIT 1
        """,
        user_id,
    )
    if not row:
        row = await fetch_one(
            """
            SELECT id FROM user_home_dashboards
            WHERE user_id = $1
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            user_id,
        )
    if not row:
        raise LookupError("No dashboard found")
    return await put_layout(user_id, str(row["id"]), layout)
