"""CRUD for zettelkasten_settings JSONB (per user)."""

import json
import logging
from datetime import datetime
from typing import Optional

from models.zettelkasten_models import (
    DailyNoteFormat,
    ZettelkastenSettings,
    ZettelkastenSettingsUpdate,
)

logger = logging.getLogger(__name__)


class ZettelkastenSettingsService:
    async def _get_pool(self):
        from services.database_manager.database_manager_service import get_database_manager

        db_manager = await get_database_manager()
        return db_manager._pool

    def _defaults(self, user_id: str) -> ZettelkastenSettings:
        return ZettelkastenSettings(
            user_id=user_id,
            enabled=False,
            daily_note_location=None,
            daily_note_format=DailyNoteFormat.ISO,
            daily_note_template="",
            note_id_prefix=False,
            backlinks_enabled=True,
            wikilink_autocomplete=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    async def get_settings(self, user_id: str) -> ZettelkastenSettings:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT settings_json, created_at, updated_at FROM zettelkasten_settings WHERE user_id = $1",
                    user_id,
                )
                if not row:
                    return self._defaults(user_id)
                raw = row["settings_json"]
                settings_dict = raw if isinstance(raw, dict) else json.loads(raw)
                settings_dict["user_id"] = user_id
                settings_dict["created_at"] = row["created_at"]
                settings_dict["updated_at"] = row["updated_at"]
                return ZettelkastenSettings(**settings_dict)
        except Exception as e:
            logger.error("Failed to load zettelkasten settings for %s: %s", user_id, e)
            return self._defaults(user_id)

    async def create_or_update_settings(
        self, user_id: str, update: ZettelkastenSettingsUpdate
    ) -> ZettelkastenSettings:
        current = await self.get_settings(user_id)
        if update.enabled is not None:
            current.enabled = update.enabled
        if update.daily_note_location is not None:
            current.daily_note_location = update.daily_note_location
        if update.daily_note_format is not None:
            current.daily_note_format = update.daily_note_format
        if update.daily_note_template is not None:
            current.daily_note_template = update.daily_note_template
        if update.note_id_prefix is not None:
            current.note_id_prefix = update.note_id_prefix
        if update.backlinks_enabled is not None:
            current.backlinks_enabled = update.backlinks_enabled
        if update.wikilink_autocomplete is not None:
            current.wikilink_autocomplete = update.wikilink_autocomplete
        current.updated_at = datetime.now()

        payload = current.model_dump(exclude={"user_id", "created_at", "updated_at"})
        settings_json = json.dumps(payload)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO zettelkasten_settings (user_id, settings_json, created_at, updated_at)
                VALUES ($1, $2::jsonb, $3, $4)
                ON CONFLICT (user_id) DO UPDATE SET
                    settings_json = EXCLUDED.settings_json,
                    updated_at = EXCLUDED.updated_at
                """,
                user_id,
                settings_json,
                current.created_at or datetime.now(),
                current.updated_at,
            )
        return await self.get_settings(user_id)

    async def delete_settings(self, user_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM zettelkasten_settings WHERE user_id = $1", user_id)
        return True


_service: Optional[ZettelkastenSettingsService] = None


async def get_zettelkasten_settings_service() -> ZettelkastenSettingsService:
    global _service
    if _service is None:
        _service = ZettelkastenSettingsService()
    return _service
