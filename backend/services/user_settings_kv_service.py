import logging
from typing import Optional

from services.database_manager.database_helpers import fetch_one, execute

logger = logging.getLogger(__name__)


async def get_user_setting(user_id: str, key: str) -> Optional[str]:
    try:
        row = await fetch_one(
            "SELECT value FROM user_settings WHERE user_id = $1 AND key = $2",
            user_id,
            key,
        )
        return row["value"] if row else None
    except Exception as e:
        logger.error(f"❌ Failed to get user setting {key} for {user_id}: {e}")
        return None


async def set_user_setting(user_id: str, key: str, value: str, data_type: str = "string") -> bool:
    try:
        # Upsert behavior
        existing = await fetch_one(
            "SELECT 1 FROM user_settings WHERE user_id = $1 AND key = $2",
            user_id,
            key,
        )
        if existing:
            await execute(
                "UPDATE user_settings SET value = $1, data_type = $2, updated_at = NOW() WHERE user_id = $3 AND key = $4",
                value,
                data_type,
                user_id,
                key,
            )
        else:
            await execute(
                "INSERT INTO user_settings (user_id, key, value, data_type, created_at, updated_at) VALUES ($1, $2, $3, $4, NOW(), NOW())",
                user_id,
                key,
                value,
                data_type,
            )
        return True
    except Exception as e:
        logger.error(f"❌ Failed to set user setting {key} for {user_id}: {e}")
        return False


