import logging
from pathlib import Path
from typing import Dict
import asyncio

from config import settings

logger = logging.getLogger(__name__)


def _user_org_base_dir(user_id: str, username: str = None) -> Path:
    """Get user's Org directory using new folder structure"""
    folder_name = username if username else user_id
    return Path(settings.UPLOAD_DIR) / "Users" / folder_name / "Org"


async def _get_username(user_id: str) -> str:
    """Get username from user_id"""
    try:
        from services.database_manager.database_helpers import fetch_one
        row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
        return row['username'] if row else user_id
    except Exception as e:
        logger.warning(f"⚠️ Could not get username for {user_id}: {e}")
        return user_id


async def ensure_user_org_files(user_id: str) -> Dict[str, str]:
    """Ensure per-user Org directory structure and seed files.

    Structure:
    /app/uploads/Users/{username}/Org/
      - inbox.org
      - Archive/
          - archive.org
    """
    # Get username for folder structure
    username = await _get_username(user_id)
    base_dir = _user_org_base_dir(user_id, username)
    archive_dir = base_dir / "Archive"

    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(parents=True, exist_ok=True)

        inbox_path = base_dir / "inbox.org"
        if not inbox_path.exists():
            inbox_path.write_text("* Inbox\n", encoding="utf-8")

        archive_path = archive_dir / "archive.org"
        if not archive_path.exists():
            archive_path.write_text("* Archive\n", encoding="utf-8")

        logger.info(
            f"✅ Ensured Org files for user {user_id} ({username}): {inbox_path} | {archive_path}"
        )
        return {
            "org_base_dir": str(base_dir),
            "inbox_path": str(inbox_path),
            "archive_path": str(archive_path),
        }
    except Exception as e:
        logger.error(f"❌ Failed ensuring Org files for user {user_id}: {e}")
        raise


async def get_user_inbox_path(user_id: str) -> str:
    info = await ensure_user_org_files(user_id)
    return info["inbox_path"]


async def get_user_archive_path(user_id: str) -> str:
    info = await ensure_user_org_files(user_id)
    return info["archive_path"]


