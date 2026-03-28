"""
Resolve optional RSS article import target folder (user setting or per-request override).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def resolve_rss_import_target_folder_id(
    folder_id: Optional[str],
    *,
    user_id: str,
    user_role: str = "user",
) -> Optional[str]:
    """
    Return a folder_id safe to pass as FilePlacementRequest.target_folder_id, or None to use default placement.

    Rules:
    - User collection folder: must exist and belong to user_id.
    - Global collection folder: only if user_role is admin.
    """
    if not folder_id or not str(folder_id).strip():
        return None

    fid = str(folder_id).strip()

    try:
        from services.service_container import get_service_container

        container = await get_service_container()
        fs = container.folder_service
        if not fs:
            logger.warning("RSS import folder resolve: folder_service unavailable")
            return None

        folder = await fs.get_folder(fid, user_id, user_role or "user")
        if not folder:
            logger.warning("RSS import folder resolve: folder not found or inaccessible: %s", fid)
            return None

        ct = (folder.collection_type or "user").lower()
        if ct == "user":
            owner = folder.user_id
            if owner and str(owner) != str(user_id):
                logger.warning(
                    "RSS import folder resolve: user %s cannot use folder %s owned by %s",
                    user_id,
                    fid,
                    owner,
                )
                return None
            return fid

        if ct == "global" and (user_role or "").lower() == "admin":
            return fid

        logger.warning(
            "RSS import folder resolve: folder %s collection_type=%s not allowed for this user",
            fid,
            ct,
        )
        return None
    except Exception as e:
        logger.warning("RSS import folder resolve failed: %s", e)
        return None
