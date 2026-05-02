"""
Background warm of music cover art into the per-user disk cache after library refresh.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from config import settings
from services.celery_app import celery_app
from services.celery_tasks.async_runner import run_async
from services.music_cover_cache import (
    fetch_cover_upstream,
    get_or_fetch,
    parse_warm_sizes,
)
from services.music_service import music_service

logger = logging.getLogger(__name__)


async def _warm_async(
    user_id: str, service_type: str, cover_ids: List[str], sizes: List[int]
) -> Dict:
    rls_context = {"user_id": user_id}

    async def one(cid: str, sz: int) -> Tuple[int, int]:
        try:

            async def upstream_fetch():
                url = await music_service.get_cover_art_url(
                    user_id, cid, service_type, sz
                )
                if not url:
                    raise RuntimeError("no_cover_art_url")
                return await fetch_cover_upstream(url)

            await get_or_fetch(
                user_id, service_type, cid, sz, rls_context, upstream_fetch
            )
            return 1, 0
        except Exception as exc:
            logger.debug(
                "Cover warm failed user=%s id=%s size=%s: %s",
                user_id,
                cid,
                sz,
                exc,
            )
            return 0, 1

    pairs = await asyncio.gather(
        *(one(cid, sz) for cid in cover_ids for sz in sizes)
    )
    ok = sum(p[0] for p in pairs)
    fail = sum(p[1] for p in pairs)
    return {"success": True, "ok": ok, "fail": fail, "covers": len(cover_ids)}


@celery_app.task(name="services.celery_tasks.music_cover_warm_tasks.warm_user_covers")
def warm_user_covers(
    user_id: str,
    service_type: str,
    cover_ids: List[str],
    sizes: Optional[List[int]] = None,
) -> Dict:
    if not getattr(settings, "MUSIC_COVER_CACHE_ENABLED", True):
        return {"skipped": True, "reason": "music_cover_cache_disabled"}
    if sizes is None:
        sizes = parse_warm_sizes(settings.MUSIC_COVER_CACHE_WARM_SIZES)
    if not cover_ids or not sizes:
        return {"skipped": True, "reason": "empty_input"}
    try:
        return run_async(_warm_async(user_id, service_type, cover_ids, sizes))
    except Exception as e:
        logger.error("warm_user_covers failed: %s", e)
        return {"success": False, "error": str(e)}


@celery_app.task(name="services.celery_tasks.music_cover_warm_tasks.sweep_music_cover_cache_lru")
def sweep_music_cover_cache_lru() -> Dict:
    """LRU eviction runs on cache insert; beat schedule not wired by default."""
    return {"skipped": True, "reason": "opportunistic_eviction_on_insert"}
