"""
Scraper Celery Tasks - Bulk URL scraping with optional image download.
Used by Data Connection Builder bulk_scrape_urls tool for 20+ URLs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from services.celery_app import celery_app, update_task_progress, TaskStatus
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)


def _safe_filename(url: str, index: int, ext: str = ".jpg") -> str:
    """Derive a safe filename from URL and index."""
    parsed = urlparse(url)
    name = (parsed.path or "image").strip("/").split("/")[-1] or "image"
    name = re.sub(r"[^\w.\-]", "_", name)[:80]
    if not name or name == "_":
        name = "image"
    base = Path(name).stem if Path(name).suffix else name
    return f"{base}_{index}{ext}"


async def _run_batch_scrape(
    task,
    urls: List[str],
    user_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Async implementation: chunk URLs, crawl_many each chunk, optionally download images."""
    from clients.crawl_service_client import get_crawl_service_client
    import httpx

    extract_images = config.get("extract_images", True)
    download_images = config.get("download_images", True)
    image_output_folder = (config.get("image_output_folder") or "bulk_scrape").strip("/")
    max_concurrent = config.get("max_concurrent", 10)
    rate_limit_seconds = config.get("rate_limit_seconds", 1.0)

    try:
        from config import settings
        base_dir = Path(settings.UPLOAD_DIR) / "web_sources" / image_output_folder / user_id
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create bulk scrape output dir: %s", e)
        base_dir = None

    chunk_size = 10
    chunks = [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]
    total_chunks = len(chunks)
    all_results: List[Dict[str, Any]] = []
    images_found = 0
    images_downloaded = 0

    client = await get_crawl_service_client()

    for chunk_idx, chunk_urls in enumerate(chunks):
        update_task_progress(
            task,
            chunk_idx + 1,
            total_chunks,
            f"Crawling batch {chunk_idx + 1}/{total_chunks} ({len(chunk_urls)} URLs)",
        )
        try:
            response = await client.crawl_many(
                urls=chunk_urls,
                max_concurrent=max_concurrent,
                rate_limit_seconds=rate_limit_seconds,
                include_metadata=True,
            )
        except Exception as e:
            logger.exception("crawl_many failed for chunk %s: %s", chunk_idx, e)
            for u in chunk_urls:
                all_results.append({"url": u, "success": False, "error": str(e)})
            continue

        for r in response.get("results", []):
            all_results.append(r)
            imgs = r.get("images", [])
            images_found += len(imgs)

            if download_images and base_dir and imgs:
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    for idx, img_url in enumerate(imgs[:20]):
                        try:
                            resp = await http_client.get(img_url)
                            if resp.status_code != 200:
                                continue
                            content_type = resp.headers.get("content-type", "").split(";")[0].strip()
                            ext = ".jpg"
                            if "png" in content_type:
                                ext = ".png"
                            elif "gif" in content_type:
                                ext = ".gif"
                            elif "webp" in content_type:
                                ext = ".webp"
                            safe_name = _safe_filename(img_url, idx, ext)
                            out_path = base_dir / safe_name
                            out_path.write_bytes(resp.content)
                            sidecar = {
                                "schema_type": "image",
                                "schema_version": "1.0",
                                "title": r.get("title") or safe_name,
                                "content": img_url,
                                "type": "other",
                                "image_filename": safe_name,
                                "custom_fields": {"source_url": r.get("url", ""), "image_url": img_url},
                            }
                            sidecar_path = base_dir / f"{out_path.stem}.metadata.json"
                            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
                            images_downloaded += 1
                        except Exception as e:
                            logger.debug("Failed to download image %s: %s", img_url[:80], e)

    return {
        "results": all_results,
        "count": len(all_results),
        "images_found": images_found,
        "images_downloaded": images_downloaded,
        "progress_current": total_chunks,
        "progress_total": total_chunks,
        "progress_message": "Complete",
    }


@celery_app.task(bind=True, name="scrapers.batch_url_scrape")
def batch_url_scrape_task(
    self,
    urls: List[str],
    user_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Scrape many URLs in chunks; optionally download images and write sidecars for file watcher."""
    try:
        if not urls:
            return {"results": [], "count": 0, "images_found": 0, "images_downloaded": 0}
        update_task_progress(self, 0, 1, "Starting bulk URL scrape...")
        result = run_async(_run_batch_scrape(self, urls, user_id or "system", config or {}))
        self.update_state(
            state=TaskStatus.SUCCESS,
            meta=result,
        )
        return result
    except Exception as e:
        logger.exception("batch_url_scrape_task failed: %s", e)
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={"error": str(e), "results": [], "count": 0, "images_found": 0, "images_downloaded": 0},
        )
        raise
