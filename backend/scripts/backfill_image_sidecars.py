"""
One-shot backfill: minimal *.metadata.json sidecar on disk + image_sidecar row
for existing non-exempt image documents.

Usage (from backend container, cwd /app or with PYTHONPATH including backend):
    python -m scripts.backfill_image_sidecars
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _run(batch_size: int = 500) -> None:
    from services.database_manager.database_helpers import fetch_all
    from services.image_sidecar_service import get_image_sidecar_service
    from services import ds_upload_library_fs as dsf
    from services.service_container import get_service_container

    container = await get_service_container()
    folder_service = container.folder_service
    document_service = container.document_service
    isvc = await get_image_sidecar_service()

    offset = 0
    processed = 0
    while True:
        rows = await fetch_all(
            """
            SELECT document_id, filename, folder_id, user_id, collection_type, team_id
            FROM document_metadata
            WHERE LOWER(COALESCE(doc_type::text, '')) = 'image'
              AND COALESCE(exempt_from_vectorization, FALSE) = FALSE
            ORDER BY upload_date
            LIMIT $1 OFFSET $2
            """,
            batch_size,
            offset,
            rls_context={"user_id": "", "user_role": "admin"},
        )
        if not rows:
            break

        for row in rows:
            row = dict(row)
            doc_id = row["document_id"]
            uid = row.get("user_id") or ""
            try:
                fp = await folder_service.get_document_file_path(
                    filename=row["filename"],
                    folder_id=row.get("folder_id"),
                    user_id=row.get("user_id"),
                    collection_type=row.get("collection_type") or "user",
                    team_id=row.get("team_id"),
                )
                fp = Path(fp)
                if not fp.exists() and not await dsf.exists(uid, fp):
                    logger.warning(
                        "Skipping missing image file document_id=%s path=%s",
                        doc_id,
                        fp,
                    )
                    continue
                await isvc.ensure_sidecar_for_image(
                    image_document_id=doc_id,
                    image_file_path=fp,
                    folder_id=row.get("folder_id"),
                    user_id=row.get("user_id"),
                    collection_type=row.get("collection_type") or "user",
                    document_service=document_service,
                )
                processed += 1
            except Exception as e:
                logger.warning("Backfill failed for document_id=%s: %s", doc_id, e)

        logger.info(
            "Image sidecar backfill batch complete offset=%s batch_size=%s total_processed=%s",
            offset,
            len(rows),
            processed,
        )
        offset += len(rows)


def main() -> None:
    asyncio.run(_run(batch_size=500))


if __name__ == "__main__":
    main()
