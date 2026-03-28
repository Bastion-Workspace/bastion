"""
Touch image metadata sidecar files so the file watcher re-processes them and adds
the is_image_sidecar payload flag. Run after deploying the image-sidecar filter
so document search excludes image sidecars.

Usage (from backend container or with PYTHONPATH set):
    python scripts/backfill_image_sidecar_flags.py

Expects UPLOAD_DIR from config (e.g. /app/uploads). Touches all *.metadata.json
files under it; the file watcher will re-ingest and ImageSidecarService will
tag new vectors with is_image_sidecar=true.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    upload_dir = Path(settings.UPLOAD_DIR)
    if not upload_dir.exists():
        logger.warning("UPLOAD_DIR %s does not exist, nothing to touch", upload_dir)
        return
    count = 0
    for path in upload_dir.rglob("*.metadata.json"):
        try:
            path.touch()
            count += 1
        except OSError as e:
            logger.warning("Could not touch %s: %s", path, e)
    logger.info("Touched %d .metadata.json file(s); file watcher will re-process them", count)


if __name__ == "__main__":
    main()
