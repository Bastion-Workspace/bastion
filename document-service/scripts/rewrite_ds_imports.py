#!/usr/bin/env python3
"""Rewrite copied backend imports to document-service ds_* packages."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

REPLACEMENTS: list[tuple[str, str]] = [
    (r"from services\.database_manager\.database_helpers import", "from ds_db.database_manager.database_helpers import"),
    (r"from services\.database_manager\.database_manager_service import", "from ds_db.database_manager.database_manager_service import"),
    (r"from services\.database_manager\.celery_database_helpers import", "from ds_db.database_manager.celery_database_helpers import"),
    (r"from services\.database_manager\.models\.", "from ds_db.database_manager.models."),
    (r"from services\.database_manager import", "from ds_db.database_manager import"),
    (r"from repositories\.document_repository_extensions import", "from ds_db.document_repository_extensions import"),
    (r"from repositories\.document_repository import", "from ds_db.document_repository import"),
    (r"from models\.vector_point import", "from ds_models.vector_point import"),
    (r"from models\.api_models import", "from ds_models.api_models import"),
    (r"from utils\.parallel_document_processor import", "from ds_processing.parallel_document_processor import"),
    (r"from utils\.document_processor import", "from ds_processing.document_processor import"),
    (r"from utils\.ocr_in_progress import", "from ds_processing.ocr_in_progress import"),
    (r"from services\.parallel_document_service import", "from ds_services.parallel_document_service import"),
    (r"from services\.document_service_v2 import", "from ds_services.document_service_v2 import"),
    (r"from services\.zip_processor_service import", "from ds_services.zip_processor_service import"),
    (r"from services\.embedding_service_wrapper import", "from ds_services.embedding_service_wrapper import"),
    (r"from services\.vector_store_service import", "from ds_services.vector_store_service import"),
    (r"from services\.knowledge_graph_service import", "from ds_services.knowledge_graph_service import"),
    (r"from services\.folder_service import", "from ds_services.folder_service import"),
    (r"from services\.link_extraction_service import", "from ds_services.link_extraction_service import"),
    (r"from services\.ocr_service import", "from ds_processing.ocr_service import"),
    (r"from clients\.vector_service_client import", "from ds_clients.vector_service_client import"),
    (r"from config import get_settings", "from ds_config import get_settings"),
    (r"from config import settings", "from ds_config import settings"),
]


def rewrite_file(path: pathlib.Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    for pat, repl in REPLACEMENTS:
        text = re.sub(pat, repl, text)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    dirs = [
        ROOT / "ds_config",
        ROOT / "ds_models",
        ROOT / "ds_db",
        ROOT / "ds_processing",
        ROOT / "ds_services",
        ROOT / "ds_clients",
    ]
    n = 0
    for d in dirs:
        for path in d.rglob("*.py"):
            if rewrite_file(path):
                n += 1
                print("updated", path.relative_to(ROOT))
    print(f"Done. {n} files updated.")


if __name__ == "__main__":
    main()
