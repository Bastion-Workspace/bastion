"""
Image metadata sidecar ingestion for document-service.

Upserts a document_metadata row for *.metadata.json (doc_type image_sidecar) and runs
the standard document processing pipeline for chunking and embeddings.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from ds_models.api_models import (
    DocumentCategory,
    DocumentInfo,
    DocumentType,
    ProcessingStatus,
    QualityMetrics,
)
from ds_services.image_sidecar_helpers import build_minimal_image_sidecar_metadata

logger = logging.getLogger(__name__)

_image_sidecar_service_instance: Optional["ImageSidecarService"] = None


class ImageSidecarService:
    """Ingest image *.metadata.json sidecars into the unified document pipeline."""

    TYPE_TO_CATEGORY: Dict[str, DocumentCategory] = {
        "comic": DocumentCategory.COMIC,
        "artwork": DocumentCategory.ENTERTAINMENT,
        "meme": DocumentCategory.ENTERTAINMENT,
        "screenshot": DocumentCategory.TECHNICAL,
        "medical": DocumentCategory.MEDICAL,
        "documentation": DocumentCategory.TECHNICAL,
        "maps": DocumentCategory.REFERENCE,
        "photo": DocumentCategory.ENTERTAINMENT,
        "other": DocumentCategory.OTHER,
    }

    def _sha256_file(self, path: Path) -> Optional[str]:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return None

    async def upsert_and_process_image_sidecar(
        self,
        *,
        file_path: str,
        metadata_data: Dict[str, Any],
        folder_id: Optional[str],
        user_id: Optional[str],
        collection_type: str,
        document_service: Any,
    ) -> Dict[str, Any]:
        """
        Create or update the sidecar document row and run standard async processing.
        """
        json_path = Path(file_path)
        if not json_path.exists():
            return {"success": False, "error": f"JSON file not found: {file_path}"}
        if not json_path.name.lower().endswith(".metadata.json"):
            return {"success": False, "error": "File must end with .metadata.json"}

        json_image_filename = (metadata_data.get("image_filename") or "").strip()
        image_filename_with_ext = json_path.name[: -len(".metadata.json")]
        if json_image_filename:
            actual_image_filename = json_image_filename
        else:
            actual_image_filename = image_filename_with_ext

        _minimal = build_minimal_image_sidecar_metadata(
            actual_image_filename,
            (metadata_data.get("type") or "photo").strip().lower(),
        )
        for _k, _v in _minimal.items():
            if _k not in metadata_data:
                metadata_data[_k] = _v

        title = (metadata_data.get("title") or "").strip()
        content = (
            metadata_data.get("content") or metadata_data.get("transcript") or ""
        ).strip()
        author = (metadata_data.get("author") or "").strip()
        date = (metadata_data.get("date") or "").strip()
        series = (metadata_data.get("series") or "").strip()
        tags = metadata_data.get("tags", []) or []
        image_type = (metadata_data.get("type") or "other").strip().lower()

        if not title:
            return {"success": False, "error": "Missing required field: title"}
        if "content" not in metadata_data and "transcript" not in metadata_data:
            return {
                "success": False,
                "error": "Missing required field: content or transcript (key must exist)",
            }

        if image_type == "map":
            image_type = "maps"
        valid_types = [
            "comic",
            "artwork",
            "meme",
            "screenshot",
            "medical",
            "documentation",
            "maps",
            "photo",
            "other",
        ]
        if image_type not in valid_types:
            logger.warning("Invalid image type '%s', defaulting to 'other'", image_type)
            image_type = "other"

        category = self.TYPE_TO_CATEGORY.get(image_type, DocumentCategory.OTHER)
        sidecar_filename = json_path.name

        meta_out: Dict[str, Any] = dict(metadata_data)
        meta_out.setdefault("has_searchable_metadata", True)
        meta_out["image_filename"] = actual_image_filename
        meta_out["type"] = image_type
        meta_out["image_type"] = image_type
        meta_out["date"] = date or None
        meta_out["author"] = author or None
        meta_out["series"] = series or None
        meta_out["content"] = content

        for key in (
            "location",
            "event",
            "medium",
            "dimensions",
            "body_part",
            "modality",
            "map_type",
            "coordinates",
            "application",
            "platform",
        ):
            if metadata_data.get(key):
                meta_out[key] = metadata_data[key]
        if metadata_data.get("llm_metadata") and isinstance(
            metadata_data["llm_metadata"], dict
        ):
            meta_out["llm_metadata"] = metadata_data["llm_metadata"]
        if metadata_data.get("faces") and isinstance(metadata_data["faces"], list):
            meta_out["faces"] = metadata_data["faces"]
        if metadata_data.get("objects") and isinstance(
            metadata_data["objects"], list
        ):
            meta_out["objects"] = metadata_data["objects"]

        metadata_json_str = json.dumps(meta_out, ensure_ascii=False)
        repo = document_service.document_repository
        existing = await repo.find_by_filename_and_context(
            filename=sidecar_filename,
            user_id=user_id,
            collection_type=collection_type,
            folder_id=folder_id,
        )

        pub_date = None
        if date:
            try:
                pub_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                pub_date = None

        file_size = json_path.stat().st_size
        file_hash = self._sha256_file(json_path)

        if existing:
            document_id = existing.document_id
            try:
                if not document_service.embedding_manager:
                    from ds_services.embedding_service_wrapper import get_embedding_service

                    document_service.embedding_manager = await get_embedding_service()
                await document_service.embedding_manager.delete_document_chunks(
                    document_id, user_id
                )
            except Exception as del_e:
                logger.warning(
                    "Could not delete old chunks before sidecar re-embed: %s", del_e
                )
            await repo.update(
                document_id,
                user_id=user_id,
                title=title,
                doc_type=DocumentType.IMAGE_SIDECAR,
                category=category,
                tags=tags if tags else [],
                description=content[:500] if content else None,
                author=author if author else None,
                publication_date=pub_date,
                metadata_json=metadata_json_str,
                file_size=file_size,
                file_hash=file_hash,
            )
            logger.info("Updated image sidecar document %s (%s)", document_id, sidecar_filename)
        else:
            document_id = str(uuid4())
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=sidecar_filename,
                title=title,
                doc_type=DocumentType.IMAGE_SIDECAR,
                category=category,
                tags=tags if tags else [],
                description=content[:500] if content else None,
                author=author if author else None,
                language="en",
                publication_date=pub_date,
                upload_date=datetime.utcnow(),
                file_size=file_size,
                file_hash=file_hash,
                status=ProcessingStatus.PROCESSING,
                quality_metrics=QualityMetrics(
                    ocr_confidence=1.0,
                    language_confidence=1.0,
                    vocabulary_score=1.0,
                    pattern_score=1.0,
                    overall_score=1.0,
                ),
                chunk_count=0,
                entity_count=0,
                collection_type=collection_type,
                user_id=user_id,
            )
            ok = await repo.create_with_folder(doc_info, folder_id)
            if not ok:
                return {"success": False, "error": "Failed to create document record"}
            await repo.update(
                document_id,
                user_id=user_id,
                metadata_json=metadata_json_str,
            )
            logger.info("Created image sidecar document %s (%s)", document_id, sidecar_filename)

        await document_service._process_document_async(
            document_id, json_path, "image_sidecar", user_id
        )
        return {
            "success": True,
            "document_id": document_id,
            "title": title,
        }

    async def ensure_sidecar_for_image(
        self,
        *,
        image_document_id: str,
        image_file_path: Path,
        folder_id: Optional[str],
        user_id: Optional[str],
        collection_type: str,
        document_service: Any,
    ) -> Dict[str, Any]:
        """
        Ensure {stem}.metadata.json exists (write-if-missing) and upsert the image_sidecar row
        through the unified pipeline. Idempotent: existing rich sidecars are not overwritten.
        """
        sidecar_path = image_file_path.parent / f"{image_file_path.stem}.metadata.json"
        wrote_file = False
        if not sidecar_path.exists():
            payload = build_minimal_image_sidecar_metadata(image_file_path.name)
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            wrote_file = True

        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                metadata_on_disk = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "ensure_sidecar_for_image: failed to read sidecar %s: %s",
                sidecar_path,
                e,
            )
            return {
                "wrote_file": wrote_file,
                "sidecar_document_id": "",
                "success": False,
            }

        result = await self.upsert_and_process_image_sidecar(
            file_path=str(sidecar_path),
            metadata_data=metadata_on_disk,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type,
            document_service=document_service,
        )
        sidecar_document_id = str(result.get("document_id") or "")
        if not result.get("success"):
            logger.warning(
                "ensure_sidecar_for_image: upsert failed for image %s: %s",
                image_document_id,
                result.get("error"),
            )
        return {
            "wrote_file": wrote_file,
            "sidecar_document_id": sidecar_document_id,
            "success": bool(result.get("success")),
        }


async def get_image_sidecar_service() -> ImageSidecarService:
    global _image_sidecar_service_instance
    if _image_sidecar_service_instance is None:
        _image_sidecar_service_instance = ImageSidecarService()
    return _image_sidecar_service_instance
