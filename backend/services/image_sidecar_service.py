"""
Image Sidecar Service - Handles ingestion of image metadata JSON sidecar files
Supports universal image types: comic, artwork, meme, screenshot, medical, documentation, maps

Ingestion (`process_image_metadata`) is triggered by the filesystem watcher, which runs in
document-service on the mounted upload tree — not on the backend (see
document-service/ds_services/file_watcher_service.py). Backend APIs use other methods here
(e.g. sync_identity_to_tags) and should read/write library files via document-service when
the backend has no ./uploads mount.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from models.api_models import (
    DocumentCategory,
    DocumentInfo,
    DocumentType,
    ProcessingStatus,
    QualityMetrics,
)
from repositories.document_repository import DocumentRepository
from services import ds_upload_library_fs as dsf
from services.image_sidecar_helpers import build_minimal_image_sidecar_metadata

logger = logging.getLogger(__name__)


class ImageSidecarService:
    """Service for processing image metadata JSON sidecar files"""
    
    # Map image type to DocumentCategory
    TYPE_TO_CATEGORY = {
        "comic": DocumentCategory.COMIC,
        "artwork": DocumentCategory.ENTERTAINMENT,
        "meme": DocumentCategory.ENTERTAINMENT,
        "screenshot": DocumentCategory.TECHNICAL,
        "medical": DocumentCategory.MEDICAL,
        "documentation": DocumentCategory.TECHNICAL,
        "maps": DocumentCategory.REFERENCE,
        "photo": DocumentCategory.ENTERTAINMENT,
        "other": DocumentCategory.OTHER
    }
    
    def __init__(self):
        self.document_repository = None
    
    def _fix_json_common_issues(self, json_text: str) -> str:
        """
        Attempt to fix common JSON issues like trailing commas.
        
        This handles:
        - Trailing commas before closing braces/brackets (safe to fix)
        
        Note: Unescaped quotes inside strings are NOT auto-fixed because it's
        too risky - we can't reliably determine which quotes should be escaped
        without a full JSON parser. Instead, we provide detailed error messages
        to help identify the issue. Python's json.loads() correctly handles
        properly escaped quotes (e.g., "He said \\"hello\\"").
        """
        # Fix trailing commas (most common and safe to fix)
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text, flags=re.MULTILINE)
        json_text = re.sub(r',\s*\n\s*([}\]])', r'\n\1', json_text)
        
        return json_text
    
    async def initialize(self):
        """Initialize the image sidecar service"""
        try:
            self.document_repository = DocumentRepository()
            await self.document_repository.initialize()
            
            logger.info("Image Sidecar Service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Image Sidecar Service: {e}")
            raise
    
    async def process_image_metadata(
        self, file_path: str, skip_embedding: bool = False
    ) -> Dict[str, Any]:
        """
        Image *.metadata.json ingestion runs in document-service (file watcher + unified pipeline).
        This backend entry point is retained only for accidental callers.
        """
        logger.warning(
            "process_image_metadata called on backend; ingestion runs in document-service (%s)",
            file_path,
        )
        return {
            "success": False,
            "error": "Image sidecar ingestion runs in document-service (file watcher).",
        }

    async def _sha256_file_dsf(self, dsf_uid: str, json_path: Path) -> Optional[str]:
        try:
            raw = await dsf.read_text(dsf_uid, json_path, encoding="utf-8")
        except Exception:
            return None
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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
        Library bytes are read/written via document-service (dsf).
        """
        json_path = Path(file_path)
        dsf_uid = user_id or ""
        if not await dsf.exists(dsf_uid, json_path):
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

        raw_text = await dsf.read_text(dsf_uid, json_path, encoding="utf-8")
        file_size = len(raw_text.encode("utf-8"))
        file_hash = await self._sha256_file_dsf(dsf_uid, json_path)

        if existing:
            document_id = existing.document_id
            try:
                if not document_service.embedding_manager:
                    from services.embedding_service_wrapper import get_embedding_service

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
            doc_info_row = DocumentInfo(
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
            ok = await repo.create_with_folder(doc_info_row, folder_id)
            if not ok:
                return {"success": False, "error": "Failed to create document record"}
            await repo.update(
                document_id,
                user_id=user_id,
                metadata_json=metadata_json_str,
            )
            logger.info("Created image sidecar document %s (%s)", document_id, sidecar_filename)

        from clients.document_service_client import get_document_service_client

        dsc = get_document_service_client()
        await dsc.initialize(required=True)
        await dsc.reprocess_via_document_service(document_id, user_id, force_reprocess=False)
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
        Ensure {stem}.metadata.json exists (write-if-missing) and upsert the image_sidecar row.
        """
        sidecar_path = image_file_path.parent / f"{image_file_path.stem}.metadata.json"
        dsf_uid = user_id or ""
        wrote_file = False
        if not await dsf.exists(dsf_uid, sidecar_path):
            payload = build_minimal_image_sidecar_metadata(image_file_path.name)
            try:
                await dsf.write_text(
                    dsf_uid,
                    sidecar_path,
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                wrote_file = True
            except Exception as w_e:
                logger.warning(
                    "ensure_sidecar_for_image: could not write sidecar %s: %s",
                    sidecar_path,
                    w_e,
                )
                return {
                    "wrote_file": False,
                    "sidecar_document_id": "",
                    "success": False,
                }

        try:
            raw = await dsf.read_text(dsf_uid, sidecar_path, encoding="utf-8")
            metadata_on_disk = json.loads(raw)
        except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
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

    async def sync_identity_to_tags(self, document_id: str, identity_name: str, user_id: str, collection_type: str = "user") -> Dict[str, Any]:
        """
        Add a face identity to both metadata.json tags and document_metadata.tags
        Ensures search index stays in sync with face detection
        
        Args:
            document_id: Document ID
            identity_name: Name of the person (e.g., 'Steve McQueen')
            user_id: User ID for access control
            collection_type: 'user' or 'global'
            
        Returns:
            Dict with success status and updated tags list
        """
        try:
            # Get document info
            doc_info = await self.document_repository.get_by_id(document_id)
            if not doc_info:
                return {
                    "success": False,
                    "error": "Document not found"
                }
            
            filename = getattr(doc_info, 'filename', '')
            folder_id = getattr(doc_info, 'folder_id', None)
            
            # Get file path
            from services.service_container import get_service_container
            container = await get_service_container()
            folder_service = container.folder_service
            
            file_path_str = await folder_service.get_document_file_path(
                filename=filename,
                folder_id=folder_id,
                user_id=user_id,
                collection_type=collection_type
            )
            image_path = Path(file_path_str)
            dsf_uid = user_id

            if not await dsf.exists(dsf_uid, image_path):
                return {
                    "success": False,
                    "error": "Image file not found"
                }
            
            # Load existing metadata.json
            # Use stem (filename without extension) to match actual sidecar naming: image.jpg -> image.metadata.json
            metadata_path = image_path.parent / f"{image_path.stem}.metadata.json"
            
            metadata_data = {}
            if await dsf.exists(dsf_uid, metadata_path):
                try:
                    raw = await dsf.read_text(dsf_uid, metadata_path, encoding="utf-8")
                    metadata_data = json.loads(raw)
                except Exception as e:
                    logger.warning(f"Failed to load existing metadata.json: {e}")
                    # Continue with empty metadata - will create new one
            
            # Get existing tags
            tags = metadata_data.get('tags', [])
            if not isinstance(tags, list):
                tags = []
            
            # Add identity to tags if not already present
            identity_lower = identity_name.lower().strip()
            existing_tag_lower = [tag.lower().strip() if isinstance(tag, str) else str(tag).lower().strip() for tag in tags]
            
            if identity_lower not in existing_tag_lower:
                tags.append(identity_name)
                metadata_data["tags"] = tags

                _base = build_minimal_image_sidecar_metadata(image_path.name)
                for _k, _v in _base.items():
                    if _k not in metadata_data:
                        metadata_data[_k] = _v

                # Write updated metadata.json
                try:
                    await dsf.write_text(
                        dsf_uid,
                        metadata_path,
                        json.dumps(metadata_data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    logger.info("Updated metadata.json with identity tag: %s", identity_name)
                except Exception as e:
                    logger.error("Failed to write metadata.json: %s", e)
                    return {
                        "success": False,
                        "error": f"Failed to update metadata.json: {str(e)}"
                    }
            
            # Update document_metadata.tags in database for search indexing
            try:
                # Get current tags from document
                current_tags = getattr(doc_info, 'tags', [])
                if not isinstance(current_tags, list):
                    current_tags = []
                
                # Add identity if not present (case-insensitive)
                current_tags_lower = [tag.lower().strip() if isinstance(tag, str) else str(tag).lower().strip() for tag in current_tags]
                if identity_lower not in current_tags_lower:
                    current_tags.append(identity_name)
                    
                    # Update document in database
                    await self.document_repository.update(
                        document_id,
                        user_id=user_id,
                        tags=current_tags
                    )
                    logger.info(f"Updated document_metadata.tags with identity: {identity_name}")
                
                return {
                    "success": True,
                    "tags": tags,
                    "message": f"Added '{identity_name}' to image tags"
                }
            except Exception as e:
                logger.error(f"Failed to update document_metadata.tags: {e}")
                # Metadata.json was updated, so partial success
                return {
                    "success": True,
                    "tags": tags,
                    "warning": f"Added to metadata.json but failed to update database: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Failed to sync identity to tags: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instance for use by file watcher
_image_sidecar_service_instance = None


async def get_image_sidecar_service() -> ImageSidecarService:
    """Get global image sidecar service instance"""
    global _image_sidecar_service_instance
    if _image_sidecar_service_instance is None:
        _image_sidecar_service_instance = ImageSidecarService()
        await _image_sidecar_service_instance.initialize()
    return _image_sidecar_service_instance
