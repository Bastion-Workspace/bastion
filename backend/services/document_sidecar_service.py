"""
Document Sidecar Service - Handles ingestion of document metadata JSON sidecar files.
Supports summary, description, key_topics, tags, notes; syncs to DB and re-embeds.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

from config import settings
from models.api_models import Chunk
from repositories.document_repository import DocumentRepository
from services.embedding_service_wrapper import get_embedding_service

logger = logging.getLogger(__name__)


class DocumentSidecarService:
    """Service for processing document metadata JSON sidecar files."""

    def __init__(self):
        self.document_repository = None
        self.embedding_manager = None

    def _fix_json_common_issues(self, json_text: str) -> str:
        """Attempt to fix common JSON issues like trailing commas."""
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text, flags=re.MULTILINE)
        json_text = re.sub(r',\s*\n\s*([}\]])', r'\n\1', json_text)
        return json_text

    async def initialize(self):
        """Initialize the document sidecar service."""
        try:
            self.document_repository = DocumentRepository()
            await self.document_repository.initialize()
            self.embedding_manager = await get_embedding_service()
            logger.info("Document Sidecar Service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Document Sidecar Service: {e}")
            raise

    async def process_document_metadata(
        self, file_path: str, document_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document metadata JSON sidecar file and sync to DB + re-embed.

        Args:
            file_path: Path to the JSON sidecar file (e.g., report.metadata.json)
            document_info: Optional pre-resolved dict with document_id, user_id,
                folder_id, collection_type, filename (from file watcher).

        Returns:
            Dict with success, document_id, and optional error.
        """
        try:
            json_path = Path(file_path)
            if not json_path.exists():
                return {"success": False, "error": f"JSON file not found: {file_path}"}

            if not json_path.name.lower().endswith(".metadata.json"):
                return {
                    "success": False,
                    "error": f"File must end with .metadata.json: {file_path}",
                }

            json_content = None
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    json_content = f.read()
                    metadata_data = json.loads(json_content)
            except json.JSONDecodeError as json_error:
                error_msg = f"Invalid JSON format in {json_path.name}: {json_error}"
                logger.error(error_msg)
                try:
                    fixed_json = self._fix_json_common_issues(json_content)
                    metadata_data = json.loads(fixed_json)
                except json.JSONDecodeError:
                    return {"success": False, "error": error_msg}
            except IOError as io_error:
                return {
                    "success": False,
                    "error": f"Failed to read JSON file {json_path.name}: {io_error}",
                }

            schema_type = (metadata_data.get("schema_type") or "").strip().lower()
            if schema_type != "document":
                return {
                    "success": False,
                    "error": "schema_type must be 'document' for document sidecars",
                }

            document_filename = (
                metadata_data.get("document_filename") or ""
            ).strip()
            if not document_filename:
                document_filename = json_path.name[:-14]

            summary = (metadata_data.get("summary") or "").strip()
            description = (metadata_data.get("description") or "").strip()
            key_topics = metadata_data.get("key_topics") or []
            tags = metadata_data.get("tags") or []
            notes = (metadata_data.get("notes") or "").strip()
            author = (metadata_data.get("author") or "").strip()
            custom_fields = metadata_data.get("custom_fields")
            if not isinstance(custom_fields, dict):
                custom_fields = {}
            llm_metadata = metadata_data.get("llm_metadata")

            if not isinstance(key_topics, list):
                key_topics = []
            if not isinstance(tags, list):
                tags = [t for t in tags] if tags else []

            has_content = bool(
                summary or description or key_topics or tags or notes
            )
            if not has_content:
                return {
                    "success": False,
                    "error": "At least one of summary, description, key_topics, tags, or notes is required",
                }

            if not document_info:
                return {
                    "success": False,
                    "error": "document_info is required (document_id, user_id, etc.)",
                }

            document_id = document_info["document_id"]
            user_id = document_info.get("user_id")
            doc_info = await self.document_repository.get_by_id(document_id)

            if not doc_info or not document_id:
                return {
                    "success": False,
                    "error": f"No existing document found for filename: {document_filename}",
                }

            if user_id is None:
                user_id = getattr(doc_info, "user_id", None)

            description_for_db = (summary or description)[:500] or None
            current_tags = list(getattr(doc_info, "tags", []) or [])
            merged_tags = list(set(current_tags) | set(tags))

            metadata_json = {
                "has_document_sidecar": True,
                "summary": summary or None,
                "description": description or None,
                "key_topics": key_topics,
                "notes": notes or None,
                "custom_fields": custom_fields if custom_fields else None,
            }
            if llm_metadata and isinstance(llm_metadata, dict):
                metadata_json["llm_metadata"] = llm_metadata

            existing_metadata = getattr(doc_info, "metadata_json", None)
            if isinstance(existing_metadata, str):
                try:
                    existing_metadata = json.loads(existing_metadata)
                except Exception:
                    existing_metadata = {}
            if not isinstance(existing_metadata, dict):
                existing_metadata = {}
            for k, v in list(metadata_json.items()):
                existing_metadata[k] = v
            metadata_json = existing_metadata

            await self.document_repository.update(
                document_id,
                user_id=user_id,
                description=description_for_db,
                tags=merged_tags,
                metadata_json=json.dumps(metadata_json),
            )
            if author:
                await self.document_repository.update(
                    document_id, user_id=user_id, author=author
                )

            searchable_parts = []
            if summary:
                searchable_parts.append(summary)
            if description:
                searchable_parts.append(description)
            if notes:
                searchable_parts.append(notes)
            if key_topics:
                searchable_parts.extend(key_topics)
            if tags:
                searchable_parts.extend(tags)
            searchable_text = "\n".join(searchable_parts)

            try:
                chunk = Chunk(
                    chunk_id=f"{document_id}_sidecar_0",
                    content=searchable_text,
                    document_id=document_id,
                    chunk_index=0,
                    quality_score=1.0,
                    method="document_sidecar",
                    metadata={
                        "summary": summary[:200] if summary else None,
                        "key_topics": key_topics,
                        "tags": tags,
                    },
                )
                doc_title = getattr(doc_info, "title", None) or document_filename
                doc_author = getattr(doc_info, "author", None) or author
                doc_category = getattr(doc_info, "category", None)
                if hasattr(doc_category, "value"):
                    doc_category = doc_category.value
                await self.embedding_manager.embed_and_store_chunks(
                    chunks=[chunk],
                    user_id=user_id,
                    document_category=doc_category,
                    document_tags=merged_tags,
                    document_title=doc_title,
                    document_author=doc_author,
                    document_filename=document_filename,
                )
            except Exception as e:
                logger.warning(f"Sidecar re-embed failed (non-fatal): {e}")
                return {
                    "success": True,
                    "document_id": document_id,
                    "warning": f"DB updated but vectorization failed: {e}",
                }

            logger.info(
                f"Processed document metadata sidecar for {document_filename}"
            )
            return {
                "success": True,
                "document_id": document_id,
            }
        except Exception as e:
            logger.error(f"Failed to process document metadata: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


_document_sidecar_service_instance = None


async def get_document_sidecar_service() -> DocumentSidecarService:
    """Get global document sidecar service instance."""
    global _document_sidecar_service_instance
    if _document_sidecar_service_instance is None:
        _document_sidecar_service_instance = DocumentSidecarService()
        await _document_sidecar_service_instance.initialize()
    return _document_sidecar_service_instance
