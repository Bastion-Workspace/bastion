"""
Image Sidecar Service - Handles ingestion of image metadata JSON sidecar files
Supports universal image types: comic, artwork, meme, screenshot, medical, documentation, maps
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from config import settings
from models.api_models import DocumentInfo, DocumentType, DocumentCategory, ProcessingStatus, QualityMetrics
from repositories.document_repository import DocumentRepository
from services.embedding_service_wrapper import get_embedding_service

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
        self.embedding_manager = None
    
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
            
            self.embedding_manager = await get_embedding_service()
            logger.info("Image Sidecar Service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Image Sidecar Service: {e}")
            raise
    
    async def process_image_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Process an image metadata JSON sidecar file and create a searchable document
        
        Args:
            file_path: Path to the JSON sidecar file (e.g., image.jpg.metadata.json)
            
        Returns:
            Dict with processing results
        """
        try:
            json_path = Path(file_path)
            if not json_path.exists():
                return {
                    "success": False,
                    "error": f"JSON file not found: {file_path}"
                }
            
            # Extract image filename from sidecar name
            # "image.jpg.metadata.json" -> "image.jpg"
            if not json_path.name.endswith('.metadata.json'):
                return {
                    "success": False,
                    "error": f"File must end with .metadata.json: {file_path}"
                }
            
            image_filename_with_ext = json_path.name[:-14]  # Remove ".metadata.json"
            
            # Read JSON sidecar
            json_content = None
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_content = f.read()
                    metadata_data = json.loads(json_content)
            except json.JSONDecodeError as json_error:
                # Provide detailed error information
                error_msg = f"Invalid JSON format in {json_path.name}: {json_error}"
                logger.error(f"❌ {error_msg}")
                
                # Show context around the error if we have the content
                if json_content and hasattr(json_error, 'pos') and json_error.pos is not None:
                    error_pos = json_error.pos
                    start = max(0, error_pos - 100)
                    end = min(len(json_content), error_pos + 100)
                    context = json_content[start:end]
                    logger.error(f"❌ JSON context around error (position {error_pos}): {repr(context)}")
                    
                    # Try to show the problematic line
                    lines = json_content[:error_pos].split('\n')
                    if lines:
                        line_num = len(lines)
                        logger.error(f"❌ Error at line {line_num}, column {json_error.colno if hasattr(json_error, 'colno') else 'unknown'}")
                        all_lines = json_content.split('\n')
                        if line_num > 0 and line_num <= len(all_lines):
                            problematic_line = all_lines[line_num - 1]
                            logger.error(f"❌ Problematic line: {repr(problematic_line)}")
                
                # Try to fix common JSON issues (trailing commas) and retry parsing
                logger.info(f"🔧 Attempting to fix common JSON issues (trailing commas) in {json_path.name}")
                try:
                    fixed_json = self._fix_json_common_issues(json_content)
                    metadata_data = json.loads(fixed_json)
                    logger.info(f"✅ Successfully fixed JSON issues in {json_path.name}")
                except json.JSONDecodeError as fix_error:
                    logger.error(f"❌ Failed to fix JSON issues: {fix_error}")
                    # Check if the error might be due to unescaped quotes
                    if '"' in json_content and 'Expecting' in str(json_error):
                        logger.error(f"💡 TIP: This error might be caused by unescaped quotes in string values.")
                        logger.error(f"💡 Quotes inside strings must be escaped: use \\\" instead of \"")
                        logger.error(f"💡 Example: \"content\": \"He said \\\"hello\\\"\" (correct)")
                        logger.error(f"💡 Example: \"content\": \"He said \"hello\"\" (incorrect)")
                    return {
                        "success": False,
                        "error": error_msg
                    }
            except IOError as io_error:
                return {
                    "success": False,
                    "error": f"Failed to read JSON file {json_path.name}: {io_error}"
                }
            
            # Extract data from universal schema (ignore empty fields)
            # Handle None values from JSON (e.g., "series": null)
            # Accept both 'content' and 'transcript' fields (treat as synonyms)
            title = (metadata_data.get('title') or '').strip()
            content = (metadata_data.get('content') or metadata_data.get('transcript') or '').strip()  # Unified field: accepts both content and transcript
            author = (metadata_data.get('author') or '').strip()
            date = (metadata_data.get('date') or '').strip()
            series = (metadata_data.get('series') or '').strip()
            tags = metadata_data.get('tags', []) or []
            image_type = (metadata_data.get('type') or 'other').strip().lower()
            json_image_filename = (metadata_data.get('image_filename') or '').strip()
            
            # Extract optional type-specific fields
            location = (metadata_data.get('location') or '').strip()
            event = (metadata_data.get('event') or '').strip()
            medium = (metadata_data.get('medium') or '').strip()
            dimensions = (metadata_data.get('dimensions') or '').strip()
            body_part = (metadata_data.get('body_part') or '').strip()
            modality = (metadata_data.get('modality') or '').strip()
            map_type = (metadata_data.get('map_type') or '').strip()
            coordinates = (metadata_data.get('coordinates') or '').strip()
            application = (metadata_data.get('application') or '').strip()
            platform = (metadata_data.get('platform') or '').strip()
            
            # Determine the actual image filename to use for the document record
            # This is what users will see in the UI
            if json_image_filename:
                # Use the image_filename from JSON metadata
                actual_image_filename = json_image_filename
            else:
                # Fallback: derive from the sidecar filename
                # "image.jpg.metadata.json" -> "image.jpg"
                actual_image_filename = json_path.name[:-14]  # Remove ".metadata.json"
            
            logger.info(f"📸 Using actual image filename for document record: {actual_image_filename}")
            
            # Validate required fields
            if not title:
                return {
                    "success": False,
                    "error": "Missing required field: title"
                }
            
            # Content or transcript field must exist (but can be empty string)
            # Check if either field exists in the JSON (don't check if it's empty)
            has_content_field = 'content' in metadata_data or 'transcript' in metadata_data
            if not has_content_field:
                return {
                    "success": False,
                    "error": "Missing required field: content or transcript (field must exist, but can be empty)"
                }
            
            # If content is empty, log it for visibility (but don't reject)
            if not content:
                logger.info(f"📝 Image sidecar has empty content (title={title}) - this is allowed")
            
            # Validate image type
            # Normalize singular/plural variations
            if image_type == "map":
                image_type = "maps"
            
            valid_types = ["comic", "artwork", "meme", "screenshot", "medical", "documentation", "maps", "photo", "other"]
            if image_type not in valid_types:
                logger.warning(f"Invalid image type '{image_type}', defaulting to 'other'")
                image_type = "other"
            
            # Get category from type
            category = self.TYPE_TO_CATEGORY.get(image_type, DocumentCategory.OTHER)
            
            # Check if document already exists for this image (to avoid duplicates with file watcher)
            # This handles the case where the image file was processed before the sidecar
            existing_doc = None
            try:
                # Query database for document with matching filename in the same directory
                from services.database_manager.database_helpers import execute, fetch_one
                
                # Use the actual image filename (not the .metadata.json filename)
                query = """
                    SELECT document_id, filename, folder_id, user_id, collection_type
                    FROM document_metadata
                    WHERE filename = $1
                    AND processing_status != 'deleted'
                    LIMIT 1
                """
                rls_context = {'user_id': '', 'user_role': 'admin'}  # Admin context for global comics
                row = await fetch_one(query, actual_image_filename, rls_context=rls_context)
                
                if row:
                    logger.info(f"✅ Found existing document for image {actual_image_filename}: {row['document_id']}")
                    existing_doc = {
                        "document_id": row['document_id'],
                        "filename": row['filename'],
                        "folder_id": row['folder_id'],
                        "user_id": row['user_id'],
                        "collection_type": row['collection_type']
                    }
            except Exception as e:
                logger.warning(f"⚠️ Failed to check for existing document: {e}")
                existing_doc = None
            
            # Find image file - try multiple locations
            image_path = None
            
            # First try: image in same directory as JSON
            image_path = json_path.parent / image_filename_with_ext
            if not image_path.exists():
                # Second try: use image_filename from JSON if provided (relative path)
                json_image_filename = (metadata_data.get('image_filename') or '').strip()
                if json_image_filename:
                    # Try relative to JSON location
                    image_path = json_path.parent / json_image_filename
                    if not image_path.exists():
                        # Try relative to uploads root
                        uploads_root = Path(settings.UPLOAD_DIR)
                        image_path = uploads_root / json_image_filename
                        if not image_path.exists():
                            image_path = None
                else:
                    image_path = None
            
            if not image_path or not image_path.exists():
                logger.warning(f"Image file not found for {json_path.name}, but continuing with metadata processing")
                # Continue processing even if image not found - metadata is still valuable
            
            # Calculate image URL based on collection type
            # Determine collection from path
            path_str = str(json_path)
            path_lower = path_str.lower()
            collection_type = "global"
            user_id = None
            
            if '/global/' in path_lower:
                collection_type = "global"
                user_id = None
                # For global images, calculate relative path from the appropriate root
                # /api/comics serves from Global/Comics, so relative path should start from Comics
                uploads_root = Path(settings.UPLOAD_DIR)
                try:
                    if image_path and image_path.exists():
                        # Find Global directory in path
                        parts = image_path.parts
                        if 'Global' in parts:
                            # Use appropriate API endpoint based on subfolder
                            if 'Comics' in parts:
                                # For /api/comics, calculate relative to Comics directory
                                comics_idx = parts.index('Comics')
                                relative_parts = parts[comics_idx + 1:]  # Skip 'Comics' itself
                                image_relative_path = Path(*relative_parts)
                                image_url = f"/api/comics/{str(image_relative_path).replace(chr(92), '/')}"
                            else:
                                # For other global images, calculate relative to Global
                                global_idx = parts.index('Global')
                                relative_parts = parts[global_idx + 1:]
                                image_relative_path = Path(*relative_parts)
                                image_url = f"/api/images/{str(image_relative_path).replace(chr(92), '/')}"
                        else:
                            image_url = f"/api/images/{image_filename_with_ext}"
                    else:
                        image_url = f"/api/images/{image_filename_with_ext}"
                except Exception as e:
                    logger.warning(f"Could not calculate relative path: {e}")
                    image_url = f"/api/images/{image_filename_with_ext}"
            else:
                # Personal collection - use document-based serving
                collection_type = "personal"
                # user_id will be determined from document record
                image_url = f"/api/images/{image_filename_with_ext}"  # Placeholder, will be updated
            
            # Extract folder hierarchy from file_path and resolve to folder_id
            # Note: Folder creation may fail due to RLS, but document creation will proceed with folder_id=None
            folder_id = None
            if collection_type == "global":
                try:
                    # Extract folder hierarchy from path (e.g., Global/Comics/Dilbert/1989/file.json)
                    uploads_base = Path(settings.UPLOAD_DIR)
                    relative_path = json_path.relative_to(uploads_base)
                    parts = relative_path.parts
                    
                    if len(parts) > 2:  # e.g., ['Global', 'Comics', 'Dilbert', '1989', 'file.json']
                        # Skip 'Global' and filename, get folder parts
                        folder_parts = parts[1:-1]  # ['Comics', 'Dilbert', '1989']
                        
                        if folder_parts:
                            # Try to resolve folder hierarchy by querying existing folders first
                            # If folders don't exist, try to create them (may fail due to RLS, but that's okay)
                            try:
                                from services.folder_service import FolderService
                                folder_service = FolderService()
                                
                                # Query existing folders first (avoids RLS issues if folders already exist)
                                folders_data = await folder_service.document_repository.get_folders_by_user(
                                    user_id=None,  # Global folders have no user_id
                                    collection_type="global"
                                )
                                
                                # Build folder map: (name, parent_folder_id) -> folder_id
                                folder_map = {
                                    (f.get('name'), f.get('parent_folder_id')): f.get('folder_id') 
                                    for f in folders_data
                                }
                                
                                # Walk through folder hierarchy, finding or creating each level
                                parent_folder_id = None
                                for folder_name in folder_parts:
                                    # First, try to find existing folder
                                    key = (folder_name, parent_folder_id)
                                    if key in folder_map:
                                        # Folder exists - use it
                                        folder_id = folder_map[key]
                                        parent_folder_id = folder_id
                                        logger.debug(f"✅ Found existing folder '{folder_name}' with folder_id={folder_id}")
                                    else:
                                        # Folder doesn't exist - try to create it
                                        try:
                                            folder = await folder_service.create_folder(
                                                name=folder_name,
                                                parent_folder_id=parent_folder_id,
                                                user_id=None,  # Global folders have no user_id
                                                collection_type="global",
                                                current_user_role="admin",
                                                admin_user_id="system_admin"
                                            )
                                            
                                            if folder and hasattr(folder, 'folder_id'):
                                                parent_folder_id = folder.folder_id
                                                folder_id = folder.folder_id
                                                logger.debug(f"✅ Created folder '{folder_name}' with folder_id={folder_id}")
                                            else:
                                                logger.debug(f"⚠️ Folder '{folder_name}' creation returned None, continuing without folder_id")
                                                folder_id = None
                                                break
                                        except Exception as create_error:
                                            # Folder creation failed (likely RLS blocking), but that's okay
                                            logger.debug(f"⚠️ Failed to create folder '{folder_name}' (may be RLS blocking): {create_error}")
                                            logger.debug(f"⚠️ Continuing without folder_id - document will still be searchable")
                                            folder_id = None
                                            break
                                
                                if folder_id:
                                    logger.info(f"✅ Resolved folder_id={folder_id} for path: {'/'.join(folder_parts)}")
                            except Exception as folder_error:
                                # Folder resolution failed, but that's okay
                                # Document will be created without folder_id and can still be found via search
                                logger.debug(f"⚠️ Folder hierarchy resolution failed: {folder_error}")
                                logger.debug(f"⚠️ Document will be created without folder_id - this is acceptable for global documents")
                                folder_id = None
                except Exception as e:
                    logger.debug(f"⚠️ Failed to extract folder hierarchy from path: {e}")
                    folder_id = None
            
            # Generate document ID (or use existing if updating)
            if existing_doc:
                document_id = existing_doc["document_id"]
                # Use existing folder_id and collection_type if available
                if existing_doc.get("folder_id"):
                    folder_id = existing_doc["folder_id"]
                if existing_doc.get("collection_type"):
                    collection_type = existing_doc["collection_type"]
                if existing_doc.get("user_id") is not None:
                    user_id = existing_doc["user_id"]
                logger.info(f"📝 Using existing document: {document_id} (folder: {folder_id}, collection: {collection_type})")
            else:
                document_id = str(uuid4())
                logger.info(f"📝 Creating new document: {document_id}")
            
            # Build searchable text from all non-empty fields (universal + type-specific)
            searchable_parts = [title]
            if content:
                searchable_parts.append(content)
            if author:
                searchable_parts.append(author)
            if series:
                searchable_parts.append(series)
            if tags:
                searchable_parts.extend(tags)
            
            # Add type-specific fields to searchable text
            if location:
                searchable_parts.append(f"Location: {location}")
            if event:
                searchable_parts.append(f"Event: {event}")
            if medium:
                searchable_parts.append(f"Medium: {medium}")
            if body_part:
                searchable_parts.append(f"Body part: {body_part}")
            if modality:
                searchable_parts.append(f"Modality: {modality}")
            if map_type:
                searchable_parts.append(f"Map type: {map_type}")
            if coordinates:
                searchable_parts.append(f"Coordinates: {coordinates}")
            if application:
                searchable_parts.append(f"Application: {application}")
            if platform:
                searchable_parts.append(f"Platform: {platform}")
            # Add face identities so they are vectorized and searchable
            faces_data = metadata_data.get("faces") or []
            if isinstance(faces_data, list):
                for face in faces_data:
                    if isinstance(face, dict):
                        name = (face.get("identity_name") or face.get("suggested_identity") or "").strip()
                        if name:
                            searchable_parts.append(name)
            # Add object labels (user_tag or class_name) so they are vectorized and searchable.
            # original_class_name is not vectorized; it is kept for audit/display only.
            objects_data = metadata_data.get("objects") or []
            if isinstance(objects_data, list):
                for obj in objects_data:
                    if isinstance(obj, dict):
                        label = (obj.get("user_tag") or obj.get("class_name") or "").strip()
                        if label:
                            searchable_parts.append(label)
            searchable_text = "\n".join(searchable_parts)
            
            # Create document record
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=actual_image_filename,  # Use actual image filename (not .metadata.json)
                title=title,
                doc_type=DocumentType.TXT,  # Sidecar is JSON text file, not the image itself
                category=category,
                tags=tags if tags else [],
                description=content[:500] if content else None,  # Use content as description
                author=author if author else None,
                language="en",
                publication_date=datetime.strptime(date, "%Y-%m-%d").date() if date else None,
                upload_date=datetime.now(),
                file_size=json_path.stat().st_size,
                file_hash=None,
                status=ProcessingStatus.PROCESSING,
                quality_metrics=QualityMetrics(
                    ocr_confidence=1.0,
                    language_confidence=1.0,
                    vocabulary_score=1.0,
                    pattern_score=1.0,
                    overall_score=1.0
                ),
                chunk_count=0,
                entity_count=0,
                collection_type=collection_type,
                user_id=user_id,
                folder_id=folder_id  # Include folder_id for proper path resolution
            )
            
            # Store metadata including image URL
            # Unified field name: always use 'content' (transcript is converted to content)
            metadata_json = {
                "image_filename": image_filename_with_ext,
                # Note: image_url is NOT stored - constructed dynamically from folder path
                "type": image_type,  # CRITICAL: Use 'type' key, not 'image_type' for consistency
                "image_type": image_type,  # Keep for backward compatibility
                "date": date if date else None,
                "author": author if author else None,
                "series": series if series else None,
                "content": content,  # Unified field name (accepts both content and transcript from input)
                "has_searchable_metadata": True
            }
            
            # Add type-specific fields to metadata_json if present
            if location:
                metadata_json["location"] = location
            if event:
                metadata_json["event"] = event
            if medium:
                metadata_json["medium"] = medium
            if dimensions:
                metadata_json["dimensions"] = dimensions
            if body_part:
                metadata_json["body_part"] = body_part
            if modality:
                metadata_json["modality"] = modality
            if map_type:
                metadata_json["map_type"] = map_type
            if coordinates:
                metadata_json["coordinates"] = coordinates
            if application:
                metadata_json["application"] = application
            if platform:
                metadata_json["platform"] = platform

            # Preserve optional llm_metadata, faces, objects when present (image description, face detection, object detection)
            if metadata_data.get("llm_metadata") and isinstance(metadata_data["llm_metadata"], dict):
                metadata_json["llm_metadata"] = metadata_data["llm_metadata"]
            if metadata_data.get("faces") and isinstance(metadata_data["faces"], list):
                metadata_json["faces"] = metadata_data["faces"]
            if metadata_data.get("objects") and isinstance(metadata_data["objects"], list):
                metadata_json["objects"] = metadata_data["objects"]

            # Create or update document in database
            if existing_doc:
                # UPDATE existing document with new metadata
                logger.info(f"✅ Updating existing document {document_id} with sidecar metadata")
                try:
                    await self.document_repository.update(
                        document_id,
                        user_id=user_id,
                        title=title,
                        category=category,
                        tags=tags if tags else [],
                        description=content[:500] if content else None,
                        author=author if author else None,
                        publication_date=datetime.strptime(date, "%Y-%m-%d").date() if date else None,
                        metadata_json=json.dumps(metadata_json)
                    )
                except Exception as update_error:
                    logger.error(f"❌ Exception during document update: {update_error}")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    return {
                        "success": False,
                        "error": f"Failed to update document record: {str(update_error)}"
                    }
            else:
                # CREATE new document
                logger.info(f"✅ Creating new document {document_id} for image metadata")
                try:
                    success = await self.document_repository.create(doc_info)
                    
                    if not success:
                        logger.error(f"❌ Document creation returned False for {document_id}")
                        return {
                            "success": False,
                            "error": "Failed to create document record"
                        }
                except Exception as create_error:
                    logger.error(f"❌ Exception during document creation: {create_error}")
                    logger.error(f"❌ Exception type: {type(create_error).__name__}")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    return {
                        "success": False,
                        "error": f"Failed to create document record: {str(create_error)}"
                    }
                
                # Store metadata_json separately (for new documents only)
                await self.document_repository.update(
                    document_id,
                    user_id=user_id,
                    metadata_json=json.dumps(metadata_json)
                )
            
            # Generate embeddings from searchable text
            try:
                from models.api_models import Chunk
                
                # If updating existing document, delete old chunks first
                if existing_doc:
                    try:
                        logger.info(f"🗑️ Deleting old chunks for document {document_id} before re-vectorizing")
                        await self.embedding_manager.delete_document_chunks(document_id)
                    except Exception as delete_error:
                        logger.warning(f"⚠️ Failed to delete old chunks (non-critical): {delete_error}")
                
                chunk = Chunk(
                    chunk_id=f"{document_id}_chunk_0",
                    content=searchable_text,
                    document_id=document_id,
                    chunk_index=0,  # Required by Chunk model
                    quality_score=1.0,  # Required by Chunk model
                    method="metadata",  # Required by Chunk model
                    metadata={
                        "title": title,
                        "date": date if date else None,
                        "author": author if author else None,
                        "series": series if series else None,
                        "tags": tags if tags else [],
                        "image_url": image_url,
                        "image_type": image_type,
                        "category": category.value
                    }
                )
                
                # Vectorize the content
                await self.embedding_manager.embed_and_store_chunks(
                    chunks=[chunk],
                    user_id=user_id,
                    document_category=category.value,
                    document_tags=tags if tags else [],
                    document_title=title,
                    document_author=author if author else None,
                    document_filename=json_path.name
                )
                
                # Update document status
                await self.document_repository.update(
                    document_id,
                    user_id=user_id,
                    status=ProcessingStatus.COMPLETED,
                    chunk_count=1
                )
                
                logger.info(f"Successfully processed image metadata: {title} (type: {image_type})")
                
                return {
                    "success": True,
                    "document_id": document_id,
                    "title": title,
                    "date": date,
                    "image_type": image_type,
                    "image_url": image_url
                }
                
            except Exception as e:
                logger.error(f"Failed to vectorize image metadata content: {e}")
                # Still mark as completed even if vectorization fails
                await self.document_repository.update(
                    document_id,
                    user_id=user_id,
                    status=ProcessingStatus.COMPLETED
                )
                return {
                    "success": True,
                    "document_id": document_id,
                    "warning": f"Document created but vectorization failed: {e}"
                }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON format: {e}"
            }
        except Exception as e:
            logger.error(f"❌ Failed to process image metadata JSON: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            logger.error(f"❌ Exception args: {e.args}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e)
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
            
            if not image_path.exists():
                return {
                    "success": False,
                    "error": "Image file not found"
                }
            
            # Load existing metadata.json
            # Use stem (filename without extension) to match actual sidecar naming: image.jpg -> image.metadata.json
            metadata_path = image_path.parent / f"{image_path.stem}.metadata.json"
            
            metadata_data = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata_data = json.load(f)
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
                metadata_data['tags'] = tags
                
                # Ensure image_filename is set (required for file watcher)
                if 'image_filename' not in metadata_data:
                    metadata_data['image_filename'] = image_path.name
                
                # Write updated metadata.json
                try:
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"✅ Updated metadata.json with identity tag: {identity_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to write metadata.json: {e}")
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
