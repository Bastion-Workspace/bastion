"""
Comic Service - Handles ingestion of comic strip images with JSON sidecar metadata

DEPRECATED: This service is deprecated. All image metadata processing (including comics)
is now handled by ImageSidecarService, which accepts both 'content' and 'transcript' fields.
The file watcher routes all .metadata.json files to ImageSidecarService, not ComicService.

This file is kept for reference but should not be used in new code.
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


class ComicService:
    """
    Service for processing comic strip JSON sidecar files
    
    DEPRECATED: Use ImageSidecarService instead. This service is no longer called
    by the file watcher. ImageSidecarService handles all image types including comics
    and accepts both 'content' and 'transcript' fields for backward compatibility.
    """
    
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
        """Initialize the comic service"""
        try:
            self.document_repository = DocumentRepository()
            await self.document_repository.initialize()
            
            self.embedding_manager = await get_embedding_service()
            logger.info("Comic Service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Comic Service: {e}")
            raise
    
    async def process_comic_json(self, file_path: str) -> Dict[str, Any]:
        """
        Process a comic JSON sidecar file and create a searchable document
        
        Args:
            file_path: Path to the JSON sidecar file
            
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
            
            # Read JSON sidecar
            json_content = None
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_content = f.read()
                    comic_data = json.loads(json_content)
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
                    comic_data = json.loads(fixed_json)
                    logger.info(f"✅ Successfully fixed JSON issues in {json_path.name}")
                except json.JSONDecodeError as fix_error:
                    logger.error(f"❌ Failed to fix JSON issues: {fix_error}")
                    # Check if the error might be due to unescaped quotes
                    if '"' in json_content and 'Expecting' in str(json_error):
                        logger.error(f"💡 TIP: This error might be caused by unescaped quotes in string values.")
                        logger.error(f"💡 Quotes inside strings must be escaped: use \\\" instead of \"")
                        logger.error(f"💡 Example: \"transcript\": \"He said \\\"hello\\\"\" (correct)")
                        logger.error(f"💡 Example: \"transcript\": \"He said \"hello\"\" (incorrect)")
                    return {
                        "success": False,
                        "error": error_msg
                    }
            except IOError as io_error:
                return {
                    "success": False,
                    "error": f"Failed to read JSON file {json_path.name}: {io_error}"
                }
            
            # Validate required fields
            required_fields = ['title', 'transcript', 'date', 'image_filename']
            missing_fields = [field for field in required_fields if field not in comic_data]
            if missing_fields:
                return {
                    "success": False,
                    "error": f"Missing required fields: {missing_fields}"
                }
            
            # Extract data
            title = comic_data.get('title', '')
            transcript = comic_data.get('transcript', '')
            date = comic_data.get('date', '')
            image_filename = comic_data.get('image_filename', '')
            description = comic_data.get('description', '')
            tags = comic_data.get('tags', [])
            series = comic_data.get('series', '')
            
            # Calculate relative path from Comics root to image file
            # First try: image in same directory as JSON
            image_path = json_path.parent / image_filename
            if not image_path.exists():
                # Second try: image_filename is relative to Comics root
                comics_root = Path(settings.UPLOAD_DIR) / "Global" / "Comics"
                image_path = comics_root / image_filename
                if not image_path.exists():
                    logger.warning(f"Image file not found: {image_path}")
            
            # Calculate relative path from Comics root for serving
            comics_root = Path(settings.UPLOAD_DIR) / "Global" / "Comics"
            try:
                image_relative_path = image_path.relative_to(comics_root)
                # Use forward slashes for URL (works on both Windows and Unix)
                image_url = f"/api/comics/{str(image_relative_path).replace(chr(92), '/')}"
            except ValueError:
                # Image is outside Comics root (shouldn't happen, but fallback)
                logger.warning(f"Image path outside Comics root: {image_path}")
                image_relative_path = Path(image_filename)  # Fallback to filename
                image_url = f"/api/comics/{image_filename}"
            
            # Generate document ID
            document_id = str(uuid4())
            
            # Build searchable text from transcript, description, and tags
            searchable_text = f"{title}\n{transcript}\n{description}\n"
            if tags:
                searchable_text += " ".join(tags)
            if series:
                searchable_text += f"\n{series}"
            
            # Create document record
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=json_path.name,
                title=title,
                doc_type=DocumentType.TEXT,
                category=DocumentCategory.COMIC,
                tags=tags,
                description=description,
                author=series if series else None,
                language="en",
                publication_date=datetime.strptime(date, "%Y-%m-%d").date() if date else None,
                upload_date=datetime.now(),
                file_size=json_path.stat().st_size,
                file_hash=None,
                status=ProcessingStatus.PENDING,
                quality_metrics=QualityMetrics(
                    overall_score=1.0,
                    readability_score=1.0,
                    completeness_score=1.0
                ),
                chunk_count=0,
                entity_count=0,
                collection_type="global",
                user_id=None
            )
            
            # Store metadata including image URL (use calculated relative path)
            metadata_json = {
                "image_filename": str(image_relative_path).replace(chr(92), '/'),  # Store relative path
                "image_url": image_url,  # Use calculated URL with subfolder support
                "date": date,
                "series": series,
                "transcript": transcript,
                "comic_type": "strip"
            }
            
            # Create document in database (user_id is None for global comics)
            success = await self.document_repository.create_with_folder(
                doc_info,
                folder_id=None
            )
            
            if not success:
                return {
                    "success": False,
                    "error": "Failed to create document record"
                }
            
            # Store metadata_json separately
            import json as json_module
            await self.document_repository.update(
                document_id,
                user_id=None,
                metadata_json=json_module.dumps(metadata_json)
            )
            
            # Generate embeddings from searchable text
            try:
                from models.api_models import Chunk
                
                chunk = Chunk(
                    chunk_id=f"{document_id}_chunk_0",
                    content=searchable_text,
                    document_id=document_id,
                    metadata={
                        "title": title,
                        "date": date,
                        "series": series,
                        "tags": tags,
                        "image_url": image_url,  # Use calculated URL with subfolder support
                        "category": "comic"
                    }
                )
                
                # Vectorize the content
                await self.embedding_manager.embed_and_store_chunks(
                    chunks=[chunk],
                    user_id=None,
                    document_category="comic",
                    document_tags=tags,
                    document_title=title,
                    document_author=series if series else None,
                    document_filename=json_path.name
                )
                
                # Update document status
                await self.document_repository.update(
                    document_id,
                    user_id=None,
                    status=ProcessingStatus.COMPLETED,
                    chunk_count=1
                )
                
                logger.info(f"Successfully processed comic: {title} ({date})")
                
                return {
                    "success": True,
                    "document_id": document_id,
                    "title": title,
                    "date": date,
                    "image_url": image_url  # Use calculated URL with subfolder support
                }
                
            except Exception as e:
                logger.error(f"Failed to vectorize comic content: {e}")
                # Still mark as completed even if vectorization fails
                await self.document_repository.update(
                    document_id,
                    user_id=None,
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
            logger.error(f"Failed to process comic JSON: {e}")
            return {
                "success": False,
                "error": str(e)
            }
