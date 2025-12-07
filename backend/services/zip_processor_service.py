"""
ZIP Processor Service
Handles individual file extraction and processing from ZIP archives
Creates separate document records for each file with parent-child relationships
"""

import asyncio
import hashlib
import logging
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

import aiofiles
from fastapi import UploadFile

from config import settings
from models.api_models import (
    DocumentInfo, DocumentType, ProcessingStatus, DocumentUploadResponse
)
from repositories.document_repository import DocumentRepository
from utils.document_processor import DocumentProcessor
from services.embedding_service_wrapper import get_embedding_service

logger = logging.getLogger(__name__)


class ZipFileInfo:
    """Information about a file within a ZIP archive"""
    def __init__(self, zip_path: str, filename: str, content: bytes, doc_type: str):
        self.zip_path = zip_path
        self.filename = filename  
        self.content = content
        self.doc_type = doc_type
        self.file_hash = self._calculate_hash()
        
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(self.content).hexdigest()


class ZipProcessingResult:
    """Result of ZIP file processing"""
    def __init__(self):
        self.parent_document_id: str = ""
        self.extracted_files: List[DocumentUploadResponse] = []
        self.successful_files: int = 0
        self.failed_files: int = 0
        self.duplicate_files: int = 0
        self.processing_time: float = 0.0
        self.message: str = ""


class ZipProcessorService:
    """Service for processing ZIP files and extracting individual documents"""
    
    def __init__(self, document_repository: DocumentRepository, websocket_manager=None):
        self.document_repository = document_repository
        self.document_processor: Optional[DocumentProcessor] = None
        self.embedding_manager = None  # EmbeddingServiceWrapper
        self.websocket_manager = websocket_manager  # For real-time UI updates
        
        # Supported file types for extraction
        self.supported_extensions = {
            '.pdf': 'pdf',
            '.txt': 'txt', 
            '.docx': 'docx',
            '.html': 'html',
            '.htm': 'html',
            '.eml': 'eml',
            '.srt': 'srt',
            '.epub': 'epub'
        }
        
    async def initialize(self):
        """Initialize the ZIP processor service"""
        try:
            if not self.document_processor:
                self.document_processor = DocumentProcessor.get_instance()
                await self.document_processor.initialize()
                
            if not self.embedding_manager:
                self.embedding_manager = await get_embedding_service()
                
            logger.info("✅ ZIP Processor Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ZIP Processor Service: {e}")
            raise
    
    async def _emit_document_status_update(self, document_id: str, status: str, user_id: str = None):
        """Emit document status update via WebSocket for real-time UI updates"""
        try:
            if self.websocket_manager:
                # Get document details to include folder_id and filename
                try:
                    document_metadata = await self.document_repository.get_document_metadata(document_id)
                    folder_id = document_metadata.get("folder_id") if document_metadata else None
                    filename = document_metadata.get("filename") if document_metadata else None
                except Exception as e:
                    logger.warning(f"⚠️ Could not get metadata for document {document_id}: {e}")
                    folder_id = None
                    filename = None
                
                await self.websocket_manager.send_document_status_update(
                    document_id=document_id,
                    status=status,
                    folder_id=folder_id,
                    user_id=user_id,
                    filename=filename
                )
                logger.info(f"📡 Emitted WebSocket status update: {document_id} -> {status}")
            else:
                logger.debug(f"📡 WebSocket manager not available for status update: {document_id} -> {status}")
        except Exception as e:
            logger.error(f"❌ Failed to emit document status update: {e}")
    
    async def process_zip_file(self, file: UploadFile, parent_metadata: Dict[str, Any] = None) -> ZipProcessingResult:
        """
        Process a ZIP file by extracting and individually processing each supported file
        """
        start_time = asyncio.get_event_loop().time()
        result = ZipProcessingResult()
        
        try:
            logger.info(f"📦 Starting ZIP file processing: {file.filename}")
            
            # Read ZIP file content
            zip_content = await file.read()
            
            # Create parent ZIP document record
            parent_doc_id = await self._create_parent_zip_record(
                file.filename, zip_content, parent_metadata
            )
            result.parent_document_id = parent_doc_id
            
            # Extract files from ZIP
            extracted_files = await self._extract_zip_files(zip_content, file.filename)
            
            if not extracted_files:
                result.message = "No supported files found in ZIP archive"
                return result
            
            logger.info(f"📦 Extracted {len(extracted_files)} supported files from ZIP")
            
            # Process each extracted file individually
            for zip_file_info in extracted_files:
                try:
                    file_result = await self._process_individual_file(
                        zip_file_info, parent_doc_id, parent_metadata
                    )
                    
                    result.extracted_files.append(file_result)
                    
                    if file_result.status == ProcessingStatus.PROCESSING:
                        result.successful_files += 1
                    elif "duplicate" in file_result.message.lower():
                        result.duplicate_files += 1
                    else:
                        result.failed_files += 1
                        
                except Exception as e:
                    logger.error(f"❌ Failed to process {zip_file_info.filename}: {e}")
                    result.failed_files += 1
                    result.extracted_files.append(DocumentUploadResponse(
                        document_id="",
                        filename=zip_file_info.filename,
                        status=ProcessingStatus.FAILED,
                        message=f"Processing failed: {str(e)}"
                    ))
            
            # Update parent ZIP status
            user_id = parent_metadata.get("user_id") if parent_metadata else None
            await self._update_parent_zip_status(parent_doc_id, result, user_id)
            
            result.processing_time = asyncio.get_event_loop().time() - start_time
            result.message = (
                f"ZIP processed: {result.successful_files} successful, "
                f"{result.duplicate_files} duplicates, {result.failed_files} failed"
            )
            
            logger.info(f"✅ ZIP processing completed: {result.message}")
            return result
            
        except Exception as e:
            logger.error(f"❌ ZIP processing failed: {e}")
            result.message = f"ZIP processing failed: {str(e)}"
            return result
    
    async def _extract_zip_files(self, zip_content: bytes, zip_filename: str) -> List[ZipFileInfo]:
        """Extract supported files from ZIP archive"""
        extracted_files = []
        
        try:
            # Create temporary file for ZIP content
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                temp_zip.write(zip_content)
                temp_zip_path = temp_zip.name
            
            try:
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_file:
                    file_list = zip_file.namelist()
                    
                    for zip_path in file_list:
                        # Skip directories and hidden files
                        if zip_path.endswith('/') or Path(zip_path).name.startswith('.'):
                            continue
                            
                        file_ext = Path(zip_path).suffix.lower()
                        if file_ext in self.supported_extensions:
                            try:
                                # Extract file content
                                file_content = zip_file.read(zip_path)
                                doc_type = self.supported_extensions[file_ext]
                                
                                # Create ZipFileInfo
                                zip_file_info = ZipFileInfo(
                                    zip_path=zip_path,
                                    filename=Path(zip_path).name,
                                    content=file_content,
                                    doc_type=doc_type
                                )
                                
                                extracted_files.append(zip_file_info)
                                logger.debug(f"📄 Extracted: {zip_path} ({doc_type})")
                                
                            except Exception as e:
                                logger.error(f"❌ Failed to extract {zip_path}: {e}")
                                continue
                        else:
                            logger.debug(f"⏭️ Skipping unsupported file: {zip_path}")
            
            finally:
                # Clean up temporary ZIP file
                try:
                    os.unlink(temp_zip_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Failed to extract ZIP files: {e}")
            
        return extracted_files
    
    async def _create_parent_zip_record(self, filename: str, content: bytes, metadata: Dict[str, Any] = None) -> str:
        """Create database record for the parent ZIP file"""
        try:
            parent_doc_id = str(uuid4())
            file_hash = hashlib.sha256(content).hexdigest()
            
            # Create parent document info
            doc_info = DocumentInfo(
                document_id=parent_doc_id,
                filename=filename,
                doc_type=DocumentType.ZIP,
                upload_date=datetime.utcnow(),
                file_size=len(content),
                file_hash=file_hash,
                status=ProcessingStatus.PROCESSING
            )
            
            # Apply provided metadata
            if metadata:
                if 'title' in metadata:
                    doc_info.title = metadata['title']
                if 'category' in metadata:
                    doc_info.category = metadata['category']
                if 'tags' in metadata:
                    doc_info.tags = metadata['tags']
                if 'description' in metadata:
                    doc_info.description = metadata['description']
                if 'author' in metadata:
                    doc_info.author = metadata['author']
                if 'publication_date' in metadata:
                    doc_info.publication_date = metadata['publication_date']
            
            # Save to database
            await self.document_repository.create(doc_info)
            
            # Mark as ZIP container
            await self.document_repository.mark_as_zip_container(parent_doc_id)
            
            logger.info(f"📦 Created parent ZIP record: {parent_doc_id}")
            return parent_doc_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create parent ZIP record: {e}")
            raise
    
    async def _process_individual_file(
        self, 
        zip_file_info: ZipFileInfo, 
        parent_doc_id: str, 
        parent_metadata: Dict[str, Any] = None
    ) -> DocumentUploadResponse:
        """Process an individual file extracted from ZIP"""
        try:
            # Check for duplicate by file hash
            duplicate_doc = await self.document_repository.find_by_hash(zip_file_info.file_hash)
            if duplicate_doc:
                logger.info(f"🔄 Duplicate file in ZIP: {zip_file_info.filename} matches {duplicate_doc.filename}")
                return DocumentUploadResponse(
                    document_id=duplicate_doc.document_id,
                    filename=zip_file_info.filename,
                    status=duplicate_doc.status,
                    message=f"Duplicate file detected. Existing document: {duplicate_doc.filename}"
                )
            
            # Generate new document ID for the individual file
            document_id = str(uuid4())
            
            # Save file to uploads directory
            upload_dir = Path(settings.UPLOAD_DIR)
            upload_dir.mkdir(exist_ok=True)
            
            file_path = upload_dir / f"{document_id}_{zip_file_info.filename}"
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(zip_file_info.content)
            
            # Create document record for individual file
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=zip_file_info.filename,
                doc_type=DocumentType(zip_file_info.doc_type),
                upload_date=datetime.utcnow(),
                file_size=len(zip_file_info.content),
                file_hash=zip_file_info.file_hash,
                status=ProcessingStatus.PROCESSING
            )
            
            # Inherit metadata from parent if specified
            if parent_metadata:
                doc_info.title = parent_metadata.get('title')
                doc_info.category = parent_metadata.get('category')
                doc_info.tags = parent_metadata.get('tags', [])
                doc_info.description = parent_metadata.get('description')
                doc_info.author = parent_metadata.get('author')
                doc_info.publication_date = parent_metadata.get('publication_date')
            
            # Save to database with parent relationship
            await self.document_repository.create(doc_info)
            await self.document_repository.set_parent_relationship(
                document_id, parent_doc_id, zip_file_info.zip_path
            )
            
            # Start async processing with user_id if provided
            user_id = parent_metadata.get("user_id") if parent_metadata else None
            asyncio.create_task(self._process_file_async(document_id, file_path, zip_file_info.doc_type, user_id))
            
            logger.info(f"📄 Individual file processed: {zip_file_info.filename} ({document_id})")
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=zip_file_info.filename,
                status=ProcessingStatus.PROCESSING,
                message="File extracted from ZIP and processing started"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to process individual file {zip_file_info.filename}: {e}")
            raise
    
    async def _process_file_async(self, document_id: str, file_path: Path, doc_type: str, user_id: str = None):
        """Asynchronously process an individual file from ZIP with user isolation support"""
        try:
            # Check if document is exempt from vectorization BEFORE processing
            is_exempt = await self.document_repository.is_document_exempt(document_id)
            if is_exempt:
                logger.info(f"🚫 Document {document_id} is exempt from vectorization - skipping all processing")
                await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
                return
            
            # Process document
            result = await self.document_processor.process_document(str(file_path), doc_type, document_id)
            
            # Update status to embedding
            await self.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
            
            # Generate and store embeddings in appropriate collection
            if result.chunks:
                # Fetch document metadata for vector filtering
                try:
                    doc_info = await self.document_repository.get_by_id(document_id)
                    document_category = doc_info.category.value if doc_info and doc_info.category else None
                    document_tags = doc_info.tags if doc_info else None
                    document_title = doc_info.title if doc_info else None
                    document_author = doc_info.author if doc_info else None
                    document_filename = doc_info.filename if doc_info else None
                except Exception as e:
                    logger.debug(f"Could not fetch document metadata: {e}")
                    document_category = None
                    document_tags = None
                    document_title = None
                    document_author = None
                    document_filename = None
                
                await self.embedding_manager.embed_and_store_chunks(
                    result.chunks, 
                    user_id=user_id,
                    document_category=document_category,
                    document_tags=document_tags,
                    document_title=document_title,
                    document_author=document_author,
                    document_filename=document_filename
                )
                collection_type = "user" if user_id else "global"
                logger.debug(f"📊 Stored {len(result.chunks)} chunks for ZIP file {document_id} in {collection_type} collection")
            
            # Update final status
            await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            await self._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            if result.quality_metrics:
                await self.document_repository.update_quality_metrics(document_id, result.quality_metrics)
            
            logger.debug(f"✅ ZIP file processing completed: {document_id}")
            
        except Exception as e:
            logger.error(f"❌ ZIP file processing failed for {document_id}: {e}")
            await self.document_repository.update_status(document_id, ProcessingStatus.FAILED)
            await self._emit_document_status_update(document_id, ProcessingStatus.FAILED.value, user_id)
    
    async def _update_parent_zip_status(self, parent_doc_id: str, result: ZipProcessingResult, user_id: str = None):
        """Update the parent ZIP document status based on processing results"""
        try:
            if result.failed_files == 0:
                status = ProcessingStatus.COMPLETED
            elif result.successful_files > 0:
                status = ProcessingStatus.COMPLETED  # Partial success still counts as completed
            else:
                status = ProcessingStatus.FAILED
                
            await self.document_repository.update_status(parent_doc_id, status)
            await self._emit_document_status_update(parent_doc_id, status.value, user_id)
            logger.info(f"📦 Updated parent ZIP status: {parent_doc_id} -> {status.value}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update parent ZIP status: {e}") 