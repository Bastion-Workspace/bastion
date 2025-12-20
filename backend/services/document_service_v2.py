"""
Document Service V2 - PostgreSQL-backed document management
Handles document upload, processing, and management using PostgreSQL storage
"""

import asyncio
import hashlib
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from uuid import uuid4

import aiofiles
from fastapi import UploadFile
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import PyPDF2
import pdfplumber

from config import settings
from models.api_models import (
    DocumentInfo, DocumentStatus, ProcessingStatus, DocumentType, DocumentCategory,
    QualityMetrics, ProcessingResult, Chunk, Entity, DocumentFilterRequest,
    DocumentUpdateRequest, BulkCategorizeRequest, DocumentListResponse,
    CategorySummary, TagSummary, DocumentCategoriesResponse, BulkOperationResponse,
    DocumentUploadResponse
)
from repositories.document_repository import DocumentRepository
from utils.document_processor import DocumentProcessor
from utils.parallel_document_processor import ParallelDocumentProcessor, ProcessingConfig, ProcessingStrategy
from services.knowledge_graph_service import KnowledgeGraphService
from services.embedding_service_wrapper import get_embedding_service

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document management and processing using PostgreSQL storage"""
    
    def __init__(self):
        self.qdrant_client = None
        self.document_processor = None
        self.embedding_manager = None
        self.kg_service = None
        self.document_repository = None
    
    async def initialize(self, shared_document_repository=None, shared_embedding_manager=None, shared_kg_service=None):
        """Initialize the document service with optional shared dependencies"""
        logger.info("🔧 Initializing Document Service V2 (PostgreSQL)...")
        
        # Use shared repository if provided, otherwise initialize new one
        if shared_document_repository:
            self.document_repository = shared_document_repository
            logger.info("✅ Using shared document repository")
        else:
            self.document_repository = DocumentRepository()
            await self.document_repository.initialize()
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        
        # Create collection if it doesn't exist
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if settings.VECTOR_COLLECTION_NAME not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=settings.VECTOR_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created Qdrant collection: {settings.VECTOR_COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant: {e}")
            raise
        
        # Initialize processors (use singleton)
        self.document_processor = DocumentProcessor.get_instance()
        await self.document_processor.initialize()
        
        # Use shared embedding manager if provided, otherwise initialize new one
        if shared_embedding_manager:
            self.embedding_manager = shared_embedding_manager
            logger.info("✅ Using shared embedding manager")
        else:
            self.embedding_manager = await get_embedding_service()
            logger.info("✅ Using embedding service wrapper")
        
        # Use shared knowledge graph service if provided, otherwise initialize new one
        if shared_kg_service:
            self.kg_service = shared_kg_service
            logger.info("✅ Using shared knowledge graph service")
        else:
            self.kg_service = KnowledgeGraphService()
            await self.kg_service.initialize()
        
        # Initialize WebSocket manager for real-time updates
        try:
            from main import websocket_manager
            self.websocket_manager = websocket_manager
            logger.info("✅ WebSocket manager connected for real-time updates")
        except (ImportError, AttributeError):
            self.websocket_manager = None
            logger.warning("⚠️ WebSocket manager not available - real-time updates disabled")
        
        logger.info("✅ Document Service V2 initialized")
    
    async def _emit_document_status_update(self, document_id: str, status: str, user_id: str = None):
        """Emit document status update via WebSocket - **BULLY!** Now with filename for toast notifications!"""
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
                    filename=filename  # **ROOSEVELT FIX**: Pass filename for UI notifications!
                )
            else:
                logger.debug(f"📡 WebSocket manager not available for status update: {document_id} -> {status}")
        except Exception as e:
            logger.error(f"❌ Failed to emit document status update: {e}")
    
    async def upload_and_process(self, file: UploadFile, doc_type: str = None, user_id: str = None, folder_id: str = None, team_id: str = None) -> DocumentUploadResponse:
        """Upload and process a document with optional user isolation"""
        start_time = time.time()
        
        try:
            # Read file content
            content = await file.read()
            
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(content)
            
            # Check for duplicate
            duplicate_doc = await self.document_repository.find_by_hash(file_hash)
            if duplicate_doc:
                logger.info(f"🔄 Duplicate file detected: {file.filename} matches {duplicate_doc.filename}")
                return DocumentUploadResponse(
                    document_id=duplicate_doc.document_id,
                    filename=file.filename,
                    status=duplicate_doc.status,
                    message=f"Duplicate file detected. Existing document: {duplicate_doc.filename} (ID: {duplicate_doc.document_id})"
                )
            
            # Generate new document ID
            document_id = str(uuid4())
            
            # Determine file path using folder structure
            # No need for ID prefix - folder isolation provides uniqueness
            if team_id:
                collection_type = "team"
            else:
                collection_type = "user" if user_id else "global"
            
            # Get folder service to determine proper file path
            from services.service_container import get_service_container
            container = await get_service_container()
            folder_service = container.folder_service
            
            file_path = await folder_service.get_document_file_path(
                filename=file.filename,
                folder_id=folder_id,
                user_id=user_id,
                collection_type=collection_type,
                team_id=team_id
            )
            
            logger.info(f"📁 Saving file to: {file_path}")
            
            # Save uploaded file (directory already created by folder_service)
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Determine document type
            if not doc_type:
                doc_type = self._detect_document_type(file.filename)
            
            # Automatically exempt .json files from vectorization
            exempt_from_vectorization = None
            if file.filename and file.filename.lower().endswith('.json'):
                exempt_from_vectorization = True
                logger.info(f"🚫 JSON file detected ({file.filename}) - automatically exempting from vectorization")
            
            # Create document record with file hash and user ownership
            # collection_type already set above
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=file.filename,
                doc_type=DocumentType(doc_type),
                upload_date=datetime.utcnow(),
                file_size=len(content),
                file_hash=file_hash,
                status=ProcessingStatus.PROCESSING,
                user_id=user_id,  # Track document ownership
                collection_type=collection_type,  # Set correct collection type
                team_id=team_id,  # Track team ownership if applicable
                exempt_from_vectorization=exempt_from_vectorization
            )
            
            # ROOSEVELT FIX: Save to database and assign folder in a single transaction
            logger.info(f"🔧 DEBUG: About to create document record: {document_id}, user_id: {user_id}, collection_type: {'user' if user_id else 'global'}")
            creation_success = await self.document_repository.create_with_folder(doc_info, folder_id)
            if not creation_success:
                logger.error(f"❌ Failed to create document record in database: {document_id}")
                raise Exception(f"Document creation failed for {document_id}")
            logger.info(f"✅ Document record created successfully in database: {document_id}")
            if folder_id:
                logger.info(f"✅ Document {document_id} assigned to folder {folder_id} within creation transaction")
            
            # **ROOSEVELT FAST-TRACK FOR ORG FILES!**
            # Org files process instantly (no vectorization) - wait for completion
            if doc_type == 'org':
                logger.info(f"⚡ BULLY! Org file fast-track - processing synchronously")
                await self._process_document_async(document_id, file_path, doc_type, user_id)
                
                logger.info(f"📄 Org file uploaded and ready: {file.filename} ({document_id})")
                
                return DocumentUploadResponse(
                    document_id=document_id,
                    filename=file.filename,
                    status=ProcessingStatus.COMPLETED,
                    message=f"Org file uploaded and ready to use"
                )
            else:
                # **ROOSEVELT FOLDER INHERITANCE**: Apply folder tags if folder has metadata
                if folder_id:
                    try:
                        from services.service_container import get_service_container
                        container = await get_service_container()
                        folder_service = container.folder_service
                        folder_metadata = await folder_service.get_folder_metadata(folder_id)
                        
                        if folder_metadata.get('inherit_tags', True):
                            folder_category = folder_metadata.get('category')
                            folder_tags = folder_metadata.get('tags', [])
                            
                            if folder_category or folder_tags:
                                from models.api_models import DocumentUpdateRequest, DocumentCategory
                                
                                # Parse category enum
                                doc_category = None
                                if folder_category:
                                    try:
                                        doc_category = DocumentCategory(folder_category)
                                    except ValueError:
                                        logger.warning(f"⚠️ Invalid folder category '{folder_category}'")
                                
                                # Update document with folder metadata
                                update_request = DocumentUpdateRequest(
                                    category=doc_category,
                                    tags=folder_tags if folder_tags else None
                                )
                                await self.update_document_metadata(document_id, update_request)
                                logger.info(f"📋 FOLDER INHERITANCE: Applied folder metadata to {document_id} - category={folder_category}, tags={folder_tags}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to apply folder metadata inheritance: {e}")
                        # Continue with upload even if inheritance fails
                
                # Start async processing with user_id for regular documents
                asyncio.create_task(self._process_document_async(document_id, file_path, doc_type, user_id))
                
                logger.info(f"📄 Document uploaded to {collection_type} collection: {file.filename} ({document_id}) - Hash: {file_hash[:8]}...")
                
                return DocumentUploadResponse(
                    document_id=document_id,
                    filename=file.filename,
                    status=ProcessingStatus.PROCESSING,
                    message=f"Document uploaded successfully to {collection_type} collection, processing started"
                )
            
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            raise
    
    async def _analyze_pdf_type(self, file_path: Path) -> dict:
        """Analyze PDF to determine processing strategy"""
        try:
            # Quick metadata check
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata or {}
                
            producer = str(metadata.get('/Producer', '')).lower()
            creator = str(metadata.get('/Creator', '')).lower()
            
            # OCR software signatures
            ocr_indicators = ['ocr', 'scan', 'abbyy', 'tesseract', 'finereader']
            has_ocr_metadata = any(indicator in producer + creator for indicator in ocr_indicators)
            
            # Quick content analysis
            with pdfplumber.open(file_path) as pdf:
                if not pdf.pages:
                    return {"type": "empty", "confidence": 1.0, "reason": "No pages found"}
                    
                first_page = pdf.pages[0]
                text = first_page.extract_text() or ""
                images = first_page.images or []
                
                # Analysis metrics
                text_length = len(text.strip())
                image_count = len(images)
                
                # Check for fonts more safely
                try:
                    has_fonts = bool(getattr(first_page, 'fonts', None))
                except:
                    has_fonts = False
                
                # Analyze text quality for OCR detection
                text_quality_score = self._analyze_text_quality(text)
                
                # Decision logic with text quality analysis
                if has_ocr_metadata:
                    return {
                        "type": "ocr_candidate",
                        "confidence": 0.9,
                        "reason": "OCR software detected in metadata",
                        "metadata": {"producer": producer, "creator": creator, "text_quality": text_quality_score}
                    }
                
                elif text_length == 0 and image_count > 0:
                    return {
                        "type": "scanned_image", 
                        "confidence": 0.95,
                        "reason": "No text layer, images present"
                    }
                
                elif text_length > 50 and text_quality_score < 0.3:
                    return {
                        "type": "ocr_candidate",
                        "confidence": 0.85,
                        "reason": f"Poor text quality detected (score: {text_quality_score:.2f}) - likely OCR",
                        "text_quality": text_quality_score
                    }
                
                elif text_length > 100 and image_count == 0 and has_fonts and text_quality_score > 0.7:
                    return {
                        "type": "native_digital",
                        "confidence": 0.9,
                        "reason": f"Clean text, no images, proper fonts (quality: {text_quality_score:.2f})",
                        "text_quality": text_quality_score
                    }
                
                elif text_length > 100 and image_count > 3:
                    return {
                        "type": "ocr_candidate",
                        "confidence": 0.7,
                        "reason": "Text + many images suggests OCR",
                        "text_quality": text_quality_score
                    }
                
                elif text_quality_score > 0.6:
                    return {
                        "type": "native_digital",
                        "confidence": 0.7,
                        "reason": f"Good text quality suggests native digital (score: {text_quality_score:.2f})",
                        "text_quality": text_quality_score
                    }
                
                else:
                    return {
                        "type": "ocr_candidate",
                        "confidence": 0.6,
                        "reason": f"Uncertain - defaulting to OCR processing (quality: {text_quality_score:.2f})",
                        "text_quality": text_quality_score
                    }
                    
        except Exception as e:
            logger.error(f"❌ PDF analysis failed: {e}")
            return {
                "type": "unknown",
                "confidence": 0.0,
                "reason": f"Analysis failed: {str(e)}"
            }
    
    def _analyze_text_quality(self, text: str) -> float:
        """Analyze text quality to detect OCR artifacts"""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Sample first 1000 characters for analysis
        sample = text[:1000]
        total_chars = len(sample)
        
        if total_chars == 0:
            return 0.0
        
        # Count various quality indicators
        alphabetic_chars = sum(1 for c in sample if c.isalpha())
        numeric_chars = sum(1 for c in sample if c.isdigit())
        space_chars = sum(1 for c in sample if c.isspace())
        punctuation_chars = sum(1 for c in sample if c in '.,!?;:()[]{}"-')
        
        # Count OCR artifacts and suspicious patterns
        suspicious_chars = sum(1 for c in sample if c in '«»°±²³¹¼½¾¿×÷')
        isolated_chars = 0
        garbled_sequences = 0
        
        # Look for isolated single characters (common OCR error)
        words = sample.split()
        for word in words:
            if len(word) == 1 and word.isalpha():
                isolated_chars += 1
        
        # Look for garbled sequences (multiple consecutive non-alphabetic chars)
        import re
        garbled_patterns = re.findall(r'[^a-zA-Z\s]{3,}', sample)
        garbled_sequences = len(garbled_patterns)
        
        # Look for excessive spacing or formatting issues
        excessive_spaces = len(re.findall(r'\s{3,}', sample))
        
        # Calculate quality score (0.0 = poor OCR, 1.0 = clean text)
        quality_score = 1.0
        
        # Penalize for suspicious characters
        if total_chars > 0:
            suspicious_ratio = suspicious_chars / total_chars
            quality_score -= suspicious_ratio * 2.0
        
        # Penalize for too many isolated characters
        if len(words) > 0:
            isolated_ratio = isolated_chars / len(words)
            quality_score -= isolated_ratio * 1.5
        
        # Penalize for garbled sequences
        quality_score -= garbled_sequences * 0.1
        
        # Penalize for excessive spacing
        quality_score -= excessive_spaces * 0.05
        
        # Bonus for good alphabetic ratio
        if total_chars > 0:
            alphabetic_ratio = alphabetic_chars / total_chars
            if alphabetic_ratio > 0.7:
                quality_score += 0.2
        
        # Bonus for reasonable punctuation
        if total_chars > 0:
            punct_ratio = punctuation_chars / total_chars
            if 0.02 <= punct_ratio <= 0.15:  # Reasonable punctuation range
                quality_score += 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, quality_score))

    async def _process_native_pdf(self, document_id: str, file_path: Path, user_id: str = None):
        """Fast processing for native digital PDFs"""
        logger.info(f"🚀 Fast-track processing native PDF: {document_id}")
        
        # Check if document is exempt from vectorization BEFORE processing
        is_exempt = await self.document_repository.is_document_exempt(document_id, user_id)
        if is_exempt:
            logger.info(f"🚫 Document {document_id} is exempt from vectorization - skipping all processing")
            await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            await self._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            return
        
        # Standard text extraction only
        result = await self.document_processor.process_document(str(file_path), 'pdf', document_id)
        
        # Update status to embedding
        await self.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
        await self._emit_document_status_update(document_id, ProcessingStatus.EMBEDDING.value, user_id)
        
        # **ROOSEVELT METADATA FIX**: Fetch document metadata for vector filtering
        doc_info = await self.document_repository.get_by_id(document_id)
        document_category = doc_info.category.value if doc_info and doc_info.category else None
        document_tags = doc_info.tags if doc_info else None
        
        # Generate and store embeddings with metadata
        if result.chunks:
            await self.embedding_manager.embed_and_store_chunks(
                result.chunks,
                user_id=user_id,
                document_category=document_category,
                document_tags=document_tags
            )
            logger.info(f"📊 Stored {len(result.chunks)} chunks for native PDF {document_id}")
        
        # Store entities in knowledge graph
        if result.entities and self.kg_service:
            await self.kg_service.store_entities(result.entities, document_id)
            logger.info(f"🔗 Stored {len(result.entities)} entities for document {document_id}")
        
        # Update final status
        await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
        await self._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
        if result.quality_metrics:
            await self.document_repository.update_quality_metrics(document_id, result.quality_metrics)
        
        logger.info(f"✅ Native PDF processed: {len(result.chunks)} chunks")

    async def _process_segmentation_candidate(self, document_id: str, file_path: Path, analysis: dict, user_id: str = None):
        """Process OCR candidates using enhanced PDF segmentation"""
        logger.info(f"🔄 OCR candidate processing: {document_id}")
        
        # Check if document is exempt from vectorization BEFORE processing
        is_exempt = await self.document_repository.is_document_exempt(document_id, user_id)
        if is_exempt:
            logger.info(f"🚫 Document {document_id} is exempt from vectorization - skipping all processing")
            await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            await self._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            return
        
        # Use enhanced PDF segmentation service for better processing
        try:
            from services.enhanced_pdf_segmentation_service import EnhancedPDFSegmentationService
            from models.segmentation_models import PDFExtractionRequest
            
            # Initialize enhanced PDF segmentation service if needed
            if not hasattr(self, 'enhanced_pdf_service') or not self.enhanced_pdf_service:
                self.enhanced_pdf_service = EnhancedPDFSegmentationService(self.document_repository, self.embedding_manager)
                await self.enhanced_pdf_service.initialize()
            
            # Extract PDF info for enhanced processing
            extraction_request = PDFExtractionRequest(
                document_id=document_id,
                extract_images=False,  # Enhanced service works directly with PDF
                image_dpi=300,
                image_format="PNG"
            )
            
            result = await self.enhanced_pdf_service.extract_pdf_info(extraction_request)
            logger.info(f"📄 Enhanced PDF extraction completed: {result.pages_extracted} pages ready for segmentation")
            
            # Also try standard text extraction as fallback
            try:
                text_result = await self.document_processor.process_document(str(file_path), 'pdf', document_id)
                if text_result.chunks:
                    # Update status to embedding
                    await self.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
                    
                    # Fetch document metadata
                    doc_info = await self.document_repository.get_by_id(document_id)
                    document_category = doc_info.category.value if doc_info and doc_info.category else None
                    document_tags = doc_info.tags if doc_info else None
                    
                    await self.embedding_manager.embed_and_store_chunks(
                        text_result.chunks,
                        user_id=user_id,
                        document_category=document_category,
                        document_tags=document_tags
                    )
                    logger.info(f"📄 Extracted {len(text_result.chunks)} text chunks as fallback")
                    
                    # Store entities in knowledge graph
                    if text_result.entities and self.kg_service:
                        await self.kg_service.store_entities(text_result.entities, document_id)
                        logger.info(f"🔗 Stored {len(text_result.entities)} entities for document {document_id}")
                    
                    # Update quality metrics
                    if text_result.quality_metrics:
                        await self.document_repository.update_quality_metrics(document_id, text_result.quality_metrics)
            except Exception as e:
                logger.warning(f"⚠️ Text extraction failed, enhanced segmentation available: {e}")
            
            # Update final status
            await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            logger.info(f"✅ Enhanced segmentation candidate ready: {result.pages_extracted} pages available for editing")
            
        except Exception as e:
            logger.error(f"❌ Enhanced PDF processing failed, falling back to standard: {e}")
            
            # Fallback to standard processing
            try:
                result = await self.document_processor.process_document(str(file_path), 'pdf', document_id)
                if result.chunks:
                    # Update status to embedding
                    await self.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
                    
                    # Fetch document metadata
                    doc_info = await self.document_repository.get_by_id(document_id)
                    document_category = doc_info.category.value if doc_info and doc_info.category else None
                    document_tags = doc_info.tags if doc_info else None
                    
                    await self.embedding_manager.embed_and_store_chunks(
                        result.chunks,
                        user_id=user_id,
                        document_category=document_category,
                        document_tags=document_tags
                    )
                    logger.info(f"📄 Extracted {len(result.chunks)} text chunks as standard fallback")
                    
                    # Store entities in knowledge graph
                    if result.entities and self.kg_service:
                        await self.kg_service.store_entities(result.entities, document_id)
                        logger.info(f"🔗 Stored {len(result.entities)} entities for document {document_id}")
                    
                    # Update quality metrics
                    if result.quality_metrics:
                        await self.document_repository.update_quality_metrics(document_id, result.quality_metrics)
                        
                # Update final status
                await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
                logger.info(f"✅ Standard processing completed as fallback")
                
            except Exception as fallback_error:
                logger.error(f"❌ All processing methods failed: {fallback_error}")
                await self.document_repository.update_status(document_id, ProcessingStatus.FAILED)
        await self._emit_document_status_update(document_id, ProcessingStatus.FAILED.value, user_id)

    async def _process_document_async(self, document_id: str, file_path: Path, doc_type: str, user_id: str = None):
        """Asynchronously process a document with intelligent routing"""
        try:
            logger.info(f"🔄 Processing document: {document_id}")
            
            # Initialize document processor if needed (use singleton)
            if not self.document_processor:
                self.document_processor = DocumentProcessor.get_instance()
                await self.document_processor.initialize()
            
            # Initialize embedding service wrapper if needed
            if not self.embedding_manager:
                self.embedding_manager = await get_embedding_service()
            
            # Process all documents using standard processing
            await self._process_standard_document(document_id, file_path, doc_type, user_id)
            
            logger.info(f"✅ Document processing completed: {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            await self.document_repository.update_status(document_id, ProcessingStatus.FAILED)
            await self._emit_document_status_update(document_id, ProcessingStatus.FAILED.value, user_id)

    async def _process_standard_document(self, document_id: str, file_path: Path, doc_type: str, user_id: str = None):
        """Standard processing for non-PDF documents with user isolation support"""
        # ROOSEVELT DOCTRINE: Org Mode files skip vectorization entirely!
        if doc_type == 'org':
            logger.info(f"📋 Org file stored for structured access: {document_id}")
            logger.info(f"Use llm-orchestrator for task management operations (OrgInboxAgent and OrgProjectAgent migrated)")
            
            # Process document for metadata only (no vectorization)
            result = await self.document_processor.process_document(str(file_path), doc_type, document_id)
            
            # Update quality metrics if available
            if result.quality_metrics:
                await self.document_repository.update_quality_metrics(document_id, result.quality_metrics)
            
            # Mark as completed immediately - no embedding needed
            await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            await self._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            logger.info(f"✅ Org file ready for structured queries: {document_id}")
            return
        
        # Check if document is exempt from vectorization BEFORE processing
        # This prevents unnecessary processing work for exempt documents
        is_exempt = await self.document_repository.is_document_exempt(document_id, user_id)
        if is_exempt:
            logger.info(f"🚫 Document {document_id} is exempt from vectorization - skipping all processing")
            await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            await self._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            return
        
        # Process document (only if not exempt)
        result = await self.document_processor.process_document(str(file_path), doc_type, document_id)
        
        # Update status to embedding for regular documents
        await self.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
        await self._emit_document_status_update(document_id, ProcessingStatus.EMBEDDING.value, user_id)
        
        # **ROOSEVELT METADATA FIX**: Fetch document category and tags for vector filtering
        doc_info = await self.document_repository.get_by_id(document_id)
        document_category = doc_info.category.value if doc_info and doc_info.category else None
        document_tags = doc_info.tags if doc_info else None
        team_id = doc_info.team_id if doc_info and hasattr(doc_info, 'team_id') else None
        
        # Generate and store embeddings in appropriate collection with metadata
        if result.chunks:
            await self.embedding_manager.embed_and_store_chunks(
                result.chunks, 
                user_id=user_id,
                team_id=team_id,
                document_category=document_category,
                document_tags=document_tags
            )
            if team_id:
                collection_type = "team"
            elif user_id:
                collection_type = "user"
            else:
                collection_type = "global"
            logger.info(f"📊 Stored {len(result.chunks)} chunks in {collection_type} collection for document {document_id}")
        
        # Store entities in knowledge graph
        if result.entities and self.kg_service:
            await self.kg_service.store_entities(result.entities, document_id)
            logger.info(f"🔗 Stored {len(result.entities)} entities for document {document_id}")
        
        # Update final status
        await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
        await self._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
        if result.quality_metrics:
            await self.document_repository.update_quality_metrics(document_id, result.quality_metrics)
    
    async def import_from_url(self, url: str, content_type: str = "html") -> DocumentUploadResponse:
        """Import content from URL"""
        document_id = str(uuid4())
        
        try:
            logger.info(f"🔗 Starting URL import: {url}")
            
            # Create document record with URL as filename
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=url,  # Use URL as filename
                doc_type=DocumentType.URL,
                upload_date=datetime.utcnow(),
                file_size=0,  # Will be updated after fetch
                status=ProcessingStatus.PROCESSING
            )
            
            # Save to database
            await self.document_repository.create(doc_info)
            
            # Start async processing
            asyncio.create_task(self._process_url_async(document_id, url, content_type))
            
            logger.info(f"🔗 URL import started: {url} ({document_id})")
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=url,
                status=ProcessingStatus.PROCESSING,
                message="URL import started, fetching content..."
            )
            
        except Exception as e:
            logger.error(f"❌ URL import failed: {e}")
            # Update status to failed
            await self.document_repository.update_status(document_id, ProcessingStatus.FAILED)
            await self._emit_document_status_update(document_id, ProcessingStatus.FAILED.value, user_id)
            raise
    
    async def _process_url_async(self, document_id: str, url: str, content_type: str, user_id: str = None):
        """Asynchronously fetch and process URL content using Crawl4AI for enhanced extraction"""
        try:
            logger.info(f"🔗 Fetching URL content: {url}")
            
            # Determine file type from URL
            url_lower = url.lower()
            if url_lower.endswith('.pdf'):
                file_type = 'pdf'
                file_extension = '.pdf'
                use_crawl4ai = False  # Use direct download for binary files
            elif url_lower.endswith('.docx'):
                file_type = 'docx'
                file_extension = '.docx'
                use_crawl4ai = False  # Use direct download for binary files
            elif url_lower.endswith('.epub'):
                file_type = 'epub'
                file_extension = '.epub'
                use_crawl4ai = False  # Use direct download for binary files
            elif url_lower.endswith('.txt'):
                file_type = 'txt'
                file_extension = '.txt'
                use_crawl4ai = False  # Use direct download for text files
            else:
                file_type = 'html'
                file_extension = '.html'
                use_crawl4ai = True  # Use Crawl4AI for web pages
            
            logger.info(f"🔍 Detected file type: {file_type}, using Crawl4AI: {use_crawl4ai}")
            
            if use_crawl4ai:
                # Use Crawl4AI for enhanced web content extraction
                content, original_html, images = await self._extract_content_with_crawl4ai(url)
                content_size = len(content.encode('utf-8'))
                
                # Store original HTML for display
                if original_html:
                    await self._store_original_html(document_id, original_html, images)
                
                logger.info(f"📥 Crawl4AI extracted {content_size} bytes from {url}")
            else:
                # Use direct download for binary/text files
                content, content_size = await self._download_file_directly(url, file_type)
                original_html = None
                images = None
            
            # Update file size in database
            await self.document_repository.update_file_size(document_id, content_size)
            
            # Create temporary file for processing
            upload_dir = Path(settings.UPLOAD_DIR)
            upload_dir.mkdir(exist_ok=True)
            
            # Create a safe filename from URL
            safe_filename = url.replace('://', '_').replace('/', '_').replace('?', '_').replace('&', '_')
            temp_file_path = upload_dir / f"{document_id}_{safe_filename}{file_extension}"
            
            # Save content to temporary file
            if file_type in ['pdf', 'docx', 'epub']:
                # Binary file
                async with aiofiles.open(temp_file_path, 'wb') as f:
                    await f.write(content)
            else:
                # Text file
                async with aiofiles.open(temp_file_path, 'w', encoding='utf-8') as f:
                    await f.write(content)
            
            logger.info(f"💾 Saved URL content to: {temp_file_path}")
            
            # Initialize processors if needed
            if not self.document_processor:
                self.document_processor = DocumentProcessor.get_instance()
                await self.document_processor.initialize()
            
            if not self.embedding_manager:
                self.embedding_manager = await get_embedding_service()
            
            # Check if document is exempt from vectorization BEFORE processing
            is_exempt = await self.document_repository.is_document_exempt(document_id, user_id)
            if is_exempt:
                logger.info(f"🚫 Document {document_id} is exempt from vectorization - skipping all processing")
                await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
                return
            
            # Process the content with the detected file type
            result = await self.document_processor.process_document(str(temp_file_path), file_type, document_id)
            
            # Update status to embedding
            await self.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
            
            # Fetch document metadata for vector filtering
            doc_info = await self.document_repository.get_by_id(document_id)
            document_category = doc_info.category.value if doc_info and doc_info.category else None
            document_tags = doc_info.tags if doc_info else None
            document_title = doc_info.title if doc_info else None
            document_author = doc_info.author if doc_info else None
            document_filename = doc_info.filename if doc_info else None
            
            # Generate and store embeddings with metadata
            if result.chunks:
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
                logger.info(f"📊 Stored {len(result.chunks)} chunks in {collection_type} collection for URL {url}")
            
            # Update final status
            await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            if result.quality_metrics:
                await self.document_repository.update_quality_metrics(document_id, result.quality_metrics)
            
            logger.info(f"✅ URL processing completed: {url}")
            
        except Exception as e:
            logger.error(f"❌ URL processing failed for {url}: {e}")
            await self.document_repository.update_status(document_id, ProcessingStatus.FAILED)
            await self._emit_document_status_update(document_id, ProcessingStatus.FAILED.value, user_id)
    
    async def _extract_content_with_crawl4ai(self, url: str) -> tuple[str, Optional[str], Optional[List[Dict[str, Any]]]]:
        """Extract content using Crawl4AI for enhanced web scraping"""
        try:
            logger.info(f"🕷️ Using Crawl4AI to extract content from {url}")
            
            # Import Crawl4AI tools
            from services.langgraph_tools.crawl4ai_web_tools import Crawl4AIWebTools
            
            # Initialize Crawl4AI tools
            crawl4ai_tools = Crawl4AIWebTools()
            
            # Extract content using Crawl4AI
            result = await crawl4ai_tools.crawl_web_content(
                urls=[url],
                extraction_strategy="markdown",  # Always markdown - we don't configure Crawl4AI with LLM
                chunking_strategy="NlpSentenceChunking",
                word_count_threshold=10
            )
            
            if result and result.get("results") and len(result["results"]) > 0:
                crawl_result = result["results"][0]
                
                # Extract clean text content
                extracted_text = ""
                content_blocks = crawl_result.get("content_blocks", [])
                
                if content_blocks:
                    # Use content_blocks for clean text
                    for block in content_blocks:
                        if block and isinstance(block, str):
                            extracted_text += block + "\n\n"
                
                if extracted_text.strip():
                    # Clean up the content for text storage
                    cleaned_content = self._clean_extracted_content(extracted_text)
                    
                    # Get original HTML content for display
                    original_html = crawl_result.get("full_content", "")
                    
                    # Extract and enhance images from the crawl result
                    raw_images = crawl_result.get("images", [])
                    enhanced_images = []
                    
                    for img in raw_images:
                        enhanced_image = {
                            "src": img.get("src") or img.get("url"),
                            "alt": img.get("alt", ""),
                            "title": img.get("title", ""),
                            "width": img.get("width"),
                            "height": img.get("height"),
                            "caption": img.get("caption", ""),
                            "position": img.get("position", "inline"),
                            "type": img.get("type", "content")
                        }
                        enhanced_images.append(enhanced_image)
                    
                    logger.info(f"✅ Crawl4AI extracted {len(cleaned_content)} characters, {len(original_html)} HTML chars, and {len(enhanced_images)} images from {url}")
                    return cleaned_content, original_html, enhanced_images
                
                # Fallback to full_content if no content_blocks available
                if crawl_result.get("full_content"):
                    content = crawl_result["full_content"]
                    # Clean up the content (this will be HTML, so we need to extract text)
                    cleaned_content = self._clean_extracted_content(content)
                    # Get original HTML content for display
                    original_html = crawl_result.get("full_content", "")
                    # Extract and enhance images from the crawl result
                    raw_images = crawl_result.get("images", [])
                    enhanced_images = []
                    
                    for img in raw_images:
                        enhanced_image = {
                            "src": img.get("src") or img.get("url"),
                            "alt": img.get("alt", ""),
                            "title": img.get("title", ""),
                            "width": img.get("width"),
                            "height": img.get("height"),
                            "caption": img.get("caption", ""),
                            "position": img.get("position", "inline"),
                            "type": img.get("type", "content")
                        }
                        enhanced_images.append(enhanced_image)
                    
                    logger.info(f"✅ Crawl4AI extracted {len(cleaned_content)} characters, {len(original_html)} HTML chars, and {len(enhanced_images)} images from {url} (fallback)")
                    return cleaned_content, original_html, enhanced_images
                
                logger.warning(f"⚠️ No content found in Crawl4AI result for {url}")
                return "", None, None
            
            logger.warning(f"⚠️ No Crawl4AI results for {url}")
            return "", None, None
            
        except Exception as e:
            logger.error(f"❌ Crawl4AI extraction failed for {url}: {e}")
            return "", None, None
    
    def _clean_extracted_content(self, content: str) -> str:
        """Clean extracted content for better readability - focus on main article content"""
        if not content:
            return ""
        
        # If content looks like HTML, try to extract text from it
        if "<html>" in content.lower() or "<body>" in content.lower():
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove navigation and website chrome elements
                for element in soup(["script", "style", "nav", "header", "footer", "aside", "menu", "sidebar"]):
                    element.decompose()
                
                # Remove common navigation and menu classes/IDs
                for element in soup.find_all(class_=re.compile(r'(nav|menu|header|footer|sidebar|breadcrumb|pagination|social|share|ad|banner|logo|widget|sidebar|column|panel)', re.I)):
                    element.decompose()
                
                for element in soup.find_all(id=re.compile(r'(nav|menu|header|footer|sidebar|breadcrumb|pagination|social|share|ad|banner|logo|widget|sidebar|column|panel)', re.I)):
                    element.decompose()
                
                # Remove common navigation patterns
                for element in soup.find_all("div", class_=re.compile(r'(navigation|navbar|menubar|toolbar|banner|advertisement|sidebar|widget|column|panel|menu|nav)', re.I)):
                    element.decompose()
                
                # Remove Hackaday-specific elements
                for element in soup.find_all("div", class_=re.compile(r'(sidebar|widget|column|panel|menu|nav|related|popular|trending|recommended)', re.I)):
                    element.decompose()
                
                # Remove elements with common sidebar/column patterns
                for element in soup.find_all("div", class_=re.compile(r'(col-|column-|sidebar-|widget-|panel-)', re.I)):
                    element.decompose()
                
                # Remove elements with specific Hackaday patterns
                for element in soup.find_all("div", class_=re.compile(r'(hackaday|hack|sidebar|widget)', re.I)):
                    element.decompose()
                
                # Get text content
                content = soup.get_text()
            except ImportError:
                # If BeautifulSoup is not available, do basic HTML tag removal
                import re
                content = re.sub(r'<[^>]+>', '', content)
        
        # Remove excessive whitespace
        import re
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Remove common web artifacts and navigation text
        artifacts_to_remove = [
            r'Share this article', r'Follow us on', r'Subscribe to', r'BBC Homepage', 
            r'Skip to content', r'Accessibility Help', r'Cookie Policy', r'Privacy Policy',
            r'Terms of Service', r'Contact Us', r'About Us', r'Home', r'News', r'Sports',
            r'Entertainment', r'Business', r'Technology', r'Science', r'Health',
            r'Search', r'Login', r'Sign up', r'Subscribe', r'Newsletter',
            r'Follow', r'Share', r'Like', r'Comment', r'Related Articles',
            r'Recommended', r'Popular', r'Trending', r'Most Read', r'Latest',
            r'Previous', r'Next', r'Back to top', r'Return to top',
            r'Advertisement', r'Ad', r'Sponsored', r'Promoted',
            r'Menu', r'Navigation', r'Breadcrumb', r'Pagination',
            r'Footer', r'Header', r'Sidebar', r'Widget',
            # Hackaday-specific patterns
            r'Hackaday', r'Hack a Day', r'Hackaday\.com', r'Hackaday Blog',
            r'Submit a Tip', r'Submit Tip', r'Submit Your Tip',
            r'Recent Posts', r'Recent Articles', r'Latest Posts', r'Latest Articles',
            r'Popular Posts', r'Popular Articles', r'Featured Posts', r'Featured Articles',
            r'Related Posts', r'Related Articles', r'You might also like',
            r'Comments', r'Comment', r'Leave a comment', r'Post a comment',
            r'Tagged with', r'Tags', r'Categories', r'Category',
            r'Posted by', r'Author', r'Written by', r'By',
            r'Posted on', r'Published on', r'Date', r'Time',
            r'Read more', r'Continue reading', r'Full article',
            r'Subscribe to Hackaday', r'Follow Hackaday', r'Hackaday Newsletter',
            r'RSS Feed', r'RSS', r'Atom Feed', r'Atom',
            r'Twitter', r'Facebook', r'Reddit', r'YouTube', r'Instagram',
            r'Email', r'Contact', r'About', r'Privacy', r'Terms'
        ]
        
        for artifact in artifacts_to_remove:
            content = re.sub(artifact, '', content, flags=re.IGNORECASE)
        
        # Remove common website chrome patterns
        content = re.sub(r'^\s*(Home|News|Sports|Entertainment|Business|Technology|Science|Health)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        content = re.sub(r'^\s*(Search|Login|Sign up|Subscribe|Follow|Share)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove Hackaday-specific patterns
        content = re.sub(r'^\s*(Hackaday|Hack a Day|Submit a Tip|Recent Posts|Popular Posts|Related Posts)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        content = re.sub(r'^\s*(Comments|Comment|Leave a comment|Posted by|Posted on|Tagged with)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove common sidebar/menu patterns
        content = re.sub(r'^\s*(Sidebar|Widget|Column|Panel|Menu|Navigation)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up any remaining excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Limit content length to prevent database issues
        if len(content) > 50000:
            content = content[:50000] + "..."
        
        return content
    
    async def _download_file_directly(self, url: str, file_type: str) -> Tuple[Union[bytes, str], int]:
        """Download file directly using httpx for binary/text files"""
        try:
            import httpx
            
            # Enhanced headers for better compatibility
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                "DNT": "1"
            }
            
            # Fetch URL content with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True, max_redirects=10) as client:
                        response = await client.get(url, headers=headers)
                        response.raise_for_status()
                        break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in [403, 429, 503] and attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise
                except httpx.RequestError as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise
            
            # Handle binary vs text content
            if file_type in ['pdf', 'docx', 'epub']:
                content = response.content  # Binary content
                content_size = len(content)
            else:
                content = response.text  # Text content
                content_size = len(content.encode('utf-8'))
            
            logger.info(f"📥 Direct download: {content_size} bytes from {url} (type: {file_type})")
            return content, content_size
            
        except Exception as e:
            logger.error(f"❌ Direct download failed for {url}: {e}")
            raise
    
    async def _store_original_html(self, document_id: str, original_html: str, images: Optional[List[Dict[str, Any]]] = None):
        """Store original HTML content and images for display"""
        try:
            # Store in a separate table or as metadata
            # For now, we'll store it as document metadata
            metadata = {
                "original_html": original_html,
                "has_original_layout": True,
                "images": images or []
            }
            
            # Update document with HTML metadata
            await self.document_repository.update_metadata(document_id, {
                "metadata": metadata
            })
            
            logger.info(f"💾 Stored original HTML for document {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store original HTML for document {document_id}: {e}")
            # Don't fail the entire process if HTML storage fails
    
    async def list_documents(self, skip: int = 0, limit: int = 100) -> List[DocumentInfo]:
        """List all documents"""
        return await self.document_repository.list_documents(skip, limit)
    
    async def filter_documents(self, filter_request: DocumentFilterRequest) -> DocumentListResponse:
        """Filter and search documents with advanced criteria"""
        try:
            logger.info(f"🔍 Filtering documents with criteria: {filter_request.dict()}")
            
            # Use repository to filter documents
            documents, total_count = await self.document_repository.filter_documents(filter_request)
            
            # Generate category and tag counts for the filtered results
            categories = {}
            tags = {}
            
            for doc in documents:
                if doc.category:
                    categories[doc.category.value] = categories.get(doc.category.value, 0) + 1
                for tag in doc.tags:
                    tags[tag] = tags.get(tag, 0) + 1
            
            return DocumentListResponse(
                documents=documents,
                total=total_count,
                categories=categories,
                tags=tags,
                filters_applied=filter_request.dict(exclude_none=True)
            )
            
        except Exception as e:
            logger.error(f"❌ Document filtering failed: {e}")
            return DocumentListResponse(documents=[], total=0)
    
    async def update_document_metadata(self, document_id: str, update_request: DocumentUpdateRequest) -> bool:
        """Update document metadata in both PostgreSQL and Qdrant vector chunks"""
        try:
            # **ROOSEVELT'S SMART RE-EXTRACTION**: Get old metadata BEFORE update to detect domain changes
            old_doc_info = await self.document_repository.get_by_id(document_id)
            old_tags = old_doc_info.tags if old_doc_info and old_doc_info.tags else []
            old_category = old_doc_info.category.value if old_doc_info and old_doc_info.category else None
            
            # Update PostgreSQL metadata
            success = await self.document_repository.update_metadata(document_id, update_request)
            if not success:
                logger.warning(f"⚠️ Document {document_id} not found for metadata update")
                return False
            
            logger.info(f"✅ Updated PostgreSQL metadata for document {document_id}")
            
            # **ROOSEVELT'S VECTOR METADATA SYNC**: Update Qdrant vector chunks with new metadata
            # This ensures tag/category filters work immediately without reprocessing!
            try:
                # Get the updated document info
                doc_info = await self.document_repository.get_by_id(document_id)
                if doc_info:
                    # Update Qdrant payloads with new metadata
                    await self._update_qdrant_metadata(
                        document_id=document_id,
                        document_category=doc_info.category.value if doc_info.category else None,
                        document_tags=doc_info.tags if doc_info.tags else None,
                        document_title=doc_info.title,
                        document_author=doc_info.author,
                        document_filename=doc_info.filename,
                        user_id=getattr(doc_info, 'user_id', None),
                        team_id=getattr(doc_info, 'team_id', None)
                    )
                    logger.info(f"✅ Updated Qdrant metadata for document {document_id}")
                else:
                    logger.warning(f"⚠️ Could not fetch updated document info for {document_id}")
            except Exception as qdrant_error:
                logger.error(f"⚠️ Failed to update Qdrant metadata: {qdrant_error}")
                # Don't fail the whole operation - PostgreSQL is updated
            
            # **ROOSEVELT'S SMART DOMAIN RE-EXTRACTION!** 🎬📊
            # Check if domain membership changed and re-extract KG if needed
            try:
                from services.domain_detector import get_domain_detector
                domain_detector = get_domain_detector()
                
                new_tags = update_request.tags if update_request.tags is not None else old_tags
                new_category = update_request.category.value if update_request.category else old_category
                
                domain_changes = domain_detector.get_domain_changes(
                    old_tags, old_category,
                    new_tags, new_category
                )
                
                if domain_changes["changed"]:
                    logger.info(f"📊 Domain change detected for {document_id}:")
                    logger.info(f"   Added domains: {domain_changes['added']}")
                    logger.info(f"   Removed domains: {domain_changes['removed']}")
                    
                    # Re-extract knowledge graph for affected domains
                    await self._reextract_knowledge_graph(document_id, domain_changes)
                
            except Exception as kg_error:
                logger.error(f"⚠️ Failed to re-extract knowledge graph for {document_id}: {kg_error}")
                # Don't fail the whole operation - metadata is already updated
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to update document metadata: {e}")
            return False
    
    async def bulk_categorize_documents(self, bulk_request: BulkCategorizeRequest) -> BulkOperationResponse:
        """Bulk categorize multiple documents"""
        try:
            success_count, failed_documents = await self.document_repository.bulk_categorize(bulk_request)
            
            message = f"Successfully categorized {success_count} documents"
            if failed_documents:
                message += f", {len(failed_documents)} documents not found"
            
            logger.info(f"📋 {message}")
            
            return BulkOperationResponse(
                success_count=success_count,
                failed_count=len(failed_documents),
                failed_documents=failed_documents,
                message=message
            )
            
        except Exception as e:
            logger.error(f"❌ Bulk categorization failed: {e}")
            return BulkOperationResponse(
                success_count=0,
                failed_count=len(bulk_request.document_ids),
                failed_documents=bulk_request.document_ids,
                message=f"Bulk categorization failed: {str(e)}"
            )
    
    async def get_document_categories_overview(self) -> DocumentCategoriesResponse:
        """Get overview of document categories and tags"""
        try:
            return await self.document_repository.get_categories_overview()
            
        except Exception as e:
            logger.error(f"❌ Failed to get categories overview: {e}")
            return DocumentCategoriesResponse(
                categories=[],
                tags=[],
                total_documents=0,
                uncategorized_count=0
            )
    
    async def get_document_status(self, doc_id: str) -> Optional[DocumentStatus]:
        """Get document processing status"""
        doc_info = await self.document_repository.get_by_id(doc_id)
        if not doc_info:
            return None
        
        # Calculate progress based on status
        progress_map = {
            ProcessingStatus.UPLOADING: 10.0,
            ProcessingStatus.PROCESSING: 50.0,
            ProcessingStatus.EMBEDDING: 80.0,
            ProcessingStatus.COMPLETED: 100.0,
            ProcessingStatus.FAILED: 0.0
        }
        
        return DocumentStatus(
            document_id=doc_id,
            status=doc_info.status,
            progress=progress_map.get(doc_info.status, 0.0),
            message=f"Document is {doc_info.status.value}",
            quality_metrics=doc_info.quality_metrics,
            chunks_processed=0,  # TODO: implement actual tracking
            entities_extracted=0  # TODO: implement actual tracking
        )
    
    async def check_qdrant_health(self) -> bool:
        """Check Qdrant health"""
        try:
            self.qdrant_client.get_collections()
            return True
        except Exception as e:
            logger.error(f"❌ Qdrant health check failed: {e}")
            return False
    
    async def _update_qdrant_metadata(
        self,
        document_id: str,
        document_category: Optional[str] = None,
        document_tags: Optional[List[str]] = None,
        document_title: Optional[str] = None,
        document_author: Optional[str] = None,
        document_filename: Optional[str] = None,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None
    ):
        """
        Update metadata in Qdrant vector payloads for all chunks of a document.
        This ensures tag/category filters work without reprocessing!
        
        Collection determination priority: team_id > user_id > global
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, SetPayload
            from services.vector_store_service import VectorStoreService
            
            vector_store = VectorStoreService()
            
            # Determine which collection to update (priority: team > user > global)
            if team_id:
                collection_name = vector_store._get_team_collection_name(team_id)
            elif user_id:
                collection_name = vector_store._get_user_collection_name(user_id)
            else:
                collection_name = settings.VECTOR_COLLECTION_NAME
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if collection_name not in collection_names:
                logger.warning(f"⚠️ Collection {collection_name} doesn't exist for metadata update")
                return
            
            # Build the metadata payload to update
            payload_updates = {}
            if document_category is not None:
                payload_updates["document_category"] = document_category
            if document_tags is not None:
                payload_updates["document_tags"] = document_tags
            if document_title is not None:
                payload_updates["document_title"] = document_title
            if document_author is not None:
                payload_updates["document_author"] = document_author
            if document_filename is not None:
                payload_updates["document_filename"] = document_filename
            
            if not payload_updates:
                logger.info(f"⚠️ No metadata to update for document {document_id}")
                return
            
            # Use Qdrant's set_payload to update all points matching document_id
            # This updates the payload without re-vectorizing!
            self.qdrant_client.set_payload(
                collection_name=collection_name,
                payload=payload_updates,
                points=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            
            logger.info(f"✅ Updated Qdrant payloads for document {document_id} in {collection_name}")
            logger.info(f"   Updated fields: {list(payload_updates.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update Qdrant metadata for {document_id}: {e}")
            raise
    
    def _detect_document_type(self, filename: str) -> str:
        """Detect document type from filename"""
        extension = Path(filename).suffix.lower()
        
        type_map = {
            '.pdf': 'pdf',
            '.txt': 'txt',
            '.md': 'md',
            '.org': 'org',
            '.docx': 'docx',
            '.doc': 'docx',
            '.epub': 'epub',
            '.html': 'html',
            '.htm': 'html',
            '.eml': 'eml',
            '.zip': 'zip',
            '.srt': 'srt',
            '.mp4': 'mp4',
            '.mkv': 'mkv',
            '.avi': 'avi',
            '.mov': 'mov',
            '.webm': 'webm'
        }
        
        return type_map.get(extension, 'txt')
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its embeddings, even if file is missing"""
        try:
            logger.info(f"🗑️  Starting deletion of document: {document_id}")
            
            # Check if document exists in database
            doc_info = await self.document_repository.get_by_id(document_id)
            if not doc_info:
                logger.warning(f"⚠️  Document {document_id} not found in database")
                return False
            
            logger.info(f"📄 Document found: {doc_info.filename} (Status: {doc_info.status})")
            
            # Always delete embeddings from vector database (even if file is missing)
            try:
                await self.embedding_manager.delete_document_chunks(document_id)
                logger.info(f"🗑️  Deleted embeddings for document {document_id}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to delete embeddings for {document_id}: {e}")
                # Continue with deletion even if embeddings fail
            
            # Try to delete original file using folder service
            from services.service_container import get_service_container
            files_deleted = 0
            
            try:
                # Get the proper file path using folder service
                container = await get_service_container()
                folder_service = container.folder_service
                
                file_path = await folder_service.get_document_file_path(
                    filename=doc_info.filename,
                    folder_id=getattr(doc_info, 'folder_id', None),
                    user_id=getattr(doc_info, 'user_id', None),
                    collection_type=getattr(doc_info, 'collection_type', 'user')
                )
                
                if file_path and file_path.exists():
                    file_path.unlink()
                    files_deleted += 1
                    logger.info(f"🗑️  Deleted file: {file_path}")
                else:
                    logger.warning(f"⚠️  File not found at expected path: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to delete file using folder service: {e}")
                
                # Fallback: try old document_id_ pattern for legacy files
                upload_dir = Path(settings.UPLOAD_DIR)
                for file_path in upload_dir.glob(f"{document_id}_*"):
                    try:
                        file_path.unlink()
                        files_deleted += 1
                        logger.info(f"🗑️  Deleted legacy file: {file_path}")
                    except Exception as del_e:
                        logger.warning(f"⚠️  Failed to delete legacy file {file_path}: {del_e}")
            
            # Log file deletion status
            if files_deleted == 0:
                logger.warning(f"⚠️  No files deleted for document {document_id} - file may not exist on disk")
            else:
                logger.info(f"✅ Deleted {files_deleted} file(s) for document {document_id}")
            
            # Always remove from database (even if file deletion failed)
            try:
                # Pass user_id for proper RLS context
                deleted = await self.document_repository.delete(document_id, doc_info.user_id)
                if deleted:
                    logger.info(f"🗑️  Removed document {document_id} from database")
                else:
                    logger.warning(f"⚠️  Document {document_id} not found in database (may have been already deleted)")
                    # Document not found is still a successful outcome for delete (idempotent)
            except Exception as e:
                logger.error(f"❌ Failed to remove document {document_id} from database: {e}")
                return False
            
            # Delete from knowledge graph if available
            if self.kg_service:
                try:
                    await self.kg_service.delete_document_entities(document_id)
                    logger.info(f"🗑️  Deleted knowledge graph entities for document {document_id}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete knowledge graph entities for {document_id}: {e}")
                    # Continue - this is not critical for deletion success
            
            logger.info(f"✅ Document {document_id} ({doc_info.filename}) deleted successfully - all data cleaned up")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document {document_id}: {e}")
            return False
    
    async def exempt_document_from_vectorization(self, document_id: str, user_id: str = None) -> bool:
        """Exempt document from vectorization and delete existing vectors/entities"""
        try:
            logger.info(f"🚫 Exempting document {document_id} from vectorization")
            
            # Mark document as exempt
            success = await self.document_repository.update_document_exemption_status(document_id, True, user_id)
            if not success:
                logger.error(f"Failed to update exemption status for document {document_id}")
                return False
            
            # Delete existing vectors
            try:
                await self.embedding_manager.delete_document_chunks(document_id, user_id)
                logger.info(f"Deleted vectors for exempted document {document_id}")
            except Exception as e:
                logger.warning(f"Failed to delete vectors for {document_id}: {e}")
            
            # Delete existing knowledge graph entities
            if self.kg_service:
                try:
                    await self.kg_service.delete_document_entities(document_id)
                    logger.info(f"Deleted knowledge graph entities for exempted document {document_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete KG entities for {document_id}: {e}")
            
            # Update status to completed since document is exempt
            # The update_status method will handle RLS context automatically
            await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED, user_id)
            
            # Emit WebSocket notification
            await self._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            
            logger.info(f"✅ Document {document_id} exempted from vectorization")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to exempt document {document_id}: {e}")
            return False
    
    async def remove_document_exemption(self, document_id: str, user_id: str = None, inherit: bool = False) -> bool:
        """
        Remove exemption and immediately re-process document.
        
        Args:
            document_id: Document ID
            user_id: User ID
            inherit: If True, set to inherit from folder (NULL). If False, set to explicit vectorize (FALSE).
        """
        try:
            if inherit:
                logger.info(f"✅ Setting document {document_id} to inherit from folder")
                exempt_status = None
            else:
                logger.info(f"✅ Removing exemption for document {document_id} - re-processing (explicit vectorize)")
                exempt_status = False
            
            # Update document exemption status
            success = await self.document_repository.update_document_exemption_status(document_id, exempt_status, user_id)
            if not success:
                logger.error(f"Failed to update exemption status for document {document_id}")
                return False
            
            # If setting to inherit, we're done - no need to re-process
            if inherit:
                logger.info(f"✅ Document {document_id} now inherits from folder")
                return True
            
            # Get document info
            doc_info = await self.document_repository.get_by_id(document_id)
            if not doc_info:
                logger.error(f"Document {document_id} not found")
                return False
            
            # Get file path
            from services.service_container import get_service_container
            container = await get_service_container()
            folder_service = container.folder_service
            
            file_path = await folder_service.get_document_file_path(
                filename=doc_info.filename,
                folder_id=doc_info.folder_id,
                user_id=doc_info.user_id,
                collection_type=doc_info.collection_type
            )
            
            if not file_path or not file_path.exists():
                logger.warning(f"File not found for document {document_id}, cannot re-process")
                return False
            
            # Re-process document
            await self.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
            
            # Process document
            result = await self.document_processor.process_document(
                str(file_path), 
                doc_info.doc_type.value, 
                document_id
            )
            
            # Generate and store embeddings
            if result.chunks:
                document_category = doc_info.category.value if doc_info.category else None
                document_tags = doc_info.tags if doc_info.tags else None
                document_title = doc_info.title if doc_info.title else None
                document_author = doc_info.author if doc_info.author else None
                document_filename = doc_info.filename if doc_info.filename else None
                
                await self.embedding_manager.embed_and_store_chunks(
                    result.chunks,
                    user_id=user_id or doc_info.user_id,
                    document_category=document_category,
                    document_tags=document_tags,
                    document_title=document_title,
                    document_author=document_author,
                    document_filename=document_filename
                )
                logger.info(f"Re-embedded {len(result.chunks)} chunks for document {document_id}")
            
            # Extract and store entities
            if self.kg_service and result.chunks:
                try:
                    entities = await self.document_processor._extract_entities(
                        result.text_content or "",
                        result.chunks or []
                    )
                    if entities:
                        await self.kg_service.store_entities(entities, document_id)
                        logger.info(f"Extracted and stored {len(entities)} entities for document {document_id}")
                except Exception as e:
                    logger.warning(f"Failed to extract/store entities for {document_id}: {e}")
            
            # Update status
            await self.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            await self.document_repository.update_chunk_count(document_id, len(result.chunks) if result.chunks else 0)
            
            # Emit WebSocket notification
            await self._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            
            logger.info(f"✅ Document {document_id} exemption removed and re-processed")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove exemption for document {document_id}: {e}")
            return False
    
    async def get_documents_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            stats = await self.document_repository.get_stats()
            
            # Get vector database stats
            vector_stats = await self.embedding_manager.get_collection_stats()
            
            return {
                **stats,
                "total_embeddings": vector_stats.get("total_points", 0),
                "vector_dimensions": vector_stats.get("vector_size", 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get documents stats: {e}")
            return {}
    
    async def cleanup_orphaned_embeddings(self) -> int:
        """Clean up embeddings for documents that no longer exist"""
        try:
            logger.info("🧹 Starting cleanup of orphaned embeddings...")
            
            # Get all document IDs from database
            registered_doc_ids = await self.document_repository.get_all_document_ids()
            registered_doc_ids = set(registered_doc_ids)
            
            # Get all document IDs from vector database
            vector_doc_ids = set()
            scroll_result = self.qdrant_client.scroll(
                collection_name=settings.VECTOR_COLLECTION_NAME,
                limit=10000,  # Adjust based on your needs
                with_payload=["document_id"]
            )
            
            for point in scroll_result[0]:
                doc_id = point.payload.get("document_id")
                if doc_id:
                    vector_doc_ids.add(doc_id)
            
            # Find orphaned embeddings
            orphaned_doc_ids = vector_doc_ids - registered_doc_ids
            
            if orphaned_doc_ids:
                logger.info(f"🧹 Found {len(orphaned_doc_ids)} orphaned document embeddings")
                
                # Delete orphaned embeddings
                for doc_id in orphaned_doc_ids:
                    await self.embedding_manager.delete_document_chunks(doc_id)
                    logger.info(f"🗑️  Cleaned up embeddings for orphaned document: {doc_id}")
                
                return len(orphaned_doc_ids)
            else:
                logger.info("✅ No orphaned embeddings found")
                return 0
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup orphaned embeddings: {e}")
            return 0
    
    def _calculate_file_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
    
    async def get_duplicate_documents(self) -> Dict[str, List[DocumentInfo]]:
        """Get all duplicate documents grouped by hash"""
        return await self.document_repository.get_duplicates()
    
    async def store_text_document(self, doc_id: str, content: str, metadata: Dict[str, Any], filename: str = None, user_id: Optional[str] = None, collection_type: str = "user", folder_id: Optional[str] = None, file_path: Optional[str] = None) -> bool:
        """Store text content directly as a document"""
        try:
            logger.info(f"📥 Storing text document: {doc_id}")
            
            # Generate filename if not provided
            if not filename:
                filename = f"{doc_id}.txt"
            
            # If we have a file_path, update the filename to match
            if file_path:
                from pathlib import Path
                filename = Path(file_path).name
                logger.info(f"📁 Using markdown filename: {filename}")
                
                # Determine document type from filename
                if filename.lower().endswith('.md'):
                    doc_type = DocumentType.MD
                elif filename.lower().endswith('.org'):
                    doc_type = DocumentType.ORG
                elif filename.lower().endswith('.html') or filename.lower().endswith('.htm'):
                    doc_type = DocumentType.HTML
                elif filename.lower().endswith('.txt'):
                    doc_type = DocumentType.TXT
                else:
                    # Fallback to extension detection
                    doc_type = DocumentType(self._detect_document_type(filename)) if hasattr(self, '_detect_document_type') else DocumentType.TXT
            else:
                # Determine document type from filename extension
                if filename.lower().endswith('.md'):
                    doc_type = DocumentType.MD
                elif filename.lower().endswith('.org'):
                    doc_type = DocumentType.ORG
                elif filename.lower().endswith('.html') or filename.lower().endswith('.htm'):
                    doc_type = DocumentType.HTML
                elif filename.lower().endswith('.txt'):
                    doc_type = DocumentType.TXT
                else:
                    doc_type = DocumentType(self._detect_document_type(filename)) if hasattr(self, '_detect_document_type') else DocumentType.TXT
            
            # Map web_search category to a valid enum value
            category_value = metadata.get("category", "other")
            if category_value == "web_search":
                category_value = "other"  # Map to valid enum value
            
            # Ensure category is a valid DocumentCategory
            try:
                category = DocumentCategory(category_value)
            except ValueError:
                logger.warning(f"Invalid category '{category_value}', using 'other'")
                category = DocumentCategory.OTHER
            
            # Automatically exempt .json files from vectorization
            exempt_from_vectorization = None
            if filename and filename.lower().endswith('.json'):
                exempt_from_vectorization = True
                logger.info(f"🚫 JSON file detected ({filename}) - automatically exempting from vectorization")
            
            # Create document info with proper fields
            document_info = DocumentInfo(
                document_id=doc_id,
                filename=filename,
                title=metadata.get("title", filename),
                doc_type=doc_type,
                category=category,
                tags=metadata.get("tags", []),
                description=metadata.get("description"),
                author=metadata.get("author"),
                language=metadata.get("language"),
                upload_date=datetime.utcnow(),
                file_size=len(content.encode('utf-8')),
                status=ProcessingStatus.PROCESSING,
                user_id=user_id,
                collection_type=collection_type,
                folder_id=folder_id,
                exempt_from_vectorization=exempt_from_vectorization
            )
            
            # Store original file to disk so content endpoint can load it
            # Skip if file_path is provided (file was already written by file manager)
            if not file_path:
                try:
                    from pathlib import Path
                    from config import settings
                    from services.service_container import get_service_container
                    
                    # Use folder service to get proper path
                    # No need for ID prefix - folder isolation provides uniqueness
                    container = await get_service_container()
                    folder_service = container.folder_service
                    
                    disk_path = await folder_service.get_document_file_path(
                        filename=filename or f"{doc_id}.txt",
                        folder_id=folder_id,
                        user_id=user_id,
                        collection_type=collection_type
                    )
                    
                    logger.info(f"📁 Saving editor file to: {disk_path}")
                    
                    with open(disk_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"📝 Wrote source file to disk: {disk_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to write file to disk for {doc_id}: {e}")
            else:
                logger.info(f"📝 File already written by caller: {file_path}")
            
            # Store in database - use create_with_folder to inherit exemption from parent folder
            await self.document_repository.create_with_folder(document_info, folder_id)
            
            # Check if document is exempt from vectorization BEFORE processing
            # This prevents unnecessary processing work for exempt documents
            is_exempt = await self.document_repository.is_document_exempt(doc_id, user_id)
            if is_exempt:
                logger.info(f"🚫 Document {doc_id} is exempt from vectorization - skipping all processing")
                await self.document_repository.update_status(doc_id, ProcessingStatus.COMPLETED)
                logger.info(f"✅ Successfully stored text document (exempt): {doc_id}")
                return True
            
            # Update status to embedding
            await self.document_repository.update_status(doc_id, ProcessingStatus.EMBEDDING)
            
            # Process content into chunks
            chunks = await self.document_processor.process_text_content(
                content, doc_id, metadata
            )
            
            # Store chunks in vector database (Note: These are direct text chunks, may not have category/tags)
            if chunks:
                await self.embedding_manager.embed_and_store_chunks(
                    chunks,
                    user_id=user_id,  # Use provided user_id instead of None
                    document_category=None,  # No category for direct text
                    document_tags=None  # No tags for direct text
                )
                logger.info(f"📊 Stored {len(chunks)} chunks for document {doc_id}")
            
            # Extract and store entities if knowledge graph is available
            if self.kg_service:
                try:
                    entities = await self.kg_service.extract_entities_from_text(content)
                    if entities:
                        await self.kg_service.store_entities(entities, doc_id)
                        logger.info(f"🔗 Stored {len(entities)} entities for document {doc_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to extract entities for {doc_id}: {e}")
            
            # Update final status to completed
            await self.document_repository.update_status(doc_id, ProcessingStatus.COMPLETED)
            
            logger.info(f"✅ Successfully stored text document: {doc_id} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store text document {doc_id}: {e}")
            # Update status to failed if document was created
            try:
                await self.document_repository.update_status(doc_id, ProcessingStatus.FAILED)
            except:
                pass  # Ignore errors when updating status
            return False
    
    async def _reextract_knowledge_graph(self, document_id: str, domain_changes: Dict[str, Any]) -> None:
        """
        Re-extract knowledge graph for a document when domains change
        
        **BULLY!** Smart re-extraction WITHOUT re-chunking/re-embedding!
        
        This method:
        1. Removes old domain-specific entities (for removed domains)
        2. Extracts and stores new domain-specific entities (for added domains)
        3. Does NOT re-chunk or re-embed the document
        """
        if not self.kg_service:
            logger.warning(f"⚠️ KG service not available for re-extraction")
            return
        
        from services.domain_detector import get_domain_detector
        from pathlib import Path
        domain_detector = get_domain_detector()
        
        # Get document info and content
        doc_info = await self.document_repository.get_by_id(document_id)
        if not doc_info:
            logger.warning(f"⚠️ Document {document_id} not found for KG re-extraction")
            return
        
        # Get document content from disk
        filename = getattr(doc_info, 'filename', None) or ""
        if not filename:
            logger.warning(f"⚠️ No filename for document {document_id}")
            return
        
        user_id = getattr(doc_info, 'user_id', None)
        folder_id = getattr(doc_info, 'folder_id', None)
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        # Get file path
        file_path = None
        try:
            from services.service_container import service_container
            folder_service = service_container.folder_service
            
            file_path_str = await folder_service.get_document_file_path(
                filename=filename,
                folder_id=folder_id,
                user_id=user_id,
                collection_type=collection_type
            )
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                logger.warning(f"⚠️ File not found at: {file_path}")
                return
        except Exception as e:
            logger.error(f"⚠️ Failed to get file path for {document_id}: {e}")
            return
        
        # Read content from disk
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content:
                logger.warning(f"⚠️ Empty content for document {document_id}")
                return
        except Exception as e:
            logger.error(f"⚠️ Failed to read file content for {document_id}: {e}")
            return
        
        # Handle removed domains - delete their entities
        for removed_domain in domain_changes.get("removed", set()):
            try:
                logger.info(f"🗑️ Removing {removed_domain} entities for {document_id}")
                # Domain-specific deletion would go here
                # For now, we'll rely on the full document deletion and re-extraction
                pass
            except Exception as e:
                logger.error(f"❌ Failed to remove {removed_domain} entities: {e}")
        
        # Handle added domains - extract and store new entities
        for added_domain in domain_changes.get("added", set()):
            try:
                logger.info(f"🎬 Extracting {added_domain} entities for {document_id}")
                
                extractor = domain_detector.get_extractor_for_domain(added_domain)
                if not extractor:
                    logger.warning(f"⚠️ No extractor found for domain: {added_domain}")
                    continue
                
                # Extract domain-specific entities and relationships
                entities, relationships = extractor.extract_entities_and_relationships(
                    content, doc_info
                )
                
                if entities or relationships:
                    # Store using domain-specific storage method
                    if added_domain == "entertainment":
                        await self.kg_service.store_entertainment_entities_and_relationships(
                            entities, relationships, document_id
                        )
                        logger.info(f"✅ Stored {len(entities)} {added_domain} entities, {len(relationships)} relationships")
                    # Future domains can be added here:
                    # elif added_domain == "business":
                    #     await self.kg_service.store_business_entities_and_relationships(...)
                    # elif added_domain == "research":
                    #     await self.kg_service.store_research_entities_and_relationships(...)
                    else:
                        logger.warning(f"⚠️ No storage method for domain: {added_domain}")
                
            except Exception as e:
                logger.error(f"❌ Failed to extract {added_domain} entities: {e}")
        
        logger.info(f"✅ Completed KG re-extraction for {document_id}")
    
    async def check_document_exists(self, doc_id: str) -> bool:
        """Check if a document with the given ID exists"""
        try:
            document = await self.document_repository.get_by_id(doc_id)
            return document is not None
        except Exception as e:
            logger.warning(f"⚠️ Failed to check document existence: {e}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        """Get document by ID - compatibility method for chat service"""
        try:
            return await self.document_repository.get_by_id(document_id)
        except Exception as e:
            logger.error(f"❌ Failed to get document {document_id}: {e}")
            return None
    
    # Compatibility properties for existing code
    @property
    def documents_db(self):
        """Compatibility property - returns empty dict since we use database now"""
        logger.warning("documents_db property is deprecated - use repository methods instead")
        return {}
    
    @property
    def _documents_cache(self):
        """Compatibility property - returns empty dict since we use database now"""
        logger.warning("_documents_cache property is deprecated - use repository methods instead")
        return {}
    
    async def close(self):
        """Clean up resources"""
        if self.document_repository:
            await self.document_repository.close()
        
        if self.qdrant_client:
            self.qdrant_client.close()
        logger.info("🔄 Document Service V2 closed")
