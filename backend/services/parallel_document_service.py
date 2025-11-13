"""
Parallel Document Service - Enhanced document service with true parallel processing
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

import aiofiles
from fastapi import UploadFile

from config import settings
from models.api_models import (
    DocumentInfo, DocumentStatus, ProcessingStatus, DocumentType,
    DocumentUploadResponse, BulkUploadResponse
)
from repositories.document_repository import DocumentRepository
from utils.parallel_document_processor import (
    ParallelDocumentProcessor, ProcessingConfig, ProcessingStrategy
)
from services.document_service_v2 import DocumentService
from services.zip_processor_service import ZipProcessorService
from services.embedding_service_wrapper import get_embedding_service

logger = logging.getLogger(__name__)


class ParallelDocumentService(DocumentService):
    """Enhanced document service with parallel processing and ZIP support capabilities"""
    
    def __init__(self):
        super().__init__()
        self.parallel_processor = None
        self.parallel_embedding_manager = None
        self.zip_processor = None  # Add ZIP processor
        self.processing_jobs = {}  # Track active processing jobs
        self.embedding_jobs = {}   # Track active embedding jobs
    
    async def initialize(self, enable_parallel: bool = True, processing_config: ProcessingConfig = None, 
                        shared_document_repository=None, shared_embedding_manager=None, shared_kg_service=None):
        """Initialize the parallel document service with shared dependencies"""
        logger.info("🔧 Initializing Parallel Document Service...")
        
        # Initialize base service with shared dependencies
        await super().initialize(shared_document_repository, shared_embedding_manager, shared_kg_service)
        
        if enable_parallel:
            # Initialize parallel processors with custom configs
            self.parallel_processor = ParallelDocumentProcessor(processing_config)
            
            # Inject dependencies into parallel processor for status updates
            self.parallel_processor.document_repository = self.document_repository
            self.parallel_processor.embedding_manager = self.embedding_manager
            self.parallel_processor.kg_service = self.kg_service
            self.parallel_processor.websocket_manager = self.websocket_manager
            
            await self.parallel_processor.initialize()
            
            # ROOSEVELT FIX: Use the same shared embedding manager instead of creating another one!
            # This eliminates the duplicate parallel embedding manager
            self.parallel_embedding_manager = self.embedding_manager
            logger.info("✅ Using shared embedding manager for parallel processing - NO DUPLICATION!")
            
            logger.info("✅ Parallel processing enabled with dependency injection")
            
            # Resume processing of incomplete documents in parallel
            await self._resume_incomplete_processing()
        else:
            logger.info("⚠️ Parallel processing disabled - using sequential processing")
            
            # Resume processing sequentially if parallel is disabled
            await self._resume_incomplete_processing_sequential()
        
        # Initialize ZIP processor for ZIP file support
        self.zip_processor = ZipProcessorService(self.document_repository, self.websocket_manager)
        await self.zip_processor.initialize()
        
        logger.info("✅ Parallel Document Service initialized with ZIP support")
    
    async def upload_and_process(self, file: UploadFile, doc_type: str = None, user_id: str = None, folder_id: str = None) -> DocumentUploadResponse:
        """Enhanced upload and process with ZIP support and user isolation"""
        start_time = time.time()
        
        try:
            # Determine document type
            if not doc_type:
                doc_type = self._detect_document_type(file.filename)
            
            # Handle ZIP files specially
            if doc_type == 'zip':
                logger.info(f"📦 Processing ZIP file: {file.filename}")
                return await self._process_zip_upload(file, user_id)
            else:
                # Use standard processing for non-ZIP files with user_id
                logger.info(f"🔧 DEBUG: ParallelDocumentService calling parent upload_and_process for {file.filename}, doc_type: {doc_type}, user_id: {user_id}, folder_id: {folder_id}")
                result = await super().upload_and_process(file, doc_type, user_id, folder_id)
                logger.info(f"🔧 DEBUG: ParallelDocumentService parent upload_and_process returned: {result.document_id}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Enhanced upload failed: {e}")
            raise
    
    async def _process_zip_upload(self, file: UploadFile, user_id: str) -> DocumentUploadResponse:
        """Process ZIP file with individual file extraction"""
        try:
            # Check for duplicate ZIP file first
            content = await file.read()
            file_hash = self._calculate_file_hash(content)
            
            duplicate_doc = await self.document_repository.find_by_hash(file_hash)
            if duplicate_doc:
                logger.info(f"🔄 Duplicate ZIP detected: {file.filename} matches {duplicate_doc.filename}")
                return DocumentUploadResponse(
                    document_id=duplicate_doc.document_id,
                    filename=file.filename,
                    status=duplicate_doc.status,
                    message=f"Duplicate ZIP detected. Existing document: {duplicate_doc.filename}"
                )
            
            # Reset file pointer for ZIP processor
            await file.seek(0)
            
            # Process ZIP with individual file extraction
            metadata = {"user_id": user_id} if user_id else None
            zip_result = await self.zip_processor.process_zip_file(file, metadata)
            
            if not zip_result.parent_document_id:
                raise Exception("Failed to create parent ZIP record")
            
            # Create response based on processing results
            status = ProcessingStatus.PROCESSING
            if zip_result.failed_files == len(zip_result.extracted_files):
                status = ProcessingStatus.FAILED
            elif zip_result.successful_files == 0 and zip_result.duplicate_files > 0:
                status = ProcessingStatus.COMPLETED  # All duplicates
            
            message = zip_result.message
            if zip_result.duplicate_files > 0:
                message += f" ({zip_result.duplicate_files} duplicates found)"
            
            logger.info(f"✅ ZIP upload completed: {file.filename} ({zip_result.parent_document_id})")
            
            return DocumentUploadResponse(
                document_id=zip_result.parent_document_id,
                filename=file.filename,
                status=status,
                message=message
            )
            
        except Exception as e:
            logger.error(f"❌ ZIP upload failed: {e}")
            raise

    async def upload_multiple_documents(self, files: List[UploadFile], enable_parallel: bool = True) -> BulkUploadResponse:
        """Upload and process multiple documents with parallel processing"""
        start_time = time.time()
        
        try:
            logger.info(f"📄 Starting bulk upload of {len(files)} documents (parallel: {enable_parallel})")
            
            if enable_parallel and self.parallel_processor:
                return await self._upload_multiple_parallel(files)
            else:
                return await self._upload_multiple_sequential(files)
                
        except Exception as e:
            logger.error(f"❌ Bulk upload failed: {e}")
            return BulkUploadResponse(
                total_files=len(files),
                successful_uploads=0,
                failed_uploads=len(files),
                upload_results=[],
                processing_time=time.time() - start_time,
                message=f"Bulk upload failed: {str(e)}"
            )
    
    async def _upload_multiple_parallel(self, files: List[UploadFile]) -> BulkUploadResponse:
        """Upload multiple documents with true parallel processing"""
        start_time = time.time()
        upload_results = []
        successful_uploads = 0
        failed_uploads = 0
        
        # Process uploads concurrently
        upload_semaphore = asyncio.Semaphore(8)  # Limit concurrent uploads
        
        async def process_single_upload(file: UploadFile):
            async with upload_semaphore:
                try:
                    result = await self._upload_single_for_parallel(file)
                    return result
                except Exception as e:
                    logger.error(f"❌ Failed to upload {file.filename}: {e}")
                    return DocumentUploadResponse(
                        document_id="",
                        filename=file.filename,
                        status=ProcessingStatus.FAILED,
                        message=f"Upload failed: {str(e)}"
                    )
        
        # Process all uploads concurrently
        upload_tasks = [process_single_upload(file) for file in files]
        upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        
        # Process results
        processing_jobs = []
        for result in upload_results:
            if isinstance(result, Exception):
                failed_uploads += 1
                upload_results.append(DocumentUploadResponse(
                    document_id="",
                    filename="unknown",
                    status=ProcessingStatus.FAILED,
                    message=f"Upload exception: {str(result)}"
                ))
            elif result.status == ProcessingStatus.FAILED:
                failed_uploads += 1
            else:
                successful_uploads += 1
                # Submit for parallel processing if we have a valid document ID
                if result.document_id and result.document_id in self.processing_jobs:
                    processing_jobs.append(self.processing_jobs[result.document_id])
        
        # Submit all processing jobs to parallel processor
        if processing_jobs:
            logger.info(f"🔄 Submitting {len(processing_jobs)} documents for parallel processing")
            
            for job_info in processing_jobs:
                await self.parallel_processor.submit_document(
                    job_info['document_id'],
                    job_info['file_path'],
                    job_info['doc_type']
                )
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Bulk upload completed: {successful_uploads} successful, {failed_uploads} failed in {processing_time:.2f}s")
        
        return BulkUploadResponse(
            total_files=len(files),
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            upload_results=upload_results,
            processing_time=processing_time,
            message=f"Uploaded {successful_uploads}/{len(files)} files successfully with parallel processing"
        )
    
    async def _upload_multiple_sequential(self, files: List[UploadFile]) -> BulkUploadResponse:
        """Upload multiple documents sequentially (fallback)"""
        start_time = time.time()
        upload_results = []
        successful_uploads = 0
        failed_uploads = 0
        
        for file in files:
            try:
                result = await self.upload_and_process(file)
                upload_results.append(result)
                
                if result.status == ProcessingStatus.FAILED:
                    failed_uploads += 1
                else:
                    successful_uploads += 1
                    
            except Exception as e:
                logger.error(f"❌ Failed to upload {file.filename}: {e}")
                failed_uploads += 1
                upload_results.append(DocumentUploadResponse(
                    document_id="",
                    filename=file.filename,
                    status=ProcessingStatus.FAILED,
                    message=f"Upload failed: {str(e)}"
                ))
        
        processing_time = time.time() - start_time
        
        return BulkUploadResponse(
            total_files=len(files),
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            upload_results=upload_results,
            processing_time=processing_time,
            message=f"Uploaded {successful_uploads}/{len(files)} files successfully with sequential processing"
        )
    
    async def _upload_single_for_parallel(self, file: UploadFile) -> DocumentUploadResponse:
        """Upload a single file and prepare for parallel processing"""
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
                    message=f"Duplicate file detected. Existing document: {duplicate_doc.filename}"
                )
            
            # Generate new document ID
            document_id = str(uuid4())
            
            # Create upload directory if it doesn't exist
            upload_dir = Path(settings.UPLOAD_DIR)
            upload_dir.mkdir(exist_ok=True)
            
            # Save uploaded file
            file_path = upload_dir / f"{document_id}_{file.filename}"
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Determine document type
            doc_type = self._detect_document_type(file.filename)
            
            # Create document record
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=file.filename,
                doc_type=DocumentType(doc_type),
                upload_date=datetime.utcnow(),
                file_size=len(content),
                file_hash=file_hash,
                status=ProcessingStatus.PROCESSING
            )
            
            # Save to database
            await self.document_repository.create(doc_info)
            
            # Store job info for parallel processing
            self.processing_jobs[document_id] = {
                'document_id': document_id,
                'file_path': str(file_path),
                'doc_type': doc_type
            }
            
            logger.info(f"📄 Document uploaded for parallel processing: {file.filename} ({document_id})")
            
            return DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                status=ProcessingStatus.PROCESSING,
                message="Document uploaded successfully, queued for parallel processing"
            )
            
        except Exception as e:
            logger.error(f"❌ Upload failed for {file.filename}: {e}")
            raise
    
    async def get_processing_status(self, document_id: str) -> Optional[DocumentStatus]:
        """Get enhanced processing status including parallel processing info"""
        try:
            # Get base status from parent class
            base_status = await super().get_document_status(document_id)
            if not base_status:
                return None
            
            # Enhance with parallel processing information
            if self.parallel_processor:
                parallel_status = await self.parallel_processor.get_processing_status(document_id)
                
                if parallel_status["status"] != "not_found":
                    # Update progress based on parallel processing status
                    if parallel_status["status"] == "processing":
                        base_status.progress = parallel_status.get("progress", 50.0)
                        base_status.message = f"Processing in parallel queue (position: {parallel_status.get('queue_position', 'unknown')})"
                    elif parallel_status["status"] == "completed":
                        base_status.progress = 100.0
                        base_status.message = "Parallel processing completed"
                    elif parallel_status["status"] == "failed":
                        base_status.progress = 0.0
                        base_status.message = f"Parallel processing failed: {parallel_status.get('error', 'Unknown error')}"
            
            # Check embedding status if available
            if self.parallel_embedding_manager and document_id in self.embedding_jobs:
                embedding_job_id = self.embedding_jobs[document_id]
                embedding_status = await self.parallel_embedding_manager.get_job_status(embedding_job_id)
                
                if embedding_status["status"] == "processing":
                    base_status.message += " | Embeddings: Processing"
                elif embedding_status["status"] == "completed":
                    base_status.message += " | Embeddings: Completed"
                elif embedding_status["status"] == "failed":
                    base_status.message += f" | Embeddings: Failed ({embedding_status.get('error', 'Unknown')})"
            
            return base_status
            
        except Exception as e:
            logger.error(f"❌ Failed to get processing status for {document_id}: {e}")
            return None
    
    async def get_parallel_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about parallel processing queues"""
        stats = {
            "parallel_processing_enabled": bool(self.parallel_processor),
            "parallel_embeddings_enabled": bool(self.parallel_embedding_manager)
        }
        
        if self.parallel_processor:
            processor_stats = await self.parallel_processor.get_queue_stats()
            stats["document_processing"] = processor_stats
        
        if self.parallel_embedding_manager:
            embedding_stats = await self.parallel_embedding_manager.get_queue_stats()
            stats["embedding_processing"] = embedding_stats
        
        return stats
    
    async def wait_for_document_completion(self, document_id: str, timeout: Optional[float] = None) -> bool:
        """Wait for a document to complete processing"""
        start_time = time.time()
        
        while True:
            status = await self.get_processing_status(document_id)
            
            if not status:
                return False
            
            if status.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                return status.status == ProcessingStatus.COMPLETED
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"⚠️ Timeout waiting for document {document_id} completion")
                return False
            
            # Wait before checking again
            await asyncio.sleep(1.0)
    
    async def wait_for_all_processing_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all queued processing to complete"""
        if not self.parallel_processor:
            return True
        
        try:
            # Wait for document processing queue
            doc_completion = await self.parallel_processor.wait_for_completion(timeout)
            
            # Wait for embedding processing queue if available
            embedding_completion = True
            if self.parallel_embedding_manager:
                # Wait for embedding queue to be empty
                start_time = time.time()
                while True:
                    stats = await self.parallel_embedding_manager.get_queue_stats()
                    if stats["embedding_queue_size"] == 0 and stats["storage_queue_size"] == 0:
                        break
                    
                    if timeout and (time.time() - start_time) > timeout:
                        embedding_completion = False
                        break
                    
                    await asyncio.sleep(1.0)
            
            return doc_completion and embedding_completion
            
        except Exception as e:
            logger.error(f"❌ Error waiting for processing completion: {e}")
            return False
    
    async def process_with_parallel_embeddings(self, document_id: str, chunks: List[Any], user_id: str = None) -> str:
        """Process chunks with parallel embedding generation"""
        # Fetch document metadata for filtering
        document_category = None
        document_tags = None
        document_title = None
        document_author = None
        document_filename = None
        try:
            doc_info = await self.document_repository.get_by_id(document_id)
            document_category = doc_info.category.value if doc_info and doc_info.category else None
            document_tags = doc_info.tags if doc_info else None
            document_title = doc_info.title if doc_info else None
            document_author = doc_info.author if doc_info else None
            document_filename = doc_info.filename if doc_info else None
        except Exception as e:
            logger.debug(f"Could not fetch document metadata: {e}")
        
        if not self.parallel_embedding_manager:
            # Fallback to sequential processing
            await self.embedding_manager.embed_and_store_chunks(
                chunks,
                user_id=user_id,
                document_category=document_category,
                document_tags=document_tags,
                document_title=document_title,
                document_author=document_author,
                document_filename=document_filename
            )
            return ""
        
        try:
            # Submit chunks for parallel embedding processing
            job_id = await self.parallel_embedding_manager.embed_and_store_chunks_parallel(chunks, document_id)
            
            # Track the embedding job
            self.embedding_jobs[document_id] = job_id
            
            logger.info(f"📊 Started parallel embedding job {job_id} for document {document_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start parallel embedding processing: {e}")
            # Fallback to sequential processing
            await self.embedding_manager.embed_and_store_chunks(
                chunks,
                user_id=user_id,
                document_category=document_category,
                document_tags=document_tags,
                document_title=document_title,
                document_author=document_author,
                document_filename=document_filename
            )
            return ""
    
    async def optimize_processing_configuration(self, document_count: int, average_file_size: int) -> ProcessingConfig:
        """Dynamically optimize processing configuration based on workload"""
        # Base configuration
        config = ProcessingConfig()
        
        # Adjust based on document count
        if document_count > 20:
            config.max_concurrent_documents = min(8, document_count // 3)
            config.strategy = ProcessingStrategy.HYBRID
        elif document_count > 10:
            config.max_concurrent_documents = 4
            config.strategy = ProcessingStrategy.ASYNC_CONCURRENT
        else:
            config.max_concurrent_documents = 2
            config.strategy = ProcessingStrategy.ASYNC_CONCURRENT
        
        # Adjust based on file size
        if average_file_size > 10 * 1024 * 1024:  # > 10MB
            config.thread_pool_size = 4
            config.enable_io_parallelism = True
        elif average_file_size > 1024 * 1024:  # > 1MB
            config.thread_pool_size = 6
        else:
            config.thread_pool_size = 8
        
        logger.info(f"🔧 Optimized processing config: {config.max_concurrent_documents} concurrent docs, {config.strategy.value} strategy")
        return config
    
    async def optimize_embedding_configuration(self, total_chunks: int) -> None:
        """
        DEPRECATED: Embedding configuration is now handled by EmbeddingServiceWrapper
        via USE_VECTOR_SERVICE flag in config.py
        
        This method is kept for backward compatibility but does nothing.
        """
        logger.info(f"📊 Embedding optimization requested for {total_chunks} chunks (handled by EmbeddingServiceWrapper)")
        return None
    
    async def _resume_incomplete_processing(self):
        """Resume processing of documents that were left in PROCESSING state (parallel)"""
        try:
            logger.info("🔄 Checking for incomplete documents to resume processing...")
            
            # Find documents in PROCESSING state
            incomplete_docs = await self.document_repository.get_documents_by_status(ProcessingStatus.PROCESSING)
            
            if not incomplete_docs:
                logger.info("✅ No incomplete documents found")
                return
            
            logger.info(f"🔄 Found {len(incomplete_docs)} incomplete documents, resuming with parallel processing...")
            
            # Process incomplete documents in parallel
            recovery_semaphore = asyncio.Semaphore(4)  # Limit concurrent recovery operations
            
            async def recover_single_document(doc_info: DocumentInfo):
                async with recovery_semaphore:
                    try:
                        # Find the original file
                        upload_dir = Path(settings.UPLOAD_DIR)
                        file_path = None
                        
                        # Look for the original file
                        for potential_file in upload_dir.glob(f"{doc_info.document_id}_*"):
                            file_path = potential_file
                            break
                        
                        if file_path and file_path.exists():
                            logger.info(f"🔄 Resuming processing for: {doc_info.filename} ({doc_info.document_id})")
                            
                            # Determine document type
                            doc_type = self._detect_document_type(doc_info.filename)
                            
                            # Submit to parallel processor with user_id
                            await self.parallel_processor.submit_document(
                                doc_info.document_id, 
                                str(file_path), 
                                doc_type,
                                user_id=getattr(doc_info, 'user_id', None)
                            )
                            
                            logger.info(f"✅ Queued for parallel processing: {doc_info.document_id}")
                            
                        elif doc_info.doc_type == DocumentType.URL:
                            logger.info(f"🔗 Resuming URL processing: {doc_info.filename} ({doc_info.document_id})")
                            
                            # Re-import from URL (maintain original collection)
                            asyncio.create_task(self._process_url_async(
                                doc_info.document_id, 
                                doc_info.filename, 
                                "html",
                                None  # Default to global collection for existing documents
                            ))
                            
                        else:
                            logger.warning(f"⚠️ File not found for document {doc_info.document_id}, marking as failed")
                            await self.document_repository.update_status(doc_info.document_id, ProcessingStatus.FAILED)
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to resume processing for {doc_info.document_id}: {e}")
                        await self.document_repository.update_status(doc_info.document_id, ProcessingStatus.FAILED)
            
            # Process all incomplete documents concurrently
            recovery_tasks = [recover_single_document(doc) for doc in incomplete_docs]
            await asyncio.gather(*recovery_tasks, return_exceptions=True)
            
            logger.info(f"✅ Startup recovery completed: {len(incomplete_docs)} documents queued for parallel processing")
            
        except Exception as e:
            logger.error(f"❌ Failed to resume incomplete processing: {e}")
    
    async def _resume_incomplete_processing_sequential(self):
        """Resume processing of documents that were left in PROCESSING state (sequential fallback)"""
        try:
            logger.info("🔄 Checking for incomplete documents to resume processing (sequential)...")
            
            # Find documents in PROCESSING state
            incomplete_docs = await self.document_repository.get_documents_by_status(ProcessingStatus.PROCESSING)
            
            if not incomplete_docs:
                logger.info("✅ No incomplete documents found")
                return
            
            logger.info(f"🔄 Found {len(incomplete_docs)} incomplete documents, resuming with sequential processing...")
            
            for doc_info in incomplete_docs:
                try:
                    # Find the original file
                    upload_dir = Path(settings.UPLOAD_DIR)
                    file_path = None
                    
                    # Look for the original file
                    for potential_file in upload_dir.glob(f"{doc_info.document_id}_*"):
                        file_path = potential_file
                        break
                    
                    if file_path and file_path.exists():
                        logger.info(f"🔄 Resuming processing for: {doc_info.filename} ({doc_info.document_id})")
                        
                        # Determine document type
                        doc_type = self._detect_document_type(doc_info.filename)
                        
                        # Start async processing with user_id
                        asyncio.create_task(super()._process_document_async(doc_info.document_id, file_path, doc_type, getattr(doc_info, 'user_id', None)))
                        
                    elif doc_info.doc_type == DocumentType.URL:
                        logger.info(f"🔗 Resuming URL processing: {doc_info.filename} ({doc_info.document_id})")
                        
                        # Re-import from URL (maintain original collection)
                        asyncio.create_task(self._process_url_async(
                            doc_info.document_id, 
                            doc_info.filename, 
                            "html",
                            None  # Default to global collection for existing documents
                        ))
                        
                    else:
                        logger.warning(f"⚠️ File not found for document {doc_info.document_id}, marking as failed")
                        await self.document_repository.update_status(doc_info.document_id, ProcessingStatus.FAILED)
                        
                except Exception as e:
                    logger.error(f"❌ Failed to resume processing for {doc_info.document_id}: {e}")
                    await self.document_repository.update_status(doc_info.document_id, ProcessingStatus.FAILED)
            
            logger.info(f"✅ Sequential startup recovery completed: {len(incomplete_docs)} documents queued for processing")
            
        except Exception as e:
            logger.error(f"❌ Failed to resume incomplete processing: {e}")

    async def _process_document_async(self, document_id: str, file_path: Path, doc_type: str, user_id: str = None):
        """Override to use parallel processing for reprocessing with user isolation support"""
        try:
            logger.info(f"🔄 Processing document with parallel processors: {document_id}")
            
            # Use parallel processor if available, otherwise fall back to base implementation
            if self.parallel_processor:
                logger.info(f"🚀 Using parallel document processor for reprocess: {document_id}")
                
                # Submit to parallel processor with user_id
                await self.parallel_processor.submit_document(document_id, str(file_path), doc_type, user_id=user_id)
                
                # Wait for completion with timeout
                success = await self.parallel_processor.wait_for_document_completion(document_id, timeout=300)
                
                if success:
                    logger.info(f"✅ Parallel reprocessing completed: {document_id}")
                else:
                    logger.error(f"❌ Parallel reprocessing failed or timed out: {document_id}")
                    await self.document_repository.update_status(document_id, ProcessingStatus.FAILED)
            else:
                logger.info(f"⚠️ Parallel processor not available, using standard processing: {document_id}")
                # Fall back to base implementation with user_id
                await super()._process_document_async(document_id, file_path, doc_type, user_id)
            
        except Exception as e:
            logger.error(f"❌ Parallel document processing failed for {document_id}: {e}")
            await self.document_repository.update_status(document_id, ProcessingStatus.FAILED)

    async def close(self):
        """Clean up resources"""
        logger.info("🔄 Shutting down Parallel Document Service...")
        
        # Close parallel processors
        if self.parallel_processor:
            await self.parallel_processor.close()
        
        if self.parallel_embedding_manager:
            await self.parallel_embedding_manager.close()
        
        # Close base service
        await super().close()
        
        logger.info("✅ Parallel Document Service shut down complete")

    # ===== ZIP HIERARCHY SUPPORT METHODS =====
    
    async def get_documents_with_hierarchy(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Get documents with hierarchy information for UI display"""
        try:
            documents = await self.document_repository.get_documents_with_hierarchy(limit, offset)
            
            # Group documents by parent for hierarchical display
            hierarchy = []
            current_parent = None
            current_group = None
            
            for doc in documents:
                if not doc['parent_document_id']:  # Top-level document
                    if current_group:
                        hierarchy.append(current_group)
                    
                    current_group = {
                        'parent': doc,
                        'children': []
                    }
                    current_parent = doc['document_id']
                elif doc['parent_document_id'] == current_parent:  # Child of current parent
                    if current_group:
                        current_group['children'].append(doc)
                else:  # Different parent, start new group
                    if current_group:
                        hierarchy.append(current_group)
                    
                    # This shouldn't happen with proper ordering, but handle gracefully
                    current_group = {
                        'parent': {'document_id': doc['parent_document_id'], 'filename': doc.get('parent_filename', 'Unknown')},
                        'children': [doc]
                    }
                    current_parent = doc['parent_document_id']
            
            # Add final group
            if current_group:
                hierarchy.append(current_group)
            
            return {
                "hierarchy": hierarchy,
                "total_documents": len(documents),
                "has_more": len(documents) == limit  # Simple pagination indicator
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get documents with hierarchy: {e}")
            return {"error": str(e)}

    async def get_zip_hierarchy(self, document_id: str) -> Dict[str, Any]:
        """Get ZIP file with its extracted children"""
        try:
            # Get parent ZIP info
            parent_doc = await self.document_repository.get_by_id(document_id)
            if not parent_doc:
                return {"error": "Document not found"}
            
            # Get children if this is a ZIP container
            children = []
            if hasattr(parent_doc, 'is_zip_container') and parent_doc.is_zip_container:
                children = await self.document_repository.get_zip_children(document_id)
            
            return {
                "parent": parent_doc.dict() if hasattr(parent_doc, 'dict') else parent_doc,
                "children": children,
                "total_children": len(children)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get ZIP hierarchy: {e}")
            return {"error": str(e)}

    async def delete_zip_with_children(self, parent_document_id: str, delete_children: bool = True) -> Dict[str, Any]:
        """Delete ZIP file and optionally its children"""
        try:
            if delete_children:
                # Delete entire hierarchy
                result = await self.document_repository.zip_extensions.delete_zip_hierarchy(parent_document_id)
                message = f"Deleted ZIP hierarchy: {result['total_deleted']} documents"
            else:
                # Delete only parent, children become orphaned (but keep their content)
                await self.document_repository.set_parent_relationship("", parent_document_id)  # Clear parent relationships
                success = await self.document_repository.delete_by_id(parent_document_id)
                result = {"deleted_parent": 1 if success else 0, "deleted_children": 0, "total_deleted": 1 if success else 0}
                message = "Deleted ZIP container, children preserved as independent documents"
            
            logger.info(f"🗑️ {message}")
            
            return {
                **result,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to delete ZIP: {e}")
            return {"error": str(e)}

    def _calculate_file_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        import hashlib
        return hashlib.sha256(content).hexdigest()
