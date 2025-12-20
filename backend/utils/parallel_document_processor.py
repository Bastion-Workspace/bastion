"""
Parallel Document Processor - Handles concurrent document processing with configurable limits
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from config import settings
from models.api_models import ProcessingResult, Chunk, Entity, QualityMetrics
from utils.document_processor import DocumentProcessor
import asyncio

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Different strategies for parallel processing"""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC_CONCURRENT = "async_concurrent"
    HYBRID = "hybrid"


@dataclass
class ProcessingJob:
    """Represents a document processing job"""
    document_id: str
    file_path: str
    doc_type: str
    priority: int = 0
    user_id: str = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing"""
    max_concurrent_documents: int = 12  # Increased from 4 to allow true parallel processing
    max_concurrent_chunks: int = 16     # Increased for better chunk processing
    max_concurrent_embeddings: int = 24 # Increased for better embedding throughput
    strategy: ProcessingStrategy = ProcessingStrategy.HYBRID
    chunk_batch_size: int = 50
    embedding_batch_size: int = 20
    enable_document_level_parallelism: bool = True
    enable_chunk_level_parallelism: bool = True
    enable_io_parallelism: bool = True
    thread_pool_size: int = 12          # Increased to match document concurrency
    process_pool_size: int = 4          # Increased for CPU-intensive tasks


class ParallelDocumentProcessor:
    """Enhanced document processor with true parallel processing capabilities"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.document_processor = None
        self.processing_queue = asyncio.Queue()
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.completed_jobs: Dict[str, ProcessingResult] = {}
        self.failed_jobs: Dict[str, Exception] = {}
        
        # Concurrency controls
        self.document_semaphore = asyncio.Semaphore(self.config.max_concurrent_documents)
        self.chunk_semaphore = asyncio.Semaphore(self.config.max_concurrent_chunks)
        self.embedding_semaphore = asyncio.Semaphore(self.config.max_concurrent_embeddings)
        
        # Thread/Process pools for CPU-intensive tasks
        self.thread_pool = None
        self.process_pool = None
        
        # Worker tasks
        self.workers = []
        self.is_running = False
        
        # WebSocket manager for real-time UI updates (injected)
        self.websocket_manager = None
        
        logger.info(f"🔧 Parallel Document Processor initialized with config: {self.config}")
    
    async def initialize(self):
        """Initialize the parallel processor"""
        logger.info("🔧 Initializing Parallel Document Processor...")
        
        # Initialize base document processor (use singleton)
        self.document_processor = DocumentProcessor.get_instance()
        await self.document_processor.initialize()
        
        # Initialize thread/process pools based on strategy
        if self.config.strategy in [ProcessingStrategy.THREAD_POOL, ProcessingStrategy.HYBRID]:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
            logger.info(f"✅ Thread pool initialized with {self.config.thread_pool_size} workers")
        
        if self.config.strategy in [ProcessingStrategy.PROCESS_POOL, ProcessingStrategy.HYBRID]:
            self.process_pool = ProcessPoolExecutor(max_workers=self.config.process_pool_size)
            logger.info(f"✅ Process pool initialized with {self.config.process_pool_size} workers")
        
        # Start worker tasks
        await self.start_workers()
        
        logger.info("✅ Parallel Document Processor initialized")
    
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
    
    async def start_workers(self):
        """Start background worker tasks"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start multiple worker tasks for document processing
        for i in range(self.config.max_concurrent_documents):
            worker_task = asyncio.create_task(self._document_worker(f"worker-{i}"))
            self.workers.append(worker_task)
        
        logger.info(f"✅ Started {len(self.workers)} document processing workers")
    
    async def stop_workers(self):
        """Stop all worker tasks"""
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        logger.info("🔄 All workers stopped")
    
    async def submit_document(self, document_id: str, file_path: str, doc_type: str, priority: int = 0, user_id: str = None) -> bool:
        """Submit a document for parallel processing"""
        try:
            job = ProcessingJob(
                document_id=document_id,
                file_path=file_path,
                doc_type=doc_type,
                priority=priority,
                user_id=user_id
            )
            
            # Add to simple FIFO queue
            await self.processing_queue.put(job)
            self.active_jobs[document_id] = job
            
            logger.info(f"📄 Submitted document for parallel processing: {document_id} (queue size: {self.processing_queue.qsize()})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to submit document {document_id}: {e}")
            return False
    
    async def get_processing_status(self, document_id: str) -> Dict[str, Any]:
        """Get the current processing status of a document"""
        if document_id in self.completed_jobs:
            return {
                "status": "completed",
                "result": self.completed_jobs[document_id],
                "progress": 100.0
            }
        elif document_id in self.failed_jobs:
            return {
                "status": "failed",
                "error": str(self.failed_jobs[document_id]),
                "progress": 0.0
            }
        elif document_id in self.active_jobs:
            return {
                "status": "processing",
                "progress": 50.0,  # TODO: Implement detailed progress tracking
                "queue_position": self._get_queue_position(document_id)
            }
        else:
            return {
                "status": "not_found",
                "progress": 0.0
            }
    
    def _get_queue_position(self, document_id: str) -> int:
        """Get the position of a document in the processing queue"""
        # This is a simplified implementation
        # In a real system, you'd track queue positions more accurately
        return self.processing_queue.qsize()
    
    async def _document_worker(self, worker_name: str):
        """Worker task that processes documents from the queue"""
        logger.info(f"🔄 Document worker {worker_name} started")
        
        while self.is_running:
            try:
                # Wait for a job with timeout to allow graceful shutdown
                job = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                logger.info(f"🔄 Worker {worker_name} processing document: {job.document_id}")
                
                # Acquire document-level semaphore
                async with self.document_semaphore:
                    try:
                        # Process the document with parallel strategies
                        result = await self._process_document_parallel(job)
                        
                        # Store successful result
                        self.completed_jobs[job.document_id] = result
                        
                        # Remove from active jobs
                        if job.document_id in self.active_jobs:
                            del self.active_jobs[job.document_id]
                        
                        logger.info(f"✅ Worker {worker_name} completed document: {job.document_id}")
                        
                    except Exception as e:
                        logger.error(f"❌ Worker {worker_name} failed to process {job.document_id}: {e}")
                        
                        # Store failed result
                        self.failed_jobs[job.document_id] = e
                        
                        # Remove from active jobs
                        if job.document_id in self.active_jobs:
                            del self.active_jobs[job.document_id]
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                # Timeout is expected for graceful shutdown
                continue
            except Exception as e:
                logger.error(f"❌ Worker {worker_name} encountered error: {e}")
                continue
        
        logger.info(f"🔄 Document worker {worker_name} stopped")
    
    async def _process_document_parallel(self, job: ProcessingJob) -> ProcessingResult:
        """Process a document using parallel strategies"""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Starting parallel processing for {job.document_id} using {self.config.strategy.value}")
            
            # Update status to processing if we have access to document repository
            if hasattr(self, 'document_repository') and self.document_repository:
                from models.api_models import ProcessingStatus
                try:
                    # ROOSEVELT FIX: Pass user_id context for status updates
                    if hasattr(job, 'user_id') and job.user_id:
                        # For user documents, set proper RLS context before update
                        from services.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', $1, false)", job.user_id)
                        await execute("SELECT set_config('app.current_user_role', 'user', false)")
                    else:
                        # For global documents, set admin context
                        from services.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', '', false)")
                        await execute("SELECT set_config('app.current_user_role', 'admin', false)")
                    
                    await self.document_repository.update_status(job.document_id, ProcessingStatus.PROCESSING)
                except Exception as e:
                    logger.warning(f"⚠️ No document found to update: {job.document_id}")
                    # Continue processing even if status update fails
            
            # Process the document
            if self.config.strategy == ProcessingStrategy.ASYNC_CONCURRENT:
                result = await self._process_async_concurrent(job)
            elif self.config.strategy == ProcessingStrategy.THREAD_POOL:
                result = await self._process_with_thread_pool(job)
            elif self.config.strategy == ProcessingStrategy.PROCESS_POOL:
                result = await self._process_with_process_pool(job)
            elif self.config.strategy == ProcessingStrategy.HYBRID:
                result = await self._process_hybrid(job)
            else:
                # Fallback to standard processing
                result = await self.document_processor.process_document(job.file_path, job.doc_type, job.document_id)
            
            # Update status to embedding phase
            if hasattr(self, 'document_repository') and self.document_repository:
                from models.api_models import ProcessingStatus
                try:
                    # ROOSEVELT FIX: Set proper RLS context before status update
                    if hasattr(job, 'user_id') and job.user_id:
                        from services.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', $1, false)", job.user_id)
                        await execute("SELECT set_config('app.current_user_role', 'user', false)")
                    else:
                        from services.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', '', false)")
                        await execute("SELECT set_config('app.current_user_role', 'admin', false)")
                        
                    await self.document_repository.update_status(job.document_id, ProcessingStatus.EMBEDDING)
                except Exception as e:
                    logger.warning(f"⚠️ No document found to update: {job.document_id}")
                    # Continue processing even if status update fails
            
            # Process embeddings if we have chunks
            if result.chunks and hasattr(self, 'embedding_manager') and self.embedding_manager:
                # Try to fetch document metadata if we have document repository access
                document_category = None
                document_tags = None
                document_title = None
                document_author = None
                document_filename = None
                if hasattr(self, 'document_repository') and self.document_repository:
                    try:
                        doc_info = await self.document_repository.get_by_id(job.document_id)
                        document_category = doc_info.category.value if doc_info and doc_info.category else None
                        document_tags = doc_info.tags if doc_info else None
                        document_title = doc_info.title if doc_info else None
                        document_author = doc_info.author if doc_info else None
                        document_filename = doc_info.filename if doc_info else None
                    except Exception as e:
                        logger.debug(f"Could not fetch document metadata: {e}")
                
                await self.embedding_manager.embed_and_store_chunks(
                    result.chunks, 
                    user_id=job.user_id,
                    document_category=document_category,
                    document_tags=document_tags,
                    document_title=document_title,
                    document_author=document_author,
                    document_filename=document_filename
                )
                collection_type = "user" if job.user_id else "global"
                logger.info(f"📊 Stored {len(result.chunks)} chunks in {collection_type} collection for document {job.document_id}")
            
            # Store entities in knowledge graph if available
            if result.entities and hasattr(self, 'kg_service') and self.kg_service:
                await self.kg_service.store_entities(result.entities, job.document_id)
                logger.info(f"🔗 Stored {len(result.entities)} entities for document {job.document_id}")
            
            # Update final status to completed
            if hasattr(self, 'document_repository') and self.document_repository:
                from models.api_models import ProcessingStatus
                try:
                    # ROOSEVELT FIX: Set proper RLS context before final status update
                    if hasattr(job, 'user_id') and job.user_id:
                        from services.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', $1, false)", job.user_id)
                        await execute("SELECT set_config('app.current_user_role', 'user', false)")
                    else:
                        from services.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', '', false)")
                        await execute("SELECT set_config('app.current_user_role', 'admin', false)")
                        
                    await self.document_repository.update_status(job.document_id, ProcessingStatus.COMPLETED)
                    
                    # Emit WebSocket notification for real-time UI update
                    await self._emit_document_status_update(job.document_id, ProcessingStatus.COMPLETED.value, job.user_id)
                    
                    # Update quality metrics if available
                    if result.quality_metrics:
                        await self.document_repository.update_quality_metrics(job.document_id, result.quality_metrics)
                except Exception as e:
                    logger.warning(f"⚠️ No document found to update: {job.document_id}")
                    # Continue processing even if status update fails
            
            logger.info(f"✅ Document {job.document_id} processing completed successfully")
            return result
                
        except Exception as e:
            logger.error(f"❌ Parallel processing failed for {job.document_id}: {e}")
            
            # Update status to failed if we have access to document repository
            if hasattr(self, 'document_repository') and self.document_repository:
                from models.api_models import ProcessingStatus
                try:
                    # ROOSEVELT FIX: Set proper RLS context before failure status update
                    if hasattr(job, 'user_id') and job.user_id:
                        from services.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', $1, false)", job.user_id)
                        await execute("SELECT set_config('app.current_user_role', 'user', false)")
                    else:
                        from services.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', '', false)")
                        await execute("SELECT set_config('app.current_user_role', 'admin', false)")
                        
                    await self.document_repository.update_status(job.document_id, ProcessingStatus.FAILED)
                    
                    # Emit WebSocket notification for real-time UI update
                    await self._emit_document_status_update(job.document_id, ProcessingStatus.FAILED.value, job.user_id)
                except Exception as e:
                    logger.warning(f"⚠️ No document found to update: {job.document_id}")
                    # Continue processing even if status update fails
            
            raise
        finally:
            processing_time = time.time() - start_time
            logger.info(f"⏱️ Document {job.document_id} processing took {processing_time:.2f} seconds")
    
    async def _process_async_concurrent(self, job: ProcessingJob) -> ProcessingResult:
        """Process document using async concurrency"""
        logger.info(f"🔄 Processing {job.document_id} with async concurrency")
        
        # Use the standard processor but with concurrent chunk processing
        result = await self.document_processor.process_document(job.file_path, job.doc_type, job.document_id)
        
        # If we have chunks, process them in parallel for embeddings
        if result.chunks and self.config.enable_chunk_level_parallelism:
            result.chunks = await self._process_chunks_parallel(result.chunks)
        
        return result
    
    async def _process_with_thread_pool(self, job: ProcessingJob) -> ProcessingResult:
        """Process document using thread pool for I/O intensive tasks"""
        logger.info(f"🔄 Processing {job.document_id} with thread pool")
        
        # Run the CPU-intensive parts in thread pool
        loop = asyncio.get_event_loop()
        
        # Text extraction in thread pool
        result = await loop.run_in_executor(
            self.thread_pool,
            self._process_document_sync,
            job.file_path,
            job.doc_type
        )
        
        return result
    
    async def _process_with_process_pool(self, job: ProcessingJob) -> ProcessingResult:
        """Process document using process pool for CPU intensive tasks"""
        logger.info(f"🔄 Processing {job.document_id} with process pool")
        
        # Run the CPU-intensive parts in process pool
        loop = asyncio.get_event_loop()
        
        # Note: Process pool requires picklable functions
        # This is a simplified implementation
        result = await loop.run_in_executor(
            self.process_pool,
            self._process_document_sync,
            job.file_path,
            job.doc_type
        )
        
        return result
    
    async def _process_hybrid(self, job: ProcessingJob) -> ProcessingResult:
        """Process document using hybrid approach (best of all strategies)"""
        logger.info(f"🔄 Processing {job.document_id} with hybrid strategy")
        
        # Use thread pool for I/O intensive text extraction
        loop = asyncio.get_event_loop()
        
        # Extract text in thread pool
        if job.doc_type in ['pdf', 'docx', 'epub']:
            # I/O intensive - use thread pool
            result = await loop.run_in_executor(
                self.thread_pool,
                self._process_document_sync,
                job.file_path,
                job.doc_type,
                job.document_id
            )
        else:
            # Light processing - use async
            result = await self.document_processor.process_document(job.file_path, job.doc_type, job.document_id)
        
        # Process chunks in parallel if enabled
        if result.chunks and self.config.enable_chunk_level_parallelism:
            result.chunks = await self._process_chunks_parallel(result.chunks)
        
        # Process entities in parallel if enabled
        if result.entities and len(result.entities) > 10:
            result.entities = await self._process_entities_parallel(result.entities)
        
        return result
    
    def _process_document_sync(self, file_path: str, doc_type: str, document_id: str) -> ProcessingResult:
        """Synchronous document processing for thread/process pools
        
        Args:
            file_path: Path to the document file
            doc_type: Type of document
            document_id: UUID of the document
        """
        # This needs to be a synchronous version for thread/process pools
        # For now, we'll use a simplified approach
        import asyncio
        
        # Create a new event loop for this thread
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Use the singleton DocumentProcessor instance
            processor = DocumentProcessor.get_instance()
            
            # Initialize the processor if needed (will be skipped if already initialized)
            logger.info(f"🔧 Using DocumentProcessor singleton for {doc_type} document in thread pool")
            loop.run_until_complete(processor.initialize())
            logger.info(f"✅ DocumentProcessor singleton ready for {doc_type} document")
            
            # Run the async processing in this thread's loop
            result = loop.run_until_complete(processor.process_document(file_path, doc_type, document_id))
            
            return result
        finally:
            loop.close()
    
    async def _process_chunks_parallel(self, chunks: List[Chunk]) -> List[Chunk]:
        """Process chunks in parallel for better performance"""
        if not chunks:
            return chunks
        
        logger.info(f"🔄 Processing {len(chunks)} chunks in parallel")
        
        # Split chunks into batches
        batch_size = self.config.chunk_batch_size
        chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        # Process batches concurrently
        processed_batches = await asyncio.gather(
            *[self._process_chunk_batch(batch) for batch in chunk_batches],
            return_exceptions=True
        )
        
        # Flatten results
        processed_chunks = []
        for batch_result in processed_batches:
            if isinstance(batch_result, Exception):
                logger.error(f"❌ Chunk batch processing failed: {batch_result}")
                continue
            processed_chunks.extend(batch_result)
        
        logger.info(f"✅ Processed {len(processed_chunks)} chunks in parallel")
        return processed_chunks
    
    async def _process_chunk_batch(self, chunk_batch: List[Chunk]) -> List[Chunk]:
        """Process a batch of chunks"""
        async with self.chunk_semaphore:
            # For now, just return the chunks as-is
            # In a real implementation, you might do additional processing here
            await asyncio.sleep(0.01)  # Simulate processing time
            return chunk_batch
    
    async def _process_entities_parallel(self, entities: List[Entity]) -> List[Entity]:
        """Process entities in parallel"""
        if not entities:
            return entities
        
        logger.info(f"🔄 Processing {len(entities)} entities in parallel")
        
        # Split entities into batches
        batch_size = 20
        entity_batches = [entities[i:i + batch_size] for i in range(0, len(entities), batch_size)]
        
        # Process batches concurrently
        processed_batches = await asyncio.gather(
            *[self._process_entity_batch(batch) for batch in entity_batches],
            return_exceptions=True
        )
        
        # Flatten results
        processed_entities = []
        for batch_result in processed_batches:
            if isinstance(batch_result, Exception):
                logger.error(f"❌ Entity batch processing failed: {batch_result}")
                continue
            processed_entities.extend(batch_result)
        
        logger.info(f"✅ Processed {len(processed_entities)} entities in parallel")
        return processed_entities
    
    async def _process_entity_batch(self, entity_batch: List[Entity]) -> List[Entity]:
        """Process a batch of entities"""
        # For now, just return the entities as-is
        # In a real implementation, you might do entity linking, validation, etc.
        await asyncio.sleep(0.01)  # Simulate processing time
        return entity_batch
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing queue"""
        return {
            "queue_size": self.processing_queue.qsize(),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "workers_running": len([w for w in self.workers if not w.done()]),
            "total_workers": len(self.workers),
            "is_running": self.is_running,
            "config": {
                "max_concurrent_documents": self.config.max_concurrent_documents,
                "max_concurrent_chunks": self.config.max_concurrent_chunks,
                "strategy": self.config.strategy.value
            }
        }
    
    async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all queued documents to complete processing"""
        try:
            await asyncio.wait_for(self.processing_queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Timeout waiting for queue completion")
            return False
    
    async def wait_for_document_completion(self, document_id: str, timeout: Optional[float] = None) -> bool:
        """Wait for a specific document to complete processing"""
        start_time = time.time()
        
        while True:
            # Check if document is completed
            if document_id in self.completed_jobs:
                logger.info(f"✅ Document {document_id} completed successfully")
                return True
            
            # Check if document failed
            if document_id in self.failed_jobs:
                logger.error(f"❌ Document {document_id} failed: {self.failed_jobs[document_id]}")
                return False
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"⚠️ Timeout waiting for document {document_id} completion")
                return False
            
            # Wait before checking again
            await asyncio.sleep(1.0)
    
    async def close(self):
        """Clean up resources"""
        logger.info("🔄 Shutting down Parallel Document Processor...")
        
        # Stop workers
        await self.stop_workers()
        
        # Close thread/process pools
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            logger.info("🔄 Thread pool shut down")
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            logger.info("🔄 Process pool shut down")
        
        logger.info("✅ Parallel Document Processor shut down complete")
