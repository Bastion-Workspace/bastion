"""
Parallel Document Processor - Handles concurrent document processing with configurable limits
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ds_config import settings
from ds_db.database_manager.rls_context import rls_context
from ds_models.api_models import ProcessingResult, Chunk, Entity, QualityMetrics, ProcessingStatus
from ds_processing.document_processor import DocumentProcessor
from ds_processing.dep_guard import get_dependency_guard
from ds_processing.failure_policy import (
    backoff_seconds,
    classify_error,
    should_retry,
)

from bastion_indexing.policy import APP_CHUNK_INDEX_SCHEMA_VERSION, is_chunk_index_eligible

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
        self._scheduler_tasks: List[asyncio.Task] = []
        
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
        
        # Only PROCESS_POOL strategy uses the process pool. HYBRID uses thread pool + async only
        # (see _process_hybrid); spawning ProcessPoolExecutor here wasted workers and could hit
        # ENOSPC on /tmp or raise EMFILE when many Celery prefork children each created a pool.
        if self.config.strategy == ProcessingStrategy.PROCESS_POOL:
            self.process_pool = ProcessPoolExecutor(max_workers=self.config.process_pool_size)
            logger.info(f"✅ Process pool initialized with {self.config.process_pool_size} workers")
        
        # Start worker tasks
        await self.start_workers()
        
        logger.info("✅ Parallel Document Processor initialized")
    
    async def _emit_document_status_update(
        self,
        document_id: str,
        status: str,
        user_id: str = None,
        content_source: str = "embedding",
    ):
        """Emit document status update via WebSocket for real-time UI updates"""
        try:
            if self.websocket_manager:
                # Get document details to include folder_id and filename
                try:
                    document_metadata = await self.document_repository.get_document_metadata(document_id)
                    folder_id = document_metadata.get("folder_id") if document_metadata else None
                    filename = document_metadata.get("filename") if document_metadata else None
                    collection_type = (
                        (document_metadata or {}).get("collection_type") or "user"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Could not get metadata for document {document_id}: {e}")
                    folder_id = None
                    filename = None
                    collection_type = "user"

                effective_user_id = user_id
                if (
                    collection_type == "team"
                    or not effective_user_id
                    or str(effective_user_id) == "system"
                ):
                    effective_user_id = None

                await self.websocket_manager.send_document_status_update(
                    document_id=document_id,
                    status=status,
                    folder_id=folder_id,
                    user_id=effective_user_id,
                    filename=filename,
                    content_source=content_source,
                )
                logger.info(f"📡 Emitted WebSocket status update: {document_id} -> {status}")
            else:
                logger.debug(f"📡 WebSocket manager not available for status update: {document_id} -> {status}")
        except Exception as e:
            logger.error(f"❌ Failed to emit document status update: {e}")

    @staticmethod
    def _compute_progress_pct(
        processing_status: Optional[str],
        stage: Optional[str],
        done: int,
        total: int,
    ) -> float:
        """Map DB stage + counters to a 0-100 UI progress value."""
        st = (processing_status or "").lower()
        if st == "completed":
            return 100.0
        if st in ("failed",):
            return 0.0
        if st == "retry_scheduled":
            return 5.0
        sg = (stage or "").lower()
        base = {"queued": 2.0, "parsing": 12.0, "chunking": 28.0, "embedding": 45.0, "kg": 92.0, "done": 100.0}.get(
            sg, 15.0 if st == "processing" else 0.0
        )
        if sg == "embedding" and total and total > 0:
            frac = max(0.0, min(1.0, float(done) / float(total)))
            return 30.0 + frac * 55.0
        return base

    async def _lease_renew_loop(
        self, document_id: str, worker_id: str, user_id: Optional[str], ttl_seconds: int
    ):
        interval = max(5, int(getattr(settings, "PROCESSING_LEASE_RENEW_INTERVAL_SECONDS", 300)))
        try:
            while True:
                await asyncio.sleep(interval)
                if not self.document_repository:
                    break
                await self.document_repository.renew_processing_lease(
                    document_id, worker_id, ttl_seconds, user_id=user_id
                )
        except asyncio.CancelledError:
            return

    async def _retry_scheduler_loop(self):
        while self.is_running:
            try:
                await asyncio.sleep(float(getattr(settings, "RETRY_SCHEDULER_INTERVAL_SECONDS", 30)))
                if not getattr(settings, "PROCESSING_RESILIENT_WORKER", True):
                    continue
                if not self.document_repository:
                    continue
                due = await self.document_repository.list_due_processing_retries(50)
                for doc in due:
                    path = await self._resolve_stored_file_path(doc.document_id, doc.filename)
                    if not path:
                        await self.document_repository.mark_processing_failed_terminal(
                            doc.document_id,
                            "Retry skipped: source file not found",
                            "terminal",
                            user_id=doc.user_id,
                        )
                        continue
                    await self.document_repository.mark_retry_picked_up(doc.document_id, user_id=doc.user_id)
                    doc_type = Path(doc.filename).suffix.lstrip(".").lower() or "txt"
                    await self.submit_document(
                        doc.document_id, path, doc_type, user_id=doc.user_id
                    )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("retry_scheduler_loop error: %s", e)

    async def _lease_reaper_loop(self):
        max_attempts = int(getattr(settings, "PROCESSING_MAX_ATTEMPTS", 5))
        while self.is_running:
            try:
                await asyncio.sleep(float(getattr(settings, "LEASE_REAPER_INTERVAL_SECONDS", 60)))
                if not getattr(settings, "PROCESSING_RESILIENT_WORKER", True):
                    continue
                if not self.document_repository:
                    continue
                expired = await self.document_repository.list_expired_processing_leases(50)
                for doc in expired:
                    uid = doc.user_id
                    await self.document_repository.close_open_processing_attempts(
                        doc.document_id, "timeout", "timeout", "Lease expired", user_id=uid
                    )
                    await self.document_repository.release_processing_lease(
                        doc.document_id, doc.locked_by or "", user_id=uid
                    )
                    if int(doc.attempt_count or 0) >= max_attempts:
                        await self.document_repository.mark_processing_failed_terminal(
                            doc.document_id,
                            "Processing lease expired",
                            "timeout",
                            user_id=uid,
                        )
                        await self._emit_document_status_update(
                            doc.document_id, ProcessingStatus.FAILED.value, uid
                        )
                    else:
                        delay = backoff_seconds(int(doc.attempt_count or 1))
                        nxt = datetime.now(timezone.utc) + timedelta(seconds=delay)
                        await self.document_repository.schedule_processing_retry(
                            doc.document_id,
                            nxt,
                            "Lease expired (worker timeout)",
                            "timeout",
                            user_id=uid,
                        )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("lease_reaper_loop error: %s", e)

    async def _resolve_stored_file_path(self, document_id: str, filename: str) -> Optional[str]:
        from ds_config import settings as ds

        upload_dir = Path(ds.UPLOAD_DIR)
        for potential_file in upload_dir.glob(f"{document_id}_*"):
            if potential_file.is_file():
                return str(potential_file)
        return None

    async def start_workers(self):
        """Start background worker tasks"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start multiple worker tasks for document processing
        for i in range(self.config.max_concurrent_documents):
            worker_task = asyncio.create_task(self._document_worker(f"worker-{i}"))
            self.workers.append(worker_task)

        if getattr(settings, "PROCESSING_RESILIENT_WORKER", True):
            self._scheduler_tasks.append(asyncio.create_task(self._retry_scheduler_loop()))
            self._scheduler_tasks.append(asyncio.create_task(self._lease_reaper_loop()))
        
        logger.info(f"✅ Started {len(self.workers)} document processing workers")
    
    async def stop_workers(self):
        """Stop all worker tasks"""
        self.is_running = False

        for t in self._scheduler_tasks:
            t.cancel()
        if self._scheduler_tasks:
            await asyncio.gather(*self._scheduler_tasks, return_exceptions=True)
        self._scheduler_tasks.clear()
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        logger.info("🔄 All workers stopped")
    
    async def submit_document(
        self,
        document_id: str,
        file_path: str,
        doc_type: str,
        priority: int = 0,
        user_id: str = None,
        *,
        force: bool = False,
    ) -> bool:
        """Submit a document for parallel processing.

        If *force* is True, an in-flight slot for the same *document_id* is cleared so
        operator reprocess can queue work even when a prior job is still marked active
        (e.g. stuck lease / duplicate queue pressure).
        """
        try:
            self.completed_jobs.pop(document_id, None)
            if document_id in self.active_jobs:
                if not force:
                    logger.info(
                        "Document %s already in parallel processing queue; skipping duplicate submit",
                        document_id,
                    )
                    return True
                del self.active_jobs[document_id]
                logger.info(
                    "Document %s: force submit clearing in-flight slot for reprocess",
                    document_id,
                )

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
        """Return processing status from DB, merged with in-memory completion cache."""
        doc = None
        if self.document_repository:
            try:
                doc = await self.document_repository.get_by_id(document_id)
            except Exception:
                doc = None

        if doc:
            st = doc.status.value if hasattr(doc.status, "value") else str(doc.status)
            pct = self._compute_progress_pct(
                st,
                doc.processing_stage,
                doc.processing_progress_done,
                doc.processing_progress_total,
            )
            out: Dict[str, Any] = {
                "status": st,
                "progress": pct,
                "queue_position": self._get_queue_position(document_id)
                if document_id in self.active_jobs
                else 0,
                "processing_stage": doc.processing_stage,
                "attempt_count": doc.attempt_count,
                "last_error_kind": doc.last_error_kind,
                "next_attempt_at": doc.next_attempt_at.isoformat()
                if doc.next_attempt_at
                else None,
                "last_error": doc.last_error,
            }
            if document_id in self.completed_jobs and st == "completed":
                out["result"] = self.completed_jobs[document_id]
            return out

        if document_id in self.completed_jobs:
            return {
                "status": "completed",
                "result": self.completed_jobs[document_id],
                "progress": 100.0,
            }
        if document_id in self.active_jobs:
            return {
                "status": "processing",
                "progress": 15.0,
                "queue_position": self._get_queue_position(document_id),
            }
        return {"status": "not_found", "progress": 0.0}
    
    def _get_queue_position(self, document_id: str) -> int:
        """Get the position of a document in the processing queue"""
        # This is a simplified implementation
        # In a real system, you'd track queue positions more accurately
        return self.processing_queue.qsize()
    
    async def _document_worker(self, worker_name: str):
        """Worker task that processes documents from the queue"""
        logger.info(f"🔄 Document worker {worker_name} started")
        worker_id = f"{socket.gethostname()}:{os.getpid()}:{worker_name}"

        while self.is_running:
            try:
                job = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                logger.info(f"🔄 Worker {worker_name} processing document: {job.document_id}")

                async with self.document_semaphore:
                    try:
                        await self._run_resilient_job(job, worker_id)
                        logger.info(f"✅ Worker {worker_name} finished job: {job.document_id}")
                    except Exception as e:
                        logger.error(
                            "Worker %s unexpected error for %s: %s",
                            worker_name,
                            job.document_id,
                            e,
                        )
                    finally:
                        if job.document_id in self.active_jobs:
                            del self.active_jobs[job.document_id]

                self.processing_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Worker {worker_name} encountered error: {e}")
                continue

        logger.info(f"🔄 Document worker {worker_name} stopped")

    async def _parse_chunk_stage(self, job: ProcessingJob) -> ProcessingResult:
        """Parse document and build chunks (no embeddings or KG writes)."""
        if self.config.strategy == ProcessingStrategy.ASYNC_CONCURRENT:
            return await self._process_async_concurrent(job)
        if self.config.strategy == ProcessingStrategy.THREAD_POOL:
            return await self._process_with_thread_pool(job)
        if self.config.strategy == ProcessingStrategy.PROCESS_POOL:
            return await self._process_with_process_pool(job)
        if self.config.strategy == ProcessingStrategy.HYBRID:
            return await self._process_hybrid(job)
        return await self.document_processor.process_document(
            job.file_path, job.doc_type, job.document_id
        )

    async def _finalize_completed(self, job: ProcessingJob, result: ProcessingResult) -> None:
        """Persist completed status, chunk counts, chunk index freshness, quality metrics."""
        if not self.document_repository:
            return
        async with rls_context(job.user_id):
            await self.document_repository.update_status(
                job.document_id, ProcessingStatus.COMPLETED, user_id=job.user_id
            )
        await self._emit_document_status_update(
            job.document_id, ProcessingStatus.COMPLETED.value, job.user_id
        )
        if result.quality_metrics:
            await self.document_repository.update_quality_metrics(
                job.document_id, result.quality_metrics, user_id=job.user_id
            )
        n_chunks = len(result.chunks) if result.chunks else 0
        await self.document_repository.update_chunk_count(job.document_id, n_chunks)
        try:
            di = await self.document_repository.get_by_id(job.document_id)
        except Exception:
            di = None
        dtp = (
            di.doc_type.value
            if di and hasattr(di.doc_type, "value")
            else (str(di.doc_type) if di else job.doc_type)
        )
        if di and is_chunk_index_eligible(dtp, getattr(di, "is_zip_container", None)):
            await self.document_repository.mark_chunk_index_fresh(
                job.document_id,
                getattr(di, "file_hash", None) or "",
                APP_CHUNK_INDEX_SCHEMA_VERSION,
            )
        async with rls_context(job.user_id):
            await self.document_repository.update(
                job.document_id,
                user_id=job.user_id,
                processing_stage="done",
                processing_progress_done=1,
                processing_progress_total=1,
                processing_completed_at=datetime.now(timezone.utc),
            )

    async def _run_resilient_job(self, job: ProcessingJob, worker_id: str) -> None:
        """Lease-backed pipeline with per-stage timeouts, retries, and attempt logging."""
        if not getattr(settings, "PROCESSING_RESILIENT_WORKER", True):
            result = await self._legacy_process_document_parallel(job)
            self.completed_jobs[job.document_id] = result
            return

        repo = self.document_repository
        if not repo:
            result = await self._legacy_process_document_parallel(job)
            self.completed_jobs[job.document_id] = result
            return

        ttl = int(getattr(settings, "PROCESSING_LEASE_TTL_SECONDS", 900))
        acquired = await repo.acquire_processing_lease(
            job.document_id, worker_id, ttl, job.user_id
        )
        if not acquired:
            if job.document_id in self.active_jobs:
                del self.active_jobs[job.document_id]
            await self.processing_queue.put(job)
            await asyncio.sleep(0.2)
            return

        start_row = await repo.start_processing_attempt_row(
            job.document_id, "parsing", worker_id, job.user_id
        )
        if not start_row:
            await repo.release_processing_lease(job.document_id, worker_id, job.user_id)
            raise RuntimeError("start_processing_attempt_row failed")

        attempt_id, attempt_num = start_row
        renew = asyncio.create_task(
            self._lease_renew_loop(job.document_id, worker_id, job.user_id, ttl)
        )
        try:
            async with rls_context(job.user_id):
                await repo.update_status(
                    job.document_id, ProcessingStatus.PROCESSING, user_id=job.user_id
                )
                await repo.update_processing_progress(
                    job.document_id, "parsing", 0, 1, job.user_id
                )

            result = await asyncio.wait_for(
                self._parse_chunk_stage(job),
                timeout=float(getattr(settings, "STAGE_TIMEOUT_PARSE_SECONDS", 300)),
            )

            async with rls_context(job.user_id):
                await repo.update_status(
                    job.document_id, ProcessingStatus.EMBEDDING, user_id=job.user_id
                )
                nchunks = len(result.chunks or [])
                await repo.update_processing_progress(
                    job.document_id, "embedding", 0, max(1, nchunks), job.user_id
                )

            if result.chunks and self.embedding_manager:
                doc_info = await repo.get_by_id(job.document_id)
                document_category = (
                    doc_info.category.value if doc_info and doc_info.category else None
                )
                document_tags = doc_info.tags if doc_info else None
                document_title = doc_info.title if doc_info else None
                document_author = doc_info.author if doc_info else None
                document_filename = doc_info.filename if doc_info else None
                team_id = doc_info.team_id if doc_info else None
                is_image_sidecar = job.doc_type == "image_sidecar"

                async def _embed_progress(done: int, total: int):
                    await repo.update_processing_progress(
                        job.document_id, "embedding", done, max(1, total), job.user_id
                    )

                emb_avail = self.embedding_manager.is_vector_stack_available()
                _embed_kw = dict(
                    chunks=result.chunks,
                    user_id=job.user_id,
                    team_id=team_id,
                    document_category=document_category,
                    document_tags=document_tags,
                    document_title=document_title,
                    document_author=document_author,
                    document_filename=document_filename,
                    is_image_sidecar=is_image_sidecar,
                    embed_progress=_embed_progress if nchunks else None,
                )
                _timeout = float(getattr(settings, "STAGE_TIMEOUT_EMBED_SECONDS", 900))
                if emb_avail:
                    async with get_dependency_guard().guard("embedding"):
                        await asyncio.wait_for(
                            self.embedding_manager.embed_and_store_chunks(**_embed_kw),
                            timeout=_timeout,
                        )
                else:
                    await asyncio.wait_for(
                        self.embedding_manager.embed_and_store_chunks(**_embed_kw),
                        timeout=_timeout,
                    )
                    logger.info(
                        "Document %s embedded with backlog path (vector stack unavailable)",
                        job.document_id,
                    )
                collection_type = "user" if job.user_id else "global"
                logger.info(
                    "Stored %s chunks in %s collection for document %s",
                    len(result.chunks),
                    collection_type,
                    job.document_id,
                )

            async with rls_context(job.user_id):
                await repo.update_processing_progress(
                    job.document_id, "kg", 0, 1, job.user_id
                )

            if result.entities and self.kg_service:
                if self.kg_service.is_connected():
                    async with get_dependency_guard().guard("neo4j"):
                        await asyncio.wait_for(
                            self.kg_service.store_entities(
                                result.entities,
                                job.document_id,
                                result.chunks,
                                user_id=getattr(job, "user_id", None),
                            ),
                            timeout=float(getattr(settings, "STAGE_TIMEOUT_KG_SECONDS", 120)),
                        )
                    logger.info(
                        "Stored %s entities for document %s",
                        len(result.entities),
                        job.document_id,
                    )
                else:
                    await self.kg_service.store_entities(
                        result.entities,
                        job.document_id,
                        result.chunks,
                        user_id=getattr(job, "user_id", None),
                    )
                    logger.info(
                        "Queued or skipped Neo4j entities for document %s (graph unavailable)",
                        job.document_id,
                    )

            await self._finalize_completed(job, result)
            await repo.finish_processing_attempt_row(
                attempt_id,
                "success",
                user_id=job.user_id,
            )
            self.completed_jobs[job.document_id] = result

        except Exception as e:
            kind = classify_error(e)
            tb = traceback.format_exc()
            await repo.finish_processing_attempt_row(
                attempt_id,
                "timeout" if kind == "timeout" else "failure",
                error_kind=kind,
                error_message=str(e)[:4000],
                error_traceback=tb[-8000:],
                user_id=job.user_id,
            )
            max_a = int(getattr(settings, "PROCESSING_MAX_ATTEMPTS", 5))
            do_retry, _reason = should_retry(kind, attempt_num, max_a)
            if do_retry:
                delay = backoff_seconds(attempt_num)
                nxt = datetime.now(timezone.utc) + timedelta(seconds=delay)
                await repo.schedule_processing_retry(
                    job.document_id,
                    nxt,
                    str(e)[:2000],
                    kind,
                    user_id=job.user_id,
                )
                await self._emit_document_status_update(
                    job.document_id,
                    ProcessingStatus.RETRY_SCHEDULED.value,
                    job.user_id,
                )
            else:
                await repo.mark_processing_failed_terminal(
                    job.document_id,
                    str(e)[:2000],
                    kind,
                    user_id=job.user_id,
                )
                await self._emit_document_status_update(
                    job.document_id,
                    ProcessingStatus.FAILED.value,
                    job.user_id,
                )
        finally:
            renew.cancel()
            try:
                await renew
            except asyncio.CancelledError:
                pass
            await repo.release_processing_lease(job.document_id, worker_id, job.user_id)
    
    async def _legacy_process_document_parallel(self, job: ProcessingJob) -> ProcessingResult:
        """Legacy full pipeline (no leases / retries). Used when PROCESSING_RESILIENT_WORKER is False."""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Starting parallel processing for {job.document_id} using {self.config.strategy.value}")
            
            # Update status to processing if we have access to document repository
            if hasattr(self, 'document_repository') and self.document_repository:
                from ds_models.api_models import ProcessingStatus
                try:
                    # ROOSEVELT FIX: Pass user_id context for status updates
                    if hasattr(job, 'user_id') and job.user_id:
                        # For user documents, set proper RLS context before update
                        from ds_db.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', $1, false)", job.user_id)
                        await execute("SELECT set_config('app.current_user_role', 'user', false)")
                    else:
                        # For global documents, set admin context
                        from ds_db.database_manager.database_helpers import execute
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
                from ds_models.api_models import ProcessingStatus
                try:
                    # ROOSEVELT FIX: Set proper RLS context before status update
                    if hasattr(job, 'user_id') and job.user_id:
                        from ds_db.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', $1, false)", job.user_id)
                        await execute("SELECT set_config('app.current_user_role', 'user', false)")
                    else:
                        from ds_db.database_manager.database_helpers import execute
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

                is_image_sidecar = job.doc_type == "image_sidecar"
                team_id_legacy = None
                if hasattr(self, "document_repository") and self.document_repository:
                    try:
                        di2 = await self.document_repository.get_by_id(job.document_id)
                        team_id_legacy = di2.team_id if di2 else None
                    except Exception:
                        team_id_legacy = None
                emb_avail = self.embedding_manager.is_vector_stack_available()
                _kw = dict(
                    chunks=result.chunks,
                    user_id=job.user_id,
                    team_id=team_id_legacy,
                    document_category=document_category,
                    document_tags=document_tags,
                    document_title=document_title,
                    document_author=document_author,
                    document_filename=document_filename,
                    is_image_sidecar=is_image_sidecar,
                )
                if emb_avail:
                    async with get_dependency_guard().guard("embedding"):
                        await self.embedding_manager.embed_and_store_chunks(**_kw)
                else:
                    await self.embedding_manager.embed_and_store_chunks(**_kw)
                collection_type = "user" if job.user_id else "global"
                logger.info(f"📊 Stored {len(result.chunks)} chunks in {collection_type} collection for document {job.document_id}")
            
            if result.entities and hasattr(self, "kg_service") and self.kg_service:
                await self.kg_service.store_entities(
                    result.entities,
                    job.document_id,
                    result.chunks,
                    user_id=getattr(job, "user_id", None),
                )
                logger.info(
                    "Processed knowledge graph store for document %s (%s entities)",
                    job.document_id,
                    len(result.entities),
                )
            
            # Update final status to completed
            if hasattr(self, 'document_repository') and self.document_repository:
                from ds_models.api_models import ProcessingStatus
                try:
                    # ROOSEVELT FIX: Set proper RLS context before final status update
                    if hasattr(job, 'user_id') and job.user_id:
                        from ds_db.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', $1, false)", job.user_id)
                        await execute("SELECT set_config('app.current_user_role', 'user', false)")
                    else:
                        from ds_db.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', '', false)")
                        await execute("SELECT set_config('app.current_user_role', 'admin', false)")
                        
                    await self.document_repository.update_status(job.document_id, ProcessingStatus.COMPLETED)
                    
                    # Emit WebSocket notification for real-time UI update
                    await self._emit_document_status_update(job.document_id, ProcessingStatus.COMPLETED.value, job.user_id)
                    
                    # Update quality metrics if available
                    if result.quality_metrics:
                        await self.document_repository.update_quality_metrics(job.document_id, result.quality_metrics)

                    n_chunks = len(result.chunks) if result.chunks else 0
                    await self.document_repository.update_chunk_count(job.document_id, n_chunks)
                    try:
                        di = await self.document_repository.get_by_id(job.document_id)
                    except Exception:
                        di = None
                    dtp = (
                        di.doc_type.value
                        if di and hasattr(di.doc_type, "value")
                        else (str(di.doc_type) if di else job.doc_type)
                    )
                    if di and is_chunk_index_eligible(dtp, getattr(di, "is_zip_container", None)):
                        await self.document_repository.mark_chunk_index_fresh(
                            job.document_id,
                            getattr(di, "file_hash", None) or "",
                            APP_CHUNK_INDEX_SCHEMA_VERSION,
                        )
                except Exception as e:
                    logger.warning(f"⚠️ No document found to update: {job.document_id}")
                    # Continue processing even if status update fails
            
            logger.info(f"✅ Document {job.document_id} processing completed successfully")
            return result
                
        except Exception as e:
            logger.error(f"❌ Parallel processing failed for {job.document_id}: {e}")
            
            # Update status to failed if we have access to document repository
            if hasattr(self, 'document_repository') and self.document_repository:
                from ds_models.api_models import ProcessingStatus
                try:
                    # ROOSEVELT FIX: Set proper RLS context before failure status update
                    if hasattr(job, 'user_id') and job.user_id:
                        from ds_db.database_manager.database_helpers import execute
                        await execute("SELECT set_config('app.current_user_id', $1, false)", job.user_id)
                        await execute("SELECT set_config('app.current_user_role', 'user', false)")
                    else:
                        from ds_db.database_manager.database_helpers import execute
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
        failure_breakdown: Dict[str, int] = {}
        if self.document_repository:
            try:
                failure_breakdown = await self.document_repository.get_processing_failure_stats()
            except Exception:
                failure_breakdown = {}
        return {
            "queue_size": self.processing_queue.qsize(),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "failure_breakdown_by_kind": failure_breakdown,
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
            in_active = document_id in self.active_jobs
            st = await self.get_processing_status(document_id)
            name = st.get("status") or ""

            if name == "completed" and not in_active:
                logger.info(f"✅ Document {document_id} completed successfully")
                return True

            if name == "failed" and not in_active:
                logger.error("Document %s failed: %s", document_id, st.get("last_error", ""))
                return False

            if name == "not_found" and not in_active and document_id not in self.completed_jobs:
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(f"⚠️ Timeout waiting for document {document_id} completion")
                    return False
                await asyncio.sleep(0.05)
                continue

            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"⚠️ Timeout waiting for document {document_id} completion")
                return False

            await asyncio.sleep(0.1)
    
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
