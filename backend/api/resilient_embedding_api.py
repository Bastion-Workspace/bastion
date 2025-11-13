"""
Resilient Embedding API - Endpoints for managing crash-resistant embedding operations
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from utils.resilient_embedding_manager import ResilientEmbeddingManager
from utils.auth_middleware import get_current_user
from models.api_models import Chunk
from repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/resilient-embedding", tags=["resilient-embedding"])

# Global resilient embedding manager instance
resilient_manager: Optional[ResilientEmbeddingManager] = None


class EmbeddingRequest(BaseModel):
    """Request to start resilient embedding for a document"""
    document_id: str
    chunks: List[Chunk]
    user_id: Optional[str] = None
    resume_existing: bool = True


class RetryRequest(BaseModel):
    """Request to retry failed chunks"""
    document_id: str
    chunks: List[Chunk]
    user_id: Optional[str] = None


class ProcessingStatusResponse(BaseModel):
    """Response with processing status"""
    status: str
    document_id: str
    total_chunks: int
    processed_chunks: int
    failed_chunks: int
    completed_chunks: int
    progress_percentage: float
    start_time: float
    last_update: float
    processing_time: float
    failed_chunk_ids: List[str]


async def get_resilient_manager() -> ResilientEmbeddingManager:
    """Get or initialize the resilient embedding manager"""
    global resilient_manager
    
    if resilient_manager is None:
        resilient_manager = ResilientEmbeddingManager()
        await resilient_manager.initialize()
    
    return resilient_manager


@router.post("/embed-document")
async def embed_document_resilient(
    request: EmbeddingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    manager: ResilientEmbeddingManager = Depends(get_resilient_manager)
):
    """
    Start resilient embedding for a document with crash recovery.
    
    This endpoint processes chunks one by one with immediate persistence,
    allowing recovery from crashes without losing progress.
    """
    try:
        logger.info(f"🔄 Starting resilient embedding for document {request.document_id}")
        
        # Validate request
        if not request.chunks:
            raise HTTPException(status_code=400, detail="No chunks provided")
        
        # Use user ID from token if not provided in request
        user_id = request.user_id or current_user.get("user_id")
        
        # Start embedding process in background
        background_tasks.add_task(
            _process_document_embedding,
            manager,
            request.chunks,
            request.document_id,
            user_id,
            request.resume_existing
        )
        
        return {
            "status": "started",
            "message": f"Resilient embedding started for document {request.document_id}",
            "document_id": request.document_id,
            "total_chunks": len(request.chunks),
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start resilient embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start embedding: {str(e)}")


@router.get("/status/{document_id}")
async def get_embedding_status(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    manager: ResilientEmbeddingManager = Depends(get_resilient_manager)
) -> ProcessingStatusResponse:
    """Get the current processing status for a document"""
    try:
        status = await manager.get_processing_status(document_id)
        
        if status["status"] == "not_found":
            raise HTTPException(status_code=404, detail="No processing record found for this document")
        
        if status["status"] == "error":
            raise HTTPException(status_code=500, detail=status.get("error", "Unknown error"))
        
        return ProcessingStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get embedding status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/active-jobs")
async def list_active_embedding_jobs(
    current_user: dict = Depends(get_current_user),
    manager: ResilientEmbeddingManager = Depends(get_resilient_manager)
):
    """List all active embedding jobs"""
    try:
        active_jobs = await manager.list_active_processing()
        
        return {
            "active_jobs": active_jobs,
            "total_active": len(active_jobs)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list active jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list active jobs: {str(e)}")


@router.post("/resume/{document_id}")
async def resume_document_embedding(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    manager: ResilientEmbeddingManager = Depends(get_resilient_manager)
):
    """
    Resume embedding for a document that was interrupted.
    
    This will automatically load the chunks from the database and resume
    processing from where it left off.
    """
    try:
        logger.info(f"🔄 Resuming embedding for document {document_id}")
        
        # Get document chunks from database
        doc_repo = DocumentRepository()
        await doc_repo.initialize()
        
        # Get document info to verify it exists
        document = await doc_repo.get_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks from vector database
        chunks_data = await manager.get_all_document_chunks(document_id)
        if not chunks_data:
            raise HTTPException(status_code=400, detail="No chunks found for this document")
        
        # Convert to Chunk objects
        chunks = []
        for chunk_data in chunks_data:
            chunk = Chunk(
                chunk_id=chunk_data["chunk_id"],
                document_id=chunk_data["document_id"],
                content=chunk_data["content"],
                chunk_index=chunk_data.get("chunk_index", 0),
                quality_score=chunk_data.get("quality_score", 1.0),
                method=chunk_data.get("method", "unknown"),
                metadata=chunk_data.get("metadata", {})
            )
            chunks.append(chunk)
        
        user_id = current_user.get("user_id")
        
        # Start resume process in background
        background_tasks.add_task(
            _resume_document_embedding,
            manager,
            document_id,
            chunks,
            user_id
        )
        
        return {
            "status": "resumed",
            "message": f"Embedding resumed for document {document_id}",
            "document_id": document_id,
            "total_chunks": len(chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to resume embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume embedding: {str(e)}")


@router.post("/retry-failed/{document_id}")
async def retry_failed_chunks(
    document_id: str,
    request: RetryRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    manager: ResilientEmbeddingManager = Depends(get_resilient_manager)
):
    """Retry processing only the failed chunks for a document"""
    try:
        logger.info(f"🔄 Retrying failed chunks for document {document_id}")
        
        user_id = request.user_id or current_user.get("user_id")
        
        # Start retry process in background
        background_tasks.add_task(
            _retry_failed_chunks,
            manager,
            document_id,
            request.chunks,
            user_id
        )
        
        return {
            "status": "retry_started",
            "message": f"Retry started for failed chunks in document {document_id}",
            "document_id": document_id,
            "total_chunks": len(request.chunks)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start retry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start retry: {str(e)}")


@router.delete("/cancel/{document_id}")
async def cancel_embedding_processing(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    manager: ResilientEmbeddingManager = Depends(get_resilient_manager)
):
    """Cancel processing for a document"""
    try:
        success = await manager.cancel_processing(document_id)
        
        if success:
            return {
                "status": "cancelled",
                "message": f"Processing cancelled for document {document_id}",
                "document_id": document_id
            }
        else:
            raise HTTPException(status_code=404, detail="No active processing found for this document")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to cancel processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel processing: {str(e)}")


@router.get("/health")
async def embedding_health_check(
    manager: ResilientEmbeddingManager = Depends(get_resilient_manager)
):
    """Health check for the resilient embedding system"""
    try:
        # Check if manager is initialized
        if not manager.base_manager:
            return {"status": "unhealthy", "message": "Base manager not initialized"}
        
        # Get basic stats
        active_jobs = await manager.list_active_processing()
        
        return {
            "status": "healthy",
            "message": "Resilient embedding system is operational",
            "active_jobs_count": len(active_jobs),
            "progress_dir_exists": manager.progress_dir.exists(),
            "base_manager_initialized": manager.base_manager is not None
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}"
        }


# Background task functions

async def _process_document_embedding(
    manager: ResilientEmbeddingManager,
    chunks: List[Chunk],
    document_id: str,
    user_id: str,
    resume_existing: bool
):
    """Background task to process document embedding"""
    try:
        result = await manager.embed_and_store_chunks_resilient(
            chunks=chunks,
            document_id=document_id,
            user_id=user_id,
            resume_existing=resume_existing
        )
        
        logger.info(f"✅ Background embedding completed for document {document_id}: {result}")
        
    except Exception as e:
        logger.error(f"❌ Background embedding failed for document {document_id}: {e}")


async def _resume_document_embedding(
    manager: ResilientEmbeddingManager,
    document_id: str,
    chunks: List[Chunk],
    user_id: str
):
    """Background task to resume document embedding"""
    try:
        result = await manager.resume_document_processing(
            document_id=document_id,
            chunks=chunks,
            user_id=user_id
        )
        
        logger.info(f"✅ Background resume completed for document {document_id}: {result}")
        
    except Exception as e:
        logger.error(f"❌ Background resume failed for document {document_id}: {e}")


async def _retry_failed_chunks(
    manager: ResilientEmbeddingManager,
    document_id: str,
    chunks: List[Chunk],
    user_id: str
):
    """Background task to retry failed chunks"""
    try:
        result = await manager.retry_failed_chunks(
            document_id=document_id,
            chunks=chunks,
            user_id=user_id
        )
        
        logger.info(f"✅ Background retry completed for document {document_id}: {result}")
        
    except Exception as e:
        logger.error(f"❌ Background retry failed for document {document_id}: {e}")
