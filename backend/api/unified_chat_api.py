"""
Unified Chat API (V2)
Compatibility routes for unified chat operations used by the frontend.
Currently implements job cancellation to align with `/api/v2/chat/unified/job/{job_id}/cancel`.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from services.celery_app import celery_app
from services.stream_run_cancel import is_stream_run_id, set_stream_cancel_requested
from utils.auth_middleware import get_current_user, AuthenticatedUserResponse


logger = logging.getLogger(__name__)


router = APIRouter(tags=["Unified Chat V2"])


@router.post("/api/v2/chat/unified/job/{job_id}/cancel")
async def cancel_unified_job(
    job_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Cancel a unified chat job.

    - Interactive chat streaming: ``job_id`` is a server-issued UUID (``run_started`` SSE).
      Sets a Redis flag consumed by the gRPC stream proxy to cancel the orchestrator RPC.
    - Background jobs: non-UUID task ids are revoked via Celery control.
    """
    try:
        logger.info(f"🛑 Unified job cancel requested by user {current_user.user_id}: {job_id}")

        if is_stream_run_id(job_id):
            ok = await set_stream_cancel_requested(job_id)
            if ok:
                return {
                    "success": True,
                    "job_id": job_id,
                    "status": "CANCELLED",
                    "message": "Stream cancellation requested",
                }
            return {
                "success": False,
                "job_id": job_id,
                "status": "ERROR",
                "message": "Could not record stream cancellation (Redis unavailable)",
            }

        # Attempt to revoke as a Celery task
        try:
            celery_app.control.revoke(job_id, terminate=True)
            logger.info(f"✅ Celery job revoked: {job_id}")
            return {
                "success": True,
                "job_id": job_id,
                "status": "CANCELLED",
                "message": "Background job cancelled successfully"
            }
        except Exception as celery_error:
            logger.warning(f"⚠️ Failed to revoke Celery task {job_id}: {celery_error}")
            # Fall back to acknowledging cancellation to avoid frontend 404s
            return {
                "success": True,
                "job_id": job_id,
                "status": "CANCELLED",
                "message": "Cancellation acknowledged (task may not have been a Celery job)"
            }

    except Exception as e:
        logger.error(f"❌ Unified job cancel error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


