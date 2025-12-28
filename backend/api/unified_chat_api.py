"""
Unified Chat API (V2)
Compatibility routes for unified chat operations used by the frontend.
Currently implements job cancellation to align with `/api/v2/chat/unified/job/{job_id}/cancel`.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from services.celery_app import celery_app
from utils.auth_middleware import get_current_user, AuthenticatedUserResponse


logger = logging.getLogger(__name__)


router = APIRouter(tags=["Unified Chat V2"])


@router.post("/api/v2/chat/unified/job/{job_id}/cancel")
async def cancel_unified_job(
    job_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """Cancel a unified chat job.

    Frontend issues requests to `/api/v2/chat/unified/job/{job_id}/cancel` for both
    streaming and background jobs. We handle both forms here:

    - Streaming jobs typically have IDs like `streaming_<timestamp>`; client disconnect
      cancels server streaming naturally, so we acknowledge cancellation.
    - Background jobs (Celery task IDs) are revoked via Celery control.
    """
    try:
        logger.info(f"🛑 Unified job cancel requested by user {current_user.user_id}: {job_id}")

        # Heuristic: streaming jobs use a `streaming_` prefix from the frontend
        if job_id.startswith("streaming_"):
            logger.info("🔌 Streaming job cancellation acknowledged (client will disconnect stream)")
            return {
                "success": True,
                "job_id": job_id,
                "status": "CANCELLED",
                "message": "Streaming job cancellation acknowledged"
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


