"""
Org Quick Capture API
REST endpoint for Emacs-style quick capture to inbox.org
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services.org_capture_service import get_org_capture_service
from models.org_capture_models import OrgCaptureRequest, OrgCaptureResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/org", tags=["Org Tools"])


@router.post("/capture", response_model=OrgCaptureResponse)
async def quick_capture(
    request: OrgCaptureRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> OrgCaptureResponse:
    """
    Quick capture to inbox.org (Emacs org-capture style)
    
    **BULLY!** Capture anything, anywhere with a hotkey!
    
    **How it works:**
    1. Press Ctrl+Shift+C from anywhere in the app
    2. Type your content (todo, note, journal entry)
    3. Hit Enter - it's instantly saved to inbox.org
    4. Continue what you were doing!
    
    **Template Types:**
    - **todo**: Creates a TODO entry with optional scheduling
    - **note**: Simple note with timestamp
    - **journal**: Journal entry with date
    - **meeting**: Meeting notes template with sections
    
    **Example:**
    ```json
    {
      "content": "Review quarterly reports",
      "template_type": "todo",
      "tags": ["work", "review"],
      "priority": "A"
    }
    ```
    
    **Returns:**
    - Success status
    - Preview of captured entry
    - File location and line number
    """
    try:
        logger.info(f"📝 CAPTURE REQUEST: {request.template_type} from user {current_user.username}")
        
        service = await get_org_capture_service()
        response = await service.capture_to_inbox(current_user.user_id, request)
        
        if response.success:
            logger.info(f"✅ CAPTURED: {response.entry_preview[:50]}...")
        else:
            logger.error(f"❌ CAPTURE FAILED: {response.message}")
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Capture API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check-inbox", response_model=Dict[str, Any])
async def check_inbox_status(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check inbox.org status for the current user
    
    **BULLY!** Detect duplicate inbox files before they cause trouble!
    
    **Returns:**
    - inbox_configured: bool - Whether inbox is configured in settings
    - inbox_location: str - Configured or discovered inbox location
    - multiple_inboxes: bool - Whether multiple inbox.org files exist
    - all_locations: list - All inbox.org files found (if multiple)
    - warning: str - Warning message if issues detected
    """
    try:
        from pathlib import Path
        from config import settings
        from services.org_settings_service import get_org_settings_service
        from services.database_manager.database_helpers import fetch_one
        
        logger.info(f"📋 INBOX CHECK: User {current_user.username}")
        
        # Get username
        row = await fetch_one("SELECT username FROM users WHERE user_id = $1", current_user.user_id)
        username = row['username'] if row else current_user.user_id
        
        # Check org-mode settings
        org_settings_service = await get_org_settings_service()
        settings_obj = await org_settings_service.get_settings(current_user.user_id)
        
        upload_dir = Path(settings.UPLOAD_DIR)
        user_base_dir = upload_dir / "Users" / username
        
        # Search for ALL inbox.org files
        inbox_files = []
        if user_base_dir.exists():
            inbox_files = list(user_base_dir.rglob("inbox.org"))
            inbox_files.sort()  # Consistent ordering
        
        # Build response
        response = {
            "inbox_configured": bool(settings_obj.inbox_file),
            "inbox_location": settings_obj.inbox_file if settings_obj.inbox_file else None,
            "multiple_inboxes": len(inbox_files) > 1,
            "inbox_count": len(inbox_files),
            "all_locations": None,
            "warning": None
        }
        
        if inbox_files:
            # Return relative paths
            response["all_locations"] = [str(f.relative_to(user_base_dir)) for f in inbox_files]
            
            if len(inbox_files) > 1:
                response["warning"] = f"⚠️ Multiple inbox.org files detected ({len(inbox_files)}). Quick capture will use the first one. Consider consolidating or configuring a specific location in Settings."
                logger.warning(f"⚠️ User {username} has {len(inbox_files)} inbox.org files: {response['all_locations']}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Inbox check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

