"""
Org Tag API - Add tags to org entries
"""

import logging
from fastapi import APIRouter, Depends

from models.org_tag_models import OrgTagRequest, OrgTagResponse
from utils.auth_middleware import AuthenticatedUserResponse, get_current_user
from services.org_tag_service import get_org_tag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/org/tag", tags=["org-mode"])


@router.post("", response_model=OrgTagResponse)
async def add_tags(
    request: OrgTagRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> OrgTagResponse:
    """
    Add tags to an org-mode heading
    
    Endpoint: POST /api/org/tag
    
    Request body:
    {
        "file_path": "inbox.org",
        "line_number": 5,
        "tags": ["@outside", "urgent"],
        "replace_existing": false
    }
    """
    service = await get_org_tag_service()
    return await service.add_tags_to_entry(current_user.user_id, request)

