"""
Email API - REST endpoints for email operations (read, search, send, reply).
Uses connections-service gRPC and user's external email connection.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from clients.connections_service_client import get_connections_service_client
from models.api_models import AuthenticatedUserResponse
from services.external_connections_service import external_connections_service
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["email"])


class SendEmailRequest(BaseModel):
    to: List[str]
    subject: str
    body: str
    cc: Optional[List[str]] = None
    body_is_html: bool = False


class ReplyEmailRequest(BaseModel):
    body: str
    reply_all: bool = False
    body_is_html: bool = False


class SearchEmailRequest(BaseModel):
    query: str
    top: int = 50
    from_address: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@router.get("/api/email/connections")
async def get_email_connections(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """List user's email connections (for UI to pick connection_id)."""
    connections = await external_connections_service.get_user_connections(
        current_user.user_id,
        connection_type="email",
        active_only=True,
    )
    return {"connections": connections}


@router.get("/api/email/messages")
async def get_emails(
    folder_id: str = Query("inbox"),
    top: int = Query(50, le=200),
    skip: int = Query(0, ge=0),
    unread_only: bool = False,
    connection_id: Optional[int] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get emails from the user's connected account."""
    client = await get_connections_service_client()
    result = await client.get_emails(
        user_id=current_user.user_id,
        connection_id=connection_id,
        folder_id=folder_id,
        top=top,
        skip=skip,
        unread_only=unread_only,
    )
    if result.get("error") and not result.get("messages"):
        raise HTTPException(status_code=502, detail=result["error"])
    return result


@router.get("/api/email/messages/{message_id}")
async def get_email(
    message_id: str,
    connection_id: Optional[int] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get a single email by id."""
    client = await get_connections_service_client()
    result = await client.get_email_by_id(
        user_id=current_user.user_id,
        message_id=message_id,
        connection_id=connection_id,
    )
    if result.get("error") and not result.get("message"):
        raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
    return result


@router.post("/api/email/search")
async def search_emails(
    body: SearchEmailRequest,
    connection_id: Optional[int] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Search emails."""
    client = await get_connections_service_client()
    result = await client.search_emails(
        user_id=current_user.user_id,
        query=body.query,
        connection_id=connection_id,
        top=body.top,
        from_address=body.from_address,
        start_date=body.start_date,
        end_date=body.end_date,
    )
    if result.get("error") and not result.get("messages"):
        raise HTTPException(status_code=502, detail=result["error"])
    return result


@router.post("/api/email/send")
async def send_email(
    body: SendEmailRequest,
    connection_id: Optional[int] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Send an email."""
    client = await get_connections_service_client()
    result = await client.send_email(
        user_id=current_user.user_id,
        to_recipients=body.to,
        subject=body.subject,
        body=body.body,
        connection_id=connection_id,
        cc_recipients=body.cc,
        body_is_html=body.body_is_html,
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=502,
            detail=result.get("error", "Failed to send email"),
        )
    return result


@router.post("/api/email/messages/{message_id}/reply")
async def reply_to_email(
    message_id: str,
    body: ReplyEmailRequest,
    connection_id: Optional[int] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Reply to an email."""
    client = await get_connections_service_client()
    result = await client.reply_to_email(
        user_id=current_user.user_id,
        message_id=message_id,
        body=body.body,
        connection_id=connection_id,
        reply_all=body.reply_all,
        body_is_html=body.body_is_html,
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=502,
            detail=result.get("error", "Failed to reply"),
        )
    return result


@router.get("/api/email/folders")
async def get_folders(
    connection_id: Optional[int] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get mailbox folders."""
    client = await get_connections_service_client()
    result = await client.get_folders(
        user_id=current_user.user_id,
        connection_id=connection_id,
    )
    if result.get("error") and not result.get("folders"):
        raise HTTPException(status_code=502, detail=result["error"])
    return result


@router.get("/api/email/statistics")
async def get_email_statistics(
    folder_id: Optional[str] = None,
    connection_id: Optional[int] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get email counts (total, unread) for a folder."""
    client = await get_connections_service_client()
    result = await client.get_email_statistics(
        user_id=current_user.user_id,
        folder_id=folder_id,
        connection_id=connection_id,
    )
    if result.get("error") and "total_count" not in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result
