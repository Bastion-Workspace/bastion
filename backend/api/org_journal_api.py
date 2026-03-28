"""
Org Journal API
REST endpoints for journal-for-the-day: get and update a single date's entry.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services.org_journal_service import get_org_journal_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Org Tools"])


class JournalEntryUpdateBody(BaseModel):
    """Body for PUT /api/org/journal/entry."""
    date: str = Field(description="'today' or YYYY-MM-DD")
    content: str = Field(description="Full body content for that day's section")


@router.get("/api/org/journal/entry")
async def get_journal_entry(
    date: str = Query(description="'today' or YYYY-MM-DD"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get one date's journal entry. Uses user's journal preferences and timezone for 'today'.
    """
    try:
        svc = await get_org_journal_service()
        result = await svc.get_journal_entry(current_user.user_id, date.strip())
        return result
    except Exception as e:
        logger.exception("get_journal_entry failed")
        return {
            "success": False,
            "content": "",
            "date": "",
            "heading": "",
            "document_id": None,
            "file_path": None,
            "has_content": False,
            "error": str(e),
        }


@router.put("/api/org/journal/entry")
async def update_journal_entry(
    body: JournalEntryUpdateBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Replace the journal body for one date. Creates the date heading and file if missing.
    """
    try:
        svc = await get_org_journal_service()
        result = await svc.update_journal_entry(
            current_user.user_id, body.date.strip(), body.content, "replace"
        )
        return result
    except Exception as e:
        logger.exception("update_journal_entry failed")
        return {"success": False, "date": body.date, "error": str(e)}
