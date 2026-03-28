"""
Org Todo API - Universal REST endpoints for todo list, create, update, toggle, delete, archive.
Uses OrgTodoService; org files on disk are the source of truth.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services.org_todo_service import get_org_todo_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Org Tools"])


class CreateTodoBody(BaseModel):
    text: str = Field(..., description="Todo title")
    file_path: Optional[str] = Field(None, description="Org file path (omit for inbox)")
    state: str = Field("TODO", description="Initial state")
    tags: Optional[List[str]] = Field(None, description="Tags")
    scheduled: Optional[str] = Field(None, description="Org timestamp")
    deadline: Optional[str] = Field(None, description="Org timestamp")
    priority: Optional[str] = Field(None, description="A, B, C")
    body: Optional[str] = Field(None, description="Body/description under the heading")
    heading_level: Optional[int] = Field(None, ge=1, le=6, description="Org stars depth 1-6")
    insert_after_line_number: Optional[int] = Field(None, ge=0, description="0-based insert position")


class UpdateTodoBody(BaseModel):
    file_path: str = Field(..., description="Org file path")
    line_number: int = Field(..., description="0-based line index")
    heading_text: Optional[str] = Field(None, description="Verify heading matches")
    new_state: Optional[str] = Field(None, description="New TODO state")
    new_text: Optional[str] = Field(None, description="New heading text")
    add_tags: Optional[List[str]] = Field(None, description="Tags to add")
    remove_tags: Optional[List[str]] = Field(None, description="Tags to remove")
    scheduled: Optional[str] = Field(None, description="Org timestamp")
    deadline: Optional[str] = Field(None, description="Org timestamp")
    priority: Optional[str] = Field(None, description="A, B, C")
    new_body: Optional[str] = Field(None, description="Replace body/description under the heading")


class ToggleTodoBody(BaseModel):
    file_path: str = Field(..., description="Org file path")
    line_number: int = Field(..., description="0-based line index")
    heading_text: Optional[str] = Field(None, description="Verify heading matches")


class DeleteTodoBody(BaseModel):
    file_path: str = Field(..., description="Org file path")
    line_number: int = Field(..., description="0-based line index")
    heading_text: Optional[str] = Field(None, description="Verify heading matches")


class ArchiveTodoBody(BaseModel):
    file_path: Optional[str] = Field(None, description="Org file path (omit for inbox)")
    line_number: Optional[int] = Field(None, ge=0, description="0-based line index to archive one entry (any state). Omit for bulk closed.")


@router.get("/api/todos")
async def list_todos(
    scope: str = Query("all", description="all, inbox, or file path"),
    states: Optional[str] = Query(None, description="Comma-separated TODO states"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    query: str = Query("", description="Search query"),
    limit: int = Query(100, ge=1, le=500),
    include_archives: bool = Query(False),
    closed_since_days: Optional[int] = Query(None, description="Only DONE items closed in the last N days (e.g. 7 for last week)"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """List todos. scope: all | inbox | file path. line_number in results is 0-based. Use closed_since_days=7 for things completed in the last week."""
    try:
        states_list = [s.strip().upper() for s in states.split(",")] if states else None
        tags_list = [t.strip() for t in tags.split(",")] if tags else None
        service = await get_org_todo_service()
        return await service.list_todos(
            user_id=current_user.user_id,
            scope=scope,
            states=states_list,
            tags=tags_list,
            query=query,
            limit=limit,
            include_archives=include_archives,
            closed_since_days=closed_since_days,
        )
    except Exception as e:
        logger.exception("list_todos failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/todos")
async def create_todo(
    body: CreateTodoBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Create a todo. Omit file_path for inbox."""
    try:
        service = await get_org_todo_service()
        return await service.create_todo(
            user_id=current_user.user_id,
            text=body.text,
            file_path=body.file_path,
            state=body.state,
            tags=body.tags,
            scheduled=body.scheduled,
            deadline=body.deadline,
            priority=body.priority,
            body=body.body,
            heading_level=body.heading_level,
            insert_after_line_number=body.insert_after_line_number,
        )
    except Exception as e:
        logger.exception("create_todo failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/todos/update")
async def update_todo(
    body: UpdateTodoBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Update a todo (state, text, tags, schedule)."""
    try:
        service = await get_org_todo_service()
        return await service.update_todo(
            user_id=current_user.user_id,
            file_path=body.file_path,
            line_number=body.line_number,
            heading_text=body.heading_text,
            new_state=body.new_state,
            new_text=body.new_text,
            add_tags=body.add_tags,
            remove_tags=body.remove_tags,
            scheduled=body.scheduled,
            deadline=body.deadline,
            priority=body.priority,
            new_body=body.new_body,
        )
    except Exception as e:
        logger.exception("update_todo failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/todos/toggle")
async def toggle_todo(
    body: ToggleTodoBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Toggle TODO <-> DONE for the given todo."""
    try:
        service = await get_org_todo_service()
        return await service.toggle_todo(
            user_id=current_user.user_id,
            file_path=body.file_path,
            line_number=body.line_number,
            heading_text=body.heading_text,
        )
    except Exception as e:
        logger.exception("toggle_todo failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/todos/delete")
async def delete_todo(
    body: DeleteTodoBody,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete the todo line (and any SCHEDULED/DEADLINE line below it)."""
    try:
        service = await get_org_todo_service()
        return await service.delete_todo(
            user_id=current_user.user_id,
            file_path=body.file_path,
            line_number=body.line_number,
            heading_text=body.heading_text,
        )
    except Exception as e:
        logger.exception("delete_todo failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/todos/archive")
async def archive_done(
    body: Optional[ArchiveTodoBody] = Body(None),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Move closed items (or one entry if line_number set) from file to archive. Omit body or file_path for inbox. Full entries (heading + subtree + properties) are always moved."""
    try:
        file_path = body.file_path if body else None
        line_number = body.line_number if body else None
        service = await get_org_todo_service()
        return await service.archive_done(user_id=current_user.user_id, file_path=file_path, line_number=line_number)
    except Exception as e:
        logger.exception("archive_done failed")
        raise HTTPException(status_code=500, detail=str(e))

