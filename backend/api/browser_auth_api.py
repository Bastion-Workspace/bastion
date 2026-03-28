"""
Browser Auth API - Interactive login capture and session management for playbook browser authentication.

Endpoints for screenshot polling, click/fill/keypress interaction, capture (save session state),
list saved sessions, and delete session.
"""

import base64
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field

from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse
from clients.crawl_service_client import get_crawl_service_client
from services.browser_session_state_service import get_browser_session_state_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/browser-auth", tags=["Browser Auth"])


class InteractRequest(BaseModel):
    """Request body for browser interact (click, fill, keypress)."""
    session_id: str = Field(..., description="Active browser session ID")
    action: str = Field(..., description="click, fill, or keypress")
    click_x: Optional[int] = Field(None, description="X coordinate for coordinate-based click")
    click_y: Optional[int] = Field(None, description="Y coordinate for coordinate-based click")
    selector: Optional[str] = Field(None, description="CSS selector for click or fill")
    value: Optional[str] = Field(None, description="Value for fill action")
    key: Optional[str] = Field(None, description="Key for keypress (e.g. Enter, Tab)")


class CaptureRequest(BaseModel):
    """Request body for capturing and saving session state."""
    site_domain: str = Field(..., description="Site domain to key the saved session (e.g. example.com)")


class StartSessionRequest(BaseModel):
    """Request body for starting an interactive session (e.g. for re-auth)."""
    site_domain: str = Field(..., description="Site domain (e.g. example.com)")
    login_url: str = Field(..., description="URL to open for login (e.g. https://example.com/login)")


@router.post("/start-session")
async def browser_auth_start_session(
    body: StartSessionRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Start a browser session for interactive re-auth: load saved state if any, navigate to login_url, return session_id."""
    site_domain = (body.site_domain or "").strip()
    login_url = (body.login_url or "").strip()
    if not site_domain or not login_url:
        raise HTTPException(status_code=400, detail="site_domain and login_url are required")
    try:
        state_svc = get_browser_session_state_service()
        state_json = await state_svc.load_session_state(
            user_id=current_user.user_id,
            site_domain=site_domain,
        )
        client = await get_crawl_service_client()
        session_id = await client.browser_create_session(
            timeout_seconds=120,
            storage_state_json=state_json,
        )
        if not session_id:
            raise HTTPException(status_code=500, detail="Failed to create browser session")
        result = await client.browser_execute_action(
            session_id=session_id,
            action="navigate",
            url=login_url,
        )
        if not result.get("success"):
            await client.browser_destroy_session(session_id)
            raise HTTPException(status_code=400, detail=result.get("error", "Navigate failed"))
        return {"success": True, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("browser_auth start_session failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/close")
async def browser_auth_close_session(
    session_id: str = Path(..., description="Active browser session ID to close"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Close/destroy an active browser session (e.g. after re-auth capture)."""
    if not session_id or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        client = await get_crawl_service_client()
        await client.browser_destroy_session(session_id.strip())
        return {"success": True}
    except Exception as e:
        logger.exception("browser_auth close_session failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interact")
async def browser_auth_interact(
    body: InteractRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Send a click, fill, or keypress action to an active browser session; returns success and screenshot after action."""
    try:
        client = await get_crawl_service_client()
        action = (body.action or "").strip().lower()
        if action == "click":
            result = await client.browser_execute_action(
                session_id=body.session_id,
                action="click",
                selector=body.selector if not (body.click_x is not None and body.click_y is not None) else None,
                click_x=body.click_x,
                click_y=body.click_y,
            )
        elif action == "fill":
            if not body.selector or body.value is None:
                raise HTTPException(status_code=400, detail="fill action requires selector and value")
            result = await client.browser_execute_action(
                session_id=body.session_id,
                action="fill",
                selector=body.selector,
                value=body.value,
            )
        elif action == "keypress":
            result = await client.browser_execute_action(
                session_id=body.session_id,
                action="keypress",
                key=body.key or "Enter",
            )
        else:
            raise HTTPException(status_code=400, detail="action must be click, fill, or keypress")

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Action failed"))

        screenshot_b64 = None
        if result.get("screenshot_png"):
            screenshot_b64 = base64.b64encode(result["screenshot_png"]).decode("utf-8")
        else:
            screenshot_result = await client.browser_execute_action(
                session_id=body.session_id,
                action="screenshot",
            )
            if screenshot_result.get("screenshot_png"):
                screenshot_b64 = base64.b64encode(screenshot_result["screenshot_png"]).decode("utf-8")

        return {"success": True, "screenshot_b64": screenshot_b64}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("browser_auth interact failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def browser_auth_list_sessions(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """List saved browser sessions for the current user (site_domain, last_used_at, is_valid)."""
    try:
        state_svc = get_browser_session_state_service()
        sessions = await state_svc.list_sessions(user_id=current_user.user_id)
        return {"sessions": sessions}
    except Exception as e:
        logger.exception("browser_auth list_sessions failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/screenshot")
async def browser_auth_screenshot(
    session_id: str = Path(..., description="Active browser session ID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get current page screenshot for an active browser session."""
    try:
        client = await get_crawl_service_client()
        result = await client.browser_execute_action(
            session_id=session_id,
            action="screenshot",
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Screenshot failed"))
        screenshot_b64 = None
        if result.get("screenshot_png"):
            screenshot_b64 = base64.b64encode(result["screenshot_png"]).decode("utf-8")
        return {"success": True, "screenshot_b64": screenshot_b64}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("browser_auth screenshot failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/capture")
async def browser_auth_capture(
    session_id: str = Path(..., description="Active browser session ID"),
    body: CaptureRequest = ...,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Capture current session state (cookies/localStorage), encrypt and save to DB for site_domain."""
    site_domain = (body.site_domain or "").strip()
    if not site_domain:
        raise HTTPException(status_code=400, detail="site_domain is required in body")
    try:
        client = await get_crawl_service_client()
        state_json = await client.browser_save_session_state(session_id=session_id)
        if not state_json:
            raise HTTPException(status_code=400, detail="Failed to get session state from browser service")
        state_svc = get_browser_session_state_service()
        ok = await state_svc.save_session_state(
            user_id=current_user.user_id,
            site_domain=site_domain,
            state_json=state_json,
        )
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to save session state")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("browser_auth capture failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{domain:path}")
async def browser_auth_delete_session(
    domain: str = Path(..., description="Site domain to invalidate"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> Dict[str, Any]:
    """Invalidate a saved browser session for the given domain."""
    if not domain or not domain.strip():
        raise HTTPException(status_code=400, detail="domain is required")
    site_domain = domain.strip()
    try:
        state_svc = get_browser_session_state_service()
        ok = await state_svc.invalidate_session_state(
            user_id=current_user.user_id,
            site_domain=site_domain,
        )
        return {"success": ok}
    except Exception as e:
        logger.exception("browser_auth delete_session failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
