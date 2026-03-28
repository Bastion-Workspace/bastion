"""
User home dashboards: persisted in `user_home_dashboards` (PostgreSQL).

Legacy `user_settings` keys `home_dashboards_v2` / `home_dashboard_v1` are migrated on first access
when the user has no dashboard rows.

Backward compatible: GET/PUT /api/home-dashboard reads/writes the default dashboard's layout.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from models.api_models import AuthenticatedUserResponse
from models.home_dashboard_models import (
    HomeDashboardLayout,
    RssHeadlinesWidget,
    UserDashboardCreateRequest,
    UserDashboardPatchRequest,
    UserDashboardsListResponse,
)
from services import user_home_dashboard_service as dash_svc
from tools_service.services.rss_service import get_rss_service
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Home Dashboard"])


async def _validate_rss_feed_access(layout: HomeDashboardLayout, user_id: str) -> None:
    rss = await get_rss_service()
    for w in layout.widgets:
        if isinstance(w, RssHeadlinesWidget) and w.config.feed_id:
            feed = await rss.get_feed(w.config.feed_id)
            if not feed:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown RSS feed: {w.config.feed_id}",
                )
            if feed.user_id is not None and feed.user_id != user_id:
                raise HTTPException(
                    status_code=400,
                    detail="RSS feed not accessible for this user",
                )


# ---------------------------------------------------------------------------
# Phase 2: multi-dashboard CRUD
# ---------------------------------------------------------------------------


@router.get("/api/home-dashboards", response_model=UserDashboardsListResponse)
async def list_home_dashboards(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> UserDashboardsListResponse:
    return await dash_svc.list_dashboards(current_user.user_id)


@router.post("/api/home-dashboards", response_model=UserDashboardsListResponse)
async def create_home_dashboard(
    body: UserDashboardCreateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> UserDashboardsListResponse:
    try:
        return await dash_svc.create_dashboard(current_user.user_id, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.patch("/api/home-dashboards/{dashboard_id}", response_model=UserDashboardsListResponse)
async def patch_home_dashboard(
    dashboard_id: str,
    body: UserDashboardPatchRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> UserDashboardsListResponse:
    try:
        return await dash_svc.patch_dashboard(current_user.user_id, dashboard_id, body)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/api/home-dashboards/{dashboard_id}", response_model=UserDashboardsListResponse)
async def delete_home_dashboard(
    dashboard_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> UserDashboardsListResponse:
    try:
        return await dash_svc.delete_dashboard(current_user.user_id, dashboard_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/api/home-dashboards/{dashboard_id}/layout", response_model=HomeDashboardLayout)
async def get_home_dashboard_layout(
    dashboard_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> HomeDashboardLayout:
    try:
        return await dash_svc.get_layout(current_user.user_id, dashboard_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.put("/api/home-dashboards/{dashboard_id}/layout", response_model=HomeDashboardLayout)
async def put_home_dashboard_layout(
    dashboard_id: str,
    body: HomeDashboardLayout,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> HomeDashboardLayout:
    await _validate_rss_feed_access(body, current_user.user_id)
    try:
        return await dash_svc.put_layout(current_user.user_id, dashboard_id, body)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Legacy: default dashboard layout only
# ---------------------------------------------------------------------------


@router.get("/api/home-dashboard", response_model=HomeDashboardLayout)
async def get_home_dashboard(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> HomeDashboardLayout:
    try:
        return await dash_svc.get_default_layout(current_user.user_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.put("/api/home-dashboard", response_model=HomeDashboardLayout)
async def put_home_dashboard(
    body: HomeDashboardLayout,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> HomeDashboardLayout:
    await _validate_rss_feed_access(body, current_user.user_id)
    try:
        return await dash_svc.put_default_layout(current_user.user_id, body)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
