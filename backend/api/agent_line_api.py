"""
Agent Line API - CRUD for autonomous agent lines and org chart.
"""

import logging
from typing import Any, Dict, List, Optional  # noqa: F401 List for thread response

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from utils.auth_middleware import get_current_user, AuthenticatedUserResponse
from services import agent_line_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lines", tags=["Agent Lines"])


class LineCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    mission_statement: Optional[str] = None
    status: str = Field(default="active", max_length=50)
    heartbeat_config: Optional[Dict[str, Any]] = None
    governance_policy: Optional[Dict[str, Any]] = None
    reference_config: Optional[Dict[str, Any]] = None
    data_workspace_config: Optional[Dict[str, Any]] = None


class LineUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    mission_statement: Optional[str] = None
    status: Optional[str] = Field(None, max_length=50)
    heartbeat_config: Optional[Dict[str, Any]] = None
    governance_policy: Optional[Dict[str, Any]] = None
    budget_config: Optional[Dict[str, Any]] = None
    handle: Optional[str] = Field(None, max_length=100)
    team_tool_packs: Optional[List[Any]] = None  # [{"pack": "name", "mode": "read"|"full"}] or legacy ["name"]
    team_skill_ids: Optional[List[str]] = None
    reference_config: Optional[Dict[str, Any]] = None
    data_workspace_config: Optional[Dict[str, Any]] = None


class MemberAdd(BaseModel):
    agent_profile_id: str
    role: str = Field(default="worker", max_length=100)
    reports_to: Optional[str] = None


class MemberUpdate(BaseModel):
    role: Optional[str] = Field(None, max_length=100)
    reports_to: Optional[str] = None
    additional_tools: Optional[List[str]] = None


@router.get("", response_model=List[Dict[str, Any]])
async def list_lines(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """List all teams for the current user with member counts."""
    return await agent_line_service.list_lines(current_user.user_id)


@router.post("", response_model=Dict[str, Any])
async def create_line(
    body: LineCreate,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Create a new agent team."""
    return await agent_line_service.create_line(
        user_id=current_user.user_id,
        name=body.name,
        description=body.description,
        mission_statement=body.mission_statement,
        status=body.status,
        heartbeat_config=body.heartbeat_config,
        governance_policy=body.governance_policy,
        reference_config=body.reference_config,
        data_workspace_config=body.data_workspace_config,
    )


@router.get("/{line_id}", response_model=Dict[str, Any])
async def get_line(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get a line by id with full membership list."""
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    return team


@router.put("/{line_id}", response_model=Dict[str, Any])
async def update_line(
    line_id: str = Path(..., description="Line UUID"),
    body: Optional[LineUpdate] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Update a team."""
    body = body or LineUpdate()
    try:
        return await agent_line_service.update_line(
            line_id=line_id,
            user_id=current_user.user_id,
            name=body.name,
            description=body.description,
            mission_statement=body.mission_statement,
            status=body.status,
            heartbeat_config=body.heartbeat_config,
            governance_policy=body.governance_policy,
            budget_config=body.budget_config,
            handle=body.handle,
            team_tool_packs=body.team_tool_packs,
            team_skill_ids=body.team_skill_ids,
            reference_config=body.reference_config,
            data_workspace_config=body.data_workspace_config,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{line_id}/chat-context")
async def get_line_chat_context(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Compact text summary for chat when user mentions @team-handle (goals, tasks, budget, timeline)."""
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    text = await agent_line_service.get_line_chat_context(line_id, current_user.user_id)
    return JSONResponse(content={"line_id": line_id, "line_name": team.get("name", "Line"), "summary": text})


@router.delete("/{line_id}", status_code=204)
async def delete_line(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Delete a line and all its memberships."""
    try:
        await agent_line_service.delete_line(line_id, current_user.user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Line not found")


@router.post("/{line_id}/members", response_model=Dict[str, Any])
async def add_member(
    line_id: str = Path(..., description="Line UUID"),
    body: Optional[MemberAdd] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Add an agent to the line."""
    if not body:
        raise HTTPException(status_code=400, detail="Request body required")
    try:
        return await agent_line_service.add_member(
            line_id=line_id,
            user_id=current_user.user_id,
            agent_profile_id=body.agent_profile_id,
            role=body.role,
            reports_to=body.reports_to,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{line_id}/members/{membership_id}", response_model=Dict[str, Any])
async def update_member(
    line_id: str = Path(..., description="Line UUID"),
    membership_id: str = Path(..., description="Membership UUID"),
    body: Optional[MemberUpdate] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Update a membership's role, reports_to, or additional_tools."""
    body = body or MemberUpdate()
    try:
        return await agent_line_service.update_member(
            line_id=line_id,
            user_id=current_user.user_id,
            membership_id=membership_id,
            role=body.role,
            reports_to=body.reports_to,
            additional_tools=body.additional_tools,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{line_id}/members/{membership_id}", status_code=204)
async def remove_member(
    line_id: str = Path(..., description="Line UUID"),
    membership_id: str = Path(..., description="Membership UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Remove a member from the line by membership id."""
    try:
        await agent_line_service.remove_member_by_membership_id(
            line_id, current_user.user_id, membership_id
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{line_id}/members/by-agent/{agent_profile_id}", status_code=204)
async def remove_member_by_agent(
    line_id: str = Path(..., description="Line UUID"),
    agent_profile_id: str = Path(..., description="Agent profile UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Remove an agent from the line by agent profile id."""
    try:
        await agent_line_service.remove_member(line_id, current_user.user_id, agent_profile_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{line_id}/org-chart", response_model=List[Dict[str, Any]])
async def get_org_chart(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get org chart as a tree (roots = no reports_to)."""
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    return await agent_line_service.get_org_chart(line_id, current_user.user_id)


@router.get("/{line_id}/budget-summary", response_model=Dict[str, Any])
async def get_budget_summary(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get aggregated budget summary for all agents in the team."""
    try:
        return await agent_line_service.get_line_budget_summary(line_id, current_user.user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class InvokeAgentBody(BaseModel):
    """Body for manual invoke-agent trigger."""
    agent_profile_id: str = Field(..., description="Agent profile UUID to invoke")
    query: str = Field(..., min_length=1, description="Input message or prompt for the agent")


class StartDiscussionBody(BaseModel):
    """Body for start-discussion: multi-agent conversation."""
    participant_ids: List[str] = Field(..., min_length=2, description="Agent profile IDs (min 2)")
    seed_message: str = Field(..., min_length=1, description="Topic or seed message")
    moderator_id: Optional[str] = Field(None, description="Optional moderator agent profile ID")
    max_turns: int = Field(10, ge=2, le=30, description="Max conversation turns")


@router.post("/{line_id}/start-discussion", response_model=Dict[str, Any])
async def start_team_discussion(
    line_id: str = Path(..., description="Line UUID"),
    body: Optional[StartDiscussionBody] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Start a multi-agent discussion. Enqueues the task and returns immediately."""
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    if team.get("status") != "active":
        raise HTTPException(status_code=400, detail="Line must be active to start a discussion")
    body = body or StartDiscussionBody(participant_ids=[], seed_message="")
    member_ids = [m.get("agent_profile_id") for m in (team.get("members") or []) if m.get("agent_profile_id")]
    for pid in body.participant_ids:
        if pid not in member_ids:
            raise HTTPException(status_code=400, detail=f"Agent {pid} is not a member of this team")
    if body.moderator_id and body.moderator_id not in member_ids:
        raise HTTPException(status_code=400, detail="Moderator must be a line member")
    members_by_id = {m.get("agent_profile_id"): m for m in (team.get("members") or []) if m.get("agent_profile_id")}
    participant_handles = []
    for pid in body.participant_ids:
        m = members_by_id.get(pid)
        handle = (m or {}).get("agent_handle")
        if not handle:
            raise HTTPException(
                status_code=400,
                detail=f"Participant {m.get('agent_name') or pid} has no @handle set. Set a handle in the agent profile.",
            )
        participant_handles.append(handle)
    moderator_handle = None
    if body.moderator_id:
        m = members_by_id.get(body.moderator_id)
        moderator_handle = (m or {}).get("agent_handle")
        if not moderator_handle:
            raise HTTPException(status_code=400, detail="Moderator must have an @handle set")
    from services.celery_tasks.agent_tasks import dispatch_start_discussion
    dispatch_start_discussion.delay(
        line_id=line_id,
        user_id=current_user.user_id,
        initiator_profile_id=body.participant_ids[0],
        participant_handles=participant_handles,
        seed_message=body.seed_message.strip(),
        moderator_handle=moderator_handle,
        max_turns=body.max_turns,
    )
    return {"ok": True, "message": "Discussion started"}


@router.post("/{line_id}/heartbeat", response_model=Dict[str, Any])
async def trigger_heartbeat(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Manually trigger a line heartbeat (CEO run). Enqueues the task and returns immediately."""
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    if (team.get("status") or "").lower() != "active":
        raise HTTPException(
            status_code=400,
            detail="Line must be active to run a heartbeat. Resume the line first.",
        )
    from services.celery_tasks.team_heartbeat_tasks import execute_team_heartbeat

    execute_team_heartbeat.apply_async(
        kwargs={
            "line_id": line_id,
            "user_id": current_user.user_id,
            "from_manual_trigger": True,
        },
        countdown=1,
    )
    return {"ok": True, "message": "Heartbeat enqueued"}


@router.post("/{line_id}/invoke-agent", response_model=Dict[str, Any])
async def invoke_team_agent(
    line_id: str = Path(..., description="Line UUID"),
    body: Optional[InvokeAgentBody] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Manually invoke an agent in the line with a query. Enqueues the task and returns immediately."""
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    body = body or InvokeAgentBody(agent_profile_id="", query="")
    if not body.agent_profile_id or not body.query.strip():
        raise HTTPException(status_code=400, detail="agent_profile_id and query are required")
    member_ids = [m.get("agent_profile_id") for m in (team.get("members") or []) if m.get("agent_profile_id")]
    if body.agent_profile_id not in member_ids:
        raise HTTPException(status_code=400, detail="Agent is not a member of this line")
    from services.celery_tasks.agent_tasks import dispatch_agent_invocation
    dispatch_agent_invocation.delay(
        agent_profile_id=body.agent_profile_id,
        input_content=body.query.strip(),
        user_id=current_user.user_id,
        source_agent_name="user",
        chain_depth=0,
        chain_path_json="[]",
        line_id=line_id,
    )
    return {"ok": True, "message": "Agent invocation enqueued"}


@router.post("/{line_id}/emergency-stop", response_model=Dict[str, Any])
async def emergency_stop(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Pause the line, turn off autonomous scheduling, and revoke any in-flight heartbeat task."""
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    task_id = team.get("active_celery_task_id")
    hb = agent_line_service._ensure_json_obj(team.get("heartbeat_config"), {})
    hb["enabled"] = False
    await agent_line_service.update_line(
        line_id,
        current_user.user_id,
        status="paused",
        heartbeat_config=hb,
    )
    if task_id:
        try:
            from services.celery_app import celery_app

            celery_app.control.revoke(str(task_id), terminate=True)
            logger.info("Revoked team heartbeat task: %s", task_id)
        except Exception as e:
            logger.warning("Failed to revoke Celery task %s: %s", task_id, e)
    await agent_line_service.set_line_active_celery_task_id(line_id, None)
    try:
        from utils.websocket_manager import get_websocket_manager
        ws_manager = get_websocket_manager()
        await ws_manager.send_to_session(
            {
                "type": "agent_notification",
                "subtype": "line_emergency_stop",
                "line_id": line_id,
                "line_name": team.get("name", ""),
            },
            current_user.user_id,
        )
        await ws_manager.send_line_timeline_update(
            line_id,
            {"type": "execution_status", "status": "idle", "agent_id": None},
        )
    except Exception as e:
        logger.warning("Emergency stop WebSocket send failed: %s", e)
    return {"ok": True, "message": "Line paused; autonomous scheduling off; in-flight heartbeat cancelled if any"}


# ---------- Timeline (agent messages) ----------

@router.get("/{line_id}/timeline", response_model=Dict[str, Any])
async def get_line_timeline(
    line_id: str = Path(..., description="Line UUID"),
    limit: int = 50,
    offset: int = 0,
    message_type: Optional[str] = None,
    agent: Optional[str] = None,
    since: Optional[str] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Paginated timeline of inter-agent messages for the line."""
    from services import agent_message_service
    return await agent_message_service.get_line_timeline(
        line_id=line_id,
        user_id=current_user.user_id,
        limit=limit,
        offset=offset,
        message_type_filter=message_type,
        agent_filter=agent,
        since=since,
    )


@router.get("/{line_id}/timeline/summary", response_model=Dict[str, Any])
async def get_line_timeline_summary(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Timeline stats: message count today, active threads, last activity."""
    from services import agent_message_service
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    return await agent_message_service.get_line_timeline_summary(line_id, current_user.user_id)


@router.get("/{line_id}/messages/{message_id}/thread", response_model=List[Dict[str, Any]])
async def get_message_thread(
    line_id: str = Path(..., description="Line UUID"),
    message_id: str = Path(..., description="Message UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get threaded conversation (root message + replies)."""
    from services import agent_message_service
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    return await agent_message_service.get_thread(message_id, current_user.user_id)


class PostTimelineMessageBody(BaseModel):
    """Body for posting a user message to the team timeline."""
    content: str = Field(..., min_length=1, description="Message content")


@router.post("/{line_id}/timeline/message", response_model=Dict[str, Any])
async def post_team_timeline_message(
    line_id: str = Path(..., description="Line UUID"),
    body: Optional[PostTimelineMessageBody] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Post a message to the line timeline as the current user (system message)."""
    from services import agent_message_service
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    body = body or PostTimelineMessageBody(content="")
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="content is required")
    msg = await agent_message_service.create_message(
        line_id=line_id,
        from_agent_id=None,
        to_agent_id=None,
        message_type="system",
        content=body.content.strip(),
        metadata={"source": "user", "user_id": current_user.user_id},
        parent_message_id=None,
        user_id=current_user.user_id,
    )
    from services.celery_tasks.team_heartbeat_tasks import dispatch_user_post_to_ceo
    dispatch_user_post_to_ceo.apply_async(
        args=[line_id, current_user.user_id, body.content.strip(), str(msg["id"])],
        countdown=1,
    )
    return msg


@router.post("/{line_id}/timeline/clear", response_model=Dict[str, Any])
async def clear_team_timeline(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Clear all timeline messages for the line. Goals and tasks are unchanged."""
    from services import agent_message_service
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    try:
        deleted = await agent_message_service.clear_line_timeline(line_id, current_user.user_id)
        return {"ok": True, "message": "Timeline cleared", "deleted_messages": deleted}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------- Shared workspace (agent blackboard) ----------

@router.get("/{line_id}/workspace", response_model=Dict[str, Any])
async def get_line_workspace(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """List all workspace keys for the line (with updated_at and updated_by_agent_name)."""
    from services import agent_workspace_service
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    return await agent_workspace_service.list_workspace(line_id, current_user.user_id)


@router.get("/{line_id}/workspace/{key:path}", response_model=Dict[str, Any])
async def get_line_workspace_entry(
    line_id: str = Path(..., description="Line UUID"),
    key: str = Path(..., description="Workspace key"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get a single workspace entry by key (value + metadata)."""
    from services import agent_workspace_service
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    return await agent_workspace_service.get_workspace_entry(line_id, key, current_user.user_id)


@router.post("/{line_id}/reset", response_model=Dict[str, Any])
async def reset_team(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Reset line runtime data: clear timeline, tasks, workspace, goal progress, and agent memory."""
    from services import (
        agent_goal_service,
        agent_message_service,
        agent_task_service,
        agent_workspace_service,
    )
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    try:
        deleted_messages = await agent_message_service.clear_line_timeline(line_id, current_user.user_id)
        deleted_tasks = await agent_task_service.delete_all_line_tasks(line_id, current_user.user_id)
        deleted_workspace = await agent_workspace_service.clear_all_workspace(line_id, current_user.user_id)
        reset_goals = await agent_goal_service.reset_line_goals_progress(line_id, current_user.user_id)
        cleared_memory = await agent_goal_service.clear_line_agent_memory(line_id, current_user.user_id)
        return {
            "ok": True,
            "message": "Line reset: timeline, tasks, workspace, goal progress, and agent memory cleared",
            "deleted_messages": deleted_messages,
            "deleted_tasks": deleted_tasks,
            "deleted_workspace": deleted_workspace,
            "reset_goals": reset_goals,
            "cleared_memory": cleared_memory,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------- Goals (Phase 3) ----------

class GoalCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    parent_goal_id: Optional[str] = None
    assigned_agent_id: Optional[str] = None
    status: str = Field(default="active", max_length=50)
    priority: int = 0
    progress_pct: int = 0
    due_date: Optional[str] = None


class GoalUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = None
    status: Optional[str] = None
    assigned_agent_id: Optional[str] = None
    priority: Optional[int] = None
    progress_pct: Optional[int] = None
    due_date: Optional[str] = None


@router.get("/{line_id}/goals", response_model=List[Dict[str, Any]])
async def get_line_goals_tree(
    line_id: str = Path(..., description="Line UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get goal tree for the line."""
    from services import agent_goal_service
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    return await agent_goal_service.get_goal_tree(line_id, current_user.user_id)


@router.post("/{line_id}/goals", response_model=Dict[str, Any])
async def create_goal(
    line_id: str = Path(..., description="Line UUID"),
    body: Optional[GoalCreate] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Create a goal."""
    if not body:
        raise HTTPException(status_code=400, detail="Request body required")
    try:
        from services import agent_goal_service
        return await agent_goal_service.create_goal(
            line_id=line_id,
            user_id=current_user.user_id,
            title=body.title,
            description=body.description,
            parent_goal_id=body.parent_goal_id,
            assigned_agent_id=body.assigned_agent_id,
            status=body.status,
            priority=body.priority,
            progress_pct=body.progress_pct,
            due_date=body.due_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{line_id}/goals/{goal_id}", response_model=Dict[str, Any])
async def update_goal(
    line_id: str = Path(..., description="Line UUID"),
    goal_id: str = Path(..., description="Goal UUID"),
    body: Optional[GoalUpdate] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Update a goal."""
    body = body or GoalUpdate()
    try:
        from services import agent_goal_service
        return await agent_goal_service.update_goal(
            goal_id=goal_id,
            user_id=current_user.user_id,
            title=body.title,
            description=body.description,
            status=body.status,
            assigned_agent_id=body.assigned_agent_id,
            priority=body.priority,
            progress_pct=body.progress_pct,
            due_date=body.due_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{line_id}/goals/{goal_id}", status_code=204)
async def delete_goal(
    line_id: str = Path(..., description="Line UUID"),
    goal_id: str = Path(..., description="Goal UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Delete a goal."""
    try:
        from services import agent_goal_service
        await agent_goal_service.delete_goal(goal_id, current_user.user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Agent tasks (Phase 4) ---

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    assigned_agent_id: Optional[str] = None
    goal_id: Optional[str] = None
    priority: int = 0
    created_by_agent_id: Optional[str] = None
    due_date: Optional[str] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = None
    status: Optional[str] = None
    assigned_agent_id: Optional[str] = None
    goal_id: Optional[str] = None
    priority: Optional[int] = None
    due_date: Optional[str] = None


@router.get("/{line_id}/tasks", response_model=List[Dict[str, Any]])
async def list_team_tasks(
    line_id: str = Path(..., description="Line UUID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    agent_id: Optional[str] = Query(None, description="Filter by assigned agent"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """List tasks for the line with optional filters."""
    from services import agent_task_service
    team = await agent_line_service.get_line(line_id, current_user.user_id)
    if not team:
        raise HTTPException(status_code=404, detail="Line not found")
    return await agent_task_service.list_line_tasks(
        line_id, current_user.user_id, status_filter=status, agent_filter=agent_id
    )


@router.post("/{line_id}/tasks", response_model=Dict[str, Any])
async def create_task(
    line_id: str = Path(..., description="Line UUID"),
    body: Optional[TaskCreate] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Create a task."""
    if not body:
        raise HTTPException(status_code=400, detail="Request body required")
    try:
        from services import agent_task_service
        return await agent_task_service.create_task(
            line_id=line_id,
            user_id=current_user.user_id,
            title=body.title,
            description=body.description,
            assigned_agent_id=body.assigned_agent_id,
            goal_id=body.goal_id,
            priority=body.priority,
            created_by_agent_id=body.created_by_agent_id,
            due_date=body.due_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{line_id}/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(
    line_id: str = Path(..., description="Line UUID"),
    task_id: str = Path(..., description="Task UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get a task with optional thread."""
    from services import agent_task_service
    task = await agent_task_service.get_task(task_id, current_user.user_id)
    if not task or task.get("line_id") != line_id:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.put("/{line_id}/tasks/{task_id}", response_model=Dict[str, Any])
async def update_task(
    line_id: str = Path(..., description="Line UUID"),
    task_id: str = Path(..., description="Task UUID"),
    body: Optional[TaskUpdate] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Update a task."""
    body = body or TaskUpdate()
    try:
        from services import agent_task_service
        task = await agent_task_service.get_task(task_id, current_user.user_id)
        if not task or task.get("line_id") != line_id:
            raise ValueError("Task not found")
        return await agent_task_service.update_task(
            task_id=task_id,
            user_id=current_user.user_id,
            title=body.title,
            description=body.description,
            status=body.status,
            assigned_agent_id=body.assigned_agent_id,
            goal_id=body.goal_id,
            priority=body.priority,
            due_date=body.due_date,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{line_id}/tasks/{task_id}/assign", response_model=Dict[str, Any])
async def assign_task(
    line_id: str = Path(..., description="Line UUID"),
    task_id: str = Path(..., description="Task UUID"),
    agent_profile_id: str = Query(..., description="Agent to assign"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Assign task to an agent (creates task_assignment message)."""
    try:
        from services import agent_task_service
        task = await agent_task_service.get_task(task_id, current_user.user_id)
        if not task or task.get("line_id") != line_id:
            raise HTTPException(status_code=404, detail="Task not found")
        return await agent_task_service.assign_task(task_id, agent_profile_id, current_user.user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{line_id}/tasks/{task_id}/transition", response_model=Dict[str, Any])
async def transition_task(
    line_id: str = Path(..., description="Line UUID"),
    task_id: str = Path(..., description="Task UUID"),
    new_status: str = Query(..., description="New status"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Transition task to a new status."""
    try:
        from services import agent_task_service
        task = await agent_task_service.get_task(task_id, current_user.user_id)
        if not task or task.get("line_id") != line_id:
            raise HTTPException(status_code=404, detail="Task not found")
        return await agent_task_service.transition_task(task_id, current_user.user_id, new_status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{line_id}/tasks/{task_id}", status_code=204)
async def delete_task(
    line_id: str = Path(..., description="Line UUID"),
    task_id: str = Path(..., description="Task UUID"),
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Delete a task."""
    try:
        from services import agent_task_service
        await agent_task_service.delete_task(task_id, current_user.user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
