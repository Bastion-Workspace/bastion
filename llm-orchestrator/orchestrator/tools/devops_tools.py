"""Azure DevOps tools via M365 Graph dispatch (connection-scoped; uses Microsoft email connection with devops service)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.tools.m365_invoke_common import invoke_m365_graph
from orchestrator.utils.action_io_registry import register_action


# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------

class DevopsConnParams(BaseModel):
    connection_id: Optional[int] = Field(default=None, description="Microsoft 365 connection id")


class ProjectRef(BaseModel):
    id: str = Field(description="Project GUID")
    name: str = Field(description="Project name")
    description: str = Field(default="", description="Project description")
    state: str = Field(default="", description="Project state")


class TeamRef(BaseModel):
    id: str = Field(description="Team GUID")
    name: str = Field(description="Team name")
    description: str = Field(default="", description="Team description")


class TeamMemberRef(BaseModel):
    id: str = Field(description="Member identity GUID")
    display_name: str = Field(description="Display name")
    unique_name: str = Field(default="", description="Email / UPN")
    is_team_admin: bool = Field(default=False)


class WorkItemRef(BaseModel):
    id: int = Field(description="Work item ID")
    title: str = Field(description="Title")
    state: str = Field(default="", description="State")
    work_item_type: str = Field(default="", description="Bug, Task, User Story, etc.")
    assigned_to: str = Field(default="", description="Assigned person display name")
    iteration_path: str = Field(default="")
    area_path: str = Field(default="")
    priority: int = Field(default=0)
    tags: str = Field(default="")


class IterationRef(BaseModel):
    id: str = Field(description="Iteration GUID")
    name: str = Field(description="Iteration name")
    path: str = Field(default="")
    start_date: str = Field(default="")
    finish_date: str = Field(default="")
    time_frame: str = Field(default="", description="past, current, or future")


class BoardRef(BaseModel):
    id: str = Field(description="Board ID")
    name: str = Field(description="Board name")


class BoardColumnRef(BaseModel):
    id: str = Field(description="Column ID")
    name: str = Field(description="Column name")
    item_limit: int = Field(default=0)
    column_type: str = Field(default="")


class RepoRef(BaseModel):
    id: str = Field(description="Repository GUID")
    name: str = Field(description="Repository name")
    default_branch: str = Field(default="")
    web_url: str = Field(default="")
    project_name: str = Field(default="")


class PullRequestRef(BaseModel):
    id: int = Field(description="PR ID")
    title: str = Field(description="PR title")
    status: str = Field(default="")
    created_by: str = Field(default="")
    source_branch: str = Field(default="")
    target_branch: str = Field(default="")
    repo_name: str = Field(default="")


class PipelineRef(BaseModel):
    id: int = Field(description="Pipeline ID")
    name: str = Field(description="Pipeline name")
    folder: str = Field(default="")


class PipelineRunRef(BaseModel):
    id: int = Field(description="Run ID")
    name: str = Field(default="")
    state: str = Field(default="")
    result: str = Field(default="")
    created_date: str = Field(default="")
    finished_date: str = Field(default="")


# ---------------------------------------------------------------------------
# Input / Output models
# ---------------------------------------------------------------------------

class ListProjectsInputs(BaseModel):
    top: int = Field(default=100, description="Max projects")

class ListProjectsOutputs(BaseModel):
    projects: List[ProjectRef] = Field(description="DevOps projects")
    count: int = Field(description="Number of projects")
    formatted: str = Field(description="Human-readable summary")


class ListTeamsInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    top: int = Field(default=100, description="Max teams")

class ListTeamsOutputs(BaseModel):
    teams: List[TeamRef] = Field(description="Teams in the project")
    count: int = Field(description="Number of teams")
    formatted: str = Field(description="Human-readable summary")


class ListTeamMembersInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    team: str = Field(description="Team name or ID")
    top: int = Field(default=100, description="Max members")

class ListTeamMembersOutputs(BaseModel):
    members: List[TeamMemberRef] = Field(description="Team members")
    count: int = Field(description="Number of members")
    formatted: str = Field(description="Human-readable summary")


class QueryWorkItemsInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    wiql: str = Field(description="WIQL query string")
    top: int = Field(default=200, description="Max results")

class QueryWorkItemsOutputs(BaseModel):
    work_items: List[WorkItemRef] = Field(description="Matching work items")
    count: int = Field(description="Number of results")
    formatted: str = Field(description="Human-readable summary")


class GetWorkItemInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    work_item_id: int = Field(description="Work item ID")

class GetWorkItemOutputs(BaseModel):
    work_item: Optional[WorkItemRef] = Field(description="Work item details")
    formatted: str = Field(description="Human-readable summary")


class ListIterationsInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    team: str = Field(default="", description="Team name (optional)")

class ListIterationsOutputs(BaseModel):
    iterations: List[IterationRef] = Field(description="Iterations / sprints")
    count: int = Field(description="Number of iterations")
    formatted: str = Field(description="Human-readable summary")


class GetIterationWorkItemsInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    iteration_id: str = Field(description="Iteration GUID")
    team: str = Field(default="", description="Team name (optional)")

class GetIterationWorkItemsOutputs(BaseModel):
    work_items: List[WorkItemRef] = Field(description="Work items in the iteration")
    count: int = Field(description="Number of items")
    formatted: str = Field(description="Human-readable summary")


class ListBoardsInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    team: str = Field(default="", description="Team name (optional)")

class ListBoardsOutputs(BaseModel):
    boards: List[BoardRef] = Field(description="Boards")
    count: int = Field(description="Number of boards")
    formatted: str = Field(description="Human-readable summary")


class GetBoardColumnsInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    board: str = Field(description="Board name or ID")
    team: str = Field(default="", description="Team name (optional)")

class GetBoardColumnsOutputs(BaseModel):
    columns: List[BoardColumnRef] = Field(description="Board columns")
    count: int = Field(description="Number of columns")
    formatted: str = Field(description="Human-readable summary")


class ListReposInputs(BaseModel):
    project: str = Field(default="", description="Project name or ID (optional)")

class ListReposOutputs(BaseModel):
    repos: List[RepoRef] = Field(description="Git repositories")
    count: int = Field(description="Number of repos")
    formatted: str = Field(description="Human-readable summary")


class ListPullRequestsInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    status: str = Field(default="active", description="active, completed, abandoned, or all")
    top: int = Field(default=50, description="Max results")

class ListPullRequestsOutputs(BaseModel):
    pull_requests: List[PullRequestRef] = Field(description="Pull requests")
    count: int = Field(description="Number of PRs")
    formatted: str = Field(description="Human-readable summary")


class ListPipelinesInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    top: int = Field(default=50, description="Max results")

class ListPipelinesOutputs(BaseModel):
    pipelines: List[PipelineRef] = Field(description="Pipelines")
    count: int = Field(description="Number of pipelines")
    formatted: str = Field(description="Human-readable summary")


class GetPipelineRunsInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    pipeline_id: int = Field(description="Pipeline ID")
    top: int = Field(default=20, description="Max results")

class GetPipelineRunsOutputs(BaseModel):
    runs: List[PipelineRunRef] = Field(description="Pipeline runs")
    count: int = Field(description="Number of runs")
    formatted: str = Field(description="Human-readable summary")


class CreateWorkItemInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    work_item_type: str = Field(description="Type: Bug, Task, User Story, Feature, Epic, etc.")
    title: str = Field(description="Work item title")
    description: str = Field(default="", description="HTML description")
    assigned_to: str = Field(default="", description="Person to assign (display name or email)")
    iteration_path: str = Field(default="", description="Iteration path")
    area_path: str = Field(default="", description="Area path")
    priority: Optional[int] = Field(default=None, description="Priority (1-4)")
    tags: str = Field(default="", description="Semicolon-separated tags")

class CreateWorkItemOutputs(BaseModel):
    success: bool = Field(description="Whether creation succeeded")
    work_item: Optional[WorkItemRef] = Field(description="Created work item")
    formatted: str = Field(description="Human-readable summary")


class UpdateWorkItemInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    work_item_id: int = Field(description="Work item ID to update")
    title: Optional[str] = Field(default=None, description="New title")
    description: Optional[str] = Field(default=None, description="New description")
    state: Optional[str] = Field(default=None, description="New state")
    assigned_to: Optional[str] = Field(default=None, description="New assignee")
    iteration_path: Optional[str] = Field(default=None, description="New iteration")
    area_path: Optional[str] = Field(default=None, description="New area path")
    priority: Optional[int] = Field(default=None, description="New priority (1-4)")
    tags: Optional[str] = Field(default=None, description="New tags")

class UpdateWorkItemOutputs(BaseModel):
    success: bool = Field(description="Whether update succeeded")
    work_item: Optional[WorkItemRef] = Field(description="Updated work item")
    formatted: str = Field(description="Human-readable summary")


class AddWorkItemCommentInputs(BaseModel):
    project: str = Field(description="Project name or ID")
    work_item_id: int = Field(description="Work item ID")
    text: str = Field(description="Comment text (HTML)")

class AddWorkItemCommentOutputs(BaseModel):
    success: bool = Field(description="Whether comment was added")
    comment_id: int = Field(default=0, description="Created comment ID")
    formatted: str = Field(description="Human-readable summary")


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_projects(data: Dict[str, Any]) -> str:
    items = data.get("projects") or []
    if not items:
        return "No Azure DevOps projects found."
    lines = [f"Found {len(items)} project(s):"]
    for i, p in enumerate(items, 1):
        lines.append(f"  {i}. {p.get('name', '?')} — {p.get('description', '')[:80]}")
    return "\n".join(lines)


def _fmt_teams(data: Dict[str, Any]) -> str:
    items = data.get("teams") or []
    if not items:
        return "No teams found."
    lines = [f"Found {len(items)} team(s):"]
    for t in items:
        lines.append(f"  - {t.get('name', '?')}")
    return "\n".join(lines)


def _fmt_members(data: Dict[str, Any]) -> str:
    items = data.get("members") or []
    if not items:
        return "No team members found."
    lines = [f"Found {len(items)} member(s):"]
    for m in items:
        admin = " (admin)" if m.get("is_team_admin") else ""
        lines.append(f"  - {m.get('display_name', '?')} <{m.get('unique_name', '')}>{ admin}")
    return "\n".join(lines)


def _fmt_work_items(data: Dict[str, Any]) -> str:
    items = data.get("work_items") or []
    if not items:
        return "No work items found."
    lines = [f"Found {len(items)} work item(s):"]
    for wi in items:
        assignee = wi.get("assigned_to") or "Unassigned"
        lines.append(f"  #{wi.get('id', 0)} [{wi.get('work_item_type', '?')}] {wi.get('title', '?')} "
                     f"— {wi.get('state', '?')} (assigned: {assignee})")
    return "\n".join(lines)


def _fmt_single_work_item(data: Dict[str, Any]) -> str:
    wi = data.get("work_item")
    if not wi:
        return "Work item not found."
    lines = [
        f"Work Item #{wi.get('id', 0)}: {wi.get('title', '?')}",
        f"  Type: {wi.get('work_item_type', '?')}  State: {wi.get('state', '?')}",
        f"  Assigned to: {wi.get('assigned_to') or 'Unassigned'}",
        f"  Iteration: {wi.get('iteration_path', '')}  Area: {wi.get('area_path', '')}",
        f"  Priority: {wi.get('priority', 0)}  Tags: {wi.get('tags', '')}",
    ]
    desc = wi.get("description") or ""
    if desc:
        lines.append(f"  Description: {desc[:500]}")
    return "\n".join(lines)


def _fmt_iterations(data: Dict[str, Any]) -> str:
    items = data.get("iterations") or []
    if not items:
        return "No iterations found."
    lines = [f"Found {len(items)} iteration(s):"]
    for it in items:
        tf = it.get("time_frame") or ""
        dates = ""
        if it.get("start_date") and it.get("finish_date"):
            dates = f" ({it['start_date'][:10]} to {it['finish_date'][:10]})"
        lines.append(f"  - {it.get('name', '?')}{dates} [{tf}]  id={it.get('id', '')}")
    return "\n".join(lines)


def _fmt_boards(data: Dict[str, Any]) -> str:
    items = data.get("boards") or []
    if not items:
        return "No boards found."
    return "Boards: " + ", ".join(b.get("name", "?") for b in items)


def _fmt_columns(data: Dict[str, Any]) -> str:
    items = data.get("columns") or []
    if not items:
        return "No columns found."
    lines = [f"Board columns ({len(items)}):"]
    for c in items:
        limit = f" (limit {c.get('item_limit')})" if c.get("item_limit") else ""
        lines.append(f"  - {c.get('name', '?')}{limit}")
    return "\n".join(lines)


def _fmt_repos(data: Dict[str, Any]) -> str:
    items = data.get("repos") or []
    if not items:
        return "No repositories found."
    lines = [f"Found {len(items)} repo(s):"]
    for r in items:
        lines.append(f"  - {r.get('name', '?')} ({r.get('default_branch', '')})")
    return "\n".join(lines)


def _fmt_pull_requests(data: Dict[str, Any]) -> str:
    items = data.get("pull_requests") or []
    if not items:
        return "No pull requests found."
    lines = [f"Found {len(items)} PR(s):"]
    for pr in items:
        lines.append(f"  #{pr.get('id', 0)} {pr.get('title', '?')} [{pr.get('status', '?')}] "
                     f"by {pr.get('created_by', '?')} — {pr.get('source_branch', '')} → {pr.get('target_branch', '')}")
    return "\n".join(lines)


def _fmt_pipelines(data: Dict[str, Any]) -> str:
    items = data.get("pipelines") or []
    if not items:
        return "No pipelines found."
    lines = [f"Found {len(items)} pipeline(s):"]
    for p in items:
        lines.append(f"  - {p.get('name', '?')} (id={p.get('id', 0)})")
    return "\n".join(lines)


def _fmt_pipeline_runs(data: Dict[str, Any]) -> str:
    items = data.get("runs") or []
    if not items:
        return "No pipeline runs found."
    lines = [f"Found {len(items)} run(s):"]
    for r in items:
        lines.append(f"  Run #{r.get('id', 0)} state={r.get('state', '?')} result={r.get('result', '?')} "
                     f"created={r.get('created_date', '')[:16]}")
    return "\n".join(lines)


def _fmt_mutation(op_name: str, data: Dict[str, Any]) -> str:
    if data.get("error"):
        return f"DevOps {op_name} failed: {data['error']}"
    wi = data.get("work_item")
    if wi:
        return f"DevOps {op_name} succeeded: #{wi.get('id', 0)} {wi.get('title', '')}"
    return f"DevOps {op_name} succeeded."


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def list_devops_projects_tool(top: int = 100, connection_id: int = 0, user_id: str = "system") -> dict:
    """List Azure DevOps projects accessible to the user."""
    r = await invoke_m365_graph("list_devops_projects", user_id, connection_id or None, {"top": top})
    r["formatted"] = _fmt_projects(r)
    return r


async def list_devops_teams_tool(project: str, top: int = 100, connection_id: int = 0, user_id: str = "system") -> dict:
    """List teams in an Azure DevOps project."""
    r = await invoke_m365_graph("list_devops_teams", user_id, connection_id or None, {"project": project, "top": top})
    r["formatted"] = _fmt_teams(r)
    return r


async def list_devops_team_members_tool(project: str, team: str, top: int = 100, connection_id: int = 0, user_id: str = "system") -> dict:
    """List members of an Azure DevOps team."""
    r = await invoke_m365_graph("list_devops_team_members", user_id, connection_id or None, {"project": project, "team": team, "top": top})
    r["formatted"] = _fmt_members(r)
    return r


async def query_devops_work_items_tool(project: str, wiql: str, top: int = 200, connection_id: int = 0, user_id: str = "system") -> dict:
    """Query Azure DevOps work items using WIQL (Work Item Query Language)."""
    r = await invoke_m365_graph("query_devops_work_items", user_id, connection_id or None, {"project": project, "wiql": wiql, "top": top})
    r["formatted"] = _fmt_work_items(r)
    return r


async def get_devops_work_item_tool(project: str, work_item_id: int, connection_id: int = 0, user_id: str = "system") -> dict:
    """Get detailed information about a single Azure DevOps work item."""
    r = await invoke_m365_graph("get_devops_work_item", user_id, connection_id or None, {"project": project, "work_item_id": work_item_id})
    r["formatted"] = _fmt_single_work_item(r)
    return r


async def list_devops_iterations_tool(project: str, team: str = "", connection_id: int = 0, user_id: str = "system") -> dict:
    """List iterations (sprints) in an Azure DevOps project."""
    r = await invoke_m365_graph("list_devops_iterations", user_id, connection_id or None, {"project": project, "team": team})
    r["formatted"] = _fmt_iterations(r)
    return r


async def get_devops_iteration_work_items_tool(project: str, iteration_id: str, team: str = "", connection_id: int = 0, user_id: str = "system") -> dict:
    """Get work items assigned to a specific iteration (sprint)."""
    r = await invoke_m365_graph("get_devops_iteration_work_items", user_id, connection_id or None, {"project": project, "iteration_id": iteration_id, "team": team})
    r["formatted"] = _fmt_work_items(r)
    return r


async def list_devops_boards_tool(project: str, team: str = "", connection_id: int = 0, user_id: str = "system") -> dict:
    """List boards in an Azure DevOps project."""
    r = await invoke_m365_graph("list_devops_boards", user_id, connection_id or None, {"project": project, "team": team})
    r["formatted"] = _fmt_boards(r)
    return r


async def get_devops_board_columns_tool(project: str, board: str, team: str = "", connection_id: int = 0, user_id: str = "system") -> dict:
    """Get columns for a specific board."""
    r = await invoke_m365_graph("get_devops_board_columns", user_id, connection_id or None, {"project": project, "board": board, "team": team})
    r["formatted"] = _fmt_columns(r)
    return r


async def list_devops_repos_tool(project: str = "", connection_id: int = 0, user_id: str = "system") -> dict:
    """List Git repositories in an Azure DevOps project."""
    r = await invoke_m365_graph("list_devops_repos", user_id, connection_id or None, {"project": project})
    r["formatted"] = _fmt_repos(r)
    return r


async def list_devops_pull_requests_tool(project: str, status: str = "active", top: int = 50, connection_id: int = 0, user_id: str = "system") -> dict:
    """List pull requests in an Azure DevOps project."""
    r = await invoke_m365_graph("list_devops_pull_requests", user_id, connection_id or None, {"project": project, "status": status, "top": top})
    r["formatted"] = _fmt_pull_requests(r)
    return r


async def list_devops_pipelines_tool(project: str, top: int = 50, connection_id: int = 0, user_id: str = "system") -> dict:
    """List pipelines in an Azure DevOps project."""
    r = await invoke_m365_graph("list_devops_pipelines", user_id, connection_id or None, {"project": project, "top": top})
    r["formatted"] = _fmt_pipelines(r)
    return r


async def get_devops_pipeline_runs_tool(project: str, pipeline_id: int, top: int = 20, connection_id: int = 0, user_id: str = "system") -> dict:
    """Get recent runs for a specific pipeline."""
    r = await invoke_m365_graph("get_devops_pipeline_runs", user_id, connection_id or None, {"project": project, "pipeline_id": pipeline_id, "top": top})
    r["formatted"] = _fmt_pipeline_runs(r)
    return r


async def create_devops_work_item_tool(
    project: str, work_item_type: str, title: str,
    description: str = "", assigned_to: str = "", iteration_path: str = "",
    area_path: str = "", priority: int = 0, tags: str = "",
    connection_id: int = 0, user_id: str = "system",
) -> dict:
    """Create a new work item in Azure DevOps."""
    p: Dict[str, Any] = {"project": project, "work_item_type": work_item_type, "title": title}
    if description:
        p["description"] = description
    if assigned_to:
        p["assigned_to"] = assigned_to
    if iteration_path:
        p["iteration_path"] = iteration_path
    if area_path:
        p["area_path"] = area_path
    if priority:
        p["priority"] = priority
    if tags:
        p["tags"] = tags
    r = await invoke_m365_graph("create_devops_work_item", user_id, connection_id or None, p)
    r["formatted"] = _fmt_mutation("create_work_item", r)
    return r


async def update_devops_work_item_tool(
    project: str, work_item_id: int,
    title: str = None, description: str = None, state: str = None,
    assigned_to: str = None, iteration_path: str = None, area_path: str = None,
    priority: int = None, tags: str = None,
    connection_id: int = 0, user_id: str = "system",
) -> dict:
    """Update an existing Azure DevOps work item."""
    p: Dict[str, Any] = {"project": project, "work_item_id": work_item_id}
    for key, val in [("title", title), ("description", description), ("state", state),
                     ("assigned_to", assigned_to), ("iteration_path", iteration_path),
                     ("area_path", area_path), ("priority", priority), ("tags", tags)]:
        if val is not None:
            p[key] = val
    r = await invoke_m365_graph("update_devops_work_item", user_id, connection_id or None, p)
    r["formatted"] = _fmt_mutation("update_work_item", r)
    return r


async def add_devops_work_item_comment_tool(
    project: str, work_item_id: int, text: str,
    connection_id: int = 0, user_id: str = "system",
) -> dict:
    """Add a comment to an Azure DevOps work item."""
    r = await invoke_m365_graph("add_devops_work_item_comment", user_id, connection_id or None,
                                {"project": project, "work_item_id": work_item_id, "text": text})
    r["formatted"] = _fmt_mutation("add_comment", r)
    return r


# ---------------------------------------------------------------------------
# Action I/O Registry registrations
# ---------------------------------------------------------------------------

_READ_TOOLS = [
    ("list_devops_projects", "devops", "List Azure DevOps projects", ListProjectsInputs, ListProjectsOutputs, list_devops_projects_tool),
    ("list_devops_teams", "devops", "List teams in a DevOps project", ListTeamsInputs, ListTeamsOutputs, list_devops_teams_tool),
    ("list_devops_team_members", "devops", "List members of a DevOps team", ListTeamMembersInputs, ListTeamMembersOutputs, list_devops_team_members_tool),
    ("query_devops_work_items", "devops", "Query work items via WIQL", QueryWorkItemsInputs, QueryWorkItemsOutputs, query_devops_work_items_tool),
    ("get_devops_work_item", "devops", "Get details of a single work item", GetWorkItemInputs, GetWorkItemOutputs, get_devops_work_item_tool),
    ("list_devops_iterations", "devops", "List iterations (sprints)", ListIterationsInputs, ListIterationsOutputs, list_devops_iterations_tool),
    ("get_devops_iteration_work_items", "devops", "Get work items in a sprint", GetIterationWorkItemsInputs, GetIterationWorkItemsOutputs, get_devops_iteration_work_items_tool),
    ("list_devops_boards", "devops", "List boards in a project", ListBoardsInputs, ListBoardsOutputs, list_devops_boards_tool),
    ("get_devops_board_columns", "devops", "Get columns for a board", GetBoardColumnsInputs, GetBoardColumnsOutputs, get_devops_board_columns_tool),
    ("list_devops_repos", "devops", "List Git repos in a project", ListReposInputs, ListReposOutputs, list_devops_repos_tool),
    ("list_devops_pull_requests", "devops", "List pull requests", ListPullRequestsInputs, ListPullRequestsOutputs, list_devops_pull_requests_tool),
    ("list_devops_pipelines", "devops", "List pipelines in a project", ListPipelinesInputs, ListPipelinesOutputs, list_devops_pipelines_tool),
    ("get_devops_pipeline_runs", "devops", "Get recent pipeline runs", GetPipelineRunsInputs, GetPipelineRunsOutputs, get_devops_pipeline_runs_tool),
]

_WRITE_TOOLS = [
    ("create_devops_work_item", "devops", "Create a new work item", CreateWorkItemInputs, CreateWorkItemOutputs, create_devops_work_item_tool),
    ("update_devops_work_item", "devops", "Update an existing work item", UpdateWorkItemInputs, UpdateWorkItemOutputs, update_devops_work_item_tool),
    ("add_devops_work_item_comment", "devops", "Add a comment to a work item", AddWorkItemCommentInputs, AddWorkItemCommentOutputs, add_devops_work_item_comment_tool),
]

for _name, _cat, _desc, _inp, _out, _fn in _READ_TOOLS + _WRITE_TOOLS:
    register_action(
        name=_name,
        category=_cat,
        description=_desc,
        inputs_model=_inp,
        outputs_model=_out,
        tool_function=_fn,
    )
