"""
Microsoft Azure DevOps REST API mixin for MicrosoftGraphProvider.

Uses the same Azure AD OAuth token as Microsoft Graph but targets
https://dev.azure.com/{organization}/_apis/ endpoints (API version 7.1).
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEVOPS_BASE = "https://dev.azure.com"
DEVOPS_API_VERSION = "7.1"
DEVOPS_TIMEOUT = 30


def _wi_to_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an Azure DevOps work item response."""
    fields = item.get("fields") or {}
    return {
        "id": item.get("id", 0),
        "rev": item.get("rev", 0),
        "url": item.get("url", ""),
        "title": fields.get("System.Title", ""),
        "state": fields.get("System.State", ""),
        "work_item_type": fields.get("System.WorkItemType", ""),
        "assigned_to": (fields.get("System.AssignedTo") or {}).get("displayName", ""),
        "assigned_to_email": (fields.get("System.AssignedTo") or {}).get("uniqueName", ""),
        "created_date": fields.get("System.CreatedDate", ""),
        "changed_date": fields.get("System.ChangedDate", ""),
        "area_path": fields.get("System.AreaPath", ""),
        "iteration_path": fields.get("System.IterationPath", ""),
        "priority": fields.get("Microsoft.VSTS.Common.Priority", 0),
        "description": fields.get("System.Description", ""),
        "tags": fields.get("System.Tags", ""),
        "story_points": fields.get("Microsoft.VSTS.Scheduling.StoryPoints"),
        "remaining_work": fields.get("Microsoft.VSTS.Scheduling.RemainingWork"),
    }


class MicrosoftDevOpsMixin:
    """Adds Azure DevOps methods; requires _headers(access_token) from the host class."""

    def _devops_headers(self, access_token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _devops_get(
        self,
        access_token: str,
        org: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        project: str = "",
    ) -> Dict[str, Any]:
        base = f"{DEVOPS_BASE}/{org}"
        if project:
            base = f"{base}/{project}"
        url = f"{base}/{path.lstrip('/')}"
        p = dict(params or {})
        p.setdefault("api-version", DEVOPS_API_VERSION)
        async with httpx.AsyncClient(timeout=DEVOPS_TIMEOUT) as client:
            resp = await client.get(url, headers=self._devops_headers(access_token), params=p)
            resp.raise_for_status()
            return resp.json()

    async def _devops_post(
        self,
        access_token: str,
        org: str,
        path: str,
        json_body: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        project: str = "",
        content_type: str = "application/json",
    ) -> Dict[str, Any]:
        base = f"{DEVOPS_BASE}/{org}"
        if project:
            base = f"{base}/{project}"
        url = f"{base}/{path.lstrip('/')}"
        p = dict(params or {})
        p.setdefault("api-version", DEVOPS_API_VERSION)
        headers = self._devops_headers(access_token)
        headers["Content-Type"] = content_type
        async with httpx.AsyncClient(timeout=DEVOPS_TIMEOUT) as client:
            resp = await client.post(url, headers=headers, json=json_body, params=p)
            resp.raise_for_status()
            return resp.json() if resp.content else {}

    async def _devops_patch(
        self,
        access_token: str,
        org: str,
        path: str,
        json_body: Any,
        params: Optional[Dict[str, Any]] = None,
        project: str = "",
        content_type: str = "application/json-patch+json",
    ) -> Dict[str, Any]:
        base = f"{DEVOPS_BASE}/{org}"
        if project:
            base = f"{base}/{project}"
        url = f"{base}/{path.lstrip('/')}"
        p = dict(params or {})
        p.setdefault("api-version", DEVOPS_API_VERSION)
        headers = self._devops_headers(access_token)
        headers["Content-Type"] = content_type
        async with httpx.AsyncClient(timeout=DEVOPS_TIMEOUT) as client:
            resp = await client.patch(url, headers=headers, json=json_body, params=p)
            resp.raise_for_status()
            return resp.json() if resp.content else {}

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    async def list_devops_projects(
        self, access_token: str, org: str, top: int = 100
    ) -> Dict[str, Any]:
        try:
            data = await self._devops_get(
                access_token, org, "_apis/projects", {"$top": top}
            )
            projects = []
            for p in data.get("value", []):
                projects.append({
                    "id": p.get("id", ""),
                    "name": p.get("name", ""),
                    "description": p.get("description", ""),
                    "state": p.get("state", ""),
                    "url": p.get("url", ""),
                })
            return {"projects": projects, "count": data.get("count", len(projects))}
        except Exception as e:
            logger.exception("list_devops_projects: %s", e)
            return {"projects": [], "count": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # Teams & Members
    # ------------------------------------------------------------------

    async def list_devops_teams(
        self, access_token: str, org: str, project: str, top: int = 100
    ) -> Dict[str, Any]:
        try:
            data = await self._devops_get(
                access_token, org, f"_apis/projects/{project}/teams", {"$top": top}
            )
            teams = []
            for t in data.get("value", []):
                teams.append({
                    "id": t.get("id", ""),
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "url": t.get("url", ""),
                })
            return {"teams": teams, "count": len(teams)}
        except Exception as e:
            logger.exception("list_devops_teams: %s", e)
            return {"teams": [], "count": 0, "error": str(e)}

    async def list_devops_team_members(
        self, access_token: str, org: str, project: str, team: str, top: int = 100
    ) -> Dict[str, Any]:
        try:
            data = await self._devops_get(
                access_token, org,
                f"_apis/projects/{project}/teams/{team}/members",
                {"$top": top},
            )
            members = []
            for m in data.get("value", []):
                identity = m.get("identity") or {}
                members.append({
                    "id": identity.get("id", ""),
                    "display_name": identity.get("displayName", ""),
                    "unique_name": identity.get("uniqueName", ""),
                    "is_team_admin": m.get("isTeamAdmin", False),
                })
            return {"members": members, "count": len(members)}
        except Exception as e:
            logger.exception("list_devops_team_members: %s", e)
            return {"members": [], "count": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # Work Items
    # ------------------------------------------------------------------

    async def query_devops_work_items(
        self, access_token: str, org: str, project: str, wiql: str, top: int = 200
    ) -> Dict[str, Any]:
        """Run a WIQL query and hydrate the returned work item IDs."""
        try:
            query_result = await self._devops_post(
                access_token, org, "_apis/wit/wiql",
                json_body={"query": wiql}, params={"$top": top},
                project=project,
            )
            wi_refs = query_result.get("workItems") or []
            if not wi_refs:
                return {"work_items": [], "count": 0}
            ids = [str(w["id"]) for w in wi_refs[:top]]
            batched = []
            for i in range(0, len(ids), 200):
                chunk = ids[i:i + 200]
                id_str = ",".join(chunk)
                items_data = await self._devops_get(
                    access_token, org,
                    "_apis/wit/workitems",
                    {"ids": id_str, "$expand": "fields"},
                    project=project,
                )
                for item in items_data.get("value", []):
                    batched.append(_wi_to_dict(item))
            return {"work_items": batched, "count": len(batched)}
        except Exception as e:
            logger.exception("query_devops_work_items: %s", e)
            return {"work_items": [], "count": 0, "error": str(e)}

    async def get_devops_work_item(
        self, access_token: str, org: str, project: str, work_item_id: int
    ) -> Dict[str, Any]:
        try:
            data = await self._devops_get(
                access_token, org,
                f"_apis/wit/workitems/{work_item_id}",
                {"$expand": "all"},
                project=project,
            )
            return {"work_item": _wi_to_dict(data)}
        except Exception as e:
            logger.exception("get_devops_work_item: %s", e)
            return {"work_item": None, "error": str(e)}

    # ------------------------------------------------------------------
    # Iterations (Sprints)
    # ------------------------------------------------------------------

    async def list_devops_iterations(
        self, access_token: str, org: str, project: str, team: str = ""
    ) -> Dict[str, Any]:
        try:
            path = f"_apis/work/teamsettings/iterations"
            if team:
                path = f"{project}/{team}/_apis/work/teamsettings/iterations"
                data = await self._devops_get(access_token, org, path)
            else:
                data = await self._devops_get(
                    access_token, org, path, project=project
                )
            iterations = []
            for it in data.get("value", []):
                attrs = it.get("attributes") or {}
                iterations.append({
                    "id": it.get("id", ""),
                    "name": it.get("name", ""),
                    "path": it.get("path", ""),
                    "start_date": attrs.get("startDate", ""),
                    "finish_date": attrs.get("finishDate", ""),
                    "time_frame": attrs.get("timeFrame", ""),
                    "url": it.get("url", ""),
                })
            return {"iterations": iterations, "count": len(iterations)}
        except Exception as e:
            logger.exception("list_devops_iterations: %s", e)
            return {"iterations": [], "count": 0, "error": str(e)}

    async def get_devops_iteration_work_items(
        self, access_token: str, org: str, project: str, iteration_id: str, team: str = ""
    ) -> Dict[str, Any]:
        try:
            path = f"_apis/work/teamsettings/iterations/{iteration_id}/workitems"
            if team:
                path = f"{project}/{team}/_apis/work/teamsettings/iterations/{iteration_id}/workitems"
                data = await self._devops_get(access_token, org, path)
            else:
                data = await self._devops_get(
                    access_token, org, path, project=project
                )
            wi_refs = data.get("workItemRelations") or []
            target_ids = []
            for rel in wi_refs:
                target = rel.get("target") or {}
                tid = target.get("id")
                if tid:
                    target_ids.append(str(tid))
            if not target_ids:
                return {"work_items": [], "count": 0}
            batched: List[Dict[str, Any]] = []
            for i in range(0, len(target_ids), 200):
                chunk = target_ids[i:i + 200]
                id_str = ",".join(chunk)
                items_data = await self._devops_get(
                    access_token, org,
                    "_apis/wit/workitems",
                    {"ids": id_str, "$expand": "fields"},
                    project=project,
                )
                for item in items_data.get("value", []):
                    batched.append(_wi_to_dict(item))
            return {"work_items": batched, "count": len(batched)}
        except Exception as e:
            logger.exception("get_devops_iteration_work_items: %s", e)
            return {"work_items": [], "count": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # Boards
    # ------------------------------------------------------------------

    async def list_devops_boards(
        self, access_token: str, org: str, project: str, team: str = ""
    ) -> Dict[str, Any]:
        try:
            path = "_apis/work/boards"
            if team:
                path = f"{project}/{team}/_apis/work/boards"
                data = await self._devops_get(access_token, org, path)
            else:
                data = await self._devops_get(
                    access_token, org, path, project=project
                )
            boards = []
            for b in data.get("value", []):
                boards.append({
                    "id": b.get("id", ""),
                    "name": b.get("name", ""),
                    "url": b.get("url", ""),
                })
            return {"boards": boards, "count": len(boards)}
        except Exception as e:
            logger.exception("list_devops_boards: %s", e)
            return {"boards": [], "count": 0, "error": str(e)}

    async def get_devops_board_columns(
        self, access_token: str, org: str, project: str, board: str, team: str = ""
    ) -> Dict[str, Any]:
        try:
            path = f"_apis/work/boards/{board}/columns"
            if team:
                path = f"{project}/{team}/_apis/work/boards/{board}/columns"
                data = await self._devops_get(access_token, org, path)
            else:
                data = await self._devops_get(
                    access_token, org, path, project=project
                )
            columns = []
            for c in data.get("value", []):
                columns.append({
                    "id": c.get("id", ""),
                    "name": c.get("name", ""),
                    "item_limit": c.get("itemLimit", 0),
                    "is_split": c.get("isSplit", False),
                    "column_type": c.get("columnType", ""),
                    "state_mappings": c.get("stateMappings") or {},
                })
            return {"columns": columns, "count": len(columns)}
        except Exception as e:
            logger.exception("get_devops_board_columns: %s", e)
            return {"columns": [], "count": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # Git Repos
    # ------------------------------------------------------------------

    async def list_devops_repos(
        self, access_token: str, org: str, project: str = ""
    ) -> Dict[str, Any]:
        try:
            data = await self._devops_get(
                access_token, org, "_apis/git/repositories",
                project=project,
            )
            repos = []
            for r in data.get("value", []):
                proj = r.get("project") or {}
                repos.append({
                    "id": r.get("id", ""),
                    "name": r.get("name", ""),
                    "default_branch": r.get("defaultBranch", ""),
                    "web_url": r.get("webUrl", ""),
                    "project_name": proj.get("name", ""),
                    "size": r.get("size", 0),
                })
            return {"repos": repos, "count": len(repos)}
        except Exception as e:
            logger.exception("list_devops_repos: %s", e)
            return {"repos": [], "count": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # Pull Requests
    # ------------------------------------------------------------------

    async def list_devops_pull_requests(
        self, access_token: str, org: str, project: str,
        status: str = "active", top: int = 50,
    ) -> Dict[str, Any]:
        try:
            data = await self._devops_get(
                access_token, org, "_apis/git/pullrequests",
                {"searchCriteria.status": status, "$top": top},
                project=project,
            )
            prs = []
            for pr in data.get("value", []):
                created_by = pr.get("createdBy") or {}
                repo = pr.get("repository") or {}
                prs.append({
                    "id": pr.get("pullRequestId", 0),
                    "title": pr.get("title", ""),
                    "status": pr.get("status", ""),
                    "created_by": created_by.get("displayName", ""),
                    "created_date": pr.get("creationDate", ""),
                    "source_branch": pr.get("sourceRefName", ""),
                    "target_branch": pr.get("targetRefName", ""),
                    "repo_name": repo.get("name", ""),
                    "url": pr.get("url", ""),
                })
            return {"pull_requests": prs, "count": len(prs)}
        except Exception as e:
            logger.exception("list_devops_pull_requests: %s", e)
            return {"pull_requests": [], "count": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # Pipelines
    # ------------------------------------------------------------------

    async def list_devops_pipelines(
        self, access_token: str, org: str, project: str, top: int = 50
    ) -> Dict[str, Any]:
        try:
            data = await self._devops_get(
                access_token, org, "_apis/pipelines",
                {"$top": top}, project=project,
            )
            pipelines = []
            for p in data.get("value", []):
                pipelines.append({
                    "id": p.get("id", 0),
                    "name": p.get("name", ""),
                    "folder": p.get("folder", ""),
                    "url": p.get("url", ""),
                })
            return {"pipelines": pipelines, "count": len(pipelines)}
        except Exception as e:
            logger.exception("list_devops_pipelines: %s", e)
            return {"pipelines": [], "count": 0, "error": str(e)}

    async def get_devops_pipeline_runs(
        self, access_token: str, org: str, project: str,
        pipeline_id: int, top: int = 20
    ) -> Dict[str, Any]:
        try:
            data = await self._devops_get(
                access_token, org,
                f"_apis/pipelines/{pipeline_id}/runs",
                {"$top": top}, project=project,
            )
            runs = []
            for r in data.get("value", []):
                runs.append({
                    "id": r.get("id", 0),
                    "name": r.get("name", ""),
                    "state": r.get("state", ""),
                    "result": r.get("result", ""),
                    "created_date": r.get("createdDate", ""),
                    "finished_date": r.get("finishedDate", ""),
                    "url": r.get("url", ""),
                })
            return {"runs": runs, "count": len(runs)}
        except Exception as e:
            logger.exception("get_devops_pipeline_runs: %s", e)
            return {"runs": [], "count": 0, "error": str(e)}

    # ------------------------------------------------------------------
    # Write Operations
    # ------------------------------------------------------------------

    async def create_devops_work_item(
        self, access_token: str, org: str, project: str,
        work_item_type: str, title: str,
        description: str = "", assigned_to: str = "",
        area_path: str = "", iteration_path: str = "",
        priority: Optional[int] = None, tags: str = "",
    ) -> Dict[str, Any]:
        """Create a new work item using JSON Patch operations."""
        try:
            ops: List[Dict[str, Any]] = [
                {"op": "add", "path": "/fields/System.Title", "value": title},
            ]
            if description:
                ops.append({"op": "add", "path": "/fields/System.Description", "value": description})
            if assigned_to:
                ops.append({"op": "add", "path": "/fields/System.AssignedTo", "value": assigned_to})
            if area_path:
                ops.append({"op": "add", "path": "/fields/System.AreaPath", "value": area_path})
            if iteration_path:
                ops.append({"op": "add", "path": "/fields/System.IterationPath", "value": iteration_path})
            if priority is not None:
                ops.append({"op": "add", "path": "/fields/Microsoft.VSTS.Common.Priority", "value": priority})
            if tags:
                ops.append({"op": "add", "path": "/fields/System.Tags", "value": tags})

            data = await self._devops_post(
                access_token, org,
                f"_apis/wit/workitems/${work_item_type}",
                json_body=ops, project=project,
                content_type="application/json-patch+json",
            )
            return {"success": True, "work_item": _wi_to_dict(data)}
        except Exception as e:
            logger.exception("create_devops_work_item: %s", e)
            return {"success": False, "work_item": None, "error": str(e)}

    async def update_devops_work_item(
        self, access_token: str, org: str, project: str,
        work_item_id: int, title: Optional[str] = None,
        description: Optional[str] = None, state: Optional[str] = None,
        assigned_to: Optional[str] = None, area_path: Optional[str] = None,
        iteration_path: Optional[str] = None, priority: Optional[int] = None,
        tags: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a work item using JSON Patch operations."""
        try:
            ops: List[Dict[str, Any]] = []
            if title is not None:
                ops.append({"op": "replace", "path": "/fields/System.Title", "value": title})
            if description is not None:
                ops.append({"op": "replace", "path": "/fields/System.Description", "value": description})
            if state is not None:
                ops.append({"op": "replace", "path": "/fields/System.State", "value": state})
            if assigned_to is not None:
                ops.append({"op": "replace", "path": "/fields/System.AssignedTo", "value": assigned_to})
            if area_path is not None:
                ops.append({"op": "replace", "path": "/fields/System.AreaPath", "value": area_path})
            if iteration_path is not None:
                ops.append({"op": "replace", "path": "/fields/System.IterationPath", "value": iteration_path})
            if priority is not None:
                ops.append({"op": "replace", "path": "/fields/Microsoft.VSTS.Common.Priority", "value": priority})
            if tags is not None:
                ops.append({"op": "replace", "path": "/fields/System.Tags", "value": tags})
            if not ops:
                return {"success": True, "work_item": None}
            data = await self._devops_patch(
                access_token, org,
                f"_apis/wit/workitems/{work_item_id}",
                json_body=ops, project=project,
            )
            return {"success": True, "work_item": _wi_to_dict(data)}
        except Exception as e:
            logger.exception("update_devops_work_item: %s", e)
            return {"success": False, "work_item": None, "error": str(e)}

    async def add_devops_work_item_comment(
        self, access_token: str, org: str, project: str,
        work_item_id: int, text: str,
    ) -> Dict[str, Any]:
        try:
            data = await self._devops_post(
                access_token, org,
                f"_apis/wit/workitems/{work_item_id}/comments",
                json_body={"text": text},
                project=project,
                params={"api-version": "7.1-preview.4"},
            )
            return {
                "success": True,
                "comment_id": data.get("id", 0),
                "text": data.get("text", ""),
            }
        except Exception as e:
            logger.exception("add_devops_work_item_comment: %s", e)
            return {"success": False, "comment_id": 0, "error": str(e)}
