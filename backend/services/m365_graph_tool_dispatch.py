"""
Dispatch Microsoft Graph tool operations (To Do, OneDrive, OneNote, Planner, Azure DevOps)
via connections-service gRPC.
"""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# operation -> enabled_services key (must match m365_oauth_utils / provider_metadata)
M365_OP_SERVICE: Dict[str, str] = {
    "list_todo_lists": "todo",
    "get_todo_tasks": "todo",
    "create_todo_task": "todo",
    "update_todo_task": "todo",
    "delete_todo_task": "todo",
    "list_drive_items": "files",
    "get_drive_item": "files",
    "search_drive": "files",
    "get_file_content": "files",
    "upload_file": "files",
    "create_drive_folder": "files",
    "move_drive_item": "files",
    "delete_drive_item": "files",
    "list_onenote_notebooks": "onenote",
    "list_onenote_sections": "onenote",
    "list_onenote_pages": "onenote",
    "get_onenote_page_content": "onenote",
    "create_onenote_page": "onenote",
    "list_planner_plans": "planner",
    "get_planner_tasks": "planner",
    "create_planner_task": "planner",
    "update_planner_task": "planner",
    "delete_planner_task": "planner",
    "list_devops_projects": "devops",
    "list_devops_teams": "devops",
    "list_devops_team_members": "devops",
    "query_devops_work_items": "devops",
    "get_devops_work_item": "devops",
    "list_devops_iterations": "devops",
    "get_devops_iteration_work_items": "devops",
    "list_devops_boards": "devops",
    "get_devops_board_columns": "devops",
    "list_devops_repos": "devops",
    "list_devops_pull_requests": "devops",
    "list_devops_pipelines": "devops",
    "get_devops_pipeline_runs": "devops",
    "create_devops_work_item": "devops",
    "update_devops_work_item": "devops",
    "add_devops_work_item_comment": "devops",
}


async def dispatch_m365_graph(
    client: Any,
    user_id: str,
    connection_id: Optional[int],
    operation: str,
    params: Dict[str, Any],
    rls_context: Optional[Dict[str, str]] = None,
) -> Tuple[bool, Any, str]:
    """
    Run one M365 Graph operation. Returns (success, data_or_none, error_message).
    """
    try:
        connections_service_pb2 = importlib.import_module("protos.connections_service_pb2")
    except ImportError:
        connections_service_pb2 = importlib.import_module("connections_service_pb2")

    op = (operation or "").strip()
    m365_svc = M365_OP_SERVICE.get(op)
    if not m365_svc:
        return False, None, f"Unknown M365 operation: {op}"

    await client.initialize()
    token, cid, prov = await client._ensure_token(
        connection_id,
        user_id,
        "microsoft",
        rls_context=rls_context,
        m365_service=m365_svc,
    )
    if not token:
        return False, None, "No valid connection, token, or requested M365 service is not enabled"

    p = params or {}

    async def _call(coro):
        return await client._with_reconnect(coro)

    try:
        if op == "list_todo_lists":

            async def _do():
                req = connections_service_pb2.ListTodoListsRequest(
                    access_token=token, provider=prov
                )
                return await client.stub.ListTodoLists(req, timeout=20.0)

            resp = await _call(_do)
            data = {
                "lists": [
                    {
                        "id": x.id,
                        "display_name": x.display_name,
                        "is_owner": x.is_owner,
                        "is_shared": x.is_shared,
                        "well_known_list_name": x.well_known_list_name or "",
                    }
                    for x in resp.lists
                ],
                "error": resp.error if resp.HasField("error") else None,
            }
            return True, data, ""

        if op == "get_todo_tasks":

            async def _do():
                req = connections_service_pb2.GetTodoTasksRequest(
                    access_token=token,
                    provider=prov,
                    list_id=p.get("list_id") or "",
                    top=int(p.get("top") or 50),
                )
                return await client.stub.GetTodoTasks(req, timeout=20.0)

            resp = await _call(_do)
            data = {
                "tasks": [
                    {
                        "id": x.id,
                        "list_id": x.list_id,
                        "title": x.title,
                        "status": x.status,
                        "body": x.body,
                        "due_datetime": x.due_datetime,
                        "importance": x.importance,
                    }
                    for x in resp.tasks
                ],
                "error": resp.error if resp.HasField("error") else None,
            }
            return True, data, ""

        if op == "create_todo_task":

            async def _do():
                req = connections_service_pb2.CreateTodoTaskRequest(
                    access_token=token,
                    provider=prov,
                    list_id=p.get("list_id") or "",
                    title=p.get("title") or "Task",
                    body=p.get("body") or "",
                    due_datetime=p.get("due_datetime") or "",
                    importance=p.get("importance") or "normal",
                )
                return await client.stub.CreateTodoTask(req, timeout=20.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return (
                resp.success,
                {"success": resp.success, "task_id": resp.task_id, "error": err or None},
                err,
            )

        if op == "update_todo_task":
            req = connections_service_pb2.UpdateTodoTaskRequest(
                access_token=token,
                provider=prov,
                list_id=p.get("list_id") or "",
                task_id=p.get("task_id") or "",
            )
            if "title" in p and p["title"] is not None:
                req.title = str(p["title"])
            if "body" in p and p["body"] is not None:
                req.body = str(p["body"])
            if "status" in p and p["status"] is not None:
                req.status = str(p["status"])
            if "due_datetime" in p and p["due_datetime"] is not None:
                req.due_datetime = str(p["due_datetime"])
            if "importance" in p and p["importance"] is not None:
                req.importance = str(p["importance"])

            async def _do():
                return await client.stub.UpdateTodoTask(req, timeout=20.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return resp.success, {"success": resp.success, "error": err or None}, err

        if op == "delete_todo_task":

            async def _do():
                req = connections_service_pb2.DeleteTodoTaskRequest(
                    access_token=token,
                    provider=prov,
                    list_id=p.get("list_id") or "",
                    task_id=p.get("task_id") or "",
                )
                return await client.stub.DeleteTodoTask(req, timeout=15.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return resp.success, {"success": resp.success, "error": err or None}, err

        if op == "list_drive_items":

            async def _do():
                req = connections_service_pb2.ListDriveItemsRequest(
                    access_token=token,
                    provider=prov,
                    parent_item_id=p.get("parent_item_id") or "",
                    top=int(p.get("top") or 50),
                )
                return await client.stub.ListDriveItems(req, timeout=20.0)

            resp = await _call(_do)
            data = {
                "items": [
                    {
                        "id": x.id,
                        "name": x.name,
                        "web_url": x.web_url,
                        "is_folder": x.is_folder,
                        "mime_type": x.mime_type,
                        "size": x.size,
                        "parent_id": x.parent_id,
                        "last_modified": x.last_modified,
                    }
                    for x in resp.items
                ],
                "error": resp.error if resp.HasField("error") else None,
            }
            return True, data, ""

        if op == "get_drive_item":

            async def _do():
                req = connections_service_pb2.GetDriveItemRequest(
                    access_token=token,
                    provider=prov,
                    item_id=p.get("item_id") or "",
                )
                return await client.stub.GetDriveItem(req, timeout=15.0)

            resp = await _call(_do)
            it = resp.item
            if not it:
                return False, None, resp.error or "Not found"
            return (
                True,
                {
                    "item": {
                        "id": it.id,
                        "name": it.name,
                        "web_url": it.web_url,
                        "is_folder": it.is_folder,
                        "mime_type": it.mime_type,
                        "size": it.size,
                        "parent_id": it.parent_id,
                        "last_modified": it.last_modified,
                    }
                },
                "",
            )

        if op == "search_drive":

            async def _do():
                req = connections_service_pb2.SearchDriveRequest(
                    access_token=token,
                    provider=prov,
                    query=p.get("query") or "",
                    top=int(p.get("top") or 25),
                )
                return await client.stub.SearchDrive(req, timeout=25.0)

            resp = await _call(_do)
            data = {
                "items": [
                    {
                        "id": x.id,
                        "name": x.name,
                        "web_url": x.web_url,
                        "is_folder": x.is_folder,
                        "mime_type": x.mime_type,
                        "size": x.size,
                        "parent_id": x.parent_id,
                        "last_modified": x.last_modified,
                    }
                    for x in resp.items
                ],
                "error": resp.error if resp.HasField("error") else None,
            }
            return True, data, ""

        if op == "get_file_content":

            async def _do():
                req = connections_service_pb2.GetFileContentRequest(
                    access_token=token,
                    provider=prov,
                    item_id=p.get("item_id") or "",
                )
                return await client.stub.GetFileContent(req, timeout=60.0)

            resp = await _call(_do)
            return (
                True,
                {
                    "content_base64": resp.content_base64,
                    "mime_type": resp.mime_type,
                    "error": resp.error if resp.HasField("error") else None,
                },
                "",
            )

        if op == "upload_file":

            async def _do():
                req = connections_service_pb2.UploadFileRequest(
                    access_token=token,
                    provider=prov,
                    parent_item_id=p.get("parent_item_id") or "",
                    name=p.get("name") or "file.bin",
                    content_base64=p.get("content_base64") or "",
                    mime_type=p.get("mime_type") or "application/octet-stream",
                )
                return await client.stub.UploadFile(req, timeout=120.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return (
                resp.success,
                {"success": resp.success, "item_id": resp.item_id, "error": err or None},
                err,
            )

        if op == "create_drive_folder":

            async def _do():
                req = connections_service_pb2.CreateDriveFolderRequest(
                    access_token=token,
                    provider=prov,
                    parent_item_id=p.get("parent_item_id") or "",
                    name=p.get("name") or "New folder",
                )
                return await client.stub.CreateDriveFolder(req, timeout=20.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return (
                resp.success,
                {"success": resp.success, "item_id": resp.item_id, "error": err or None},
                err,
            )

        if op == "move_drive_item":

            async def _do():
                req = connections_service_pb2.MoveDriveItemRequest(
                    access_token=token,
                    provider=prov,
                    item_id=p.get("item_id") or "",
                    new_parent_item_id=p.get("new_parent_item_id") or "",
                )
                return await client.stub.MoveDriveItem(req, timeout=20.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return resp.success, {"success": resp.success, "error": err or None}, err

        if op == "delete_drive_item":

            async def _do():
                req = connections_service_pb2.DeleteDriveItemRequest(
                    access_token=token,
                    provider=prov,
                    item_id=p.get("item_id") or "",
                )
                return await client.stub.DeleteDriveItem(req, timeout=15.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return resp.success, {"success": resp.success, "error": err or None}, err

        if op == "list_onenote_notebooks":

            async def _do():
                req = connections_service_pb2.ListOneNoteNotebooksRequest(
                    access_token=token, provider=prov
                )
                return await client.stub.ListOneNoteNotebooks(req, timeout=20.0)

            resp = await _call(_do)
            data = {
                "notebooks": [
                    {
                        "id": x.id,
                        "display_name": x.display_name,
                        "web_url": x.web_url,
                    }
                    for x in resp.notebooks
                ],
                "error": resp.error if resp.HasField("error") else None,
            }
            return True, data, ""

        if op == "list_onenote_sections":

            async def _do():
                req = connections_service_pb2.ListOneNoteSectionsRequest(
                    access_token=token,
                    provider=prov,
                    notebook_id=p.get("notebook_id") or "",
                )
                return await client.stub.ListOneNoteSections(req, timeout=20.0)

            resp = await _call(_do)
            data = {
                "sections": [
                    {
                        "id": x.id,
                        "display_name": x.display_name,
                        "notebook_id": x.notebook_id,
                        "web_url": x.web_url,
                    }
                    for x in resp.sections
                ],
                "error": resp.error if resp.HasField("error") else None,
            }
            return True, data, ""

        if op == "list_onenote_pages":

            async def _do():
                req = connections_service_pb2.ListOneNotePagesRequest(
                    access_token=token,
                    provider=prov,
                    section_id=p.get("section_id") or "",
                    top=int(p.get("top") or 50),
                )
                return await client.stub.ListOneNotePages(req, timeout=20.0)

            resp = await _call(_do)
            data = {
                "pages": [
                    {
                        "id": x.id,
                        "title": x.title,
                        "section_id": x.section_id,
                        "web_url": x.web_url,
                        "created_time": x.created_time,
                    }
                    for x in resp.pages
                ],
                "error": resp.error if resp.HasField("error") else None,
            }
            return True, data, ""

        if op == "get_onenote_page_content":

            async def _do():
                req = connections_service_pb2.GetOneNotePageContentRequest(
                    access_token=token,
                    provider=prov,
                    page_id=p.get("page_id") or "",
                )
                return await client.stub.GetOneNotePageContent(req, timeout=30.0)

            resp = await _call(_do)
            return (
                True,
                {
                    "html_content": resp.html_content,
                    "error": resp.error if resp.HasField("error") else None,
                },
                "",
            )

        if op == "create_onenote_page":

            async def _do():
                req = connections_service_pb2.CreateOneNotePageRequest(
                    access_token=token,
                    provider=prov,
                    section_id=p.get("section_id") or "",
                    html=p.get("html") or "",
                    title=p.get("title") or "",
                )
                return await client.stub.CreateOneNotePage(req, timeout=30.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return (
                resp.success,
                {"success": resp.success, "page_id": resp.page_id, "error": err or None},
                err,
            )

        if op == "list_planner_plans":

            async def _do():
                req = connections_service_pb2.ListPlannerPlansRequest(
                    access_token=token, provider=prov
                )
                return await client.stub.ListPlannerPlans(req, timeout=25.0)

            resp = await _call(_do)
            data = {
                "plans": [
                    {"id": x.id, "title": x.title, "owner": x.owner}
                    for x in resp.plans
                ],
                "error": resp.error if resp.HasField("error") else None,
            }
            return True, data, ""

        if op == "get_planner_tasks":

            async def _do():
                req = connections_service_pb2.GetPlannerTasksRequest(
                    access_token=token,
                    provider=prov,
                    plan_id=p.get("plan_id") or "",
                )
                return await client.stub.GetPlannerTasks(req, timeout=25.0)

            resp = await _call(_do)
            data = {
                "tasks": [
                    {
                        "id": x.id,
                        "plan_id": x.plan_id,
                        "title": x.title,
                        "percent_complete": x.percent_complete,
                        "due_datetime": x.due_datetime,
                    }
                    for x in resp.tasks
                ],
                "error": resp.error if resp.HasField("error") else None,
            }
            return True, data, ""

        if op == "create_planner_task":

            async def _do():
                req = connections_service_pb2.CreatePlannerTaskRequest(
                    access_token=token,
                    provider=prov,
                    plan_id=p.get("plan_id") or "",
                    title=p.get("title") or "Task",
                    bucket_id=p.get("bucket_id") or "",
                )
                return await client.stub.CreatePlannerTask(req, timeout=20.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return (
                resp.success,
                {"success": resp.success, "task_id": resp.task_id, "error": err or None},
                err,
            )

        if op == "update_planner_task":
            req = connections_service_pb2.UpdatePlannerTaskRequest(
                access_token=token,
                provider=prov,
                task_id=p.get("task_id") or "",
            )
            if "title" in p and p["title"] is not None:
                req.title = str(p["title"])
            if "percent_complete" in p and p["percent_complete"] is not None:
                req.percent_complete = int(p["percent_complete"])
            if "due_datetime" in p and p["due_datetime"] is not None:
                req.due_datetime = str(p["due_datetime"])

            async def _do():
                return await client.stub.UpdatePlannerTask(req, timeout=20.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return resp.success, {"success": resp.success, "error": err or None}, err

        if op == "delete_planner_task":

            async def _do():
                req = connections_service_pb2.DeletePlannerTaskRequest(
                    access_token=token,
                    provider=prov,
                    task_id=p.get("task_id") or "",
                    etag=p.get("etag") or "",
                )
                return await client.stub.DeletePlannerTask(req, timeout=15.0)

            resp = await _call(_do)
            err = resp.error if resp.HasField("error") else ""
            return resp.success, {"success": resp.success, "error": err or None}, err

        # --- Azure DevOps operations (direct REST, no proto needed) ---
        if m365_svc == "devops":
            return await _dispatch_devops(token, cid, op, p, rls_context)

        return False, None, f"Unhandled operation: {op}"
    except Exception as e:
        logger.exception("M365 graph dispatch failed: %s", e)
        return False, None, str(e)


async def _get_devops_org(connection_id: int, rls_context: Optional[Dict[str, str]]) -> str:
    """Read devops_organization from the connection's provider_metadata."""
    from services.external_connections_service import external_connections_service

    conn = await external_connections_service.get_connection_by_id(
        connection_id, rls_context=rls_context
    )
    if not conn:
        return ""
    meta = external_connections_service._parse_provider_metadata(conn.get("provider_metadata"))
    return (meta.get("devops_organization") or "").strip()


async def _dispatch_devops(
    token: str,
    connection_id: int,
    operation: str,
    params: Dict[str, Any],
    rls_context: Optional[Dict[str, str]],
) -> Tuple[bool, Any, str]:
    """Dispatch Azure DevOps operations using direct REST calls."""
    import httpx

    org = params.get("organization") or await _get_devops_org(connection_id, rls_context)
    if not org:
        return False, None, "Azure DevOps organization not configured on this connection"

    base = f"https://dev.azure.com/{org}"
    api_ver = "7.1"
    project = params.get("project") or ""
    team = params.get("team") or ""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def _url(path: str, proj: str = "") -> str:
        b = f"{base}/{proj}" if proj else base
        return f"{b}/{path.lstrip('/')}"

    async def _get(path: str, extra_params: Optional[Dict[str, Any]] = None, proj: str = "") -> Dict[str, Any]:
        p = dict(extra_params or {})
        p.setdefault("api-version", api_ver)
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.get(_url(path, proj), headers=headers, params=p)
            resp.raise_for_status()
            return resp.json()

    async def _post(path: str, body: Any, extra_params: Optional[Dict[str, Any]] = None,
                    proj: str = "", ct: str = "application/json") -> Dict[str, Any]:
        p = dict(extra_params or {})
        p.setdefault("api-version", api_ver)
        h = dict(headers)
        h["Content-Type"] = ct
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.post(_url(path, proj), headers=h, json=body, params=p)
            resp.raise_for_status()
            return resp.json() if resp.content else {}

    async def _patch(path: str, body: Any, extra_params: Optional[Dict[str, Any]] = None,
                     proj: str = "", ct: str = "application/json-patch+json") -> Dict[str, Any]:
        p = dict(extra_params or {})
        p.setdefault("api-version", api_ver)
        h = dict(headers)
        h["Content-Type"] = ct
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.patch(_url(path, proj), headers=h, json=body, params=p)
            resp.raise_for_status()
            return resp.json() if resp.content else {}

    def _wi(item: Dict[str, Any]) -> Dict[str, Any]:
        f = item.get("fields") or {}
        return {
            "id": item.get("id", 0), "rev": item.get("rev", 0),
            "url": item.get("url", ""),
            "title": f.get("System.Title", ""),
            "state": f.get("System.State", ""),
            "work_item_type": f.get("System.WorkItemType", ""),
            "assigned_to": (f.get("System.AssignedTo") or {}).get("displayName", ""),
            "assigned_to_email": (f.get("System.AssignedTo") or {}).get("uniqueName", ""),
            "created_date": f.get("System.CreatedDate", ""),
            "changed_date": f.get("System.ChangedDate", ""),
            "area_path": f.get("System.AreaPath", ""),
            "iteration_path": f.get("System.IterationPath", ""),
            "priority": f.get("Microsoft.VSTS.Common.Priority", 0),
            "description": f.get("System.Description", ""),
            "tags": f.get("System.Tags", ""),
            "story_points": f.get("Microsoft.VSTS.Scheduling.StoryPoints"),
            "remaining_work": f.get("Microsoft.VSTS.Scheduling.RemainingWork"),
        }

    try:
        op = operation

        if op == "list_devops_projects":
            data = await _get("_apis/projects", {"$top": params.get("top", 100)})
            projects = [{"id": p.get("id", ""), "name": p.get("name", ""),
                         "description": p.get("description", ""), "state": p.get("state", "")}
                        for p in data.get("value", [])]
            return True, {"projects": projects, "count": len(projects)}, ""

        if op == "list_devops_teams":
            data = await _get(f"_apis/projects/{project}/teams", {"$top": params.get("top", 100)})
            teams = [{"id": t.get("id", ""), "name": t.get("name", ""),
                       "description": t.get("description", "")} for t in data.get("value", [])]
            return True, {"teams": teams, "count": len(teams)}, ""

        if op == "list_devops_team_members":
            data = await _get(f"_apis/projects/{project}/teams/{team}/members",
                              {"$top": params.get("top", 100)})
            members = []
            for m in data.get("value", []):
                ident = m.get("identity") or {}
                members.append({"id": ident.get("id", ""), "display_name": ident.get("displayName", ""),
                                "unique_name": ident.get("uniqueName", ""),
                                "is_team_admin": m.get("isTeamAdmin", False)})
            return True, {"members": members, "count": len(members)}, ""

        if op == "query_devops_work_items":
            wiql = params.get("wiql") or ""
            top = int(params.get("top") or 200)
            qr = await _post("_apis/wit/wiql", {"query": wiql}, {"$top": top}, proj=project)
            wi_refs = qr.get("workItems") or []
            if not wi_refs:
                return True, {"work_items": [], "count": 0}, ""
            ids = [str(w["id"]) for w in wi_refs[:top]]
            batched = []
            for i in range(0, len(ids), 200):
                chunk = ",".join(ids[i:i + 200])
                items = await _get("_apis/wit/workitems", {"ids": chunk, "$expand": "fields"}, proj=project)
                batched.extend(_wi(x) for x in items.get("value", []))
            return True, {"work_items": batched, "count": len(batched)}, ""

        if op == "get_devops_work_item":
            wid = int(params.get("work_item_id") or 0)
            data = await _get(f"_apis/wit/workitems/{wid}", {"$expand": "all"}, proj=project)
            return True, {"work_item": _wi(data)}, ""

        if op == "list_devops_iterations":
            path = "_apis/work/teamsettings/iterations"
            if team:
                path = f"{project}/{team}/_apis/work/teamsettings/iterations"
                data = await _get(path)
            else:
                data = await _get(path, proj=project)
            iters = []
            for it in data.get("value", []):
                attrs = it.get("attributes") or {}
                iters.append({"id": it.get("id", ""), "name": it.get("name", ""),
                              "path": it.get("path", ""),
                              "start_date": attrs.get("startDate", ""),
                              "finish_date": attrs.get("finishDate", ""),
                              "time_frame": attrs.get("timeFrame", "")})
            return True, {"iterations": iters, "count": len(iters)}, ""

        if op == "get_devops_iteration_work_items":
            it_id = params.get("iteration_id") or ""
            path = f"_apis/work/teamsettings/iterations/{it_id}/workitems"
            if team:
                path = f"{project}/{team}/_apis/work/teamsettings/iterations/{it_id}/workitems"
                data = await _get(path)
            else:
                data = await _get(path, proj=project)
            rels = data.get("workItemRelations") or []
            tids = [str(r.get("target", {}).get("id")) for r in rels if r.get("target", {}).get("id")]
            if not tids:
                return True, {"work_items": [], "count": 0}, ""
            batched = []
            for i in range(0, len(tids), 200):
                chunk = ",".join(tids[i:i + 200])
                items = await _get("_apis/wit/workitems", {"ids": chunk, "$expand": "fields"}, proj=project)
                batched.extend(_wi(x) for x in items.get("value", []))
            return True, {"work_items": batched, "count": len(batched)}, ""

        if op == "list_devops_boards":
            path = "_apis/work/boards"
            if team:
                path = f"{project}/{team}/_apis/work/boards"
                data = await _get(path)
            else:
                data = await _get(path, proj=project)
            boards = [{"id": b.get("id", ""), "name": b.get("name", "")} for b in data.get("value", [])]
            return True, {"boards": boards, "count": len(boards)}, ""

        if op == "get_devops_board_columns":
            board = params.get("board") or ""
            path = f"_apis/work/boards/{board}/columns"
            if team:
                path = f"{project}/{team}/_apis/work/boards/{board}/columns"
                data = await _get(path)
            else:
                data = await _get(path, proj=project)
            cols = [{"id": c.get("id", ""), "name": c.get("name", ""),
                     "item_limit": c.get("itemLimit", 0), "column_type": c.get("columnType", "")}
                    for c in data.get("value", [])]
            return True, {"columns": cols, "count": len(cols)}, ""

        if op == "list_devops_repos":
            data = await _get("_apis/git/repositories", proj=project)
            repos = [{"id": r.get("id", ""), "name": r.get("name", ""),
                      "default_branch": r.get("defaultBranch", ""), "web_url": r.get("webUrl", ""),
                      "project_name": (r.get("project") or {}).get("name", "")}
                     for r in data.get("value", [])]
            return True, {"repos": repos, "count": len(repos)}, ""

        if op == "list_devops_pull_requests":
            status = params.get("status") or "active"
            top = int(params.get("top") or 50)
            data = await _get("_apis/git/pullrequests",
                              {"searchCriteria.status": status, "$top": top}, proj=project)
            prs = []
            for pr in data.get("value", []):
                cb = pr.get("createdBy") or {}
                repo = pr.get("repository") or {}
                prs.append({"id": pr.get("pullRequestId", 0), "title": pr.get("title", ""),
                            "status": pr.get("status", ""), "created_by": cb.get("displayName", ""),
                            "created_date": pr.get("creationDate", ""),
                            "source_branch": pr.get("sourceRefName", ""),
                            "target_branch": pr.get("targetRefName", ""),
                            "repo_name": repo.get("name", "")})
            return True, {"pull_requests": prs, "count": len(prs)}, ""

        if op == "list_devops_pipelines":
            top = int(params.get("top") or 50)
            data = await _get("_apis/pipelines", {"$top": top}, proj=project)
            pipes = [{"id": p.get("id", 0), "name": p.get("name", ""),
                      "folder": p.get("folder", "")} for p in data.get("value", [])]
            return True, {"pipelines": pipes, "count": len(pipes)}, ""

        if op == "get_devops_pipeline_runs":
            pid = int(params.get("pipeline_id") or 0)
            top = int(params.get("top") or 20)
            data = await _get(f"_apis/pipelines/{pid}/runs", {"$top": top}, proj=project)
            runs = [{"id": r.get("id", 0), "name": r.get("name", ""),
                     "state": r.get("state", ""), "result": r.get("result", ""),
                     "created_date": r.get("createdDate", ""),
                     "finished_date": r.get("finishedDate", "")}
                    for r in data.get("value", [])]
            return True, {"runs": runs, "count": len(runs)}, ""

        if op == "create_devops_work_item":
            wit = params.get("work_item_type") or "Task"
            ops = [{"op": "add", "path": "/fields/System.Title", "value": params.get("title") or "New item"}]
            for field, key in [("System.Description", "description"), ("System.AssignedTo", "assigned_to"),
                               ("System.AreaPath", "area_path"), ("System.IterationPath", "iteration_path"),
                               ("System.Tags", "tags")]:
                if params.get(key):
                    ops.append({"op": "add", "path": f"/fields/{field}", "value": params[key]})
            if params.get("priority") is not None:
                ops.append({"op": "add", "path": "/fields/Microsoft.VSTS.Common.Priority",
                            "value": int(params["priority"])})
            data = await _post(f"_apis/wit/workitems/${wit}", ops,
                               proj=project, ct="application/json-patch+json")
            return True, {"success": True, "work_item": _wi(data)}, ""

        if op == "update_devops_work_item":
            wid = int(params.get("work_item_id") or 0)
            ops = []
            for field, key in [("System.Title", "title"), ("System.Description", "description"),
                               ("System.State", "state"), ("System.AssignedTo", "assigned_to"),
                               ("System.AreaPath", "area_path"), ("System.IterationPath", "iteration_path"),
                               ("System.Tags", "tags")]:
                if key in params and params[key] is not None:
                    ops.append({"op": "replace", "path": f"/fields/{field}", "value": params[key]})
            if "priority" in params and params["priority"] is not None:
                ops.append({"op": "replace", "path": "/fields/Microsoft.VSTS.Common.Priority",
                            "value": int(params["priority"])})
            if not ops:
                return True, {"success": True, "work_item": None}, ""
            data = await _patch(f"_apis/wit/workitems/{wid}", ops, proj=project)
            return True, {"success": True, "work_item": _wi(data)}, ""

        if op == "add_devops_work_item_comment":
            wid = int(params.get("work_item_id") or 0)
            text = params.get("text") or ""
            data = await _post(f"_apis/wit/workitems/{wid}/comments", {"text": text},
                               extra_params={"api-version": "7.1-preview.4"}, proj=project)
            return True, {"success": True, "comment_id": data.get("id", 0)}, ""

        return False, None, f"Unhandled DevOps operation: {op}"
    except Exception as e:
        logger.exception("DevOps dispatch failed: %s", e)
        return False, None, str(e)


def parse_m365_params(params_json: str) -> Dict[str, Any]:
    if not (params_json or "").strip():
        return {}
    try:
        out = json.loads(params_json)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}
