"""
Microsoft Graph: To Do, OneDrive, OneNote, Planner (mixin for MicrosoftGraphProvider).
"""

import base64
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# LLMs often pass "flagged" or "tasks" instead of the Graph list id; map to wellknownListName.
_TODO_LIST_ID_ALIASES: Dict[str, str] = {
    "flagged": "flaggedEmails",
    "flaggedemails": "flaggedEmails",
    "flagged_emails": "flaggedEmails",
    "flagged email": "flaggedEmails",
    "flagged emails": "flaggedEmails",
    "default": "defaultList",
    "defaultlist": "defaultList",
    "tasks": "defaultList",
    "task": "defaultList",
}


def _encode_todo_path_segment(segment: str) -> str:
    """Graph list/task ids may contain +, /, = — encode for URL path segments."""
    return quote((segment or "").strip(), safe="")


def _todo_list_to_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id", ""),
        "display_name": item.get("displayName", ""),
        "is_owner": item.get("isOwner", False),
        "is_shared": item.get("isShared", False),
        "well_known_list_name": item.get("wellknownListName") or "",
    }


def _todo_task_to_dict(item: Dict[str, Any], list_id: str = "") -> Dict[str, Any]:
    due = (item.get("dueDateTime") or {}) or {}
    return {
        "id": item.get("id", ""),
        "list_id": list_id or "",
        "title": item.get("title", ""),
        "status": item.get("status", ""),
        "body": ((item.get("body") or {}) or {}).get("content", ""),
        "due_datetime": due.get("dateTime", ""),
        "importance": item.get("importance", "normal"),
    }


def _drive_item_to_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    folder = item.get("folder")
    is_folder = folder is not None
    parent_ref = (item.get("parentReference") or {}) or {}
    return {
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "web_url": item.get("webUrl", ""),
        "is_folder": is_folder,
        "mime_type": item.get("file", {}).get("mimeType", "") if item.get("file") else "",
        "size": int(item.get("size") or 0),
        "parent_id": parent_ref.get("id", ""),
        "last_modified": item.get("lastModifiedDateTime", ""),
    }


class MicrosoftGraphM365Mixin:
    """Adds To Do, OneDrive, OneNote, Planner; requires _get, _post, _patch, _delete, _headers."""

    async def _resolve_todo_list_id(self, access_token: str, list_id: str) -> str:
        """
        Map common mistaken ids (e.g. "flagged") to the real list id via wellknownListName.
        """
        raw = (list_id or "").strip()
        if not raw:
            return raw
        wk_target = _TODO_LIST_ID_ALIASES.get(raw.lower())
        if not wk_target:
            return raw
        try:
            data = await self._get(access_token, "/me/todo/lists", {"$top": "100"})
            for item in data.get("value", []):
                wk = (item.get("wellknownListName") or "").strip()
                if wk.lower() == wk_target.lower():
                    resolved = (item.get("id") or "").strip()
                    if resolved:
                        return resolved
            logger.warning(
                "todo list alias %r (%s) not found in /me/todo/lists",
                raw,
                wk_target,
            )
        except Exception as e:
            logger.warning("todo list id resolve failed for %r: %s", raw, e)
        return raw

    async def list_todo_lists(self, access_token: str) -> Dict[str, Any]:
        try:
            data = await self._get(access_token, "/me/todo/lists", {"$top": "100"})
            value = data.get("value", [])
            return {"lists": [_todo_list_to_dict(x) for x in value]}
        except Exception as e:
            logger.exception("list_todo_lists: %s", e)
            return {"lists": [], "error": str(e)}

    async def get_todo_tasks(
        self, access_token: str, list_id: str, top: int = 50
    ) -> Dict[str, Any]:
        try:
            resolved_id = await self._resolve_todo_list_id(access_token, list_id)
            enc = _encode_todo_path_segment(resolved_id)
            path = f"/me/todo/lists/{enc}/tasks"
            params = {"$top": str(min(top, 100))}
            data = await self._get(access_token, path, params)
            value = data.get("value", [])
            return {"tasks": [_todo_task_to_dict(x, resolved_id) for x in value]}
        except Exception as e:
            logger.exception("get_todo_tasks: %s", e)
            return {"tasks": [], "error": str(e)}

    async def create_todo_task(
        self,
        access_token: str,
        list_id: str,
        title: str,
        body: str = "",
        due_datetime: str = "",
        importance: str = "normal",
    ) -> Dict[str, Any]:
        try:
            payload: Dict[str, Any] = {"title": title or "Task", "importance": importance or "normal"}
            if body:
                payload["body"] = {"content": body, "contentType": "text"}
            if due_datetime:
                payload["dueDateTime"] = {"dateTime": due_datetime, "timeZone": "UTC"}
            resolved_id = await self._resolve_todo_list_id(access_token, list_id)
            enc = _encode_todo_path_segment(resolved_id)
            data = await self._post(
                access_token, f"/me/todo/lists/{enc}/tasks", json=payload
            )
            return {"success": True, "task_id": data.get("id", "")}
        except Exception as e:
            logger.exception("create_todo_task: %s", e)
            return {"success": False, "task_id": "", "error": str(e)}

    async def update_todo_task(
        self,
        access_token: str,
        list_id: str,
        task_id: str,
        title: Optional[str] = None,
        body: Optional[str] = None,
        status: Optional[str] = None,
        due_datetime: Optional[str] = None,
        importance: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            payload: Dict[str, Any] = {}
            if title is not None:
                payload["title"] = title
            if body is not None:
                payload["body"] = {"content": body, "contentType": "text"}
            if status is not None:
                payload["status"] = status
            if due_datetime is not None:
                payload["dueDateTime"] = {"dateTime": due_datetime, "timeZone": "UTC"}
            if importance is not None:
                payload["importance"] = importance
            if payload:
                resolved_id = await self._resolve_todo_list_id(access_token, list_id)
                le = _encode_todo_path_segment(resolved_id)
                te = _encode_todo_path_segment(task_id)
                await self._patch(
                    access_token, f"/me/todo/lists/{le}/tasks/{te}", payload
                )
            return {"success": True}
        except Exception as e:
            logger.exception("update_todo_task: %s", e)
            return {"success": False, "error": str(e)}

    async def delete_todo_task(
        self, access_token: str, list_id: str, task_id: str
    ) -> Dict[str, Any]:
        try:
            resolved_id = await self._resolve_todo_list_id(access_token, list_id)
            le = _encode_todo_path_segment(resolved_id)
            te = _encode_todo_path_segment(task_id)
            await self._delete(access_token, f"/me/todo/lists/{le}/tasks/{te}")
            return {"success": True}
        except Exception as e:
            logger.exception("delete_todo_task: %s", e)
            return {"success": False, "error": str(e)}

    async def list_drive_items(
        self, access_token: str, parent_item_id: str = "", top: int = 50
    ) -> Dict[str, Any]:
        try:
            if parent_item_id:
                path = f"/me/drive/items/{parent_item_id}/children"
            else:
                path = "/me/drive/root/children"
            data = await self._get(access_token, path, {"$top": str(min(top, 200))})
            value = data.get("value", [])
            return {"items": [_drive_item_to_dict(x) for x in value]}
        except Exception as e:
            logger.exception("list_drive_items: %s", e)
            return {"items": [], "error": str(e)}

    async def get_drive_item(self, access_token: str, item_id: str) -> Dict[str, Any]:
        try:
            data = await self._get(access_token, f"/me/drive/items/{item_id}")
            return {"item": _drive_item_to_dict(data)}
        except Exception as e:
            logger.exception("get_drive_item: %s", e)
            return {"item": None, "error": str(e)}

    async def search_drive(
        self, access_token: str, query: str, top: int = 25
    ) -> Dict[str, Any]:
        try:
            safe_q = (query or "").replace("'", "''")
            path = f"/me/drive/root/search(q='{safe_q}')"
            data = await self._get(access_token, path, {"$top": str(min(top, 50))})
            value = data.get("value", [])
            return {"items": [_drive_item_to_dict(x) for x in value]}
        except Exception as e:
            logger.exception("search_drive: %s", e)
            return {"items": [], "error": str(e)}

    async def get_file_content(self, access_token: str, item_id: str) -> Dict[str, Any]:
        try:
            from config.settings import settings

            base = settings.MICROSOFT_GRAPH_BASE.rstrip("/")
            url = f"{base}/me/drive/items/{item_id}/content"
            async with httpx.AsyncClient(timeout=settings.GRAPH_REQUEST_TIMEOUT) as client:
                resp = await client.get(url, headers=self._headers(access_token))
                resp.raise_for_status()
                raw = resp.content
                mime = resp.headers.get("content-type", "application/octet-stream")
            return {
                "content_base64": base64.b64encode(raw).decode("ascii"),
                "mime_type": mime.split(";")[0].strip(),
            }
        except Exception as e:
            logger.exception("get_file_content: %s", e)
            return {"content_base64": "", "mime_type": "", "error": str(e)}

    async def upload_file(
        self,
        access_token: str,
        parent_item_id: str,
        name: str,
        content_base64: str,
        mime_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        try:
            raw = base64.b64decode(content_base64)
            from config.settings import settings

            base = settings.MICROSOFT_GRAPH_BASE.rstrip("/")
            enc_name = quote(name or "file.bin", safe="")
            if not parent_item_id:
                url = f"{base}/me/drive/root:/{enc_name}:/content"
            else:
                url = f"{base}/me/drive/items/{parent_item_id}:/{enc_name}:/content"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": mime_type or "application/octet-stream",
            }
            async with httpx.AsyncClient(timeout=settings.GRAPH_REQUEST_TIMEOUT) as client:
                resp = await client.put(url, headers=headers, content=raw)
                resp.raise_for_status()
                data = resp.json() if resp.content else {}
            return {"success": True, "item_id": data.get("id", "")}
        except Exception as e:
            logger.exception("upload_file: %s", e)
            return {"success": False, "item_id": "", "error": str(e)}

    async def create_drive_folder(
        self, access_token: str, parent_item_id: str, name: str
    ) -> Dict[str, Any]:
        try:
            parent = parent_item_id or "root"
            payload = {
                "name": name,
                "folder": {},
                "@microsoft.graph.conflictBehavior": "rename",
            }
            data = await self._post(
                access_token, f"/me/drive/items/{parent}/children", json=payload
            )
            return {"success": True, "item_id": data.get("id", "")}
        except Exception as e:
            logger.exception("create_drive_folder: %s", e)
            return {"success": False, "item_id": "", "error": str(e)}

    async def move_drive_item(
        self, access_token: str, item_id: str, new_parent_item_id: str
    ) -> Dict[str, Any]:
        try:
            payload = {
                "parentReference": {"id": new_parent_item_id},
            }
            await self._patch(access_token, f"/me/drive/items/{item_id}", payload)
            return {"success": True}
        except Exception as e:
            logger.exception("move_drive_item: %s", e)
            return {"success": False, "error": str(e)}

    async def delete_drive_item(self, access_token: str, item_id: str) -> Dict[str, Any]:
        try:
            await self._delete(access_token, f"/me/drive/items/{item_id}")
            return {"success": True}
        except Exception as e:
            logger.exception("delete_drive_item: %s", e)
            return {"success": False, "error": str(e)}

    async def list_onenote_notebooks(self, access_token: str) -> Dict[str, Any]:
        try:
            data = await self._get(access_token, "/me/onenote/notebooks", {"$top": "100"})
            value = data.get("value", [])
            out = []
            for n in value:
                out.append({
                    "id": n.get("id", ""),
                    "display_name": n.get("displayName", ""),
                    "web_url": n.get("links", {}).get("oneNoteWebUrl", {}).get("href", ""),
                })
            return {"notebooks": out}
        except Exception as e:
            logger.exception("list_onenote_notebooks: %s", e)
            return {"notebooks": [], "error": str(e)}

    async def list_onenote_sections(
        self, access_token: str, notebook_id: str
    ) -> Dict[str, Any]:
        try:
            data = await self._get(
                access_token, f"/me/onenote/notebooks/{notebook_id}/sections", {"$top": "100"}
            )
            value = data.get("value", [])
            out = []
            for s in value:
                out.append({
                    "id": s.get("id", ""),
                    "display_name": s.get("displayName", ""),
                    "notebook_id": notebook_id,
                    "web_url": s.get("links", {}).get("oneNoteWebUrl", {}).get("href", ""),
                })
            return {"sections": out}
        except Exception as e:
            logger.exception("list_onenote_sections: %s", e)
            return {"sections": [], "error": str(e)}

    async def list_onenote_pages(
        self, access_token: str, section_id: str, top: int = 50
    ) -> Dict[str, Any]:
        try:
            data = await self._get(
                access_token,
                f"/me/onenote/sections/{section_id}/pages",
                {"$top": str(min(top, 100))},
            )
            value = data.get("value", [])
            out = []
            for p in value:
                out.append({
                    "id": p.get("id", ""),
                    "title": p.get("title", ""),
                    "section_id": section_id,
                    "web_url": p.get("links", {}).get("oneNoteWebUrl", {}).get("href", ""),
                    "created_time": p.get("createdDateTime", ""),
                })
            return {"pages": out}
        except Exception as e:
            logger.exception("list_onenote_pages: %s", e)
            return {"pages": [], "error": str(e)}

    async def get_onenote_page_content(
        self, access_token: str, page_id: str
    ) -> Dict[str, Any]:
        try:
            from config.settings import settings

            base = settings.MICROSOFT_GRAPH_BASE.rstrip("/")
            url = f"{base}/me/onenote/pages/{page_id}/content"
            async with httpx.AsyncClient(timeout=settings.GRAPH_REQUEST_TIMEOUT) as client:
                resp = await client.get(url, headers=self._headers(access_token))
                resp.raise_for_status()
                html = resp.text
            return {"html_content": html}
        except Exception as e:
            logger.exception("get_onenote_page_content: %s", e)
            return {"html_content": "", "error": str(e)}

    async def create_onenote_page(
        self, access_token: str, section_id: str, html: str, title: str = ""
    ) -> Dict[str, Any]:
        try:
            from config.settings import settings

            base = settings.MICROSOFT_GRAPH_BASE.rstrip("/")
            url = f"{base}/me/onenote/sections/{section_id}/pages"
            body_html = html
            if title and "<title>" not in body_html.lower():
                body_html = f"<!DOCTYPE html><html><head><title>{title}</title></head><body>{body_html}</body></html>"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "text/html",
            }
            async with httpx.AsyncClient(timeout=settings.GRAPH_REQUEST_TIMEOUT) as client:
                resp = await client.post(url, headers=headers, content=body_html.encode("utf-8"))
                resp.raise_for_status()
                data = resp.json() if resp.content else {}
            return {"success": True, "page_id": data.get("id", "")}
        except Exception as e:
            logger.exception("create_onenote_page: %s", e)
            return {"success": False, "page_id": "", "error": str(e)}

    async def list_planner_plans(self, access_token: str) -> Dict[str, Any]:
        try:
            data = await self._get(access_token, "/me/planner/tasks", {"$top": "100"})
            value = data.get("value", [])
            seen: Dict[str, Dict[str, Any]] = {}
            for t in value:
                pid = (t.get("planId") or "").strip()
                if pid and pid not in seen:
                    try:
                        plan = await self._get(access_token, f"/planner/plans/{pid}")
                        seen[pid] = {
                            "id": pid,
                            "title": plan.get("title", pid),
                            "owner": plan.get("owner", ""),
                        }
                    except Exception:
                        seen[pid] = {"id": pid, "title": pid, "owner": ""}
            return {"plans": list(seen.values())}
        except Exception as e:
            logger.exception("list_planner_plans: %s", e)
            return {"plans": [], "error": str(e)}

    async def get_planner_tasks(self, access_token: str, plan_id: str) -> Dict[str, Any]:
        try:
            data = await self._get(
                access_token, f"/planner/plans/{plan_id}/tasks", {"$top": "100"}
            )
            value = data.get("value", [])
            out = []
            for t in value:
                due_obj = t.get("dueDateTime")
                due = ""
                if isinstance(due_obj, dict):
                    due = due_obj.get("dateTime") or ""
                elif isinstance(due_obj, str):
                    due = due_obj
                out.append({
                    "id": t.get("id", ""),
                    "plan_id": plan_id,
                    "title": t.get("title", ""),
                    "percent_complete": int(t.get("percentComplete") or 0),
                    "due_datetime": due,
                })
            return {"tasks": out}
        except Exception as e:
            logger.exception("get_planner_tasks: %s", e)
            return {"tasks": [], "error": str(e)}

    async def create_planner_task(
        self,
        access_token: str,
        plan_id: str,
        title: str,
        bucket_id: str = "",
    ) -> Dict[str, Any]:
        try:
            bid = bucket_id
            if not bid:
                buckets = await self._get(access_token, f"/planner/plans/{plan_id}/buckets")
                bval = buckets.get("value") or []
                if not bval:
                    return {"success": False, "task_id": "", "error": "No buckets on plan"}
                bid = bval[0].get("id", "")
            payload = {"planId": plan_id, "bucketId": bid, "title": title or "Task"}
            data = await self._post(access_token, "/planner/tasks", json=payload)
            return {"success": True, "task_id": data.get("id", "")}
        except Exception as e:
            logger.exception("create_planner_task: %s", e)
            return {"success": False, "task_id": "", "error": str(e)}

    async def update_planner_task(
        self,
        access_token: str,
        task_id: str,
        title: Optional[str] = None,
        percent_complete: Optional[int] = None,
        due_datetime: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            detail = await self._get(access_token, f"/planner/tasks/{task_id}")
            etag = detail.get("@odata.etag", "")
            payload: Dict[str, Any] = {}
            if title is not None:
                payload["title"] = title
            if percent_complete is not None:
                payload["percentComplete"] = int(percent_complete)
            if due_datetime is not None:
                payload["dueDateTime"] = {
                    "dateTime": due_datetime,
                    "timeZone": "UTC",
                }
            if not payload:
                return {"success": True}
            from config.settings import settings

            base = settings.MICROSOFT_GRAPH_BASE.rstrip("/")
            url = f"{base}/planner/tasks/{task_id}"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "If-Match": etag,
            }
            async with httpx.AsyncClient(timeout=settings.GRAPH_REQUEST_TIMEOUT) as client:
                resp = await client.patch(url, headers=headers, json=payload)
                resp.raise_for_status()
            return {"success": True}
        except Exception as e:
            logger.exception("update_planner_task: %s", e)
            return {"success": False, "error": str(e)}

    async def delete_planner_task(
        self, access_token: str, task_id: str, etag: str = ""
    ) -> Dict[str, Any]:
        try:
            et = etag
            if not et:
                detail = await self._get(access_token, f"/planner/tasks/{task_id}")
                et = detail.get("@odata.etag", "")
            from config.settings import settings

            base = settings.MICROSOFT_GRAPH_BASE.rstrip("/")
            url = f"{base}/planner/tasks/{task_id}"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "If-Match": et,
            }
            async with httpx.AsyncClient(timeout=settings.GRAPH_REQUEST_TIMEOUT) as client:
                resp = await client.delete(url, headers=headers)
                if resp.status_code not in (200, 204):
                    resp.raise_for_status()
            return {"success": True}
        except Exception as e:
            logger.exception("delete_planner_task: %s", e)
            return {"success": False, "error": str(e)}
