"""
HTTP client for Bastion backend: internal service API and user JWT API.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class BackendClient:
    def __init__(self) -> None:
        self._base = settings.BACKEND_URL.rstrip("/")
        self._key = settings.INTERNAL_SERVICE_KEY
        self._chat_timeout = settings.EXTERNAL_CHAT_TIMEOUT

    def _internal_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Internal-Service-Key": self._key,
        }

    def _user_headers(self, jwt: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt}",
        }

    async def login(self, username: str, password: str) -> Dict[str, Any]:
        url = f"{self._base}/api/auth/login"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    json={"username": username, "password": password},
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code >= 400:
                    return {"error": resp.json().get("detail", resp.text) or f"HTTP {resp.status_code}"}
                return resp.json()
        except Exception as e:
            logger.exception("login failed: %s", e)
            return {"error": str(e)}

    async def refresh_token(self, jwt: str) -> Dict[str, Any]:
        url = f"{self._base}/api/auth/refresh"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {jwt}"},
                )
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_current_user(self, jwt: str) -> Dict[str, Any]:
        url = f"{self._base}/api/auth/me"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers={"Authorization": f"Bearer {jwt}"})
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_bbs_wallpaper(
        self,
        jwt: str,
        term_cols: Optional[int] = None,
        term_rows: Optional[int] = None,
    ) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
        """Resolved wallpaper text, cycle flag, optional animation dict (frames, fps, loop)."""
        url = f"{self._base}/api/settings/user/bbs-wallpaper"
        params: Dict[str, Any] = {}
        if term_cols is not None:
            params["cols"] = int(term_cols)
        if term_rows is not None:
            params["rows"] = int(term_rows)
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.get(url, headers=self._user_headers(jwt), params=params)
                if resp.status_code >= 400:
                    return "", False, None
                data = resp.json()
                if isinstance(data, dict):
                    anim = data.get("animation")
                    if isinstance(anim, dict) and isinstance(anim.get("frames"), list):
                        return (
                            str(data.get("wallpaper") or ""),
                            bool(data.get("cycling")),
                            anim,
                        )
                    return str(data.get("wallpaper") or ""), bool(data.get("cycling")), None
                return "", False, None
        except Exception as e:
            logger.warning("get_bbs_wallpaper failed: %s", e)
            return "", False, None

    async def send_external_message(
        self,
        user_id: str,
        conversation_id: str,
        query: str,
        platform: str,
        platform_chat_id: str,
        sender_name: str,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/internal/external-chat"
        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query": query,
            "platform": platform,
            "platform_chat_id": platform_chat_id,
            "sender_name": sender_name,
        }
        try:
            async with httpx.AsyncClient(timeout=self._chat_timeout) as client:
                resp = await client.post(url, json=payload, headers=self._internal_headers())
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            return {"error": (e.response.text or str(e))[:500]}
        except Exception as e:
            return {"error": str(e)}

    async def start_new_conversation(
        self,
        user_id: str,
        conversation_id: str,
        platform: str,
        platform_chat_id: str,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/internal/external-chat"
        payload = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query": "",
            "platform": platform,
            "platform_chat_id": platform_chat_id,
            "sender_name": "",
            "start_new_conversation": True,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=self._internal_headers())
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def list_models(self, user_id: str, conversation_id: str = "") -> Dict[str, Any]:
        url = f"{self._base}/api/internal/external-chat-models"
        payload: Dict[str, Any] = {"user_id": user_id}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload, headers=self._internal_headers())
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def set_model(self, user_id: str, conversation_id: str, model_index: int) -> Dict[str, Any]:
        url = f"{self._base}/api/internal/external-chat-set-model"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    url,
                    json={
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "model_index": model_index,
                    },
                    headers=self._internal_headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def list_user_conversations(self, user_id: str, limit: int = 20) -> Dict[str, Any]:
        url = f"{self._base}/api/internal/user-conversations"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    url,
                    params={"user_id": user_id, "limit": limit},
                    headers=self._internal_headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            return {"error": str(e), "conversations": []}

    async def validate_conversation(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        url = f"{self._base}/api/internal/validate-conversation"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    url,
                    params={"user_id": user_id, "conversation_id": conversation_id},
                    headers=self._internal_headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            return {"valid": False, "title": "", "error": str(e)}

    async def get_conversation_messages(
        self,
        jwt: str,
        conversation_id: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/conversations/{conversation_id}/messages"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    url,
                    params={"most_recent": "true", "limit": limit},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    return {"error": resp.text, "messages": []}
                return resp.json()
        except Exception as e:
            return {"error": str(e), "messages": []}

    async def messaging_list_rooms(self, jwt: str, limit: int = 30) -> Dict[str, Any]:
        url = f"{self._base}/api/messaging/rooms"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    url,
                    params={"limit": limit},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    return {"error": resp.text, "rooms": []}
                return resp.json()
        except Exception as e:
            return {"error": str(e), "rooms": []}

    async def messaging_get_room_messages(
        self,
        jwt: str,
        room_id: str,
        limit: int = 50,
        before_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/messaging/rooms/{room_id}/messages"
        params: Dict[str, Any] = {"limit": limit}
        if before_message_id:
            params["before_message_id"] = before_message_id
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, params=params, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text, "messages": []}
                return resp.json()
        except Exception as e:
            return {"error": str(e), "messages": []}

    async def messaging_send_message(self, jwt: str, room_id: str, content: str) -> Dict[str, Any]:
        url = f"{self._base}/api/messaging/rooms/{room_id}/messages"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    json={"content": content, "message_type": "text"},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    detail = resp.text
                    try:
                        body = resp.json()
                        if isinstance(body, dict) and body.get("detail") is not None:
                            detail = str(body["detail"])
                    except Exception:
                        pass
                    return {"error": detail}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def messaging_create_room(
        self,
        jwt: str,
        participant_ids: List[str],
        room_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/messaging/rooms"
        payload: Dict[str, Any] = {"participant_ids": participant_ids}
        if room_name:
            payload["room_name"] = room_name
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, json=payload, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    detail = resp.text
                    try:
                        body = resp.json()
                        if isinstance(body, dict) and body.get("detail") is not None:
                            detail = str(body["detail"])
                    except Exception:
                        pass
                    return {"error": detail}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def messaging_list_users(self, jwt: str) -> Dict[str, Any]:
        url = f"{self._base}/api/messaging/users"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text, "users": []}
                return resp.json()
        except Exception as e:
            return {"error": str(e), "users": []}

    async def list_user_documents(self, jwt: str, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        url = f"{self._base}/api/user/documents"
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.get(
                    url,
                    params={"skip": skip, "limit": limit},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    return {"error": resp.text, "documents": []}
                return resp.json()
        except Exception as e:
            return {"error": str(e), "documents": []}

    async def search_documents(self, jwt: str, query: str, limit: int = 15) -> Dict[str, Any]:
        url = f"{self._base}/api/user/documents/search"
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.post(
                    url,
                    json={"query": query, "search_mode": "hybrid", "limit": limit},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    return {"error": resp.text, "results": []}
                return resp.json()
        except Exception as e:
            return {"error": str(e), "results": []}

    async def get_document_content(self, jwt: str, doc_id: str) -> Dict[str, Any]:
        url = f"{self._base}/api/documents/{doc_id}/content"
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_document_content_for_editor(self, jwt: str, doc_id: str) -> Dict[str, Any]:
        """GET document body with extended timeout for large files."""
        url = f"{self._base}/api/documents/{doc_id}/content"
        try:
            async with httpx.AsyncClient(timeout=settings.BBS_EDITOR_HTTP_TIMEOUT) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def put_document_content(self, jwt: str, doc_id: str, content: str) -> Dict[str, Any]:
        url = f"{self._base}/api/documents/{doc_id}/content"
        try:
            async with httpx.AsyncClient(timeout=settings.BBS_EDITOR_HTTP_TIMEOUT) as client:
                resp = await client.put(
                    url,
                    json={"content": content},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    detail = resp.text
                    try:
                        body = resp.json()
                        if isinstance(body, dict) and body.get("detail") is not None:
                            detail = str(body["detail"])
                    except Exception:
                        pass
                    return {"error": detail}
                if resp.content:
                    try:
                        return resp.json()
                    except Exception:
                        pass
                return {"success": True}
        except Exception as e:
            logger.exception("put_document_content failed: %s", e)
            return {"error": str(e)}

    async def get_folder_tree(
        self,
        jwt: str,
        collection_type: str = "user",
        shallow: bool = True,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/folders/tree"
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.get(
                    url,
                    params={
                        "collection_type": collection_type,
                        "shallow": "true" if shallow else "false",
                    },
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    return {"error": resp.text, "folders": []}
                return resp.json()
        except Exception as e:
            return {"error": str(e), "folders": []}

    async def get_folder_contents(
        self,
        jwt: str,
        folder_id: str,
        limit: int = 120,
        offset: int = 0,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/folders/{folder_id}/contents"
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.get(
                    url,
                    params={"limit": limit, "offset": offset},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def list_rss_feeds(self, jwt: str) -> List[Dict[str, Any]]:
        url = f"{self._base}/api/rss/feeds"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return []
                data = resp.json()
                return data if isinstance(data, list) else []
        except Exception:
            return []

    async def get_rss_articles(self, jwt: str, feed_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        url = f"{self._base}/api/rss/feeds/{feed_id}/articles"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, params={"limit": limit}, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return []
                data = resp.json()
                return data if isinstance(data, list) else []
        except Exception:
            return []

    async def rss_unread_by_feed(self, jwt: str) -> Dict[str, int]:
        """Returns {feed_id: unread_count}."""
        url = f"{self._base}/api/rss/unread-count"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {}
                j = resp.json()
                if isinstance(j, dict):
                    return {str(k): int(v) for k, v in j.items() if str(k)}
                return {}
        except Exception:
            return {}

    async def mark_article_read(self, jwt: str, article_id: str) -> bool:
        url = f"{self._base}/api/rss/articles/{article_id}/read"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.put(url, headers=self._user_headers(jwt))
                return resp.status_code < 400
        except Exception:
            return False

    async def rss_mark_all_read(self, jwt: str) -> int:
        """Mark all unread RSS articles for the user as read. Returns count updated."""
        url = f"{self._base}/api/rss/mark-all-read"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return 0
                j = resp.json()
                return int(j.get("count", 0)) if isinstance(j, dict) else 0
        except Exception:
            return 0

    async def rss_mark_feed_all_read(self, jwt: str, feed_id: str) -> int:
        """Mark all unread articles in this feed as read. Returns count updated."""
        url = f"{self._base}/api/rss/feeds/{feed_id}/mark-all-read"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return 0
                j = resp.json()
                return int(j.get("count", 0)) if isinstance(j, dict) else 0
        except Exception:
            return 0

    async def list_workspaces(self, jwt: str) -> List[Dict[str, Any]]:
        url = f"{self._base}/api/data/workspaces"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return []
                data = resp.json()
                return data if isinstance(data, list) else []
        except Exception:
            return []

    async def list_databases(self, jwt: str, workspace_id: str) -> List[Dict[str, Any]]:
        url = f"{self._base}/api/data/workspaces/{workspace_id}/databases"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return []
                data = resp.json()
                return data if isinstance(data, list) else []
        except Exception:
            return []

    async def list_tables(self, jwt: str, database_id: str) -> List[Dict[str, Any]]:
        url = f"{self._base}/api/data/databases/{database_id}/tables"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return []
                data = resp.json()
                return data if isinstance(data, list) else []
        except Exception:
            return []

    async def get_table_data(self, jwt: str, table_id: str, limit: int = 50) -> Dict[str, Any]:
        url = f"{self._base}/api/data/tables/{table_id}/data"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(
                    url,
                    params={"limit": limit, "offset": 0},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def run_sql_query(
        self, jwt: str, workspace_id: str, sql: str, limit: int = 100
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/data/workspaces/{workspace_id}/query/sql"
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    url,
                    json={"query": sql, "limit": limit, "params": None},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def list_users(self, jwt: str, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        url = f"{self._base}/api/auth/users"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    url,
                    params={"skip": skip, "limit": limit},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    return {"error": resp.text, "users": []}
                return resp.json()
        except Exception as e:
            return {"error": str(e), "users": []}

    async def create_user(
        self,
        jwt: str,
        username: str,
        email: str,
        password: str,
        display_name: Optional[str],
        role: str,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/auth/users"
        body = {
            "username": username,
            "email": email,
            "password": password,
            "display_name": display_name or username,
            "role": role,
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, json=body, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    return {"error": str(detail)}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def update_user(
        self, jwt: str, user_id: str, is_active: Optional[bool] = None, role: Optional[str] = None
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/auth/users/{user_id}"
        body: Dict[str, Any] = {}
        if is_active is not None:
            body["is_active"] = is_active
        if role is not None:
            body["role"] = role
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.put(url, json=body, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def send_chat_with_editor_context(
        self,
        jwt: str,
        conversation_id: str,
        query: str,
        active_editor: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        POST to /api/async/orchestrator/stream with active_editor context.
        Consumes the SSE stream and returns {"response": full_text} or {"error": ...}.
        """
        url = f"{self._base}/api/async/orchestrator/stream"
        payload = {
            "query": query,
            "conversation_id": conversation_id,
            "session_id": "bbs",
            "active_editor": active_editor,
            "editor_preference": "prefer",
        }
        try:
            accumulated = []
            async with httpx.AsyncClient(timeout=self._chat_timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=self._user_headers(jwt),
                ) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        return {"error": body.decode("utf-8", errors="replace")[:500]}
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        try:
                            event = json.loads(raw)
                        except Exception:
                            continue
                        event_type = event.get("type", "")
                        if event_type == "content":
                            accumulated.append(event.get("content", ""))
                        elif event_type == "error":
                            return {"error": event.get("content") or event.get("message") or "Unknown error"}
                        elif event_type == "complete":
                            break
            return {"response": "".join(accumulated)}
        except httpx.TimeoutException:
            return {"error": "Request timed out. The assistant is taking too long to respond."}
        except Exception as e:
            logger.exception("send_chat_with_editor_context failed: %s", e)
            return {"error": str(e)}

    # ── Org / todos / agenda (user JWT) ───────────────────

    async def list_todos(
        self,
        jwt: str,
        *,
        scope: str = "all",
        states: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        query: str = "",
        limit: int = 300,
        include_archives: bool = False,
        closed_since_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/todos"
        params: Dict[str, Any] = {
            "scope": scope,
            "limit": min(max(limit, 1), 500),
            "query": query or "",
            "include_archives": str(include_archives).lower(),
        }
        if states:
            params["states"] = ",".join(s.strip().upper() for s in states if s)
        if tags:
            params["tags"] = ",".join(t.strip() for t in tags if t)
        if closed_since_days is not None:
            params["closed_since_days"] = closed_since_days
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.get(url, params=params, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"success": False, "error": resp.text, "results": [], "count": 0}
                return resp.json()
        except Exception as e:
            logger.exception("list_todos failed: %s", e)
            return {"success": False, "error": str(e), "results": [], "count": 0}

    async def create_todo(
        self,
        jwt: str,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/todos"
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.post(url, json=body, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    return {"success": False, "error": str(detail)}
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def toggle_todo(
        self,
        jwt: str,
        file_path: str,
        line_number: int,
        heading_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/todos/toggle"
        payload: Dict[str, Any] = {"file_path": file_path, "line_number": line_number}
        if heading_text:
            payload["heading_text"] = heading_text
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.post(url, json=payload, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"success": False, "error": resp.text}
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def update_todo(
        self,
        jwt: str,
        file_path: str,
        line_number: int,
        updates: Dict[str, Any],
        heading_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/todos/update"
        payload: Dict[str, Any] = {"file_path": file_path, "line_number": line_number, **updates}
        if heading_text:
            payload["heading_text"] = heading_text
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.post(url, json=payload, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"success": False, "error": resp.text}
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def delete_todo(
        self,
        jwt: str,
        file_path: str,
        line_number: int,
        heading_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/todos/delete"
        payload: Dict[str, Any] = {"file_path": file_path, "line_number": line_number}
        if heading_text:
            payload["heading_text"] = heading_text
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.post(url, json=payload, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"success": False, "error": resp.text}
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def archive_todo(
        self,
        jwt: str,
        file_path: str,
        line_number: int,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/todos/archive"
        payload = {"file_path": file_path, "line_number": line_number}
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.post(url, json=payload, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"success": False, "error": resp.text}
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_org_agenda(
        self,
        jwt: str,
        *,
        days_ahead: int = 14,
        include_scheduled: bool = True,
        include_deadlines: bool = True,
        include_appointments: bool = True,
        include_org_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/org/agenda"
        params: List[Tuple[str, Any]] = [
            ("days_ahead", max(1, min(days_ahead, 90))),
            ("include_scheduled", include_scheduled),
            ("include_deadlines", include_deadlines),
            ("include_appointments", include_appointments),
        ]
        if include_org_files is not None:
            for f in include_org_files:
                if (f or "").strip():
                    params.append(("include_org_files", f.strip()))
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.get(url, params=params, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"success": False, "error": resp.text, "agenda_items": [], "count": 0}
                return resp.json()
        except Exception as e:
            logger.exception("get_org_agenda failed: %s", e)
            return {"success": False, "error": str(e), "agenda_items": [], "count": 0}

    async def discover_refile_targets(self, jwt: str) -> Dict[str, Any]:
        url = f"{self._base}/api/org/discover-targets"
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"success": False, "error": resp.text, "targets": []}
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e), "targets": []}

    async def refile_entry(
        self,
        jwt: str,
        source_file: str,
        source_line: int,
        target_file: str,
        target_heading_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base}/api/org/refile"
        body: Dict[str, Any] = {
            "source_file": source_file,
            "source_line": source_line,
            "target_file": target_file,
            "target_heading_line": target_heading_line,
        }
        try:
            async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
                resp = await client.post(url, json=body, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    return {"success": False, "error": str(detail)}
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def lookup_org_document(self, jwt: str, filename: str) -> Dict[str, Any]:
        url = f"{self._base}/api/org/lookup-document"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    url,
                    params={"filename": filename},
                    headers=self._user_headers(jwt),
                )
                if resp.status_code >= 400:
                    return {"success": False, "error": resp.text}
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_org_todo_states(self, jwt: str) -> Dict[str, Any]:
        url = f"{self._base}/api/org/settings/todo-states"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"success": False, "error": resp.text}
                return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Oregon Trail ──────────────────────────────────────

    async def ot_get_models(self, jwt: str) -> Dict[str, Any]:
        url = f"{self._base}/api/games/oregon-trail/models"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def ot_new_game(self, jwt: str, leader_name: str, party_names: List[str],
                          profession: str, model_id: str) -> Dict[str, Any]:
        url = f"{self._base}/api/games/oregon-trail/new"
        body = {"leader_name": leader_name, "party_names": party_names,
                "profession": profession, "model_id": model_id}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=body, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def ot_get_state(self, jwt: str, game_id: str) -> Dict[str, Any]:
        url = f"{self._base}/api/games/oregon-trail/{game_id}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def ot_action(self, jwt: str, game_id: str, action: str,
                        detail: Optional[str] = None, quantity: Optional[int] = None) -> Dict[str, Any]:
        url = f"{self._base}/api/games/oregon-trail/{game_id}/action"
        body: Dict[str, Any] = {"action": action}
        if detail:
            body["detail"] = detail
        if quantity is not None:
            body["quantity"] = quantity
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=body, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def ot_talk(self, jwt: str, game_id: str, message: str) -> Dict[str, Any]:
        url = f"{self._base}/api/games/oregon-trail/{game_id}/talk"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json={"message": message},
                                         headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def ot_journal(self, jwt: str, game_id: str) -> Dict[str, Any]:
        url = f"{self._base}/api/games/oregon-trail/{game_id}/journal"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def ot_list_saves(self, jwt: str) -> Dict[str, Any]:
        url = f"{self._base}/api/games/oregon-trail/saves"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def ot_delete_game(self, jwt: str, game_id: str) -> Dict[str, Any]:
        url = f"{self._base}/api/games/oregon-trail/{game_id}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.delete(url, headers=self._user_headers(jwt))
                if resp.status_code >= 400:
                    return {"error": resp.text}
                return resp.json()
        except Exception as e:
            return {"error": str(e)}
