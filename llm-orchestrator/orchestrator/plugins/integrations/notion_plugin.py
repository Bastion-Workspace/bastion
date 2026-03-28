"""
Notion plugin - search pages, get page, create page, query database for Agent Factory (Zone 4).

Uses Notion API. Requires an integration token (Internal Integration secret) in connection config.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.plugins.base_plugin import BasePlugin, PluginToolSpec


class SearchPagesInputs(BaseModel):
    """Inputs for searching Notion pages."""

    query: str = Field(default="", description="Search query; empty returns recent")
    limit: int = Field(default=20, description="Max pages to return")


class PageRef(BaseModel):
    """Reference to a Notion page."""

    id: str = Field(description="Page ID")
    title: str = Field(description="Title (plain text)")
    url: Optional[str] = Field(default=None, description="Notion URL")
    object_type: str = Field(default="page", description="page or database")


class SearchPagesOutputs(BaseModel):
    """Outputs for Notion search pages tool."""

    pages: List[PageRef] = Field(description="List of pages/databases")
    count: int = Field(description="Number of results")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetPageInputs(BaseModel):
    """Inputs for getting a Notion page."""

    page_id: str = Field(description="Notion page ID (UUID)")


class GetPageOutputs(BaseModel):
    """Outputs for get Notion page tool."""

    page_id: str = Field(description="Page ID")
    title: str = Field(description="Title")
    url: Optional[str] = Field(default=None)
    content_preview: Optional[str] = Field(default=None, description="First 500 chars of content summary")
    success: bool = Field(description="Whether fetch succeeded")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class CreatePageInputs(BaseModel):
    """Inputs for creating a Notion page."""

    parent_id: str = Field(description="Parent page ID or database ID")
    parent_type: str = Field(default="page", description="page or database")
    title: str = Field(description="Page title")
    content: str = Field(default="", description="Initial content (plain text or markdown)")


class CreatePageOutputs(BaseModel):
    """Outputs for create Notion page tool."""

    page_id: str = Field(description="Created page ID")
    url: Optional[str] = Field(default=None)
    success: bool = Field(description="Whether creation succeeded")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class QueryDatabaseInputs(BaseModel):
    """Inputs for querying a Notion database."""

    database_id: str = Field(description="Notion database ID (UUID)")
    limit: int = Field(default=20, description="Max results")
    filter_json: Optional[str] = Field(default=None, description="Optional filter (JSON string)")


class DatabaseRowRef(BaseModel):
    """Reference to a database row (page)."""

    id: str = Field(description="Page/row ID")
    title: str = Field(description="Row title or first rich text")
    url: Optional[str] = Field(default=None)


class QueryDatabaseOutputs(BaseModel):
    """Outputs for Notion query database tool."""

    rows: List[DatabaseRowRef] = Field(description="List of rows")
    count: int = Field(description="Number of rows")
    has_more: bool = Field(default=False, description="Whether more results exist")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class NotionPlugin(BasePlugin):
    """Notion integration - search pages, get page, create page, query database."""

    NOTION_VERSION = "2022-06-28"

    @property
    def plugin_name(self) -> str:
        return "notion"

    @property
    def plugin_version(self) -> str:
        return "0.1.0"

    def get_connection_requirements(self) -> Dict[str, str]:
        return {
            "token": "Notion Integration Token (Internal Integration secret)",
        }

    def get_tools(self) -> List[PluginToolSpec]:
        return [
            PluginToolSpec(
                name="notion_search_pages",
                category="plugin:notion",
                description="Search Notion pages and databases",
                inputs_model=SearchPagesInputs,
                outputs_model=SearchPagesOutputs,
                tool_function=self._search_pages,
            ),
            PluginToolSpec(
                name="notion_get_page",
                category="plugin:notion",
                description="Get a Notion page by ID",
                inputs_model=GetPageInputs,
                outputs_model=GetPageOutputs,
                tool_function=self._get_page,
            ),
            PluginToolSpec(
                name="notion_create_page",
                category="plugin:notion",
                description="Create a Notion page under a parent page or database",
                inputs_model=CreatePageInputs,
                outputs_model=CreatePageOutputs,
                tool_function=self._create_page,
            ),
            PluginToolSpec(
                name="notion_query_database",
                category="plugin:notion",
                description="Query a Notion database",
                inputs_model=QueryDatabaseInputs,
                outputs_model=QueryDatabaseOutputs,
                tool_function=self._query_database,
            ),
        ]

    def _headers(self) -> Dict[str, str]:
        config = getattr(self, "_config", None) or {}
        token = config.get("token", "")
        return {
            "Authorization": f"Bearer {token}" if token else "",
            "Notion-Version": self.NOTION_VERSION,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _rich_text_to_plain(blocks: List[Dict[str, Any]]) -> str:
        """Extract plain text from Notion rich_text array."""
        if not blocks:
            return ""
        return "".join(b.get("plain_text", "") for b in blocks)

    @staticmethod
    def _page_title(obj: Dict[str, Any]) -> str:
        """Get page title from Notion page object."""
        props = obj.get("properties", {})
        title_prop = props.get("title") or props.get("Name") or next((p for p in props.values() if p.get("type") == "title"), None)
        if not title_prop or title_prop.get("type") != "title":
            return ""
        return NotionPlugin._rich_text_to_plain(title_prop.get("title", []))

    async def _search_pages(self, query: str = "", limit: int = 20) -> Dict[str, Any]:
        """Search Notion pages and databases."""
        if not getattr(self, "_config", None) or not (getattr(self, "_config", None) or {}).get("token"):
            return {"pages": [], "count": 0, "formatted": "Notion plugin: configure token to search."}
        try:
            import aiohttp
            url = "https://api.notion.com/v1/search"
            payload = {"page_size": min(limit, 100)}
            if query:
                payload["query"] = query
            out = []
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self._headers()) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return {"pages": [], "count": 0, "formatted": f"Notion API error ({resp.status}): {text[:200]}"}
                    data = await resp.json()
            for r in data.get("results", [])[:limit]:
                obj_type = r.get("object", "page")
                title = self._page_title(r) if r.get("properties") else (r.get("title", [{}])[0].get("plain_text", "") if r.get("title") else "")
                if not title and r.get("object") == "database":
                    title = (r.get("title", []) or [{}])[0].get("plain_text", "Database") if isinstance(r.get("title"), list) else "Database"
                out.append(PageRef(
                    id=r.get("id", ""),
                    title=title or "Untitled",
                    url=r.get("url"),
                    object_type=r.get("object", "page"),
                ))
            formatted = f"Found {len(out)} result(s)." if out else "No results."
            return {"pages": [p.model_dump() for p in out], "count": len(out), "formatted": formatted}
        except ImportError:
            return {"pages": [], "count": 0, "formatted": "Notion plugin: aiohttp not installed."}
        except Exception as e:
            return {"pages": [], "count": 0, "formatted": f"Notion search failed: {e}"}

    async def _get_page(self, page_id: str) -> Dict[str, Any]:
        """Get a Notion page by ID."""
        if not getattr(self, "_config", None) or not (getattr(self, "_config", None) or {}).get("token"):
            return {"page_id": page_id, "title": "", "url": None, "content_preview": None, "success": False, "formatted": "Notion plugin: configure token."}
        try:
            import aiohttp
            url = f"https://api.notion.com/v1/pages/{page_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._headers()) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return {
                            "page_id": page_id,
                            "title": "",
                            "url": None,
                            "content_preview": None,
                            "success": False,
                            "formatted": f"Notion API error ({resp.status}): {text[:200]}",
                        }
                    data = await resp.json()
            title = self._page_title(data)
            preview = None
            if data.get("properties"):
                for v in data["properties"].values():
                    if v.get("type") == "rich_text" and v.get("rich_text"):
                        preview = self._rich_text_to_plain(v["rich_text"])[:500]
                        break
            formatted = f"Page: {title}" + (f" — {preview}..." if preview else "")
            return {
                "page_id": data.get("id", page_id),
                "title": title or "Untitled",
                "url": data.get("url"),
                "content_preview": preview,
                "success": True,
                "formatted": formatted,
            }
        except ImportError:
            return {"page_id": page_id, "title": "", "url": None, "content_preview": None, "success": False, "formatted": "Notion plugin: aiohttp not installed."}
        except Exception as e:
            return {"page_id": page_id, "title": "", "url": None, "content_preview": None, "success": False, "formatted": f"Notion get page failed: {e}"}

    async def _create_page(self, parent_id: str, parent_type: str = "page", title: str = "", content: str = "") -> Dict[str, Any]:
        """Create a Notion page under a parent page or database."""
        if not getattr(self, "_config", None) or not (getattr(self, "_config", None) or {}).get("token"):
            return {"page_id": "", "url": None, "success": False, "formatted": "Notion plugin: configure token."}
        try:
            import aiohttp
            url = "https://api.notion.com/v1/pages"
            parent_id_clean = parent_id.replace("-", "")
            parent_key = "database_id" if (parent_type or "").strip().lower() == "database" else "page_id"
            payload = {
                "parent": {parent_key: parent_id_clean},
                "properties": {
                    "title": {"title": [{"type": "text", "text": {"content": (title or "Untitled")[:2000]}}]},
                },
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self._headers()) as resp:
                    if resp.status not in (200, 201):
                        text = await resp.text()
                        return {"page_id": "", "url": None, "success": False, "formatted": f"Notion API error ({resp.status}): {text[:200]}"}
                    data = await resp.json()
            page_id = data.get("id", "")
            page_url = data.get("url")
            formatted = f"Created page: {title} ({page_url})" if page_url else f"Created page: {title}"
            return {"page_id": page_id, "url": page_url, "success": True, "formatted": formatted}
        except ImportError:
            return {"page_id": "", "url": None, "success": False, "formatted": "Notion plugin: aiohttp not installed."}
        except Exception as e:
            return {"page_id": "", "url": None, "success": False, "formatted": f"Notion create page failed: {e}"}

    async def _query_database(self, database_id: str, limit: int = 20, filter_json: Optional[str] = None) -> Dict[str, Any]:
        """Query a Notion database."""
        if not getattr(self, "_config", None) or not (getattr(self, "_config", None) or {}).get("token"):
            return {"rows": [], "count": 0, "has_more": False, "formatted": "Notion plugin: configure token."}
        try:
            import aiohttp
            import json
            db_id_clean = database_id.replace("-", "")
            url = f"https://api.notion.com/v1/databases/{db_id_clean}/query"
            payload = {"page_size": min(limit, 100)}
            if filter_json:
                try:
                    payload["filter"] = json.loads(filter_json)
                except json.JSONDecodeError:
                    pass
            out = []
            has_more = False
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self._headers()) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return {"rows": [], "count": 0, "has_more": False, "formatted": f"Notion API error ({resp.status}): {text[:200]}"}
                    data = await resp.json()
            has_more = data.get("has_more", False)
            for r in data.get("results", []):
                title = self._page_title(r)
                out.append(DatabaseRowRef(id=r.get("id", ""), title=title or "Untitled", url=r.get("url")))
                if len(out) >= limit:
                    break
            formatted = f"Found {len(out)} row(s)." if out else "No rows."
            return {"rows": [x.model_dump() for x in out], "count": len(out), "has_more": has_more, "formatted": formatted}
        except ImportError:
            return {"rows": [], "count": 0, "has_more": False, "formatted": "Notion plugin: aiohttp not installed."}
        except Exception as e:
            return {"rows": [], "count": 0, "has_more": False, "formatted": f"Notion query database failed: {e}"}
