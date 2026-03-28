"""
Granular browser automation tools for playbooks.
Session-scoped: open_session returns session_id, other tools take session_id.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── browser_open_session ───────────────────────────────────────────────────

class BrowserOpenSessionInputs(BaseModel):
    site_domain: str = Field(description="Site domain for session restore key (e.g. example.com)")
class BrowserOpenSessionParams(BaseModel):
    timeout_seconds: int = Field(default=30, description="Session timeout in seconds")
class BrowserOpenSessionOutputs(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Session handle for subsequent steps")
    formatted: str = Field(description="Human-readable summary")

async def browser_open_session_tool(
    site_domain: str,
    timeout_seconds: int = 30,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Open a browser session for a site. Restores saved cookies/localStorage if available."""
    try:
        client = await get_backend_tool_client()
        result = await client.browser_open_session(
            site_domain=site_domain,
            user_id=user_id,
            timeout_seconds=timeout_seconds,
        )
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {"session_id": None, "formatted": f"Failed to open session: {err}"}
        sid = result.get("session_id") or ""
        return {"session_id": sid, "formatted": f"Session opened for {site_domain}. session_id: {sid}"}
    except Exception as e:
        logger.error(f"browser_open_session_tool error: {e}")
        return {"session_id": None, "formatted": f"Error: {str(e)}"}

register_action(
    name="browser_open_session",
    category="browser",
    description="Open a browser session for a site. Restores saved login state if available. Use session_id in later steps.",
    inputs_model=BrowserOpenSessionInputs,
    params_model=BrowserOpenSessionParams,
    outputs_model=BrowserOpenSessionOutputs,
    tool_function=browser_open_session_tool,
)


# ── browser_navigate ───────────────────────────────────────────────────────

class BrowserNavigateInputs(BaseModel):
    session_id: str = Field(description="Session from browser_open_session")
    url: str = Field(description="URL to open")
class BrowserNavigateOutputs(BaseModel):
    page_title: Optional[str] = None
    current_url: Optional[str] = None
    formatted: str = Field(description="Human-readable summary")

async def browser_navigate_tool(session_id: str, url: str, user_id: str = "system") -> Dict[str, Any]:
    try:
        client = await get_backend_tool_client()
        result = await client.browser_navigate(session_id=session_id, url=url, user_id=user_id)
        if not result.get("success"):
            return {"page_title": None, "current_url": None, "formatted": result.get("error", "Navigate failed")}
        return {
            "page_title": result.get("page_title"),
            "current_url": result.get("current_url") or url,
            "formatted": f"Navigated to {result.get('current_url') or url}",
        }
    except Exception as e:
        logger.error(f"browser_navigate_tool error: {e}")
        return {"page_title": None, "current_url": None, "formatted": str(e)}

register_action(
    name="browser_navigate",
    category="browser",
    description="Navigate the browser session to a URL.",
    inputs_model=BrowserNavigateInputs,
    params_model=None,
    outputs_model=BrowserNavigateOutputs,
    tool_function=browser_navigate_tool,
)


# ── browser_click ──────────────────────────────────────────────────────────

class BrowserClickInputs(BaseModel):
    session_id: str = Field(description="Session from browser_open_session")
    selector: str = Field(description="CSS or Playwright selector (e.g. button:has-text(\"Submit\"))")
class BrowserClickOutputs(BaseModel):
    success: bool = False
    formatted: str = Field(description="Human-readable summary")

async def browser_click_tool(session_id: str, selector: str, user_id: str = "system") -> Dict[str, Any]:
    try:
        client = await get_backend_tool_client()
        result = await client.browser_click(session_id=session_id, selector=selector, user_id=user_id)
        ok = result.get("success", False)
        return {"success": ok, "formatted": "Click completed" if ok else (result.get("error") or "Click failed")}
    except Exception as e:
        logger.error(f"browser_click_tool error: {e}")
        return {"success": False, "formatted": str(e)}

register_action(
    name="browser_click",
    category="browser",
    description="Click an element in the browser session.",
    inputs_model=BrowserClickInputs,
    params_model=None,
    outputs_model=BrowserClickOutputs,
    tool_function=browser_click_tool,
)


# ── browser_fill ───────────────────────────────────────────────────────────

class BrowserFillInputs(BaseModel):
    session_id: str = Field(description="Session from browser_open_session")
    selector: str = Field(description="Selector for the input element")
    value: str = Field(description="Text to type")
class BrowserFillOutputs(BaseModel):
    success: bool = False
    formatted: str = Field(description="Human-readable summary")

async def browser_fill_tool(session_id: str, selector: str, value: str, user_id: str = "system") -> Dict[str, Any]:
    try:
        client = await get_backend_tool_client()
        result = await client.browser_fill(session_id=session_id, selector=selector, value=value, user_id=user_id)
        ok = result.get("success", False)
        return {"success": ok, "formatted": "Fill completed" if ok else (result.get("error") or "Fill failed")}
    except Exception as e:
        logger.error(f"browser_fill_tool error: {e}")
        return {"success": False, "formatted": str(e)}

register_action(
    name="browser_fill",
    category="browser",
    description="Fill a text input or textarea in the browser session.",
    inputs_model=BrowserFillInputs,
    params_model=None,
    outputs_model=BrowserFillOutputs,
    tool_function=browser_fill_tool,
)


# ── browser_wait ────────────────────────────────────────────────────────────

class BrowserWaitInputs(BaseModel):
    session_id: str = Field(description="Session from browser_open_session")
class BrowserWaitParams(BaseModel):
    selector: Optional[str] = Field(default=None, description="Wait for this selector to appear")
    timeout_seconds: Optional[int] = Field(default=None, description="Or wait this many seconds")
class BrowserWaitOutputs(BaseModel):
    found: bool = False
    formatted: str = Field(description="Human-readable summary")

async def browser_wait_tool(
    session_id: str,
    selector: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    try:
        client = await get_backend_tool_client()
        result = await client.browser_wait(
            session_id=session_id,
            user_id=user_id,
            selector=selector,
            timeout_seconds=timeout_seconds,
        )
        found = result.get("found", False)
        return {"found": found, "formatted": "Wait completed" if result.get("success") else (result.get("error") or "Wait failed")}
    except Exception as e:
        logger.error(f"browser_wait_tool error: {e}")
        return {"found": False, "formatted": str(e)}

register_action(
    name="browser_wait",
    category="browser",
    description="Wait for a selector to appear or for a fixed number of seconds.",
    inputs_model=BrowserWaitInputs,
    params_model=BrowserWaitParams,
    outputs_model=BrowserWaitOutputs,
    tool_function=browser_wait_tool,
)


# ── browser_scroll ─────────────────────────────────────────────────────────

class BrowserScrollInputs(BaseModel):
    session_id: str = Field(description="Session from browser_open_session")
class BrowserScrollParams(BaseModel):
    direction: str = Field(default="down", description="'down' or 'up'")
    amount_pixels: int = Field(default=800, description="Pixels to scroll (e.g. for infinite scroll)")
class BrowserScrollOutputs(BaseModel):
    success: bool = False
    formatted: str = Field(description="Human-readable summary")

async def browser_scroll_tool(
    session_id: str,
    direction: str = "down",
    amount_pixels: int = 800,
    user_id: str = "system",
) -> Dict[str, Any]:
    """Scroll the page. Use for infinite-scroll pages to load more content before extract."""
    try:
        client = await get_backend_tool_client()
        result = await client.browser_scroll(
            session_id=session_id,
            user_id=user_id,
            direction=direction or "down",
            amount_pixels=amount_pixels if amount_pixels > 0 else 800,
        )
        ok = result.get("success", False)
        return {"success": ok, "formatted": f"Scrolled {direction} {amount_pixels}px" if ok else (result.get("error") or "Scroll failed")}
    except Exception as e:
        logger.error(f"browser_scroll_tool error: {e}")
        return {"success": False, "formatted": str(e)}

register_action(
    name="browser_scroll",
    category="browser",
    description="Scroll the page down or up by a number of pixels. Use for infinite-scroll pages to load more content.",
    inputs_model=BrowserScrollInputs,
    params_model=BrowserScrollParams,
    outputs_model=BrowserScrollOutputs,
    tool_function=browser_scroll_tool,
)


# ── browser_extract ────────────────────────────────────────────────────────

class BrowserExtractInputs(BaseModel):
    session_id: str = Field(description="Session from browser_open_session")
    selector: str = Field(default="", description="CSS selector; empty = full page body")
class BrowserExtractOutputs(BaseModel):
    extracted_text: Optional[str] = None
    formatted: str = Field(description="Human-readable summary")

async def browser_extract_tool(session_id: str, selector: str = "", user_id: str = "system") -> Dict[str, Any]:
    try:
        client = await get_backend_tool_client()
        result = await client.browser_extract(session_id=session_id, selector=selector, user_id=user_id)
        text = result.get("extracted_text") or ""
        preview = (text[:500] + "…") if len(text) > 500 else text
        return {"extracted_text": text, "formatted": f"Extracted: {preview}" if text else (result.get("error") or "Extract failed")}
    except Exception as e:
        logger.error(f"browser_extract_tool error: {e}")
        return {"extracted_text": None, "formatted": str(e)}

register_action(
    name="browser_extract",
    category="browser",
    description="Extract text from the page or from a selected element.",
    inputs_model=BrowserExtractInputs,
    params_model=None,
    outputs_model=BrowserExtractOutputs,
    tool_function=browser_extract_tool,
)


# ── browser_inspect ────────────────────────────────────────────────────────

class BrowserInspectInputs(BaseModel):
    session_id: str = Field(description="Session from browser_open_session")
class BrowserInspectOutputs(BaseModel):
    page_structure: Optional[str] = Field(default=None, description="JSON: accessibility tree and interactive elements with selectors")
    formatted: str = Field(description="Human-readable summary")

async def browser_inspect_tool(session_id: str, user_id: str = "system") -> Dict[str, Any]:
    """Return page structure (accessibility tree + interactive elements with selectors) for playbook authoring."""
    try:
        client = await get_backend_tool_client()
        result = await client.browser_inspect(session_id=session_id, user_id=user_id)
        if not result.get("success"):
            return {"page_structure": None, "formatted": result.get("error", "Inspect failed")}
        raw = result.get("page_structure") or "{}"
        return {"page_structure": raw, "formatted": f"Page structure retrieved ({len(raw)} chars). Use page_structure to find selectors for click/fill/extract."}
    except Exception as e:
        logger.error(f"browser_inspect_tool error: {e}")
        return {"page_structure": None, "formatted": str(e)}

register_action(
    name="browser_inspect",
    category="browser",
    description="Get page structure (accessibility tree and interactive elements with CSS selectors). Use when building playbooks to discover selectors.",
    inputs_model=BrowserInspectInputs,
    params_model=None,
    outputs_model=BrowserInspectOutputs,
    tool_function=browser_inspect_tool,
)


# ── browser_screenshot ────────────────────────────────────────────────────

class BrowserScreenshotInputs(BaseModel):
    session_id: str = Field(description="Session from browser_open_session")
class BrowserScreenshotParams(BaseModel):
    folder_path: str = Field(default="", description="Save PNG to this folder in My Documents; empty = inline only")
    tags: Optional[List[str]] = None
    title: Optional[str] = None
class BrowserScreenshotOutputs(BaseModel):
    images_markdown: Optional[str] = None
    document_id: Optional[str] = None
    filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    formatted: str = Field(description="Human-readable summary")

async def browser_screenshot_tool(
    session_id: str,
    folder_path: str = "",
    tags: Optional[List[str]] = None,
    title: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    try:
        client = await get_backend_tool_client()
        result = await client.browser_screenshot(
            session_id=session_id,
            user_id=user_id,
            folder_path=folder_path or "",
            tags=tags or [],
            title=title,
        )
        if not result.get("success"):
            return {"images_markdown": None, "document_id": None, "filename": None, "file_size_bytes": None, "formatted": result.get("error", "Screenshot failed")}
        md = result.get("images_markdown")
        return {
            "images_markdown": md,
            "document_id": result.get("document_id"),
            "filename": result.get("filename"),
            "file_size_bytes": result.get("file_size_bytes"),
            "formatted": "Screenshot captured" + (f" and saved as {result.get('filename')}" if result.get("filename") else ""),
        }
    except Exception as e:
        logger.error(f"browser_screenshot_tool error: {e}")
        return {"images_markdown": None, "document_id": None, "filename": None, "file_size_bytes": None, "formatted": str(e)}

register_action(
    name="browser_screenshot",
    category="browser",
    description="Capture a screenshot of the current page. Optionally save to folder_path.",
    inputs_model=BrowserScreenshotInputs,
    params_model=BrowserScreenshotParams,
    outputs_model=BrowserScreenshotOutputs,
    tool_function=browser_screenshot_tool,
)


# ── browser_download (granular; session-scoped) ─────────────────────────────

class BrowserDownloadInputs(BaseModel):
    session_id: str = Field(description="Session from browser_open_session")
    selector: str = Field(description="Selector of element to click to trigger download")
    folder_path: str = Field(description="Folder in My Documents to save the file")
class BrowserDownloadParams(BaseModel):
    tags: Optional[List[str]] = None
    title: Optional[str] = None
class BrowserDownloadOutputs(BaseModel):
    document_id: Optional[str] = None
    filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    formatted: str = Field(description="Human-readable summary")

async def browser_download_file_tool(
    session_id: str,
    selector: str,
    folder_path: str,
    tags: Optional[List[str]] = None,
    title: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    try:
        client = await get_backend_tool_client()
        result = await client.browser_download_file(
            session_id=session_id,
            selector=selector,
            folder_path=folder_path,
            user_id=user_id,
            tags=tags or [],
            title=title,
        )
        if not result.get("success"):
            return {"document_id": None, "filename": None, "file_size_bytes": None, "formatted": result.get("error", "Download failed")}
        return {
            "document_id": result.get("document_id"),
            "filename": result.get("filename"),
            "file_size_bytes": result.get("file_size_bytes"),
            "formatted": f"Downloaded {result.get('filename', 'file')} to {folder_path}",
        }
    except Exception as e:
        logger.error(f"browser_download_tool error: {e}")
        return {"document_id": None, "filename": None, "file_size_bytes": None, "formatted": str(e)}

register_action(
    name="browser_download_file",
    category="browser",
    description="Click an element to trigger a file download and save it to folder_path in My Documents. Use with an open session.",
    inputs_model=BrowserDownloadInputs,
    params_model=BrowserDownloadParams,
    outputs_model=BrowserDownloadOutputs,
    tool_function=browser_download_file_tool,
)


# ── browser_close_session ──────────────────────────────────────────────────

class BrowserCloseSessionInputs(BaseModel):
    session_id: str = Field(description="Session to close")
    site_domain: str = Field(description="Site domain (for saving state key)")
class BrowserCloseSessionParams(BaseModel):
    save_state: bool = Field(default=False, description="Persist cookies/localStorage for next open")
class BrowserCloseSessionOutputs(BaseModel):
    session_saved: bool = False
    formatted: str = Field(description="Human-readable summary")

async def browser_close_session_tool(
    session_id: str,
    site_domain: str,
    save_state: bool = False,
    user_id: str = "system",
) -> Dict[str, Any]:
    try:
        client = await get_backend_tool_client()
        result = await client.browser_close_session(
            session_id=session_id,
            site_domain=site_domain,
            save_state=save_state,
            user_id=user_id,
        )
        saved = result.get("session_saved", False)
        return {"session_saved": saved, "formatted": "Session closed" + (" and state saved" if saved else "")}
    except Exception as e:
        logger.error(f"browser_close_session_tool error: {e}")
        return {"session_saved": False, "formatted": str(e)}

register_action(
    name="browser_close_session",
    category="browser",
    description="Close the browser session. Use save_state=true to persist login for next time.",
    inputs_model=BrowserCloseSessionInputs,
    params_model=BrowserCloseSessionParams,
    outputs_model=BrowserCloseSessionOutputs,
    tool_function=browser_close_session_tool,
)
