"""
Playwright browser automation tools.
Navigate, interact, then perform a final action (download, click, extract, screenshot)
via crawl4ai-service BrowserSessionService; persist files into Bastion via create_user_file.
"""

import base64
import logging
import time
from typing import Any, Dict, List, Optional

from clients.crawl_service_client import get_crawl_service_client
from services.langgraph_tools.file_creation_tools import create_user_file

logger = logging.getLogger(__name__)


async def _run_steps(
    client: Any,
    session_id: str,
    url: str,
    steps: List[Dict[str, Any]],
) -> Optional[str]:
    """Execute steps; return None on success, or error message on failure."""
    if not steps:
        result = await client.browser_execute_action(session_id, "navigate", url=url)
        if not result.get("success"):
            return result.get("error", "Navigate failed")
        return None
    for step in steps:
        action = (step.get("action") or "").lower()
        if action == "navigate":
            u = step.get("url") or url
            result = await client.browser_execute_action(session_id, "navigate", url=u)
        elif action == "click":
            result = await client.browser_execute_action(
                session_id, "click", selector=step.get("selector")
            )
        elif action == "fill":
            result = await client.browser_execute_action(
                session_id, "fill",
                selector=step.get("selector"),
                value=step.get("value", ""),
            )
        elif action == "wait":
            result = await client.browser_execute_action(
                session_id, "wait",
                wait_selector=step.get("wait_for"),
                wait_timeout_seconds=step.get("timeout_seconds"),
            )
        else:
            continue
        if not result.get("success"):
            return result.get("error", f"Step {action} failed")
    return None


async def browser_run(
    user_id: str,
    url: str,
    final_action_type: str,
    final_selector: str,
    folder_path: str = "",
    steps: Optional[List[Dict[str, Any]]] = None,
    connection_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    title: Optional[str] = None,
    goal: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a browser session: execute steps, then perform the final action (download, click,
    extract, or screenshot). Returns a unified dict with success, error, and action-specific
    fields (document_id, filename, file_size_bytes for download/screenshot; extracted_text,
    message for click/extract).
    """
    steps = steps or []
    tags = tags or []
    action_type = (final_action_type or "download").lower()
    client = await get_crawl_service_client()
    session_id = None
    try:
        session_id = await client.browser_create_session(timeout_seconds=60)
        if not session_id:
            return {"success": False, "error": "Failed to create browser session"}

        err = await _run_steps(client, session_id, url, steps)
        if err:
            return {"success": False, "error": err}

        if action_type == "download":
            if not final_selector:
                return {"success": False, "error": "final_selector required for download"}
            if not folder_path:
                return {"success": False, "error": "folder_path required for download"}
            download_result = await client.browser_download_file(
                session_id=session_id,
                trigger_selector=final_selector,
                timeout_seconds=60,
            )
            if not download_result.get("success"):
                return {
                    "success": False,
                    "error": download_result.get("error", "Download failed"),
                }
            file_content = download_result.get("file_content") or b""
            filename = download_result.get("filename") or "download"
            if not file_content:
                return {"success": False, "error": "Download produced no content"}
            create_result = await create_user_file(
                filename=filename,
                content="",
                folder_path=folder_path,
                title=title or filename,
                tags=tags,
                user_id=user_id,
                content_bytes=file_content,
            )
            if not create_result.get("success"):
                return {
                    "success": False,
                    "error": create_result.get("error", "Failed to save file"),
                }
            return {
                "success": True,
                "document_id": create_result.get("document_id", ""),
                "filename": create_result.get("filename", filename),
                "file_size_bytes": len(file_content),
            }

        if action_type == "click":
            if not final_selector:
                return {"success": False, "error": "final_selector required for click"}
            result = await client.browser_execute_action(
                session_id, "click", selector=final_selector
            )
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "Click failed")}
            return {
                "success": True,
                "message": "Click completed",
                "extracted_text": result.get("extracted_content"),
            }

        if action_type == "extract":
            result = await client.browser_execute_action(
                session_id, "extract", selector=final_selector or ""
            )
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "Extract failed")}
            return {
                "success": True,
                "extracted_text": result.get("extracted_content") or "",
            }

        if action_type == "screenshot":
            result = await client.browser_execute_action(session_id, "screenshot")
            if not result.get("success"):
                return {"success": False, "error": result.get("error", "Screenshot failed")}
            png_bytes = result.get("screenshot_png") or b""
            if not png_bytes:
                return {"success": False, "error": "Screenshot produced no image"}
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            images_markdown = f"![Screenshot](data:image/png;base64,{b64})"
            if folder_path:
                filename = title or f"screenshot_{int(time.time())}.png"
                create_result = await create_user_file(
                    filename=filename,
                    content="",
                    folder_path=folder_path,
                    title=title or filename,
                    tags=tags,
                    user_id=user_id,
                    content_bytes=png_bytes,
                )
                if not create_result.get("success"):
                    return {
                        "success": False,
                        "error": create_result.get("error", "Failed to save screenshot"),
                    }
                return {
                    "success": True,
                    "images_markdown": images_markdown,
                    "document_id": create_result.get("document_id", ""),
                    "filename": create_result.get("filename", filename),
                    "file_size_bytes": len(png_bytes),
                }
            return {
                "success": True,
                "images_markdown": images_markdown,
                "file_size_bytes": len(png_bytes),
            }

        return {"success": False, "error": f"Unknown final_action_type: {final_action_type}"}
    except Exception as e:
        logger.error(f"browser_run failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if session_id:
            await client.browser_destroy_session(session_id)


async def browser_download(
    user_id: str,
    url: str,
    download_selector: str,
    folder_path: str,
    steps: Optional[List[Dict[str, Any]]] = None,
    connection_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    title: Optional[str] = None,
    goal: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a browser session and trigger a file download (saved to folder_path).
    Thin wrapper around browser_run with final_action_type=download.
    Returns dict with success, document_id, filename, file_size_bytes, error.
    """
    result = await browser_run(
        user_id=user_id,
        url=url,
        final_action_type="download",
        final_selector=download_selector,
        folder_path=folder_path,
        steps=steps,
        connection_id=connection_id,
        tags=tags,
        title=title,
        goal=goal,
    )
    return result
