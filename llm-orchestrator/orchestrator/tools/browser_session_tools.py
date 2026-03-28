"""
Browser session tools - Playwright-based automation via backend.
Navigate, interact, then perform a final action (download, click, extract, screenshot) with structured I/O.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for browser_run_tool (universal) ─────────────────────────────

class BrowserRunInputs(BaseModel):
    """Required inputs for browser run."""
    url: str = Field(description="URL to open (and optionally navigate from)")
    final_action_type: str = Field(
        description=(
            "The final action to perform after any steps complete. "
            "Use 'screenshot' to capture the page visually — the image is always returned inline in chat "
            "(folder_path optionally saves a copy to the library). "
            "Use 'extract' to read text or structured data from the page. "
            "Use 'click' to press a button or trigger an action. "
            "Use 'download' to save a file to folder_path."
        )
    )
    final_selector: str = Field(
        default="",
        description="For download: element to click to trigger download; click: element to click; extract: selector (empty=body); screenshot: optional element (empty=full page)",
    )
    folder_path: str = Field(
        default="",
        description="Required for download; optional for screenshot (where to save PNG); unused for click/extract",
    )


class BrowserRunParams(BaseModel):
    """Optional configuration for browser_run_tool."""
    steps: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional steps to run before the final action. Each step: "
            "action (navigate/click/fill/wait), selector, value, url, wait_for, timeout_seconds. "
            "Examples: "
            "Login: [{\"action\":\"fill\",\"selector\":\"#email\",\"value\":\"user@example.com\"}, "
            "{\"action\":\"fill\",\"selector\":\"#password\",\"value\":\"pass\"}, "
            "{\"action\":\"click\",\"selector\":\"[type=submit]\"}] "
            "then final_action_type='screenshot' to capture the logged-in page. "
            "Amazon Add to Cart: [{\"action\":\"navigate\",\"url\":\"https://amazon.com/dp/ASIN\"}, "
            "{\"action\":\"click\",\"selector\":\"#add-to-cart-button\"}] then final_action_type='extract'. "
            "Accept cookies then read: [{\"action\":\"click\",\"selector\":\".cookie-accept\"}] "
            "then final_action_type='extract'."
        ),
    )
    connection_id: Optional[str] = Field(default=None, description="External connection ID for credentials (e.g. login)")
    tags: Optional[List[str]] = Field(default=None, description="Tags for created documents (download/screenshot)")
    title: Optional[str] = Field(default=None, description="Document title override (download/screenshot)")
    goal: Optional[str] = Field(default=None, description="Short description for logging")


class BrowserRunOutputs(BaseModel):
    """Outputs for browser_run_tool; fields vary by final_action_type."""
    document_id: Optional[str] = Field(default=None, description="ID of created document (download or screenshot when saved)")
    filename: Optional[str] = Field(default=None, description="Saved filename")
    file_size_bytes: Optional[int] = Field(default=None, description="Size in bytes")
    extracted_text: Optional[str] = Field(default=None, description="Extracted text (click or extract)")
    message: Optional[str] = Field(default=None, description="Success message (e.g. click)")
    images_markdown: Optional[str] = Field(default=None, description="Markdown with inline screenshot (data URI) for chat/Telegram/Discord")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def browser_run_tool(
    url: str,
    final_action_type: str,
    final_selector: str = "",
    folder_path: str = "",
    steps: Optional[List[Dict[str, Any]]] = None,
    connection_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    title: Optional[str] = None,
    goal: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Automate a real browser: optionally run steps (navigate, click, fill, wait), then perform a final action.

    final_action_type values:
      'screenshot' — capture the page as a PNG and return it inline in chat. Use for any request
                     to 'show', 'screenshot', 'capture', 'photograph', or 'visualise' a URL.
                     Optionally saves to library if folder_path is set.
      'extract'    — return all text/content from the page (or a CSS-selected element).
                     Use for prices, articles, search results, live data, tables.
      'click'      — click a button or interactive element (Add to Cart, submit, subscribe).
      'download'   — trigger a file download and save it to folder_path in the user's library.

    Use steps for any prerequisite interactions: login, search, navigation, cookie acceptance.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.browser_run(
            url=url,
            final_action_type=final_action_type or "download",
            final_selector=final_selector or "",
            folder_path=folder_path or "",
            user_id=user_id,
            steps=steps,
            connection_id=connection_id,
            tags=tags,
            title=title,
            goal=goal,
        )
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {
                "document_id": None,
                "filename": None,
                "file_size_bytes": None,
                "extracted_text": None,
                "message": None,
                "images_markdown": None,
                "formatted": f"Browser run failed: {err}",
            }
        parts = []
        if result.get("document_id"):
            parts.append(f"Document ID: {result['document_id']}")
        if result.get("filename"):
            parts.append(f"Filename: {result['filename']}")
        if result.get("file_size_bytes") is not None:
            parts.append(f"Size: {result['file_size_bytes']} bytes")
        if result.get("extracted_text") is not None:
            text = result["extracted_text"]
            preview = (text[:500] + "…") if len(text) > 500 else text
            parts.append(f"Extracted: {preview}")
        if result.get("message"):
            parts.append(result["message"])
        formatted = "; ".join(parts) if parts else "Browser run completed successfully."
        images_markdown = result.get("images_markdown")
        if images_markdown:
            formatted = formatted + "\n\n" + images_markdown
        return {
            "document_id": result.get("document_id"),
            "filename": result.get("filename"),
            "file_size_bytes": result.get("file_size_bytes"),
            "extracted_text": result.get("extracted_text"),
            "message": result.get("message"),
            "images_markdown": images_markdown,
            "formatted": formatted,
        }
    except Exception as e:
        logger.error(f"Browser run tool error: {e}")
        return {
            "document_id": None,
            "filename": None,
            "file_size_bytes": None,
            "extracted_text": None,
            "message": None,
            "images_markdown": None,
            "formatted": f"Error: {str(e)}",
        }


register_action(
    name="browser_run",
    category="browser",
    description=(
        "Use a real browser to interact with any website. "
        "Choose the right final_action_type for the task:\n"
        "- 'screenshot': ALWAYS use this when the user asks to 'take a screenshot', "
        "'show me what X looks like', 'capture the page', 'photograph this URL', "
        "'what does this site look like', or any visual inspection request. "
        "Returns the image inline in chat. folder_path is optional (saves a copy to the library).\n"
        "- 'extract': Use to read text from a page — prices, product details, article content, "
        "search results, sports scores, stock prices, weather, tracking status, table data, "
        "or any structured/unstructured content from a live URL.\n"
        "- 'click': Use to click a button or trigger an action — Add to Cart, place order, "
        "subscribe, submit a form, accept cookies, or any single button press.\n"
        "- 'download': Use to save a file to the user's library — PDFs, reports, invoices, "
        "research papers, CSV exports, or any downloadable file. Requires folder_path.\n"
        "Use the 'steps' parameter to build multi-step sequences before the final action: "
        "login (fill username/password, click submit), navigate to a product, "
        "fill out a search box, accept a cookie banner, or any prerequisite interaction. "
        "Examples: price comparison across sites, reading paywalled content (with login), "
        "booking confirmation extraction, checking stock availability, "
        "government tracking portals, live dashboards, visual regression checks."
    ),
    inputs_model=BrowserRunInputs,
    params_model=BrowserRunParams,
    outputs_model=BrowserRunOutputs,
    tool_function=browser_run_tool,
)


# ── I/O models for browser_download_tool (alias) ─────────────────────────────

class BrowserDownloadInputs(BaseModel):
    """Required inputs for browser download."""
    url: str = Field(description="URL to open (and optionally navigate from)")
    download_selector: str = Field(description="CSS selector of the element to click to trigger the download")
    folder_path: str = Field(description="Folder path in My Documents where the file will be saved (e.g. Research/Reports)")


class BrowserDownloadParams(BaseModel):
    """Optional configuration for browser_download_tool."""
    steps: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of steps before download: each with action (navigate/click/fill/wait), selector, value, url, wait_for, timeout_seconds"
    )
    connection_id: Optional[str] = Field(default=None, description="External connection ID for credentials (e.g. login)")
    tags: Optional[List[str]] = Field(default=None, description="Tags for the created document")
    title: Optional[str] = Field(default=None, description="Document title override")
    goal: Optional[str] = Field(default=None, description="Short description of what this download achieves (for logging)")


class BrowserDownloadOutputs(BaseModel):
    """Outputs for browser_download_tool."""
    document_id: str = Field(description="ID of the created document")
    filename: str = Field(description="Saved filename")
    file_size_bytes: int = Field(description="Size of the downloaded file in bytes")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def browser_download_tool(
    url: str,
    download_selector: str,
    folder_path: str,
    steps: Optional[List[Dict[str, Any]]] = None,
    connection_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    title: Optional[str] = None,
    goal: Optional[str] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Execute a browser automation sequence and save the downloaded file into Bastion.
    Optionally run steps (navigate, click, fill, wait) before clicking the download element.
    The file is stored in the user's folder and ingested into the document pipeline.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.browser_download(
            url=url,
            download_selector=download_selector,
            folder_path=folder_path,
            user_id=user_id,
            steps=steps,
            connection_id=connection_id,
            tags=tags,
            title=title,
            goal=goal,
        )
        if not result.get("success"):
            err = result.get("error", "Unknown error")
            return {
                "document_id": "",
                "filename": "",
                "file_size_bytes": 0,
                "formatted": f"Browser download failed: {err}",
            }
        doc_id = result.get("document_id", "")
        filename = result.get("filename", "")
        size = result.get("file_size_bytes", 0)
        formatted = f"Downloaded and indexed '{filename}' ({size} bytes) → {folder_path}. Document ID: {doc_id}"
        return {
            "document_id": doc_id,
            "filename": filename,
            "file_size_bytes": size,
            "formatted": formatted,
        }
    except Exception as e:
        logger.error(f"Browser download tool error: {e}")
        return {
            "document_id": "",
            "filename": "",
            "file_size_bytes": 0,
            "formatted": f"Error: {str(e)}",
        }


register_action(
    name="browser_download",
    category="browser",
    description=(
        "Navigate a website, interact with UI elements (login, click, fill forms), "
        "then trigger a file download and save it to folder_path in the user's library. "
        "Use for: research PDFs, invoices, CSV exports, reports, government documents, "
        "or any file that requires browser interaction before downloading."
    ),
    inputs_model=BrowserDownloadInputs,
    params_model=BrowserDownloadParams,
    outputs_model=BrowserDownloadOutputs,
    tool_function=browser_download_tool,
)
