"""
Browser Session Service - Playwright-based interactive automation.
Manages persistent browser sessions for click, fill, download, and extract actions.
"""

import asyncio
import json
import logging
import random
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

logger = logging.getLogger(__name__)

# Session TTL: 10 minutes idle; sessions are cleaned on next operation if expired
SESSION_IDLE_TIMEOUT_SECONDS = 600
# Random delay range (seconds) between actions to appear more human-like
ACTION_DELAY_MIN = 0.05
ACTION_DELAY_MAX = 0.2


class BrowserSessionServiceImplementation:
    """Playwright browser session management and action execution."""

    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._sessions: Dict[str, Tuple[Browser, Page]] = {}
        self._session_last_used: Dict[str, float] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Start Playwright. Call once at service startup."""
        if self._initialized:
            return
        try:
            import time
            self._playwright = await async_playwright().start()
            self._initialized = True
            logger.info("Browser session service initialized (Playwright started)")
        except Exception as e:
            logger.error(f"Failed to initialize browser session service: {e}")
            raise

    async def cleanup(self) -> None:
        """Close all sessions and stop Playwright."""
        for session_id in list(self._sessions.keys()):
            await self._destroy_session(session_id)
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._initialized = False
        logger.info("Browser session service cleaned up")

    def _touch_session(self, session_id: str) -> None:
        self._session_last_used[session_id] = time.time()

    async def _cleanup_expired_sessions(self) -> None:
        now = time.time()
        expired = [
            sid for sid, last in self._session_last_used.items()
            if (now - last) > SESSION_IDLE_TIMEOUT_SECONDS
        ]
        for session_id in expired:
            try:
                await self._destroy_session(session_id)
            except Exception as e:
                logger.warning(f"Failed to cleanup expired session {session_id}: {e}")

    async def create_session(self, request: Any) -> Any:
        """Create a new browser session and return session_id. Optionally restore from storage_state_json."""
        from protos import crawl_service_pb2
        await self._cleanup_expired_sessions()
        if not self._playwright:
            return crawl_service_pb2.BrowserSessionRef(session_id="")
        try:
            timeout_ms = (request.timeout_seconds * 1000) if request.timeout_seconds and request.timeout_seconds > 0 else 30000
            browser = await self._playwright.chromium.launch(headless=True)
            context_options = {
                "ignore_https_errors": True,
            }
            if request.user_agent:
                context_options["user_agent"] = request.user_agent
            storage_state_json = getattr(request, "storage_state_json", None) or ""
            if storage_state_json and storage_state_json.strip():
                try:
                    state_dict = json.loads(storage_state_json)
                    context_options["storage_state"] = state_dict
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid storage_state_json: {e}")
            context = await browser.new_context(**context_options)
            try:
                from playwright_stealth import Stealth
                stealth = Stealth()
                await stealth.apply_stealth_async(context)
            except Exception as e:
                logger.debug("playwright_stealth not applied: %s", e)
            context.set_default_timeout(timeout_ms)
            page = await context.new_page()
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = (browser, page)
            self._session_last_used[session_id] = time.time()
            logger.info(f"Created browser session {session_id}")
            return crawl_service_pb2.BrowserSessionRef(session_id=session_id)
        except Exception as e:
            logger.error(f"CreateSession failed: {e}")
            return crawl_service_pb2.BrowserSessionRef(session_id="")

    def _get_page(self, session_id: str) -> Optional[Page]:
        entry = self._sessions.get(session_id)
        if not entry:
            return None
        return entry[1]

    def _get_context(self, session_id: str) -> Optional[BrowserContext]:
        """Get the browser context for a session (for storage_state)."""
        entry = self._sessions.get(session_id)
        if not entry:
            return None
        _, page = entry
        return page.context

    async def _destroy_session(self, session_id: str) -> None:
        entry = self._sessions.pop(session_id, None)
        self._session_last_used.pop(session_id, None)
        if entry:
            browser, page = entry
            try:
                await page.close()
            except Exception as e:
                logger.warning(f"Error closing page for session {session_id}: {e}")
            try:
                await browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser for session {session_id}: {e}")
            logger.info(f"Destroyed browser session {session_id}")

    async def execute_action(self, request: Any) -> Any:
        """Execute a single browser action (navigate, click, fill, wait, extract, screenshot)."""
        from protos import crawl_service_pb2
        await asyncio.sleep(random.uniform(ACTION_DELAY_MIN, ACTION_DELAY_MAX))
        page = self._get_page(request.session_id)
        if not page:
            return crawl_service_pb2.BrowserActionResult(success=False, error="Session not found or expired")
        self._touch_session(request.session_id)
        try:
            which = request.WhichOneof("action")
            if which == "navigate":
                await page.goto(request.navigate.url, wait_until="domcontentloaded", timeout=30000)
                return crawl_service_pb2.BrowserActionResult(success=True)
            if which == "click":
                c = request.click
                if c.HasField("click_x") and c.HasField("click_y"):
                    await page.mouse.click(c.click_x, c.click_y)
                elif c.selector:
                    await page.click(c.selector, timeout=10000)
                else:
                    return crawl_service_pb2.BrowserActionResult(
                        success=False, error="Click requires selector or click_x/click_y"
                    )
                return crawl_service_pb2.BrowserActionResult(success=True)
            if which == "fill":
                await page.fill(request.fill.selector, request.fill.value, timeout=10000)
                return crawl_service_pb2.BrowserActionResult(success=True)
            if which == "wait":
                w = request.wait
                if w.selector:
                    await page.wait_for_selector(w.selector, timeout=(w.timeout_seconds or 10) * 1000)
                elif w.timeout_seconds:
                    await page.wait_for_timeout(w.timeout_seconds * 1000)
                return crawl_service_pb2.BrowserActionResult(success=True)
            if which == "extract":
                if request.extract.selector:
                    el = await page.query_selector(request.extract.selector)
                    content = await el.inner_text() if el else ""
                else:
                    content = await page.inner_text("body")
                return crawl_service_pb2.BrowserActionResult(success=True, extracted_content=content)
            if which == "screenshot":
                png = await page.screenshot(type="png", full_page=False)
                return crawl_service_pb2.BrowserActionResult(success=True, screenshot_png=png)
            if which == "scroll":
                s = request.scroll
                direction = (s.direction or "down").strip().lower()
                amount = s.amount_pixels if s.amount_pixels and s.amount_pixels > 0 else 800
                delta_y = -amount if direction == "up" else amount
                await page.evaluate(f"window.scrollBy({{ left: 0, top: {delta_y}, behavior: 'smooth' }})")
                await page.wait_for_timeout(300)
                return crawl_service_pb2.BrowserActionResult(success=True)
            if which == "keypress":
                key = (request.keypress.key or "").strip() or "Enter"
                await page.keyboard.press(key)
                return crawl_service_pb2.BrowserActionResult(success=True)
            return crawl_service_pb2.BrowserActionResult(success=False, error="Unknown action")
        except Exception as e:
            logger.error(f"ExecuteAction failed: {e}")
            return crawl_service_pb2.BrowserActionResult(success=False, error=str(e))

    async def download_file(self, request: Any) -> Any:
        """Click the trigger selector and capture the downloaded file."""
        from protos import crawl_service_pb2
        await asyncio.sleep(random.uniform(ACTION_DELAY_MIN, ACTION_DELAY_MAX))
        page = self._get_page(request.session_id)
        if not page:
            return crawl_service_pb2.BrowserDownloadResult(success=False, error="Session not found or expired")
        self._touch_session(request.session_id)
        timeout_ms = (request.timeout_seconds * 1000) if request.timeout_seconds and request.timeout_seconds > 0 else 30000
        try:
            if request.fallback_url:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(request.fallback_url, timeout=timeout_ms / 1000)
                    resp.raise_for_status()
                    content = resp.content
                    filename = request.fallback_url.rstrip("/").split("/")[-1].split("?")[0] or "download"
                    ctype = resp.headers.get("content-type", "application/octet-stream")
                    if ";" in ctype:
                        ctype = ctype.split(";")[0].strip()
                return crawl_service_pb2.BrowserDownloadResult(
                    success=True,
                    file_content=content,
                    filename=filename,
                    mime_type=ctype,
                    file_size_bytes=len(content),
                )
            async with page.expect_download(timeout=timeout_ms) as download_info:
                await page.click(request.trigger_selector, timeout=10000)
            download = await download_info.value
            path = await download.path()
            with open(path, "rb") as f:
                content = f.read()
            filename = download.suggested_filename or "download"
            return crawl_service_pb2.BrowserDownloadResult(
                success=True,
                file_content=content,
                filename=filename,
                mime_type=download.suggested_filename and _mime_for_filename(download.suggested_filename) or "application/octet-stream",
                file_size_bytes=len(content),
            )
        except Exception as e:
            logger.error(f"DownloadFile failed: {e}")
            return crawl_service_pb2.BrowserDownloadResult(success=False, error=str(e))

    async def destroy_session(self, request: Any) -> Any:
        """Close the browser session."""
        from protos import crawl_service_pb2
        await self._destroy_session(request.session_id)
        return crawl_service_pb2.BrowserEmpty()

    async def save_session_state(self, request: Any) -> Any:
        """Serialize current session cookies/localStorage to JSON for persistence."""
        from protos import crawl_service_pb2
        ctx = self._get_context(request.session_id)
        if not ctx:
            return crawl_service_pb2.BrowserSessionStateResult(
                success=False,
                error="Session not found or expired",
            )
        self._touch_session(request.session_id)
        try:
            state = await ctx.storage_state()
            state_json = json.dumps(state)
            return crawl_service_pb2.BrowserSessionStateResult(
                success=True,
                state_json=state_json,
            )
        except Exception as e:
            logger.error(f"SaveSessionState failed: {e}")
            return crawl_service_pb2.BrowserSessionStateResult(
                success=False,
                error=str(e),
            )

    async def inspect_page(self, request: Any) -> Any:
        """Return page structure (accessibility tree + interactive elements with selectors) for LLM playbook authoring."""
        from protos import crawl_service_pb2
        page = self._get_page(request.session_id)
        if not page:
            return crawl_service_pb2.BrowserInspectResult(
                success=False,
                error="Session not found or expired",
            )
        self._touch_session(request.session_id)
        try:
            snapshot = await page.accessibility.snapshot()
            interactive = await page.evaluate("""() => {
                const interactive = [];
                const sel = 'a[href], button, input, select, textarea, [role="button"], [role="link"], [role="menuitem"], [onclick], [data-testid]';
                document.querySelectorAll(sel).forEach((el, idx) => {
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role') || (tag === 'a' ? 'link' : tag === 'input' ? 'textbox' : tag);
                    let name = (el.getAttribute('aria-label') || el.textContent || el.value || '').trim().slice(0, 80);
                    let selector = null;
                    if (el.getAttribute('data-testid')) selector = '[data-testid="' + el.getAttribute('data-testid') + '"]';
                    else if (el.getAttribute('aria-label')) selector = '[aria-label="' + el.getAttribute('aria-label').replace(/"/g, '\\\\"') + '"]';
                    else if (el.id && !el.id.match(/^[0-9]/)) selector = '#' + el.id;
                    else if (tag === 'button' && name) selector = 'button:has-text("' + name.slice(0, 50).replace(/"/g, '\\\\"') + '")';
                    else if (tag === 'a' && name) selector = 'a:has-text("' + name.slice(0, 50).replace(/"/g, '\\\\"') + '")';
                    else selector = tag + ':nth-of-type(' + (Array.from(el.parentNode.querySelectorAll(tag)).indexOf(el) + 1) + ')';
                    interactive.push({ role, name, selector });
                });
                return interactive;
            }""")
            result = {
                "accessibility_snapshot": snapshot,
                "interactive_elements": interactive,
            }
            return crawl_service_pb2.BrowserInspectResult(
                success=True,
                page_structure=json.dumps(result),
            )
        except Exception as e:
            logger.error(f"InspectPage failed: {e}")
            return crawl_service_pb2.BrowserInspectResult(
                success=False,
                error=str(e),
            )


def _mime_for_filename(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime = {
        "pdf": "application/pdf",
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
        "json": "application/json",
        "xml": "application/xml",
        "txt": "text/plain",
        "html": "text/html",
        "zip": "application/zip",
    }
    return mime.get(ext, "application/octet-stream")
