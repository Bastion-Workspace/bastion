"""
ScraperContext - Sandboxed API for user-written Data Factory scrapers.
Provides crawl(), write_*(), cursor_state, and browser download_file().
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ScraperContext:
    """
    Context passed to user scraper code. Provides web fetch, file output, and browser download.
    When the full Data Factory is implemented, crawl() and write_* will be wired here.
    """

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        scraper_name: str,
        cursor_state: Optional[dict] = None,
    ):
        self._output_dir = output_dir
        self._run_id = run_id
        self._scraper_name = scraper_name
        self._cursor_state = cursor_state or {}
        self._browser_session_id: Optional[str] = None
        self._crawl_client = None

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def scraper_name(self) -> str:
        return self._scraper_name

    @property
    def cursor_state(self) -> dict:
        return self._cursor_state

    def log(self, message: str) -> None:
        logger.info(f"[{self._scraper_name}] {message}")

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        logger.info(f"[{self._scraper_name}] Progress {current}/{total}: {message}")

    def should_stop(self) -> bool:
        return False

    async def _get_crawl_client(self):
        if self._crawl_client is None:
            from clients.crawl_service_client import get_crawl_service_client
            self._crawl_client = await get_crawl_service_client()
        return self._crawl_client

    async def _ensure_browser_session(self) -> Optional[str]:
        """Create a browser session if none exists. Returns session_id or None."""
        if self._browser_session_id:
            return self._browser_session_id
        client = await self._get_crawl_client()
        self._browser_session_id = await client.browser_create_session(timeout_seconds=60)
        return self._browser_session_id

    async def download_file(
        self,
        trigger_selector: str,
        filename_hint: Optional[str] = None,
        url: Optional[str] = None,
    ) -> bytes:
        """
        Click a download button and capture the file. If url is provided, navigate there first.
        Otherwise the current browser session must already be on the target page.
        Returns the raw file bytes.
        """
        session_id = await self._ensure_browser_session()
        if not session_id:
            raise RuntimeError("Could not create browser session for download")
        client = await self._get_crawl_client()
        if url:
            nav = await client.browser_execute_action(session_id, "navigate", url=url)
            if not nav.get("success"):
                raise RuntimeError(nav.get("error", "Navigate failed"))
        result = await client.browser_download_file(
            session_id=session_id,
            trigger_selector=trigger_selector,
            timeout_seconds=60,
        )
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Download failed"))
        return result.get("file_content") or b""

    async def close_browser_session(self) -> None:
        """Release the browser session. Call when the scraper is done with browser automation."""
        if self._browser_session_id and self._crawl_client:
            await self._crawl_client.browser_destroy_session(self._browser_session_id)
            self._browser_session_id = None
