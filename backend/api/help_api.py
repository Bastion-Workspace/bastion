"""
Help API - Serves help documentation from backend markdown files.
Supports subdirectories as categories; topic id = path relative to help_docs (e.g. getting-started/01-welcome).
"""

import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Help"])

_API_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _API_DIR.parent
_HELP_DOCS_DIR = _BACKEND_DIR / "help_docs"
if not _HELP_DOCS_DIR.exists():
    _HELP_DOCS_DIR = Path("/app/help_docs")

_topics_cache: list | None = None


class HelpTopicSummary(BaseModel):
    """Summary of a help topic for sidebar listing"""
    id: str
    title: str
    order: int
    category: str = ""


class HelpTopicContent(BaseModel):
    """Full help topic with markdown content"""
    id: str
    title: str
    content: str


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split frontmatter and body. Returns (frontmatter_dict, body_str)."""
    if not content.strip().startswith("---"):
        return {}, content.strip()
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content.strip()
    frontmatter_str = parts[1].strip()
    body = parts[2].strip()
    fm = {}
    for line in frontmatter_str.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            fm[key.strip().lower()] = value.strip()
    return fm, body


def _order_from_path(path: Path, fm: dict) -> int:
    """Derive order: frontmatter 'order' else numeric prefix from stem (01-, 02-) else 999."""
    order = fm.get("order")
    if order is not None:
        try:
            return int(order)
        except (TypeError, ValueError):
            pass
    stem = path.stem
    match = re.match(r"^(\d+)[-_]", stem)
    if match:
        return int(match.group(1))
    return 999


def _category_from_path(rel_path: Path) -> str:
    """Category = parent dir name (title-cased); empty for root-level files."""
    parts = rel_path.parts
    if len(parts) <= 1:
        return ""
    return parts[0].replace("-", " ").replace("_", " ").title()


def _list_topics() -> list[dict]:
    """Recursively scan help_docs for .md files; return list of {id, title, order, category}."""
    global _topics_cache
    if _topics_cache is not None:
        return _topics_cache
    if not _HELP_DOCS_DIR.exists():
        logger.warning("Help docs directory not found: %s", _HELP_DOCS_DIR)
        _topics_cache = []
        return _topics_cache
    topics = []
    for path in sorted(_HELP_DOCS_DIR.rglob("*.md")):
        try:
            rel = path.relative_to(_HELP_DOCS_DIR)
            if rel.parts[0].startswith("."):
                continue
            raw = path.read_text(encoding="utf-8")
            fm, _ = _parse_frontmatter(raw)
            topic_id = str(rel.with_suffix("")).replace("\\", "/")
            title = fm.get("title", path.stem.replace("-", " ").replace("_", " ").title())
            order = _order_from_path(path, fm)
            category = _category_from_path(rel)
            topics.append({"id": topic_id, "title": title, "order": order, "category": category})
        except Exception as e:
            logger.warning("Skipping help file %s: %s", path, e)
    topics.sort(key=lambda t: (t["category"], t["order"], t["id"]))
    _topics_cache = topics
    return topics


def _topic_id_safe(topic_id: str) -> bool:
    """True if topic_id is safe (no path traversal)."""
    if not topic_id or ".." in topic_id:
        return False
    if topic_id.startswith("/") or "\\" in topic_id:
        return False
    return True


def _get_topic_content(topic_id: str) -> HelpTopicContent | None:
    """Load one topic's markdown; topic_id may be path like getting-started/01-welcome."""
    if not _HELP_DOCS_DIR.exists():
        return None
    if not _topic_id_safe(topic_id):
        return None
    path = (_HELP_DOCS_DIR / topic_id).with_suffix(".md")
    try:
        path = path.resolve()
        path.relative_to(_HELP_DOCS_DIR)
    except (ValueError, OSError):
        return None
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    fm, body = _parse_frontmatter(raw)
    title = fm.get("title", path.stem.replace("-", " ").replace("_", " ").title())
    return HelpTopicContent(id=topic_id, title=title, content=body)


@router.get("/api/help/topics", response_model=list[HelpTopicSummary])
async def get_help_topics(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> list[HelpTopicSummary]:
    """List help topics for the sidebar (grouped by category in UI)."""
    topics = _list_topics()
    return [HelpTopicSummary(**t) for t in topics]


@router.get("/api/help/topics/{topic_id:path}", response_model=HelpTopicContent)
async def get_help_topic(
    topic_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
) -> HelpTopicContent:
    """Get a single help topic's markdown content. topic_id may include slashes (e.g. getting-started/01-welcome)."""
    topic = _get_topic_content(topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Help topic not found")
    return topic
