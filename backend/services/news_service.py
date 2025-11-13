"""
News Service - Roosevelt's Newsroom

Stores synthesized headlines and articles, provides retrieval APIs, and computes breaking severity.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from models.news_models import NewsHeadline, NewsArticleSynth, NewsSourceRef
from services.database_manager.database_helpers import fetch_all, fetch_one, execute
from services.settings_service import settings_service
from utils.file_utils import ensure_dir_exists
import os
import json
from datetime import datetime
import re
import html as html_lib

logger = logging.getLogger(__name__)


class NewsService:
    def __init__(self):
        self._initialized = False

    async def initialize(self, shared_db_pool=None):
        # Ensure settings service is initialized
        try:
            if not getattr(settings_service, "_initialized", False):
                await settings_service.initialize()
        except Exception:
            pass
        self._initialized = True
        logger.info("🗞️ BULLY! NewsService initialized (SQL-backed)")

    async def upsert_article(self, article: NewsArticleSynth) -> None:
        # **BULLY!** Use lighter sanitization since RSS processing already cleaned content
        def _to_plain_text(value: str) -> str:
            if not value:
                return ""
            # Decode entities first
            decoded = html_lib.unescape(str(value))
            
            # **By George!** Only strip remaining HTML tags - don't duplicate RSS cleaning
            # RSS processing already handled image extraction and content cleaning
            no_tags = re.sub(r"<[^>]+>", " ", decoded)
            
            # Remove common WordPress class name artifacts that may leak into text
            no_tags = re.sub(r"\battachment-post[^\s]*", " ", no_tags, flags=re.IGNORECASE)
            # Collapse whitespace
            collapsed = re.sub(r"\s+", " ", no_tags).strip()
            return collapsed

        def _strip_boilerplate(text: str) -> str:
            if not text:
                return ""
            cleaned = text
            # Remove common WordPress footer lines like: "The post ... appeared first on ..."
            cleaned = re.sub(r"\bThe post\s+.*?\s+appeared first on\s+.*$", "", cleaned, flags=re.IGNORECASE)
            # Remove trailing site credit variants
            cleaned = re.sub(r"\s*appeared first on\s+.*$", "", cleaned, flags=re.IGNORECASE)
            # Remove generic "Read more" tails
            cleaned = re.sub(r"\b(Read more|Continue reading)\b.*$", "", cleaned, flags=re.IGNORECASE)
            # Remove ad/paid content markers
            cleaned = re.sub(r"\bADVERTISEMENT\b", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"(?:Paid\s*Content\s*:*)+", " ", cleaned, flags=re.IGNORECASE)
            # **By George!** Minimal cleanup since RSS processing already handled JSON ads upstream
            return cleaned.strip()

        safe_title = _to_plain_text(article.title)
        safe_lede = _strip_boilerplate(_to_plain_text(article.lede))
        safe_body = _strip_boilerplate(_to_plain_text(article.balanced_body))

        # De-duplicate title in lede/body if present at the start
        if safe_lede.lower().startswith(safe_title.lower()):
            safe_lede = safe_lede[len(safe_title):].lstrip(" -:|#").strip()
        if safe_body.lower().startswith(safe_title.lower()):
            safe_body = safe_body[len(safe_title):].lstrip(" -:|#").strip()

        # Write Markdown file to disk with YAML frontmatter
        base_dir = "/app/processed/news"
        dt = datetime.utcnow()
        subdir = os.path.join(base_dir, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))
        ensure_dir_exists(subdir)
        file_path = os.path.join(subdir, f"{article.id}.md")

        frontmatter = {
            "id": article.id,
            "title": article.title,
            "severity": article.severity,
            "diversity_score": float(article.diversity_score or 0.0),
            "updated_at": article.updated_at,
            "created_at": article.created_at,
            "key_points": article.key_points,
            "citations": [c.dict() for c in (article.citations or [])],
            "images": article.images or [],
        }

        import yaml
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("---\n")
                yaml.safe_dump(frontmatter, f, sort_keys=False, allow_unicode=True)
                f.write("---\n\n")
                # Compose Markdown body (title is stored separately; don't duplicate here)
                f.write(f"{safe_lede}\n\n")
                f.write(f"{safe_body}\n")
                
                # **BULLY!** Add images section if images exist
                if article.images:
                    f.write("\n## Images\n\n")
                    for i, img in enumerate(article.images, 1):
                        img_src = img.get('src', '')
                        img_alt = img.get('alt', f'Image {i}')
                        img_caption = img.get('caption', '')
                        
                        f.write(f"![{img_alt}]({img_src})\n")
                        if img_caption:
                            f.write(f"*{img_caption}*\n")
                        f.write("\n")
        except Exception as e:
            logger.error(f"❌ Failed writing news markdown: {e}")
            raise

        # Upsert metadata into SQL (no long-form content)
        # Parse created_at string to datetime for SQL (or use NOW())
        created_at_dt = None
        try:
            if article.created_at:
                created_str = str(article.created_at)
                if created_str.endswith('Z'):
                    created_str = created_str.replace('Z', '+00:00')
                created_at_dt = datetime.fromisoformat(created_str)
        except Exception:
            created_at_dt = None

        await execute(
            """
            INSERT INTO news_articles (id, title, lede, file_path, key_points, citations, diversity_score, severity, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, COALESCE($9, NOW()), NOW())
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                lede = EXCLUDED.lede,
                file_path = EXCLUDED.file_path,
                key_points = EXCLUDED.key_points,
                citations = EXCLUDED.citations,
                diversity_score = EXCLUDED.diversity_score,
                severity = EXCLUDED.severity,
                updated_at = NOW()
            """,
            article.id,
            safe_title,
            safe_lede,
            file_path,
            json.dumps(article.key_points or []),
            json.dumps([c.dict() for c in (article.citations or [])]),
            float(article.diversity_score or 0.0),
            article.severity,
            created_at_dt,
        )

    async def get_latest_headlines(self, user_id: Optional[str], limit: int = 20) -> List[NewsHeadline]:
        rows = await fetch_all(
            """
            SELECT id, title, lede, key_points, citations, diversity_score, severity, updated_at
            FROM news_articles
            ORDER BY updated_at DESC
            LIMIT $1
            """,
            limit,
        )
        headlines: List[NewsHeadline] = []
        for r in rows:
            def _to_plain_text(value: str) -> str:
                if not value:
                    return ""
                decoded = html_lib.unescape(str(value))
                no_tags = re.sub(r"<[^>]+>", " ", decoded)
                return re.sub(r"\s+", " ", no_tags).strip()

            # Normalize JSON fields from SQL (may come back as strings)
            key_points_raw = r.get("key_points") if isinstance(r, dict) else None
            if isinstance(key_points_raw, str):
                try:
                    key_points = json.loads(key_points_raw) or []
                except Exception:
                    key_points = []
            elif isinstance(key_points_raw, list):
                key_points = key_points_raw
            else:
                key_points = []

            citations_raw = r.get("citations") if isinstance(r, dict) else None
            if isinstance(citations_raw, str):
                try:
                    citations_list = json.loads(citations_raw) or []
                except Exception:
                    citations_list = []
            elif isinstance(citations_raw, list):
                citations_list = citations_raw
            else:
                citations_list = []

            headlines.append(
                NewsHeadline(
                    id=r["id"],
                    title=_to_plain_text(r["title"]),
                    summary=_to_plain_text(r["lede"]),
                    key_points=key_points,
                    sources_count=len(citations_list),
                    diversity_score=float(r.get("diversity_score") or 0.0),
                    severity=r.get("severity") or "normal",
                    updated_at=(r.get("updated_at") or datetime.utcnow()).isoformat(),
                )
            )
        return headlines

    async def get_article(self, news_id: str, user_id: Optional[str]) -> Optional[NewsArticleSynth]:
        r = await fetch_one(
            """
            SELECT id, title, lede, file_path, key_points, citations, diversity_score, severity, created_at, updated_at
            FROM news_articles
            WHERE id = $1
            """,
            news_id,
        )
        if not r:
            return None
        # Read markdown from disk
        file_path = r.get("file_path")
        body = ""
        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Strip frontmatter if present
                if content.startswith("---\n"):
                    end = content.find("\n---\n")
                    if end != -1:
                        body = content[end + 5:]
                    else:
                        body = content
                else:
                    body = content
        except Exception as e:
            logger.error(f"❌ Failed reading news markdown: {e}")
            body = ""

        # Normalize JSON fields from SQL (may come back as strings)
        key_points_raw = r.get("key_points") if isinstance(r, dict) else None
        if isinstance(key_points_raw, str):
            try:
                key_points = json.loads(key_points_raw) or []
            except Exception:
                key_points = []
        elif isinstance(key_points_raw, list):
            key_points = key_points_raw
        else:
            key_points = []

        citations_raw = r.get("citations") if isinstance(r, dict) else None
        if isinstance(citations_raw, str):
            try:
                citations_list = json.loads(citations_raw) or []
            except Exception:
                citations_list = []
        elif isinstance(citations_raw, list):
            citations_list = citations_raw
        else:
            citations_list = []

        def _to_plain_text(value: str) -> str:
            if not value:
                return ""
            decoded = html_lib.unescape(str(value))
            no_tags = re.sub(r"<[^>]+>", " ", decoded)
            return re.sub(r"\s+", " ", no_tags).strip()

        return NewsArticleSynth(
            id=r["id"],
            title=_to_plain_text(r["title"]),
            lede=_to_plain_text(r["lede"]),
            balanced_body=_to_plain_text(body.strip()),
            key_points=key_points,
            citations=[NewsSourceRef(**c) for c in citations_list],
            diversity_score=float(r.get("diversity_score") or 0.0),
            severity=r.get("severity") or "normal",
            created_at=(r.get("created_at") or datetime.utcnow()).isoformat(),
            updated_at=(r.get("updated_at") or datetime.utcnow()).isoformat(),
        )

    # Simple severity calculation placeholder (to be replaced with robust signals)
    def compute_severity(self, citations: List[NewsSourceRef], published_times: List[datetime]) -> str:
        try:
            unique_sources = len({(c.name or c.url or '') for c in citations if (c.name or c.url)})
            recent_count = sum(1 for t in published_times if (datetime.utcnow() - t) <= timedelta(minutes=45))
            if unique_sources >= 5 and recent_count >= 5:
                return "breaking"
            if unique_sources >= 3 and recent_count >= 3:
                return "urgent"
            return "normal"
        except Exception:
            return "normal"


