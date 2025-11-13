"""
Site Crawl Subgraph - Roosevelt's Focused Domain Recon
Implements a compact LangGraph subgraph: discover → crawl → filter → synthesize
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


async def site_crawl_discovery_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        shared = state.get("shared_memory", {}) or {}
        seed_url = shared.get("seed_url") or state.get("seed_url")
        query = state.get("current_query") or ""
        if not seed_url:
            raise ValueError("seed_url missing in state/shared_memory")
        # Calculate path prefix scope from seed
        from urllib.parse import urlparse
        p = urlparse(seed_url)
        allowed_prefix = p.path if isinstance(p.path, str) and len(p.path) > 1 else None
        # Pass through
        return { **state, "_site_crawl": {"seed_url": seed_url, "query": query, "allowed_path_prefix": allowed_prefix} }
    except Exception as e:
        logger.error(f"❌ site_crawl_discovery_node failed: {e}")
        return { **state, "agent_results": {"response": f"Site crawl discovery failed: {e}"}, "is_complete": True }


async def site_crawl_extract_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        internal = state.get("_site_crawl", {})
        seed_url = internal.get("seed_url")
        query = internal.get("query") or state.get("current_query") or ""
        allowed = internal.get("allowed_path_prefix")
        from services.langgraph_tools.crawl4ai_web_tools import Crawl4AIWebTools
        crawler = Crawl4AIWebTools()
        result = await crawler.crawl_site(
            seed_url=seed_url,
            query_criteria=query,
            max_pages=120,
            max_depth=3,
            allowed_path_prefix=allowed,
            include_pdfs=False,
        )
        return { **state, "_site_crawl": {**internal, "raw_results": result} }
    except Exception as e:
        logger.error(f"❌ site_crawl_extract_node failed: {e}")
        return { **state, "agent_results": {"response": f"Site crawl failed: {e}"}, "is_complete": True }


async def site_crawl_filter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        internal = state.get("_site_crawl", {})
        raw = internal.get("raw_results", {})
        items = raw.get("results", []) if isinstance(raw, dict) else []
        # Keep successful items, scoped to newsroom and 2025, drop error/404 pages
        def looks_valid(i):
            if not i.get("success"):
                return False
            url = (i.get("url") or "").lower()
            title = ((i.get("metadata") or {}).get("title") or "").lower()
            if "page not found" in title:
                return False
            if "/newsroom" not in url:
                return False
            # Soft prefer 2025 mentions
            content = (i.get("full_content") or "").lower()
            return True if "2025" in content or "2025" in title else True

        filtered = [i for i in items if looks_valid(i)]
        filtered.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        # Trim and de-duplicate by URL
        seen = set()
        top: List[Dict[str, Any]] = []
        for i in filtered:
            u = i.get("url")
            if u and u not in seen:
                seen.add(u)
                top.append(i)
            if len(top) >= 40:
                break
        return { **state, "_site_crawl": {**internal, "filtered": top} }
    except Exception as e:
        logger.error(f"❌ site_crawl_filter_node failed: {e}")
        return { **state, "agent_results": {"response": f"Filtering failed: {e}"}, "is_complete": True }


async def site_crawl_synthesize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        internal = state.get("_site_crawl", {})
        top = internal.get("filtered", [])
        seed_url = internal.get("seed_url")
        # Simple synthesis stub: list pages with brief bullets
        lines: List[str] = []
        lines.append(f"## Focused findings from {seed_url}\n")
        for i, item in enumerate(top, 1):
            title = ((item.get("metadata") or {}).get("title") or "No title").strip() or "No title"
            url = item.get("url", "")
            score = item.get("relevance_score", 0.0)
            lines.append(f"- {i}. **{title}** (relevance {score:.2f}) — {url}")
        findings = "\n".join(lines)
        return { **state, "agent_results": {"response": findings}, "latest_response": findings, "is_complete": True }
    except Exception as e:
        logger.error(f"❌ site_crawl_synthesize_node failed: {e}")
        return { **state, "agent_results": {"response": f"Synthesis failed: {e}"}, "is_complete": True }


