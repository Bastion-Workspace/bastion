"""
SiteCrawlAgent - Roosevelt's Focused Link-Scoped Research Cavalry
Given a link and a query, crawl that site and synthesize relevant pages.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

from .base_agent import BaseAgent
from models.agent_response_models import ResearchTaskResult, TaskStatus, CitationSource

logger = logging.getLogger(__name__)


class SiteCrawlAgent(BaseAgent):
    def __init__(self):
        super().__init__("research_agent")

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("🕷️ BULLY! SiteCrawlAgent charging forward...")

            shared = state.get("shared_memory", {}) or {}
            seed_url = shared.get("seed_url") or self._extract_seed_url_from_message(state)
            if not seed_url:
                return self._error(state, "No seed URL provided. Include a link to the site to crawl.")

            query = state.get("current_query") or self._get_latest_user_message(state)
            if not query:
                return self._error(state, "No query provided.")

            # Execute compact subgraph nodes inline
            from services.langgraph_subgraphs.site_crawl_subgraph import (
                site_crawl_discovery_node, site_crawl_extract_node, site_crawl_filter_node, site_crawl_synthesize_node
            )

            st = { **state, "seed_url": seed_url }
            st = await site_crawl_discovery_node(st)
            st = await site_crawl_extract_node(st)
            st = await site_crawl_filter_node(st)

            # LLM-based synthesis over filtered results to answer user's question
            internal = st.get("_site_crawl", {})
            filtered = internal.get("filtered", [])
            synth_findings = await self._synthesize_answer_with_llm(query, filtered, seed_url)

            # Also keep simple list rendering for context
            st = await site_crawl_synthesize_node(st)
            simple_list = st.get("latest_response", "")

            findings = f"{synth_findings}\n\n---\n\n{simple_list}" if synth_findings else simple_list

            # Convert to structured result with citations
            citations: List[CitationSource] = []
            for idx, item in enumerate(filtered[:10], 1):
                citations.append(CitationSource(
                    id=idx,
                    title=((item.get("metadata") or {}).get("title") or "Webpage"),
                    type="webpage",
                    url=item.get("url", ""),
                    author=None,
                    date=None,
                    excerpt=None,
                ))

            structured = ResearchTaskResult(
                task_status=TaskStatus.COMPLETE,
                findings=findings,
                citations=citations,
                sources_searched=["domain_crawl"],
                permission_request=None,
                confidence_level=0.85,
                next_steps=None,
            )

            st["agent_results"] = {
                "structured_response": structured.dict(),
                "research_mode": "site_crawl",
                "timestamp": datetime.now().isoformat(),
            }
            st["latest_response"] = structured.findings
            st["is_complete"] = True
            return st
        except Exception as e:
            logger.error(f"❌ SiteCrawlAgent failed: {e}")
            return self._error(state, str(e))

    async def _synthesize_answer_with_llm(self, query: str, pages: List[Dict[str, Any]], seed_url: str) -> str:
        try:
            # Prepare a compact context of top pages
            def compact(item: Dict[str, Any]) -> str:
                title = ((item.get("metadata") or {}).get("title") or "No title")
                url = item.get("url", "")
                content = (item.get("full_content") or "")
                snippet = content[:1800]
                return f"TITLE: {title}\nURL: {url}\nCONTENT_SNIPPET:\n{snippet}\n"

            top_context_blocks = "\n\n".join(compact(p) for p in pages[:12])

            system_prompt = (
                "You are Roosevelt's Site Crawl Synthesis Officer. Given the user query and content snippets "
                "from pages strictly within the target site, produce a concise, well-structured answer with "
                "clear headings and bullet lists where appropriate. Use ONLY the provided snippets. Do not "
                "invent facts; explicitly note when data is missing. Include inline numeric citations like (1), (2) "
                "mapping to the order of the snippets provided."
            )

            user_prompt = f"""
TARGET SITE: {seed_url}
USER QUERY: {query}

PAGE CONTEXT (snippets in order):
{top_context_blocks}

TASK:
1) Answer the USER QUERY strictly using only the content in the snippets above.
2) If the query asks for lists, counts, rankings, or tables, aggregate best-effort numbers from the text and label them as estimates when uncertain.
3) If the query asks for qualitative analysis or synthesis, provide a clear, well-organized summary addressing each requested facet.
4) Always include inline citations like (1), (2) pointing to the relevant snippet(s).
5) If critical data to answer the query is missing in the snippets, state what is missing and suggest what additional on-site pages might be needed.
"""

            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            response = await chat_service.openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model_name,
                temperature=0.2,
            )

            content = ""
            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0].message, 'content'):
                content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.warning(f"⚠️ LLM synthesis failed: {e}")
            return ""

    def _extract_seed_url_from_message(self, state: Dict[str, Any]) -> str:
        try:
            import re
            text = self._get_latest_user_message(state)
            if not text:
                return ""
            m = re.search(r'https?://[^\s)>\"]+', text)
            return m.group(0) if m else ""
        except Exception:
            return ""

    def _get_latest_user_message(self, state: Dict[str, Any]) -> str:
        try:
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == "human":
                    return msg.content
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
            return state.get("current_query", "")
        except Exception:
            return state.get("current_query", "")

    def _error(self, state: Dict[str, Any], message: str) -> Dict[str, Any]:
        result = ResearchTaskResult(
            task_status=TaskStatus.ERROR,
            findings=f"Site crawl failed: {message}",
            sources_searched=["domain_crawl"],
            confidence_level=0.0,
        )
        state["agent_results"] = {
            "structured_response": result.dict(),
            "error": True,
            "error_message": message,
        }
        state["latest_response"] = result.findings
        state["is_complete"] = True
        return state


