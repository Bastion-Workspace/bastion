"""
Factual Query Subgraph - Smart path for factual questions (who/what/when about an entity).

Quick local document check (small vector search); if no reference-style content found,
falls back to web search and synthesis.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage
from orchestrator.models.query_classification_models import QueryPlan
from orchestrator.models.agent_response_contract import AgentResponse, TaskStatus
from orchestrator.tools.document_tools import search_documents_structured, get_document_content_tool
from orchestrator.tools.web_tools import search_web_structured
from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

FACTUAL_CATEGORIES = {"reference", "education", "research", "academic", "news", "technical", "manual"}


def _has_substantial_factual_content(results: List[Dict[str, Any]], min_preview_len: int = 100) -> bool:
    for doc in results[:5]:
        cat = (doc.get("category") or "").lower().strip()
        if cat in FACTUAL_CATEGORIES:
            preview = (doc.get("content_preview") or doc.get("text") or "")
            if len(preview) >= min_preview_len:
                return True
    return False


async def execute_factual_query(
    plan: QueryPlan,
    query: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    user_id = state.get("user_id", "system")

    try:
        local_result = await search_documents_structured(
            query=query,
            limit=5,
            user_id=user_id,
        )
    except Exception as e:
        logger.warning(f"Factual path: local search failed: {e}")
        local_result = {"results": [], "total_count": 0}

    results = local_result.get("results") or []
    if _has_substantial_factual_content(results):
        logger.info("Factual path: found local reference content, synthesizing from local")
        try:
            content_parts = []
            for doc in results[:3]:
                doc_id = doc.get("document_id")
                if not doc_id:
                    continue
                preview = doc.get("content_preview") or doc.get("text") or ""
                if len(preview) > 500:
                    content_parts.append(f"**{doc.get('title', 'Untitled')}**\n{preview[:3000]}")
                else:
                    raw = await get_document_content_tool(document_id=doc_id, user_id=user_id)
                    if raw and not raw.startswith("Error"):
                        content_parts.append(f"**{doc.get('title', 'Untitled')}**\n{raw[:3000]}")
            if not content_parts:
                raise ValueError("No local content to synthesize")
            context = "\n\n".join(content_parts)
            base_agent = BaseAgent("factual_query")
            llm = base_agent._get_llm(temperature=0.3, state=state)
            response = await llm.ainvoke([
                SystemMessage(content="Answer the user's question using ONLY the provided local document excerpts. Be concise and factual. If the excerpts do not contain the answer, say so."),
                HumanMessage(content=f"Question: {query}\n\nExcerpts:\n{context}\n\nAnswer:"),
            ])
            text = response.content if hasattr(response, "content") else str(response)
            return AgentResponse(
                response=text,
                task_status=TaskStatus.COMPLETE,
                agent_type="research_agent",
                timestamp=datetime.utcnow().isoformat() + "Z",
                sources=["local documents"],
            ).model_dump(exclude_none=True)
        except Exception as e:
            logger.warning(f"Factual path: local synthesis failed: {e}, falling back to web")

    logger.info("Factual path: no sufficient local reference content, searching web")
    try:
        web_results = await search_web_structured(query=query, max_results=8)
    except Exception as e:
        logger.warning(f"Factual path: web search failed: {e}")
        return AgentResponse(
            response=f"I couldn't find local reference material and web search failed: {str(e)}. Try rephrasing or asking again.",
            task_status=TaskStatus.ERROR,
            agent_type="research_agent",
            timestamp=datetime.utcnow().isoformat() + "Z",
            error=str(e),
        ).model_dump(exclude_none=True)

    formatted = []
    for i, res in enumerate(web_results[:6], 1):
        formatted.append(f"{i}. **{res.get('title', 'No Title')}** - {res.get('url', '')}\n   {res.get('snippet', '')}")
    context = "\n\n".join(formatted)
    base_agent = BaseAgent("factual_query")
    llm = base_agent._get_llm(temperature=0.3, state=state)
    response = await llm.ainvoke([
        SystemMessage(content="Answer the user's question using the provided web search results. Be concise, factual, and cite sources where relevant."),
        HumanMessage(content=f"Question: {query}\n\nSearch results:\n{context}\n\nAnswer:"),
    ])
    text = response.content if hasattr(response, "content") else str(response)
    sources = [r.get("url", "") for r in web_results[:5] if r.get("url")]

    return AgentResponse(
        response=text,
        task_status=TaskStatus.COMPLETE,
        agent_type="research_agent",
        timestamp=datetime.utcnow().isoformat() + "Z",
        sources=sources,
    ).model_dump(exclude_none=True)
