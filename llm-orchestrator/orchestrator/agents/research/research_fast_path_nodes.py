"""
Fast path research nodes: quick answer check, quick vector search,
tier detection, fast path.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, TYPE_CHECKING

from pydantic import ValidationError
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.research.research_state import ResearchState, ResearchRound
from orchestrator.agents.research.research_helpers import detect_research_tier, doc_display_name, build_fast_path_response

if TYPE_CHECKING:
    from orchestrator.agents.research.full_research_agent import FullResearchAgent

logger = logging.getLogger(__name__)


async def quick_vector_search(
    agent: "FullResearchAgent",
    query: str,
    limit: int = 8,
    user_id: str = "system",
    metadata: Dict[str, Any] = None,
    shared_memory: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """Perform fast vector search for quick answer context using intelligent retrieval subgraph."""
    try:
        from orchestrator.subgraphs.intelligent_document_retrieval_subgraph import retrieve_documents_intelligently

        try:
            result = await asyncio.wait_for(
                retrieve_documents_intelligently(
                    query=query,
                    user_id=user_id,
                    mode="fast",
                    max_results=limit,
                    small_doc_threshold=5000,
                    metadata=metadata,
                    shared_memory=shared_memory,
                ),
                timeout=5.0,
            )

            retrieved_docs = result.get("retrieved_documents", [])

            formatted_results = []
            for doc in retrieved_docs:
                formatted_results.append({
                    "document_id": doc.get("document_id"),
                    "title": doc.get("title", doc.get("filename", "Unknown")),
                    "filename": doc.get("filename", ""),
                    "content_preview": doc.get("content_preview", ""),
                    "relevance_score": doc.get("relevance_score", 0.0),
                    "metadata": doc.get("metadata", {}),
                })

            logger.info(f"Quick vector search found {len(formatted_results)} results via intelligent retrieval")
            return formatted_results

        except asyncio.TimeoutError:
            logger.warning("Quick vector search timed out after 5 seconds - falling back to basic search")
            from orchestrator.tools import search_documents_structured
            search_result = await search_documents_structured(query=query, limit=limit, user_id=user_id)
            results = search_result.get("results", [])
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "document_id": doc.get("document_id"),
                    "title": doc.get("title", doc.get("filename", "Unknown")),
                    "filename": doc.get("filename", ""),
                    "content_preview": doc.get("content_preview", ""),
                    "relevance_score": doc.get("relevance_score", 0.0),
                    "metadata": doc.get("metadata", {}),
                })
            return formatted_results

    except Exception as e:
        logger.warning(f"Quick vector search failed: {e} - continuing without vector results")
        return []


async def quick_answer_check_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Check if query can be answered quickly from general knowledge without searching"""
    try:
        from orchestrator.models import QuickAnswerAssessment

        query = state["query"]
        skip_quick_answer = state.get("skip_quick_answer", False)

        if skip_quick_answer:
            logger.info("Skipping quick answer check - proceeding to full research")
            return {
                "quick_answer_provided": False,
                "quick_answer_content": "",
                "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
                "quick_vector_results": [],
                "quick_vector_relevance": None,
                "skill_config": state.get("skill_config", {}),
            }

        logger.info(f"Quick answer check for: {query}")

        evaluation_prompt = f"""Evaluate whether this query can be answered accurately from general knowledge without searching documents or the web.

USER QUERY: {query}

Consider:
1. Is this a well-known, factual query? (e.g., "What is the best water temperature for tea?")
2. Can it be answered accurately from general knowledge?
3. Does it require specific, current, or user-specific information that would need searching?
4. Is the answer likely to be stable and well-established? (not time-sensitive or controversial)
5. CRITICAL: Is the user asking about THEIR OWN content? If they are asking whether WE/they HAVE photos, documents, files, or anything in their library/collection, you CANNOT answer quickly. We must search their local documents and images first. Set can_answer_quickly=false for any query like "Do we have any photos of X?", "What documents do I have?", "Our collection", "Do I have anything about Y?", etc.

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "can_answer_quickly": boolean,
    "confidence": number (0.0-1.0),
    "quick_answer": "string (provide the answer if can_answer_quickly=true, otherwise null)",
    "reasoning": "brief explanation of why this can or cannot be answered quickly"
}}"""

        llm = agent._get_llm(temperature=0.3, state=state)
        datetime_context = agent._get_datetime_context()

        shared_memory = state.get("shared_memory", {})
        handoff_context = shared_memory.get("handoff_context", {})
        handoff_note = ""

        if handoff_context:
            source_agent = handoff_context.get("source_agent", "unknown")
            reference_doc = handoff_context.get("reference_document", {})
            if reference_doc.get("has_content"):
                ref_content = reference_doc.get("content", "")[:1000]
                ref_filename = reference_doc.get("filename", "unknown")
                handoff_note = f"""

**AGENT HANDOFF CONTEXT**:
- Delegated by: {source_agent}
- User has reference document: {ref_filename}
- Document preview: {ref_content}{"..." if len(reference_doc.get("content", "")) > 1000 else ""}

When answering, you can reference data from the user's document above."""
                logger.info(f"Handoff context detected from {source_agent}")

        messages_for_llm = [
            SystemMessage(content="You are a query evaluator. Always respond with valid JSON matching the exact schema provided."),
            SystemMessage(content=datetime_context),
        ]

        conversation_messages = state.get("messages", [])
        if conversation_messages:
            messages_for_llm.extend(conversation_messages)
            logger.info(f"Including {len(conversation_messages)} conversation messages for context")

        full_prompt = evaluation_prompt + handoff_note
        messages_for_llm.append(HumanMessage(content=full_prompt))

        async def llm_evaluation_task():
            return await llm.ainvoke(messages_for_llm)

        async def vector_search_task():
            user_id = shared_memory.get("user_id", "system") if shared_memory else "system"
            logger.debug(f"Quick vector search using user_id: {user_id}")
            metadata = state.get("metadata", {})
            return await quick_vector_search(agent, query, limit=8, user_id=user_id, metadata=metadata, shared_memory=shared_memory)

        vector_results = []

        logger.info("Running LLM evaluation and vector search in parallel...")
        try:
            llm_response, vector_results = await asyncio.gather(
                llm_evaluation_task(),
                vector_search_task(),
                return_exceptions=True,
            )

            if isinstance(llm_response, Exception):
                logger.error(f"LLM evaluation failed: {llm_response}")
                raise llm_response
            if isinstance(vector_results, Exception):
                logger.warning(f"Vector search failed: {vector_results} - continuing without vector results")
                vector_results = []

            response = llm_response
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            try:
                response = await llm.ainvoke(messages_for_llm)
                vector_results = []
            except Exception as llm_error:
                logger.error(f"LLM fallback also failed: {llm_error}")
                raise llm_error

        try:
            text = response.content.strip()
            if "```json" in text:
                m = re.search(r"```json\s*\n([\s\S]*?)\n```", text)
                if m:
                    text = m.group(1).strip()
            elif "```" in text:
                m = re.search(r"```\s*\n([\s\S]*?)\n```", text)
                if m:
                    text = m.group(1).strip()

            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                text = json_match.group(0)

            assessment = QuickAnswerAssessment.parse_raw(text)

            vector_relevance = "none"
            high_relevance_docs = []
            medium_relevance_docs = []

            if vector_results:
                for doc in vector_results:
                    score = doc.get("relevance_score", 0.0)
                    if score >= 0.7:
                        high_relevance_docs.append(doc)
                    elif score >= 0.5:
                        medium_relevance_docs.append(doc)

                if high_relevance_docs:
                    vector_relevance = "high"
                elif medium_relevance_docs:
                    vector_relevance = "medium"
                elif vector_results:
                    vector_relevance = "low"

                logger.info(f"Vector search relevance: {vector_relevance} ({len(high_relevance_docs)} high, {len(medium_relevance_docs)} medium)")

            if assessment.can_answer_quickly and assessment.quick_answer:
                formatted_answer = assessment.quick_answer

                if vector_relevance == "high" and high_relevance_docs:
                    citations_text = "\n\n**Sources from your documents:**\n"
                    for doc in high_relevance_docs[:3]:
                        title = doc.get("title", doc.get("filename", "Unknown"))
                        filename = doc.get("filename", "")
                        citations_text += f"- {title}"
                        if filename and filename != title:
                            citations_text += f" ({filename})"
                        citations_text += "\n"
                    formatted_answer += citations_text
                    logger.info(f"Included {len(high_relevance_docs[:3])} high-relevance document citations")
                elif vector_relevance == "medium" and medium_relevance_docs:
                    doc_mention = f"\n\n*Note: I found {len(medium_relevance_docs)} potentially relevant document(s) in your knowledge base. "
                    doc_mention += "Would you like me to search deeper for more specific information from your documents?*"
                    formatted_answer += doc_mention
                    logger.info(f"Mentioned {len(medium_relevance_docs)} medium-relevance documents")

                formatted_answer += "\n\n---\n*Would you like me to perform a deeper search for more detailed information, sources, or alternative perspectives? Just let me know!*"

                logger.info(f"Quick answer provided: confidence={assessment.confidence}, vector_relevance={vector_relevance}")

                shared_memory = state.get("shared_memory", {}) or {}
                shared_memory["primary_agent_selected"] = "research_agent"
                shared_memory["last_agent"] = "research_agent"

                return {
                    "quick_answer_provided": True,
                    "quick_answer_content": formatted_answer,
                    "final_response": formatted_answer,
                    "research_complete": True,
                    "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
                    "quick_vector_results": vector_results,
                    "quick_vector_relevance": vector_relevance,
                    "shared_memory": shared_memory,
                    "skill_config": state.get("skill_config", {}),
                }
            else:
                logger.info(f"Query requires full research: {assessment.reasoning}")
                vector_relevance = "none"
                if vector_results:
                    has_high = any(doc.get("relevance_score", 0.0) >= 0.7 for doc in vector_results)
                    has_medium = any(doc.get("relevance_score", 0.0) >= 0.5 for doc in vector_results)
                    if has_high:
                        vector_relevance = "high"
                    elif has_medium:
                        vector_relevance = "medium"
                    else:
                        vector_relevance = "low"

                return {
                    "quick_answer_provided": False,
                    "quick_answer_content": "",
                    "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
                    "quick_vector_results": vector_results,
                    "quick_vector_relevance": vector_relevance,
                    "skill_config": state.get("skill_config", {}),
                }

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.warning(f"Failed to parse quick answer assessment: {e}")
            logger.warning(f"Raw response: {response.content[:500]}")
            logger.info("Quick answer assessment parsing failed - proceeding to full research")
            vector_relevance = "none"
            if vector_results:
                has_high = any(doc.get("relevance_score", 0.0) >= 0.7 for doc in vector_results)
                has_medium = any(doc.get("relevance_score", 0.0) >= 0.5 for doc in vector_results)
                if has_high:
                    vector_relevance = "high"
                elif has_medium:
                    vector_relevance = "medium"
                else:
                    vector_relevance = "low"

            return {
                "quick_answer_provided": False,
                "quick_answer_content": "",
                "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
                "quick_vector_results": vector_results if "vector_results" in locals() else [],
                "quick_vector_relevance": vector_relevance,
                "skill_config": state.get("skill_config", {}),
            }

    except Exception as e:
        logger.error(f"Quick answer check error: {e}")
        return {
            "quick_answer_provided": False,
            "quick_answer_content": "",
            "current_round": ResearchRound.QUICK_ANSWER_CHECK.value,
            "quick_vector_results": [],
            "quick_vector_relevance": "none",
            "skill_config": state.get("skill_config", {}),
        }


async def tier_detection_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Detect research tier and set routing flag. Skill config can force tier (e.g. web-only for security_analysis)."""
    query = state.get("query", "")
    metadata = state.get("metadata", {})
    skill_config = state.get("skill_config", {})

    if not skill_config.get("local_search", True):
        tier = "web"
        logger.info(f"TIER DETECTION: Skill config local_search=False, forcing tier 'web'")
    elif not skill_config.get("web_search", True):
        tier = "standard"
        logger.info(f"TIER DETECTION: Skill config web_search=False, forcing tier 'standard'")
    else:
        tier = detect_research_tier(query, metadata)
        logger.info(f"TIER DETECTION: Query classified as '{tier}' tier")

    return {
        "research_tier": tier,
        "force_web_search": tier == "web",
        "query": query,
        "metadata": metadata,
        "user_id": state.get("user_id", "system"),
        "shared_memory": state.get("shared_memory", {}),
        "messages": state.get("messages", []),
        "skill_config": skill_config,
    }


async def fast_path_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Ultra-fast research path for simple existence queries. Skips expansion, gap analysis, web, etc."""
    query = state.get("query", "")
    user_id = state.get("user_id", "system")
    metadata = state.get("metadata", {})

    logger.info(f"FAST PATH: Processing simple query: {query[:100]}")

    from orchestrator.subgraphs.intelligent_document_retrieval_subgraph import retrieve_documents_intelligently

    retrieval_result = await retrieve_documents_intelligently(
        query=query,
        user_id=user_id,
        mode="fast",
        max_results=10,
        small_doc_threshold=5000,
        metadata=metadata,
        messages=state.get("messages", []),
        shared_memory=state.get("shared_memory", {}),
        skip_sufficiency_check=True,
    )

    documents = retrieval_result.get("retrieved_documents", [])
    formatted_context = retrieval_result.get("formatted_context", "")
    image_results = retrieval_result.get("image_search_results")
    structured_images = retrieval_result.get("structured_images")

    response_text, structured_images, images_markdown = build_fast_path_response(
        documents=documents,
        structured_images=structured_images,
        image_results=image_results,
    )

    logger.info(f"FAST PATH: Completed - found {len(documents)} documents, {len(structured_images or [])} images")

    if images_markdown and not structured_images:
        response_text = f"{response_text}\n\n{images_markdown}".strip()

    return {
        "final_response": response_text,
        "sources_used": [doc_display_name(doc) for doc in documents[:5]],
        "citations": [],
        "image_search_results": image_results,
        "structured_images": structured_images,
        "round1_results": {"formatted_context": formatted_context, "documents": documents},
        "research_complete": True,
        "routing_recommendation": "research_agent",
        "query": query,
        "metadata": metadata,
        "user_id": user_id,
        "shared_memory": state.get("shared_memory", {}),
        "messages": state.get("messages", []),
        "skill_config": state.get("skill_config", {}),
    }
