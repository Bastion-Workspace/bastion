"""
Research Workflow Subgraph

Reusable subgraph extracted from FullResearchAgent for multi-round research:
- Cache check
- Query expansion
- Round 1 parallel search (local + web)
- Gap analysis
- Round 2 gap filling
- Web search rounds

Can be used by knowledge_builder_agent and other research-focused agents.
"""

import logging
import json
import re
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from orchestrator.tools import (
    search_web_tool,
    crawl_web_content_tool,
    expand_query_tool,
    search_conversation_cache_tool,
    get_document_content_tool,
    search_images_tool
)
from orchestrator.models import ResearchAssessmentResult, ResearchGapAnalysis
from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.subgraphs.gap_analysis_subgraph import build_gap_analysis_subgraph
from config.settings import settings

logger = logging.getLogger(__name__)


# Use Dict[str, Any] for compatibility with main agent state
ResearchSubgraphState = Dict[str, Any]


async def intelligent_research_with_classification(
    query: str,
    user_id: str,
    metadata: Dict[str, Any],
    messages: List[Any],
) -> Optional[Dict[str, Any]]:
    """
    Classify query and run fast path (collection_search or factual_query) if applicable.
    Returns response dict for collection/factual paths, or None to use full exploratory workflow.
    """
    try:
        from orchestrator.utils.query_resolver import resolve_follow_up_query
        from orchestrator.utils.query_classifier import classify_query_intent
        from orchestrator.subgraphs.collection_search_subgraph import execute_collection_search
        from orchestrator.subgraphs.factual_query_subgraph import execute_factual_query

        resolved_query = await resolve_follow_up_query(query, messages, metadata)
        plan = await classify_query_intent(
            query=resolved_query,
            user_model=metadata.get("user_chat_model"),
            metadata=metadata,
            messages=messages,
        )
        logger.info(f"Query classified as: {plan.query_type}, reasoning: {plan.reasoning[:80]}...")

        shared_memory = dict(metadata.get("shared_memory") or {})
        if metadata.get("user_chat_model") and "user_chat_model" not in shared_memory:
            shared_memory["user_chat_model"] = metadata["user_chat_model"]
        state = {
            "query": resolved_query,
            "user_id": user_id,
            "metadata": metadata,
            "messages": messages,
            "shared_memory": shared_memory,
        }

        if plan.query_type == "collection_search":
            return await execute_collection_search(plan, state, resolved_query)
        if plan.query_type == "factual_query":
            return await execute_factual_query(plan, resolved_query, state)
        return None
    except Exception as e:
        logger.warning(f"Query classification or fast path failed: {e}, falling back to full workflow")
        return None


async def cache_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check conversation cache for previous research"""
    try:
        query = state.get("query", "")
        logger.info(f"Checking cache for: {query}")
        
        # Track tool usage
        shared_memory = state.get("shared_memory", {})
        previous_tools = shared_memory.get("previous_tools_used", [])
        if "search_conversation_cache_tool" not in previous_tools:
            previous_tools.append("search_conversation_cache_tool")
            shared_memory["previous_tools_used"] = previous_tools
            state["shared_memory"] = shared_memory
        
        # Search cache
        cache_result = await search_conversation_cache_tool(query=query, freshness_hours=24)
        logger.info("Tool used: search_conversation_cache_tool (cache check)")
        
        if cache_result.get("cache_hit") and cache_result.get("entries"):
            cached_context = "\n\n".join([
                f"[{entry['agent_name']}]: {entry['content']}"
                for entry in cache_result["entries"]
            ])
            
            logger.info(f"Cache HIT - found {len(cache_result['entries'])} cached entries")
            
            return {
                "cache_hit": True,
                "cached_context": cached_context,
                "research_findings": {"cached": True, "content": cached_context},
                "query": query,  # Preserve query in state
                # CRITICAL: Preserve metadata for user model selection
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", [])
            }
        
        logger.info("Cache MISS - no previous research found")
        return {
            "cache_hit": False,
            "cached_context": "",
            "research_findings": {},
            "query": query,  # Preserve query in state
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"Cache check error: {e}")
        return {
            "cache_hit": False,
            "cached_context": "",
            "research_findings": {},
            "query": state.get("query", ""),  # Preserve query in state
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }


async def query_expansion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Expand query with semantic variations"""
    try:
        query = state.get("query", "")
        logger.info(f"Expanding query: {query}")

        # Skip expansion for simple queries, but not for follow-ups (they need context)
        query_length = len(query.split())
        follow_up_indicators = ["more", "another", "also", "again", "else", "additional", "next", "other"]
        query_words = set(query.lower().split())
        is_follow_up = any(word in query_words for word in follow_up_indicators)
        is_simple = query_length <= 5 and not is_follow_up
        shared_memory = state.get("shared_memory", {})
        skip_expansion = shared_memory.get("skip_query_expansion", False) or is_simple

        if skip_expansion:
            logger.info(f"Skipping query expansion for simple query: {query}")
            return {
                "expanded_queries": [query],
                "key_entities": [],
                "original_query": query,
                "query": query,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": shared_memory,
                "messages": state.get("messages", []),
            }

        # If provided_queries exist and skip_expansion is implied, just pass them through
        provided_queries = state.get("provided_queries", None)
        if provided_queries:
            logger.info(f"Using provided queries (skipping expansion): {len(provided_queries)} queries")
            return {
                "expanded_queries": provided_queries,
                "key_entities": [],
                "original_query": query,
                "query": query,  # Preserve query in state
                # CRITICAL: Preserve metadata for user model selection
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", [])
            }
        
        # Track tool usage
        shared_memory = state.get("shared_memory", {})
        previous_tools = shared_memory.get("previous_tools_used", [])
        if "expand_query_tool" not in previous_tools:
            previous_tools.append("expand_query_tool")
            shared_memory["previous_tools_used"] = previous_tools
            state["shared_memory"] = shared_memory
        
        # Extract conversation context
        conversation_context = None
        conversation_messages = state.get("messages", [])
        logger.info(f"🔍 DEBUG: query_expansion_node received {len(conversation_messages)} messages from state")
        if conversation_messages:
            for idx, msg in enumerate(conversation_messages[:3]):  # Log first 3 for debugging
                msg_type = type(msg).__name__
                if isinstance(msg, dict):
                    logger.info(f"   Message {idx}: dict with keys={list(msg.keys())}, role={msg.get('role', 'N/A')}")
                else:
                    logger.info(f"   Message {idx}: {msg_type}")
        
        if conversation_messages and len(conversation_messages) >= 2:
            last_messages = conversation_messages[-2:]
            context_parts = []
            
            for msg in last_messages:
                if isinstance(msg, HumanMessage):
                    context_parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_parts.append(f"Assistant: {msg.content}")
                elif isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user" or role == "human":
                        context_parts.append(f"User: {content}")
                    elif role == "assistant" or role == "ai":
                        context_parts.append(f"Assistant: {content}")
            
            if context_parts:
                conversation_context = "\n".join(context_parts)
                logger.info(f"Including conversation context for query expansion: {len(context_parts)} messages")
            else:
                logger.warning(f"⚠️ Had {len(conversation_messages)} messages but extracted 0 context_parts - message format issue?")
        
        # Expand query
        expansion_result = await expand_query_tool(
            query=query, 
            num_variations=3,
            conversation_context=conversation_context
        )
        logger.info("Tool used: expand_query_tool (query expansion)")
        
        expanded_queries = expansion_result.get("expanded_queries", [])
        key_entities = expansion_result.get("key_entities", [])
        
        # Ensure we always have at least the original query
        if not expanded_queries:
            expanded_queries = [query]
            logger.warning(f"Query expansion returned 0 variations, using original query: {query}")
        
        logger.info(f"Generated {len(expanded_queries)} query variations, {len(key_entities)} entities")
        
        return {
            "expanded_queries": expanded_queries,
            "key_entities": key_entities,
            "original_query": query,
            "query": query,  # Preserve query in state
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"Query expansion error: {e}")
        query = state.get("query", "")
        return {
            "expanded_queries": [query] if query else [],
            "key_entities": [],
            "original_query": query,
            "query": query,  # Preserve query in state
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }


async def round1_local_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Round 1 LOCAL: Documents + Entity Graph + Images (NO web search yet)"""
    try:
        query = state.get("query", "")
        # Preserve query throughout this node
        # Support using provided queries directly (e.g., for Round 2 with gap queries)
        provided_queries = state.get("provided_queries", None)
        if provided_queries:
            expanded_queries = provided_queries
            logger.info(f"Using provided queries for search: {len(expanded_queries)} queries")
        else:
            expanded_queries = state.get("expanded_queries", [])
        
        # Fallback to original query if no expanded queries available
        if not expanded_queries or not any(q and q.strip() for q in expanded_queries):
            if query and query.strip():
                expanded_queries = [query]
                logger.info(f"No valid expanded queries, using original query: {query}")
            else:
                logger.error("No valid query available for search - both expanded_queries and original query are empty")
                return {
                    "round1_results": {"error": "No valid query", "search_results": ""},
                    "web_round1_results": {"error": "No valid query", "content": ""},
                    "sources_found": [],
                    "research_findings": {},
                    "query": query,  # Preserve query in state
                    # CRITICAL: Preserve metadata for user model selection
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", [])
                }
        
        shared_memory = state.get("shared_memory", {})
        
        logger.info(f"Round 1: Parallel search - local + web with {len(expanded_queries)} queries")
        
        # Import intelligent retrieval
        from orchestrator.subgraphs.intelligent_document_retrieval_subgraph import retrieve_documents_intelligently
        
        async def local_search_task():
            """Local search task"""
            try:
                shared_memory = state.get("shared_memory", {})
                previous_tools = shared_memory.get("previous_tools_used", [])
                if "retrieve_documents_intelligently" not in previous_tools:
                    previous_tools.append("retrieve_documents_intelligently")
                    shared_memory["previous_tools_used"] = previous_tools
                    state["shared_memory"] = shared_memory
                
                user_id = shared_memory.get("user_id", "system")
                
                # Note: Image search is handled automatically by retrieve_documents_intelligently
                # (via intelligent_document_retrieval_subgraph._vector_search_node)
                # No need to call it separately - same approach as Chat Agent
                
                # Run all expanded queries in parallel (filter out empty queries)
                query_tasks = []
                valid_queries = [q for q in expanded_queries[:3] if q and q.strip()]
                if not valid_queries:
                    logger.warning("No valid queries for local search, using original query")
                    valid_queries = [query] if query and query.strip() else []
                
                # Get metadata and shared_memory for model selection (same as Chat Agent)
                metadata = state.get("metadata", {})
                shared_memory_for_search = dict(state.get("shared_memory", {}))

                # Run image analyzer ONCE before parallel search (reuse for all query variants)
                image_analysis_cache = None
                if any(kw in query.lower() for kw in ["photo", "image", "picture", "comic"]):
                    from orchestrator.tools.image_query_analyzer import analyze_image_query
                    try:
                        image_analysis_cache = await analyze_image_query(
                            query=query,
                            user_id=user_id,
                            metadata=metadata,
                            shared_memory=shared_memory_for_search,
                        )
                        shared_memory_for_search["image_analysis_cache"] = image_analysis_cache
                        logger.info("Single image analysis completed, reusing for all queries")
                    except Exception as e:
                        logger.warning(f"Image analysis failed, continuing without cache: {e}")

                for q in valid_queries:
                    task = retrieve_documents_intelligently(
                        query=q,
                        user_id=user_id,
                        mode="comprehensive",
                        max_results=10,
                        small_doc_threshold=15000,
                        metadata=metadata,  # Pass metadata for model selection in image search
                        shared_memory=shared_memory_for_search  # Pass shared_memory (includes image_analysis_cache)
                    )
                    query_tasks.append(task)
                
                results = await asyncio.gather(*query_tasks, return_exceptions=True)
                
                # Combine results
                # Note: Image search results are already included in formatted_context
                # from retrieve_documents_intelligently (same as Chat Agent)
                all_formatted_contexts = []
                all_documents = []
                all_image_results = []  # Collect image search results (markdown)
                all_structured_images = []  # Collect structured image data for AgentResponse contract
                seen_doc_ids = set()
                
                for result in results:
                    if isinstance(result, Exception):
                        continue
                    
                    if result.get("success") and result.get("formatted_context"):
                        all_formatted_contexts.append(result.get("formatted_context", ""))
                        
                        # Collect image search results (base64 images markdown)
                        image_results = result.get("image_search_results")
                        if image_results:
                            all_image_results.append(image_results)
                            logger.info(f"Collected image search results: {len(image_results)} characters")
                        
                        # Collect structured images for AgentResponse contract
                        structured_images = result.get("structured_images")
                        if structured_images:
                            all_structured_images.extend(structured_images)
                            logger.info(f"Collected {len(structured_images)} structured image(s)")
                        
                        for doc in result.get("retrieved_documents", []):
                            doc_id = doc.get('document_id')
                            if doc_id and doc_id not in seen_doc_ids:
                                seen_doc_ids.add(doc_id)
                                all_documents.append(doc)
                
                combined_results = "\n\n".join(all_formatted_contexts)
                combined_image_results = "\n\n".join(all_image_results) if all_image_results else None
                round1_document_ids = list(seen_doc_ids)
                
                logger.info(f"Tool used: retrieve_documents_intelligently (parallel local search with integrated image search)")
                logger.info(f"Parallel search: {len([r for r in results if not isinstance(r, Exception)])} queries succeeded, {len(all_documents)} unique documents found")
                if combined_image_results:
                    logger.info(f"Image search results preserved: {len(combined_image_results)} characters")
                
                return {
                    "search_results": combined_results,
                    "queries_used": expanded_queries[:3],
                    "result_count": len([r for r in results if not isinstance(r, Exception)]),
                    "documents_found": len(all_documents),
                    "round1_document_ids": round1_document_ids,
                    "image_search_results": combined_image_results,  # Pass through image results (markdown)
                    "structured_images": all_structured_images if all_structured_images else None  # Pass through structured images
                }
            except Exception as e:
                logger.error(f"Local search error: {e}")
                return {"error": str(e), "search_results": ""}
        
        async def entity_graph_search_task():
            """NEW: Entity relationship search via knowledge graph"""
            try:
                logger.info("Round 1 Entity Graph: Starting parallel entity search")
                
                # Get or build entity relationship subgraph
                from orchestrator.subgraphs import build_entity_relationship_subgraph
                
                # Build subgraph (no checkpointer needed for this subgraph)
                entity_sg = build_entity_relationship_subgraph(checkpointer=None)
                
                # Prepare state for entity subgraph
                # NOTE: vector_results will be empty on first call, but that's OK
                # Entity extraction will use query text
                entity_state = {
                    "query": query,
                    "vector_results": [],  # Will be populated after vector search in future iteration
                    "user_id": shared_memory.get("user_id", "system"),
                    "metadata": state.get("metadata", {}),
                    "shared_memory": shared_memory,
                    "messages": state.get("messages", [])
                }
                
                # Run subgraph (no config needed since no checkpointer)
                result = await entity_sg.ainvoke(entity_state)
                
                logger.info(f"Entity graph search complete: {result.get('kg_document_count', 0)} documents")
                
                return result
                
            except Exception as e:
                logger.error(f"Entity graph search error: {e}")
                return {
                    "kg_documents": [],
                    "kg_formatted_results": "",
                    "kg_success": False,
                    "error": str(e)
                }
        
        # Execute LOCAL searches only (local + entity) in parallel - NO WEB YET
        local_result, entity_result = await asyncio.gather(
            local_search_task(),
            entity_graph_search_task(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(local_result, Exception):
            logger.error(f"Local search exception: {local_result}")
            local_result = {"error": str(local_result), "search_results": ""}
        
        if isinstance(entity_result, Exception):
            logger.error(f"Entity graph search exception: {entity_result}")
            entity_result = {"error": str(entity_result), "kg_formatted_results": "", "kg_success": False}
        
        logger.info(f"Local search complete: local={bool(local_result.get('search_results'))}, entity={bool(entity_result.get('kg_success'))}")
        
        # Build sources list from local results only
        sources_found = []
        if local_result.get("round1_document_ids"):
            for doc_id in local_result.get("round1_document_ids", [])[:5]:
                sources_found.append({
                    "type": "document",
                    "document_id": doc_id,
                    "source": "local"
                })
        
        # Entity graph sources
        if entity_result and not isinstance(entity_result, Exception) and entity_result.get("kg_success"):
            kg_count = entity_result.get("kg_document_count", 0)
            if kg_count > 0:
                kg_docs = entity_result.get("kg_documents", [])
                for doc in kg_docs[:5]:
                    sources_found.append({
                        "type": "document",
                        "document_id": doc.get("document_id"),
                        "source": "knowledge_graph"
                    })
        
        # Combine local results
        combined_findings = {}
        
        # Local results
        if local_result and not isinstance(local_result, Exception):
            combined_findings["local_results"] = local_result.get("search_results", "")
            # Preserve image search results from local search
            if local_result.get("image_search_results"):
                combined_findings["image_search_results"] = local_result.get("image_search_results")
            # Preserve structured images from local search
            if local_result.get("structured_images"):
                combined_findings["structured_images"] = local_result.get("structured_images")
        
        # Entity graph results
        if entity_result and not isinstance(entity_result, Exception) and entity_result.get("kg_success"):
            kg_formatted = entity_result.get("kg_formatted_results", "")
            kg_count = entity_result.get("kg_document_count", 0)
            
            if kg_formatted:
                combined_findings["entity_graph_results"] = kg_formatted
                logger.info(f"Added {kg_count} knowledge graph documents to Round 1 LOCAL results")
        
        # Log what we're returning
        local_content_len = len(local_result.get("search_results", "")) if local_result else 0
        entity_content_len = len(entity_result.get("kg_formatted_results", "")) if entity_result and not isinstance(entity_result, Exception) else 0
        image_content_len = len(local_result.get("image_search_results", "")) if local_result and local_result.get("image_search_results") else 0
        logger.info(f"📊 Round 1 LOCAL complete: local={local_content_len} chars, entity={entity_content_len} chars, images={image_content_len} chars, sources={len(sources_found)}")
        
        return {
            "round1_results": {
                "search_results": local_result.get("search_results", "") if local_result else "",
                "entity_graph_results": entity_result.get("kg_formatted_results", "") if entity_result and not isinstance(entity_result, Exception) else "",
                "documents_found": local_result.get("documents_found", 0) if local_result else 0,
                "kg_documents_found": entity_result.get("kg_document_count", 0) if entity_result and not isinstance(entity_result, Exception) else 0,
                "round1_document_ids": local_result.get("round1_document_ids", []) if local_result else [],
                "image_search_results": local_result.get("image_search_results") if local_result else None,
                "structured_images": local_result.get("structured_images") if local_result else None
            },
            "sources_found": sources_found,
            "research_findings": combined_findings,
            "query": query,  # Preserve query in state
            "expanded_queries": expanded_queries,  # Preserve expanded queries in state
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"Round 1 local search error: {e}")
        return {
            "round1_results": {"error": str(e), "search_results": ""},
            "sources_found": [],
            "research_findings": {},
            "query": state.get("query", ""),  # Preserve query in state
            "expanded_queries": state.get("expanded_queries", []),  # Preserve expanded queries in state
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }


async def assess_local_results_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assess ONLY local results (documents + entity graph + images) for sufficiency"""
    try:
        query = state.get("query", "")
        round1_results = state.get("round1_results", {})
        
        local_results = round1_results.get("search_results", "")
        entity_results = round1_results.get("entity_graph_results", "")
        image_results = round1_results.get("image_search_results")
        
        logger.info("Assessing LOCAL results only (no web yet)")
        
        # Build combined local context
        local_context_parts = []
        if local_results:
            local_context_parts.append(f"LOCAL DOCUMENT RESULTS:\n{local_results}")
        if entity_results:
            local_context_parts.append(f"ENTITY GRAPH RESULTS:\n{entity_results}")
        if image_results:
            # Count images in markdown
            image_count = image_results.count("![") if isinstance(image_results, str) else 0
            local_context_parts.append(f"IMAGE RESULTS: Found {image_count} relevant images")
        
        combined_local = "\n\n".join(local_context_parts) if local_context_parts else "No local results found."
        
        # Use LLM to assess quality
        assessment_prompt = f"""Assess the quality and sufficiency of these LOCAL search results for answering the user's query.

USER QUERY: {query}

LOCAL RESULTS (Documents + Entity Graph + Images):
{combined_local}

Evaluate:
1. Do the local results contain relevant information?
2. Is there enough detail to answer the query comprehensively from LOCAL sources alone?
3. Would WEB search add significant value, or are local results sufficient?
4. What specific information (if any) is missing that web search might provide?

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "sufficient": boolean (true if local results can answer the query comprehensively),
    "has_relevant_info": boolean,
    "missing_info": ["list", "of", "specific", "gaps"],
    "confidence": float (0.0 to 1.0),
    "best_source": "local" | "need_web" | "none",
    "needs_web_search": boolean (true if web search would add significant value),
    "reasoning": "explanation of assessment"
}}"""
        
        # Get LLM for assessment
        from orchestrator.agents.base_agent import BaseAgent
        base_agent = BaseAgent("research_subgraph")
        assessment_llm = base_agent._get_llm(temperature=0.7, state=state)
        
        response = await assessment_llm.ainvoke([{"role": "user", "content": assessment_prompt}])
        
        # Parse response
        content = response.content if hasattr(response, 'content') else str(response)
        
        try:
            # Extract JSON from markdown code blocks if present
            json_text = content.strip()
            if '```json' in json_text:
                match = re.search(r'```json\s*\n(.*?)\n```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
            elif '```' in json_text:
                match = re.search(r'```\s*\n(.*?)\n```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
            
            assessment = json.loads(json_text)
            
            sufficient = assessment.get("sufficient", False)
            confidence = assessment.get("confidence", 0.5)
            needs_web = assessment.get("needs_web_search", True)
            best_source = assessment.get("best_source", "local" if sufficient else "need_web")
            
            logger.info(f"Local assessment: sufficient={sufficient}, confidence={confidence}, needs_web={needs_web}, best_source={best_source}")
            logger.info(f"Assessment reasoning: {assessment.get('reasoning', 'N/A')}")
            
            return {
                "local_sufficient": sufficient,
                "local_assessment": assessment,
                "needs_web_search": needs_web,
                "research_sufficient": sufficient,  # If local is sufficient, research is sufficient
                "query": query,
                # CRITICAL: Preserve round1_results so they flow to next nodes
                "round1_results": round1_results,
                "sources_found": state.get("sources_found", []),
                "research_findings": state.get("research_findings", {}),
                # CRITICAL: Preserve metadata for user model selection
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "expanded_queries": state.get("expanded_queries", [])
            }
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Local assessment parsing failed: {e}, defaulting to needs_web=True")
            # Default to needing web search if assessment fails
            return {
                "local_sufficient": False,
                "local_assessment": {
                    "sufficient": False,
                    "has_relevant_info": bool(local_results or entity_results or image_results),
                    "missing_info": ["Assessment failed"],
                    "confidence": 0.5,
                    "best_source": "need_web",
                    "needs_web_search": True,
                    "reasoning": f"Assessment parsing failed: {str(e)}"
                },
                "needs_web_search": True,
                "research_sufficient": False,
                "query": state.get("query", ""),
                # CRITICAL: Preserve round1_results
                "round1_results": state.get("round1_results", {}),
                "sources_found": state.get("sources_found", []),
                "research_findings": state.get("research_findings", {}),
                # CRITICAL: Preserve metadata for user model selection
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "expanded_queries": state.get("expanded_queries", [])
            }
        
    except Exception as e:
        logger.error(f"Local assessment error: {e}")
        return {
            "local_sufficient": False,
            "local_assessment": {},
            "needs_web_search": True,
            "research_sufficient": False,
            "query": state.get("query", ""),
            # CRITICAL: Preserve round1_results
            "round1_results": state.get("round1_results", {}),
            "sources_found": state.get("sources_found", []),
            "research_findings": state.get("research_findings", {}),
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "expanded_queries": state.get("expanded_queries", [])
        }


async def round1_web_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Round 1 WEB: Web search (only runs if local results insufficient). Skill config can disable web search."""
    try:
        skill_config = state.get("skill_config", {})
        if not skill_config.get("web_search", True):
            logger.info("Round 1 WEB: Skipping web search - skill config has web_search=False")
            return {
                "web_round1_results": {"content": "", "skipped": True},
                "query": state.get("query", ""),
                "round1_results": state.get("round1_results", {}),
                "sources_found": state.get("sources_found", []),
                "research_findings": state.get("research_findings", {}),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "expanded_queries": state.get("expanded_queries", []),
                "skill_config": skill_config,
            }

        query = state.get("query", "")
        expanded_queries = state.get("expanded_queries", [])
        shared_memory = state.get("shared_memory", {})
        
        logger.info(f"Round 1 WEB: Local results insufficient - running web search")
        
        # Track tool usage
        previous_tools = shared_memory.get("previous_tools_used", [])
        if "search_web_tool" not in previous_tools:
            previous_tools.append("search_web_tool")
            previous_tools.append("crawl_web_content_tool")
            shared_memory["previous_tools_used"] = previous_tools
        
        # Use first expanded query (or original query as fallback)
        search_query = expanded_queries[0] if expanded_queries and expanded_queries[0] else query
        if not search_query:
            logger.warning("No valid query for web search, skipping")
            return {
                "web_round1_results": {"error": "No valid query", "content": ""},
                "query": state.get("query", ""),
                "round1_results": state.get("round1_results", {}),
                "sources_found": state.get("sources_found", []),
                "research_findings": state.get("research_findings", {}),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": shared_memory,
                "messages": state.get("messages", []),
                "expanded_queries": state.get("expanded_queries", []),
                "skill_config": state.get("skill_config", {}),
            }
        
        # Search web using structured search for better URL prioritization
        from orchestrator.tools import search_web_structured
        structured_results = await search_web_structured(query=search_query, max_results=10)
        logger.info("Tool used: search_web_structured (web search)")
        
        # Format results for display
        formatted_parts = []
        for i, res in enumerate(structured_results, 1):
            formatted_parts.append(f"\n{i}. **{res.get('title', 'No Title')}**")
            formatted_parts.append(f"   URL: {res.get('url', 'No URL')}")
            if res.get('snippet'):
                formatted_parts.append(f"   {res['snippet']}")
        search_result = '\n'.join(formatted_parts) if formatted_parts else ""
        
        # Prioritize URLs based on relevance scores
        from urllib.parse import urlparse
        official_keywords = ["allin.com", "allinpodcast.co", "youtube.com/@allin", "podcasts.apple.com"]
        
        priority_urls = []
        seen_domains = set()
        
        # Sort by relevance
        sorted_results = sorted(
            structured_results,
            key=lambda x: x.get('relevance_score', 0.0),
            reverse=True
        )
        
        for result in sorted_results:
            url = result.get('url')
            if not url:
                continue
            
            domain = urlparse(url).netloc
            relevance = result.get('relevance_score', 0.0)
            is_official = any(kw in url for kw in official_keywords)
            is_high_relevance = relevance >= 0.7
            
            if is_official and url not in priority_urls:
                priority_urls.insert(0, url)
            elif is_high_relevance and domain not in seen_domains and url not in priority_urls:
                priority_urls.append(url)
                seen_domains.add(domain)
            elif url not in priority_urls and domain not in seen_domains:
                priority_urls.append(url)
                seen_domains.add(domain)
        
        # Smart limit: crawl more if many high-relevance results
        high_relevance_count = sum(1 for r in structured_results if r.get('relevance_score', 0.0) >= 0.7)
        if high_relevance_count >= 5:
            effective_limit = min(7, len(priority_urls))
        elif high_relevance_count >= 3:
            effective_limit = min(5, len(priority_urls))
        else:
            effective_limit = min(3, len(priority_urls))
        
        top_urls = priority_urls[:effective_limit] if priority_urls else []
        
        crawled_content = ""
        if top_urls:
            crawl_result = await crawl_web_content_tool(urls=top_urls)
            logger.info("Tool used: crawl_web_content_tool (crawled top results)")
            crawled_content = f"\n\n=== Crawled Content ===\n{crawl_result}"
        
        combined_result = f"{search_result}{crawled_content}"
        
        web_result = {
            "content": combined_result,
            "query_used": search_query
        }
        
        # Add web sources to sources_found
        sources_found = list(state.get("sources_found", []))  # Copy existing sources
        if web_result.get("content"):
            urls = re.findall(r'URL: (https?://[^\s]+)', web_result.get("content", ""))
            for url in urls[:5]:
                sources_found.append({
                    "type": "web",
                    "url": url,
                    "source": "web"
                })
        
        # Update research_findings with web results
        research_findings = dict(state.get("research_findings", {}))  # Copy existing findings
        research_findings["web_results"] = web_result.get("content", "")
        # Preserve structured_images if they exist
        if "structured_images" not in research_findings and state.get("research_findings", {}).get("structured_images"):
            research_findings["structured_images"] = state.get("research_findings", {}).get("structured_images")
        
        logger.info(f"📊 Round 1 WEB complete: web={len(web_result.get('content', ''))} chars, sources={len(sources_found)}")
        
        return {
            "web_round1_results": web_result,
            "sources_found": sources_found,
            "research_findings": research_findings,
            "query": state.get("query", ""),
            "round1_results": state.get("round1_results", {}),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": shared_memory,
            "messages": state.get("messages", []),
            "expanded_queries": state.get("expanded_queries", []),
            "skill_config": state.get("skill_config", {}),
        }
        
    except Exception as e:
        logger.error(f"Round 1 web search error: {e}")
        return {
            "web_round1_results": {"error": str(e), "content": ""},
            "query": state.get("query", ""),
            "round1_results": state.get("round1_results", {}),
            "sources_found": state.get("sources_found", []),
            "research_findings": state.get("research_findings", {}),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "expanded_queries": state.get("expanded_queries", []),
            "skill_config": state.get("skill_config", {}),
        }


async def assess_combined_round1_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assess combined Round 1 results for quality and sufficiency"""
    try:
        query = state.get("query", "")
        round1_results = state.get("round1_results", {})
        web_round1_results = state.get("web_round1_results", {})
        
        local_results = round1_results.get("search_results", "")
        web_results = web_round1_results.get("content", "")
        
        logger.info("Assessing combined Round 1 results (local + web)")
        
        # Use LLM to assess quality - NO TRUNCATION since we're using markdown extraction
        assessment_prompt = f"""Assess the quality and sufficiency of these combined search results (local documents + web search) for answering the user's query.

USER QUERY: {query}

LOCAL DOCUMENT RESULTS:
{local_results if local_results else "No local results found."}

WEB SEARCH RESULTS:
{web_results if web_results else "No web results found."}

Evaluate:
1. Do the results (local + web combined) contain relevant information?
2. Is there enough detail to answer the query comprehensively?
3. What information is still missing (if any)?
4. Which source (local vs web) provides better information?

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "sufficient": boolean,
    "has_relevant_info": boolean,
    "missing_info": ["list", "of", "specific", "gaps"],
    "confidence": number (0.0-1.0),
    "reasoning": "brief explanation of assessment",
    "best_source": "local" | "web" | "both",
    "needs_more_local": boolean,
    "needs_more_web": boolean
}}"""
        
        # Get LLM
        from orchestrator.agents.base_agent import BaseAgent
        base_agent = BaseAgent("research_subgraph")
        llm = base_agent._get_llm(temperature=0.7, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        assessment_messages = [
            SystemMessage(content="You are a research quality assessor. Always respond with valid JSON."),
            SystemMessage(content=datetime_context)
        ]
        
        conversation_messages = state.get("messages", [])
        if conversation_messages:
            assessment_messages.extend(conversation_messages)
        
        assessment_messages.append(HumanMessage(content=assessment_prompt))
        
        response = await llm.ainvoke(assessment_messages)
        
        # Parse response
        try:
            text = response.content.strip()
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                text = json_match.group(0)
            
            assessment = ResearchAssessmentResult.parse_raw(text)
            assessment_dict = json.loads(text) if isinstance(text, str) else text
            
            sufficient = assessment.sufficient
            best_source = assessment_dict.get("best_source", "both")
            needs_more_local = assessment_dict.get("needs_more_local", False)
            needs_more_web = assessment_dict.get("needs_more_web", False)
            
            logger.info(f"Combined Round 1 assessment: sufficient={sufficient}, confidence={assessment.confidence}, best_source={best_source}")
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse assessment: {e}")
            sufficient = False
            best_source = "both"
            needs_more_local = True
            needs_more_web = False
        
        return {
            "round1_sufficient": sufficient,
            "round1_assessment": {
                "sufficient": sufficient,
                "has_relevant_info": assessment.has_relevant_info if 'assessment' in locals() else True,
                "confidence": assessment.confidence if 'assessment' in locals() else 0.5,
                "reasoning": assessment.reasoning if 'assessment' in locals() else response.content[:200],
                "best_source": best_source,
                "needs_more_local": needs_more_local,
                "needs_more_web": needs_more_web
            },
            "research_sufficient": sufficient,
            "query": query,  # Preserve query in state
            # CRITICAL: Preserve round1_results and web_round1_results so they flow to next nodes
            "round1_results": round1_results,
            "web_round1_results": web_round1_results,
            "sources_found": state.get("sources_found", []),  # Also preserve sources
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"Combined Round 1 assessment error: {e}")
        return {
            "round1_sufficient": False,
            "round1_assessment": {
                "sufficient": False,
                "has_relevant_info": False,
                "confidence": 0.0,
                "reasoning": f"Assessment error: {str(e)}",
                "best_source": "both",
                "needs_more_local": True,
                "needs_more_web": False
            },
            "research_sufficient": False,
            "query": state.get("query", ""),  # Preserve query in state
            # CRITICAL: Preserve round1_results and web_round1_results even on error
            "round1_results": state.get("round1_results", {}),
            "web_round1_results": state.get("web_round1_results", {}),
            "sources_found": state.get("sources_found", []),
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }




async def synthesize_findings_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize research findings into structured format"""
    try:
        round1_results = state.get("round1_results", {})
        web_round1_results = state.get("web_round1_results", {})
        sources_found = state.get("sources_found", [])

        local_results = round1_results.get("search_results", "")
        web_results = web_round1_results.get("content", "") if web_round1_results else ""
        image_search_results = round1_results.get("image_search_results")  # Get image results (markdown)
        structured_images = round1_results.get("structured_images")  # Get structured image data

        # Build citations
        citations = []
        for source in sources_found:
            if source.get("type") == "document":
                citations.append({
                    "type": "document",
                    "document_id": source.get("document_id"),
                    "source": "local"
                })
            elif source.get("type") == "web":
                citations.append({
                    "type": "web",
                    "url": source.get("url"),
                    "source": "web"
                })

        # Combine results (web may be empty if local was sufficient)
        combined_parts = [local_results] if local_results else []
        if web_results:
            combined_parts.append(web_results)
        
        research_findings = {
            "local_results": local_results,
            "web_results": web_results,
            "combined_results": "\n\n".join(combined_parts) if combined_parts else "",
            "sources_count": len(sources_found),
            "citations": citations,
            "image_search_results": image_search_results,  # Include image results (markdown)
            "structured_images": structured_images  # Include structured image data
        }

        # Log what we're synthesizing
        image_count = len(image_search_results) if image_search_results else 0
        web_status = "included" if web_results else "skipped (local sufficient)"
        logger.info(f"📊 Synthesizing findings: local={len(local_results)} chars, web={len(web_results)} chars ({web_status}), images={image_count} chars, sources={len(sources_found)}")
        
        # Pass through gap analysis and round1 assessment for routing decisions
        gap_analysis = state.get("gap_analysis", {})
        round1_assessment = state.get("round1_assessment", {})
        round1_sufficient = state.get("round1_sufficient", False)
        research_sufficient = state.get("research_sufficient", False)
        
        return {
            "research_findings": research_findings,
            "citations": citations,
            "sources_found": sources_found,
            "research_sufficient": research_sufficient or round1_sufficient,
            "round1_sufficient": round1_sufficient,
            "round1_assessment": round1_assessment,
            "gap_analysis": gap_analysis,
            "identified_gaps": state.get("identified_gaps", []),
            "query": state.get("query", ""),  # Preserve query in state
            # CRITICAL: Preserve round1_results and web_round1_results so they flow back to main agent
            "round1_results": round1_results,
            "web_round1_results": web_round1_results,
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"Synthesize findings error: {e}")
        return {
            "research_findings": {},
            "citations": [],
            "sources_found": [],
            "query": state.get("query", ""),  # Preserve query in state
            # CRITICAL: Preserve metadata for user model selection
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", [])
        }


def build_research_workflow_subgraph(checkpointer, skip_cache: bool = False, skip_expansion: bool = False) -> StateGraph:
    """
    Build research workflow subgraph
    
    Args:
        checkpointer: LangGraph checkpointer
        skip_cache: If True, skip cache check and go straight to query expansion
        skip_expansion: If True, skip query expansion and use query directly (or provided_queries from state)
    """
    # Build gap analysis subgraph once (reused by gap_analysis_node)
    gap_analysis_sg = build_gap_analysis_subgraph(checkpointer)
    
    # Create gap analysis node that uses the subgraph
    async def gap_analysis_node_with_subgraph(state: Dict[str, Any]) -> Dict[str, Any]:
        """Gap analysis node that uses the universal gap analysis subgraph"""
        try:
            query = state.get("query", "")
            round1_results = state.get("round1_results", {})
            web_round1_results = state.get("web_round1_results", {})
            
            logger.info("🔍 Performing gap analysis using universal subgraph")
            
            # Prepare combined results for gap analysis
            local_results = round1_results.get("search_results", "")
            web_results = web_round1_results.get("content", "")
            combined_results = f"{local_results}\n\n{web_results}".strip() if web_results else local_results
            
            # Prepare subgraph state
            gap_subgraph_state = {
                "query": query,
                "results": combined_results,
                "context": "",  # Can add conversation context if needed
                "domain": "research",
                "messages": state.get("messages", []),
                "metadata": state.get("metadata", {})
            }
            
            # Run gap analysis subgraph
            from orchestrator.agents.base_agent import BaseAgent
            base_agent = BaseAgent("research_subgraph")
            config = base_agent._get_checkpoint_config(state.get("metadata", {}))
            result = await gap_analysis_sg.ainvoke(gap_subgraph_state, config)
            
            # Extract results
            gap_analysis = result.get("gap_analysis", {})
            identified_gaps = result.get("identified_gaps", [])
            
            logger.info(f"✅ Gap analysis complete: severity={gap_analysis.get('gap_severity')}, web_search={gap_analysis.get('needs_web_search')}, gaps={len(identified_gaps)}")
            
            return {
                "gap_analysis": gap_analysis,
                "identified_gaps": identified_gaps,
                "query": query,  # Preserve query in state
                # CRITICAL: Preserve all state
                "round1_results": state.get("round1_results", {}),
                "web_round1_results": state.get("web_round1_results", {}),
                "sources_found": state.get("sources_found", []),
                "research_findings": state.get("research_findings", {}),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "expanded_queries": state.get("expanded_queries", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Gap analysis error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "gap_analysis": {
                    "has_gaps": False,
                    "needs_web_search": False,
                    "needs_local_search": False,
                    "gap_severity": "minor",
                    "reasoning": f"Gap analysis failed: {str(e)}"
                },
                "identified_gaps": [],
                "query": state.get("query", ""),  # Preserve query in state
                # CRITICAL: Preserve all state even on error
                "round1_results": state.get("round1_results", {}),
                "web_round1_results": state.get("web_round1_results", {}),
                "sources_found": state.get("sources_found", []),
                "research_findings": state.get("research_findings", {}),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "expanded_queries": state.get("expanded_queries", [])
            }
    
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes - NEW SEQUENTIAL FLOW
    subgraph.add_node("cache_check", cache_check_node)
    subgraph.add_node("query_expansion", query_expansion_node)
    subgraph.add_node("round1_local_search", round1_local_search_node)  # Local only
    subgraph.add_node("assess_local_results", assess_local_results_node)  # NEW: Assess local first
    subgraph.add_node("round1_web_search", round1_web_search_node)  # NEW: Conditional web search
    subgraph.add_node("assess_combined_round1", assess_combined_round1_node)  # Only after web search
    subgraph.add_node("gap_analysis", gap_analysis_node_with_subgraph)
    subgraph.add_node("synthesize_findings", synthesize_findings_node)
    
    # Set entry point based on skip_cache flag
    if skip_cache:
        subgraph.set_entry_point("query_expansion" if not skip_expansion else "round1_local_search")
    else:
        subgraph.set_entry_point("cache_check")
    
    # Flow - SEQUENTIAL with CONDITIONAL web search
    if not skip_cache:
        subgraph.add_conditional_edges(
            "cache_check",
            lambda state: "synthesize_findings" if state.get("cache_hit") else "query_expansion",
            {
                "synthesize_findings": "synthesize_findings",
                "query_expansion": "query_expansion"
            }
        )
    
    if not skip_expansion:
        subgraph.add_edge("query_expansion", "round1_local_search")
    
    # After local search, assess local results
    subgraph.add_edge("round1_local_search", "assess_local_results")
    
    # Conditional routing: local sufficient -> synthesize; else web search only if permission granted
    def _route_after_local_assessment(state: Dict[str, Any]) -> str:
        if state.get("local_sufficient"):
            return "synthesize_findings"
        if not state.get("shared_memory", {}).get("web_search_permission"):
            return "synthesize_findings"  # No web permission: synthesize from local only (same as direct Research routing)
        return "round1_web_search"

    subgraph.add_conditional_edges(
        "assess_local_results",
        _route_after_local_assessment,
        {
            "synthesize_findings": "synthesize_findings",
            "round1_web_search": "round1_web_search"
        }
    )
    
    # After web search, assess combined results
    subgraph.add_edge("round1_web_search", "assess_combined_round1")
    
    # After combined assessment, either synthesize or do gap analysis
    subgraph.add_conditional_edges(
        "assess_combined_round1",
        lambda state: "synthesize_findings" if state.get("round1_sufficient") else "gap_analysis",
        {
            "synthesize_findings": "synthesize_findings",
            "gap_analysis": "gap_analysis"
        }
    )
    
    subgraph.add_edge("gap_analysis", "synthesize_findings")
    subgraph.add_edge("synthesize_findings", END)
    
    return subgraph.compile(checkpointer=checkpointer)

