"""
Core research workflow nodes: research subgraph, full doc analysis decision,
gap analysis check, full doc analysis subgraph, round2 parallel.
"""

import json
import logging
import re
from typing import Any, Dict, List, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.research.research_state import ResearchState, ResearchRound

if TYPE_CHECKING:
    from orchestrator.agents.research.full_research_agent import FullResearchAgent

logger = logging.getLogger(__name__)


async def call_research_subgraph_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Call research workflow subgraph to perform core research"""
    try:
        logger.info("Calling research workflow subgraph")

        workflow = await agent._get_workflow()
        checkpointer = workflow.checkpointer
        research_sg = agent._get_research_subgraph(checkpointer, skip_cache=False, skip_expansion=False)

        query = state.get("query", "")
        messages = state.get("messages", [])
        shared_memory = state.get("shared_memory", {})
        skill_config = state.get("skill_config", {})
        subgraph_state = {
            "query": query,
            "original_query": query,
            "shared_memory": shared_memory,
            "messages": messages,
            "user_id": shared_memory.get("user_id", "system"),
            "metadata": state.get("metadata", {}),
            "skill_config": skill_config,
        }

        logger.info(f"Passing query to research subgraph: '{query[:100]}'")
        logger.info(f"DEBUG: Passing {len(messages)} messages to research subgraph")

        config = agent._get_checkpoint_config(state.get("metadata", {}))
        result = await research_sg.ainvoke(subgraph_state, config)

        logger.info("Research subgraph completed")

        research_findings = result.get("research_findings", {})
        sources_found = result.get("sources_found", [])
        citations = result.get("citations", [])
        research_sufficient = result.get("research_sufficient", False)
        round1_sufficient = result.get("round1_sufficient", False)

        subgraph_round1_results = result.get("round1_results", {})
        subgraph_web_round1_results = result.get("web_round1_results", {})

        if subgraph_round1_results and subgraph_round1_results.get("search_results"):
            round1_results = subgraph_round1_results
            logger.info(f"Using round1_results from subgraph: {len(round1_results.get('search_results', ''))} chars")
        else:
            round1_results = {
                "search_results": research_findings.get("local_results", ""),
                "documents_found": len([s for s in sources_found if s.get("source") == "local"]),
                "round1_document_ids": [s.get("document_id") for s in sources_found if s.get("source") == "local" and s.get("document_id")],
                "structured_images": research_findings.get("structured_images"),
            }
            logger.info(f"Using fallback round1_results from research_findings: {len(round1_results.get('search_results', ''))} chars")

        if subgraph_web_round1_results and subgraph_web_round1_results.get("content"):
            web_round1_results = subgraph_web_round1_results
            logger.info(f"Using web_round1_results from subgraph: {len(web_round1_results.get('content', ''))} chars")
        else:
            web_round1_results = {
                "content": research_findings.get("web_results", ""),
                "sources_found": [s for s in sources_found if s.get("source") == "web"],
            }
            logger.info(f"Using fallback web_round1_results from research_findings: {len(web_round1_results.get('content', ''))} chars")

        round1_assessment = result.get("round1_assessment", {})
        gap_analysis = result.get("gap_analysis", {})
        identified_gaps = result.get("identified_gaps", [])
        cache_hit = result.get("cache_hit", False)
        cached_context = result.get("cached_context", "")

        return {
            "round1_results": round1_results,
            "web_round1_results": web_round1_results,
            "sources_found": sources_found,
            "citations": citations,
            "round1_sufficient": round1_sufficient or research_sufficient,
            "round1_assessment": round1_assessment,
            "gap_analysis": gap_analysis,
            "identified_gaps": identified_gaps,
            "cache_hit": cache_hit,
            "cached_context": cached_context,
            "current_round": ResearchRound.ROUND_2_PARALLEL.value if not (round1_sufficient or research_sufficient) else ResearchRound.FINAL_SYNTHESIS.value,
            "research_findings": research_findings,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "skill_config": skill_config,
        }

    except Exception as e:
        logger.error(f"Research subgraph error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "round1_results": {"error": str(e), "search_results": ""},
            "web_round1_results": {"error": str(e), "content": ""},
            "sources_found": [],
            "citations": [],
            "round1_sufficient": False,
            "cache_hit": False,
            "cached_context": "",
            "current_round": ResearchRound.ROUND_2_PARALLEL.value,
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "skill_config": state.get("skill_config", {}),
        }


async def full_document_analysis_decision_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Hybrid decision: rules + LLM to determine if full docs needed. Skill config can disable full doc analysis."""
    try:
        skill_config = state.get("skill_config", {})
        if not skill_config.get("full_doc_analysis", True):
            logger.info("Skip full doc: skill config has full_doc_analysis=False")
            return {
                "full_doc_analysis_needed": False,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "skill_config": skill_config,
            }

        query = state.get("query", "")
        round1_results = state.get("round1_results", {})
        sources_found = state.get("sources_found", [])

        MIN_QUALITY_CHUNKS = 3
        MIN_CHUNKS_PER_DOC = 3
        MIN_CHUNK_SCORE = 0.7
        MAX_DOC_TOKENS = 100000
        LLM_CONFIDENCE_THRESHOLD = 0.6
        MAX_DOCS_TO_ANALYZE = 2
        MAX_PARALLEL_QUERIES = 4

        logger.info("Evaluating if full document analysis is needed")

        if len(query.split()) < 5:
            logger.info("Skip full doc: Query too simple (< 5 words)")
            return {"full_doc_analysis_needed": False, "skill_config": state.get("skill_config", {}), "metadata": state.get("metadata", {}), "user_id": state.get("user_id", "system"), "shared_memory": state.get("shared_memory", {}), "messages": state.get("messages", []), "query": state.get("query", "")}

        document_ids = []
        for source in sources_found:
            if source.get("type") == "document" and source.get("document_id"):
                document_ids.append(source.get("document_id"))

        if not document_ids:
            logger.info("Skip full doc: No document IDs found in sources")
            return {"full_doc_analysis_needed": False, "skill_config": state.get("skill_config", {}), "metadata": state.get("metadata", {}), "user_id": state.get("user_id", "system"), "shared_memory": state.get("shared_memory", {}), "messages": state.get("messages", []), "query": state.get("query", "")}

        round1_document_ids = round1_results.get("round1_document_ids", [])
        if not round1_document_ids:
            round1_document_ids = document_ids[:5]

        from orchestrator.subgraphs.intelligent_document_retrieval_subgraph import retrieve_documents_intelligently

        shared_memory = state.get("shared_memory", {})
        user_id = shared_memory.get("user_id", "system") if shared_memory else "system"

        try:
            chunk_result = await retrieve_documents_intelligently(
                query=query,
                user_id=user_id,
                mode="fast",
                max_results=5,
                small_doc_threshold=15000,
            )

            retrieved_docs = chunk_result.get("retrieved_documents", [])

            if not retrieved_docs:
                logger.info("Skip full doc: No documents retrieved")
                return {"full_doc_analysis_needed": False, "skill_config": state.get("skill_config", {}), "metadata": state.get("metadata", {}), "user_id": state.get("user_id", "system"), "shared_memory": state.get("shared_memory", {}), "messages": state.get("messages", []), "query": state.get("query", "")}

            doc_chunk_counts = {}
            high_quality_chunks = []
            cross_ref_signals = []

            for doc in retrieved_docs:
                doc_id = doc.get("document_id")
                if not doc_id:
                    continue

                strategy = doc.get("retrieval_strategy", "")

                if strategy == "full_document":
                    continue

                if strategy == "multi_chunk":
                    chunks = doc.get("chunks", [])
                    high_quality = [c for c in chunks if c.get("relevance_score", 0.0) >= MIN_CHUNK_SCORE]
                    if len(high_quality) >= MIN_CHUNKS_PER_DOC:
                        doc_chunk_counts[doc_id] = len(high_quality)
                        high_quality_chunks.extend(high_quality)

                        for chunk in high_quality:
                            content = chunk.get("content", "").lower()
                            if any(signal in content for signal in ["see section", "as discussed", "mentioned earlier", "refer to chapter", "in part"]):
                                cross_ref_signals.append(doc_id)
                                break

            promising_docs = {
                doc_id: count for doc_id, count in doc_chunk_counts.items()
                if count >= MIN_CHUNKS_PER_DOC
            }

            if not promising_docs:
                logger.info("Skip full doc: No document with 3+ quality chunks")
                return {"full_doc_analysis_needed": False, "skill_config": state.get("skill_config", {}), "metadata": state.get("metadata", {}), "user_id": state.get("user_id", "system"), "shared_memory": state.get("shared_memory", {}), "messages": state.get("messages", []), "query": state.get("query", "")}

            from orchestrator.backend_tool_client import get_backend_tool_client
            client = await get_backend_tool_client()

            reasonable_docs = {}
            for doc_id in promising_docs.keys():
                try:
                    doc_size = await client.get_document_size(doc_id, user_id)
                    estimated_tokens = doc_size // 4
                    if estimated_tokens < MAX_DOC_TOKENS:
                        reasonable_docs[doc_id] = promising_docs[doc_id]
                except Exception as e:
                    logger.warning(f"Could not check size for doc {doc_id}: {e}")
                    reasonable_docs[doc_id] = promising_docs[doc_id]

            if not reasonable_docs:
                logger.info("Skip full doc: All promising docs too large")
                return {"full_doc_analysis_needed": False, "skill_config": state.get("skill_config", {}), "metadata": state.get("metadata", {}), "user_id": state.get("user_id", "system"), "shared_memory": state.get("shared_memory", {}), "messages": state.get("messages", []), "query": state.get("query", "")}

            if len(high_quality_chunks) < MIN_QUALITY_CHUNKS:
                logger.info(f"Skip full doc: Too few quality chunks ({len(high_quality_chunks)} < {MIN_QUALITY_CHUNKS})")
                return {"full_doc_analysis_needed": False, "skill_config": state.get("skill_config", {}), "metadata": state.get("metadata", {}), "user_id": state.get("user_id", "system"), "shared_memory": state.get("shared_memory", {}), "messages": state.get("messages", []), "query": state.get("query", "")}

        except Exception as e:
            logger.warning(f"Error checking chunk patterns: {e}, skipping full doc analysis")
            return {"full_doc_analysis_needed": False, "skill_config": state.get("skill_config", {}), "metadata": state.get("metadata", {}), "user_id": state.get("user_id", "system"), "shared_memory": state.get("shared_memory", {}), "messages": state.get("messages", []), "query": state.get("query", "")}

        logger.info(f"Rules passed: {len(reasonable_docs)} promising docs, asking LLM...")

        chunk_preview = "\n\n".join([
            f"[Doc {c.get('document_id', 'unknown')[:8]}] Score: {c.get('relevance_score', 0.0):.2f}\n{c.get('content', '')[:200]}..."
            for c in high_quality_chunks[:5]
        ])

        top_doc_id = max(reasonable_docs.items(), key=lambda x: x[1])[0]
        top_doc_count = reasonable_docs[top_doc_id]

        decision_prompt = f"""You are evaluating whether to retrieve full documents for deeper analysis.

QUERY: {query}

CHUNK ANALYSIS:
- Found {len(high_quality_chunks)} high-quality chunks across {len(reasonable_docs)} documents
- Top document has {top_doc_count} relevant chunks
- Cross-reference signals found: {len(cross_ref_signals)} document(s)

CHUNK PREVIEW:
{chunk_preview}

EVALUATION CRITERIA:
1. Do chunks appear fragmentary or incomplete?
2. Does the query require synthesis across sections? (e.g., "relationship between X and Y", "how did X evolve")
3. Would full document context significantly improve answer quality?
4. Are chunks referencing other sections ("see chapter X", "as discussed earlier")?

Respond with ONLY valid JSON matching this exact schema:
{{
    "should_retrieve_full_docs": boolean,
    "confidence": number (0.0-1.0),
    "reasoning": "brief explanation",
    "suggested_queries": ["query1", "query2", "query3"]
}}

Your decision:"""

        llm = agent._get_llm(temperature=0.3, state=state)
        datetime_context = agent._get_datetime_context()

        decision_messages = [
            SystemMessage(content="You are a document analysis decision maker. Always respond with valid JSON matching the exact schema provided."),
            SystemMessage(content=datetime_context),
        ]

        conversation_messages = state.get("messages", [])
        if conversation_messages:
            decision_messages.extend(conversation_messages)

        decision_messages.append(HumanMessage(content=decision_prompt))

        response = await llm.ainvoke(decision_messages)

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

            decision = json.loads(text)

            if decision.get("should_retrieve_full_docs") and decision.get("confidence", 0.0) >= LLM_CONFIDENCE_THRESHOLD:
                suggested_queries = decision.get("suggested_queries", [])
                if not suggested_queries:
                    suggested_queries = [
                        f"What specific details about {query}?",
                        "How does this relate to the main topic?",
                        "What context is missing from the chunks?",
                    ]

                sorted_docs = sorted(reasonable_docs.items(), key=lambda x: x[1], reverse=True)
                doc_ids_to_analyze = [doc_id for doc_id, _ in sorted_docs[:MAX_DOCS_TO_ANALYZE]]

                logger.info(f"LLM decided YES (confidence={decision.get('confidence', 0.0):.2f}): {decision.get('reasoning', '')}")
                logger.info(f"Will analyze {len(doc_ids_to_analyze)} documents with {len(suggested_queries[:MAX_PARALLEL_QUERIES])} queries")

                return {
                    "full_doc_analysis_needed": True,
                    "document_ids_to_analyze": doc_ids_to_analyze,
                    "analysis_queries": suggested_queries[:MAX_PARALLEL_QUERIES],
                    "full_doc_decision_reasoning": decision.get("reasoning", ""),
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                }
            else:
                logger.info(f"LLM decided NO (confidence={decision.get('confidence', 0.0):.2f}): {decision.get('reasoning', '')}")
                return {"full_doc_analysis_needed": False, "skill_config": state.get("skill_config", {}), "metadata": state.get("metadata", {}), "user_id": state.get("user_id", "system"), "shared_memory": state.get("shared_memory", {}), "messages": state.get("messages", []), "query": state.get("query", "")}

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM decision parsing failed: {e}, defaulting to NO")
            return {"full_doc_analysis_needed": False, "skill_config": state.get("skill_config", {}), "metadata": state.get("metadata", {}), "user_id": state.get("user_id", "system"), "shared_memory": state.get("shared_memory", {}), "messages": state.get("messages", []), "query": state.get("query", "")}

    except Exception as e:
        logger.error(f"Full document analysis decision error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"full_doc_analysis_needed": False, "skill_config": state.get("skill_config", {}), "metadata": state.get("metadata", {}), "user_id": state.get("user_id", "system"), "shared_memory": state.get("shared_memory", {}), "messages": state.get("messages", []), "query": state.get("query", "")}


async def gap_analysis_check_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Pass-through node; routing logic handles the decision. Preserve skill_config."""
    return {"skill_config": state.get("skill_config", {})}


async def call_full_document_analysis_subgraph_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Call full document analysis subgraph to analyze full documents"""
    try:
        logger.info("Calling full document analysis subgraph")

        document_ids = state.get("document_ids_to_analyze", [])
        analysis_queries = state.get("analysis_queries", [])
        original_query = state.get("query", "")

        if not document_ids or not analysis_queries:
            logger.warning("Missing required inputs for full document analysis")
            return {
                "full_doc_insights": {},
                "documents_analyzed": [],
                "synthesis": "",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }

        workflow = await agent._get_workflow()
        checkpointer = workflow.checkpointer

        if not hasattr(agent, "_full_doc_analysis_subgraph") or agent._full_doc_analysis_subgraph is None:
            from orchestrator.subgraphs import build_full_document_analysis_subgraph
            agent._full_doc_analysis_subgraph = build_full_document_analysis_subgraph(checkpointer)

        shared_memory = state.get("shared_memory", {})
        user_id = shared_memory.get("user_id", "system") if shared_memory else "system"

        subgraph_state = {
            "document_ids": document_ids,
            "analysis_queries": analysis_queries,
            "original_query": original_query,
            "chunk_context": [],
            "user_id": user_id,
            "messages": state.get("messages", []),
            "metadata": state.get("metadata", {}),
        }

        logger.info(f"Analyzing {len(document_ids)} documents with {len(analysis_queries)} queries")

        config = agent._get_checkpoint_config(state.get("metadata", {}))
        result = await agent._full_doc_analysis_subgraph.ainvoke(subgraph_state, config)

        logger.info("Full document analysis subgraph completed")

        return {
            "full_doc_insights": result.get("full_doc_insights", {}),
            "documents_analyzed": result.get("documents_analyzed", []),
            "synthesis": result.get("synthesis", ""),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }

    except Exception as e:
        logger.error(f"Full document analysis subgraph error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "full_doc_insights": {},
            "documents_analyzed": [],
            "synthesis": "",
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def round2_parallel_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Round 2: Parallel local and web gap-filling searches based on gap analysis flags"""
    try:
        import asyncio

        query = state["query"]
        gap_analysis = state.get("gap_analysis", {})
        identified_gaps = state.get("identified_gaps", [])

        needs_local = gap_analysis.get("needs_local_search", False)
        needs_web = gap_analysis.get("needs_web_search", False)

        logger.info(f"Round 2 Parallel: local={needs_local}, web={needs_web}, gaps={len(identified_gaps)}")

        gap_queries = identified_gaps[:3] if identified_gaps else [query]

        async def local_search_task():
            if not needs_local:
                logger.info("Skipping Round 2 Local - not needed per gap analysis")
                return None

            try:
                logger.info(f"Round 2 Local: Searching for {len(gap_queries)} gaps")

                workflow = await agent._get_workflow()
                checkpointer = workflow.checkpointer
                research_sg = agent._get_research_subgraph(checkpointer, skip_cache=True, skip_expansion=True)

                shared_memory = state.get("shared_memory", {})
                subgraph_state = {
                    "query": gap_queries[0],
                    "provided_queries": gap_queries,
                    "shared_memory": shared_memory,
                    "messages": state.get("messages", []),
                    "user_id": shared_memory.get("user_id", "system"),
                    "metadata": state.get("metadata", {}),
                }

                config = agent._get_checkpoint_config(state.get("metadata", {}))
                result = await research_sg.ainvoke(subgraph_state, config)

                logger.info("Round 2 Local complete")
                return result

            except Exception as e:
                logger.error(f"Round 2 Local error: {e}")
                return {"error": str(e)}

        async def web_search_task():
            skill_config = state.get("skill_config", {})
            if not skill_config.get("web_search", True):
                logger.info("Skipping Round 2 Web - skill config has web_search=False")
                return None
            if not needs_web:
                logger.info("Skipping Round 2 Web - not needed per gap analysis")
                return None

            try:
                logger.info(f"Round 2 Web: Searching for {len(gap_queries)} gaps")

                workflow = await agent._get_workflow()
                checkpointer = workflow.checkpointer if workflow else None
                web_research_sg = agent._get_web_research_subgraph(checkpointer)

                web_subgraph_state = {
                    "query": gap_queries[0],
                    "queries": gap_queries if len(gap_queries) > 1 else None,
                    "max_results": 10,
                    "crawl_top_n": 5,
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "metadata": state.get("metadata", {}),
                }

                config = agent._get_checkpoint_config(state.get("metadata", {}))
                result = await web_research_sg.ainvoke(web_subgraph_state, config)

                logger.info("Round 2 Web complete")
                return result

            except Exception as e:
                logger.error(f"Round 2 Web error: {e}")
                return {"error": str(e)}

        local_result, web_result = await asyncio.gather(
            local_search_task(),
            web_search_task(),
            return_exceptions=True,
        )

        if isinstance(local_result, Exception):
            logger.error(f"Local search exception: {local_result}")
            local_result = None

        if isinstance(web_result, Exception):
            logger.error(f"Web search exception: {web_result}")
            web_result = None

        combined_sources = []
        combined_findings = {}

        if local_result:
            local_sources = local_result.get("sources_found", [])
            combined_sources.extend(local_sources)
            combined_findings["local_round2"] = local_result.get("research_findings", {}).get("local_results", "")

        if web_result:
            web_sources = web_result.get("sources_found", [])
            combined_sources.extend(web_sources)
            combined_findings["web_round2"] = web_result.get("web_results", {}).get("content", "")

        logger.info(f"Round 2 Parallel complete: {len(combined_sources)} total sources")

        return {
            "round2_results": {
                "local_result": local_result,
                "web_result": web_result,
                "combined_sources": len(combined_sources),
            },
            "sources_found": combined_sources,
            "research_findings": {
                **state.get("research_findings", {}),
                **combined_findings,
            },
            "round2_sufficient": True,
            "current_round": ResearchRound.FINAL_SYNTHESIS.value,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }

    except Exception as e:
        logger.error(f"Round 2 Parallel error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "round2_results": {"error": str(e)},
            "sources_found": [],
            "round2_sufficient": True,
            "current_round": ResearchRound.FINAL_SYNTHESIS.value,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
