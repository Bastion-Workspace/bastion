"""
Synthesis and post-processing nodes: detect query type, final synthesis,
post-process results, combine post-processed output.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import ValidationError
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.agents.research.research_state import ResearchState, ResearchRound
from orchestrator.utils.formatting_detection import detect_post_processing_needs

if TYPE_CHECKING:
    from orchestrator.agents.research.full_research_agent import FullResearchAgent

logger = logging.getLogger(__name__)


async def detect_query_type_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Detect query type to determine synthesis approach: objective (synthesize) vs subjective (present options)"""
    try:
        from orchestrator.models import QueryTypeDetection

        query = state["query"]

        logger.info(f"Detecting query type for: {query}")

        detection_prompt = f"""Analyze this query to determine whether it should receive a synthesized single answer or multiple distinct options.

USER QUERY: {query}

Consider:
1. **Objective queries** (synthesize single answer): Factual, process, historical, scientific - clear objective answers.
2. **Subjective queries** (present 2-3 options): Preference-based, style choices, personal decisions, creative, recipes - multiple valid answers.
3. **Mixed queries** (synthesize with alternatives mentioned): Primary answer but notable alternatives.

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{{
    "query_type": "objective" | "subjective" | "mixed",
    "confidence": number (0.0-1.0),
    "reasoning": "brief explanation",
    "should_present_options": boolean,
    "num_options": number (2-3, only relevant if should_present_options=true)
}}"""

        llm = agent._get_llm(temperature=0.3, state=state)
        datetime_context = agent._get_datetime_context()

        detection_messages = [
            SystemMessage(content="You are a query type classifier. Always respond with valid JSON matching the exact schema provided."),
            SystemMessage(content=datetime_context),
        ]

        conversation_messages = state.get("messages", [])
        if conversation_messages:
            detection_messages.extend(conversation_messages)

        detection_messages.append(HumanMessage(content=detection_prompt))

        response = await llm.ainvoke(detection_messages)

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

            detection = QueryTypeDetection.parse_raw(text)

            logger.info(f"Query type detected: {detection.query_type}, confidence={detection.confidence}")

            return {
                "query_type": detection.query_type,
                "query_type_detection": {
                    "query_type": detection.query_type,
                    "confidence": detection.confidence,
                    "reasoning": detection.reasoning,
                    "should_present_options": detection.should_present_options,
                    "num_options": detection.num_options,
                },
                "should_present_options": detection.should_present_options,
                "num_options": detection.num_options,
                "skill_config": state.get("skill_config", {}),
            }

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.warning(f"Failed to parse query type detection: {e}")
            logger.info("Query type detection parsing failed - defaulting to objective (synthesize)")
            return {
                "query_type": "objective",
                "query_type_detection": {
                    "query_type": "objective",
                    "confidence": 0.5,
                    "reasoning": "Detection parsing failed - defaulting to objective",
                    "should_present_options": False,
                    "num_options": None,
                },
                "should_present_options": False,
                "num_options": None,
                "skill_config": state.get("skill_config", {}),
            }

    except Exception as e:
        logger.error(f"Query type detection error: {e}")
        return {
            "query_type": "objective",
            "query_type_detection": {
                "query_type": "objective",
                "confidence": 0.5,
                "reasoning": f"Detection error: {str(e)}",
                "should_present_options": False,
                "num_options": None,
            },
            "should_present_options": False,
            "num_options": None,
            "skill_config": state.get("skill_config", {}),
        }


async def final_synthesis_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Final synthesis with all gathered information"""
    try:
        if state.get("research_complete") and state.get("final_response"):
            logger.info("Using pre-computed response from fast path (skipping synthesis)")
            return {
                "final_response": state.get("final_response", ""),
                "formatting_recommendations": state.get("formatting_recommendations"),
                "query": state.get("query", ""),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "structured_images": state.get("structured_images"),
                "image_search_results": state.get("image_search_results"),
                "skill_config": state.get("skill_config", {}),
            }

        query = state["query"]
        skill_config = state.get("skill_config", {})
        synthesis_style = skill_config.get("synthesis_style", "comprehensive")

        cached_context = state.get("cached_context", "")
        round1_results = state.get("round1_results", {})
        round2_results = state.get("round2_results", {})
        web_round1_results = state.get("web_round1_results", {})
        web_round2_results = state.get("web_round2_results", {})
        web_results = state.get("web_search_results", {}) or web_round1_results
        full_doc_insights = state.get("full_doc_insights", {})
        full_doc_synthesis = full_doc_insights.get("synthesis", "")

        round1_content_len = len(round1_results.get("search_results", "")) if round1_results else 0
        round1_entity_len = len(round1_results.get("entity_graph_results", "")) if round1_results else 0
        round2_content_len = len(round2_results.get("gap_results", "")) if round2_results else 0
        web1_content_len = len(web_round1_results.get("content", "")) if web_round1_results else 0
        web2_content_len = len(web_round2_results.get("content", "")) if web_round2_results else 0
        full_doc_content_len = len(full_doc_synthesis) if full_doc_synthesis else 0
        logger.info(f"Synthesis node received: round1={round1_content_len} chars, round1_entity={round1_entity_len} chars, round2={round2_content_len} chars, web1={web1_content_len} chars, web2={web2_content_len} chars, full_doc={full_doc_content_len} chars")

        logger.info("Synthesizing final response from all sources")

        context_parts = []

        if cached_context:
            context_parts.append(f"CACHED RESEARCH:\n{cached_context}")

        if round1_results:
            local_content = round1_results.get("search_results", "")
            entity_graph_content = round1_results.get("entity_graph_results", "")
            if local_content:
                context_parts.append(f"LOCAL SEARCH ROUND 1:\n{local_content[:20000]}")
            if entity_graph_content:
                context_parts.append(f"ENTITY GRAPH SEARCH (Knowledge Graph):\n{entity_graph_content[:15000]}")

        if round2_results:
            context_parts.append(f"LOCAL SEARCH ROUND 2:\n{round2_results.get('gap_results', '')[:20000]}")

        if web_round1_results:
            web_content = web_round1_results.get("content", "")
            context_parts.append(f"WEB SEARCH ROUND 1:\n{web_content if isinstance(web_content, str) else str(web_content)}")

        if web_round2_results:
            web2_content = web_round2_results.get("content", "")
            context_parts.append(f"WEB SEARCH ROUND 2:\n{web2_content if isinstance(web2_content, str) else str(web2_content)}")

        if full_doc_synthesis:
            context_parts.append(f"FULL DOCUMENT ANALYSIS:\n{full_doc_synthesis}")

        full_context = "\n\n".join(context_parts)

        query_type = state.get("query_type", "objective")
        should_present_options = state.get("should_present_options", False)
        num_options = state.get("num_options", 3)
        query_type_detection = state.get("query_type_detection", {})
        reasoning = query_type_detection.get("reasoning", "")

        logger.info(f"Synthesizing response with query_type={query_type}, should_present_options={should_present_options}, synthesis_style={synthesis_style}")

        if synthesis_style == "analytical":
            synthesis_prompt = f"""Based on all available research (local documents only; do not cite web sources), provide an analytical, comparison-focused response to the user's query.

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

Provide a well-organized analytical response that compares and analyzes information from the research. Focus on local/document sources. Do not cite web URLs. Only include information directly relevant to the query.

Your analytical response:"""

        elif synthesis_style == "verification":
            synthesis_prompt = f"""Based on all available research, provide a fact-checking and verification response. For each key claim or finding, indicate confidence (high/medium/low) and source.

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

Provide a verification-style response that emphasizes accuracy and confidence ratings. Cite sources. Only include information directly relevant to the query.

Your verification response:"""

        elif synthesis_style == "security_report":
            synthesis_prompt = f"""Based on all available research, provide a security-focused report. Highlight vulnerabilities, risks, and severity where applicable.

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

Provide a security report that emphasizes vulnerabilities, severity ratings, and recommendations. Use clear structure (e.g. findings, severity, mitigation). Only include information directly relevant to the query.

Your security report:"""

        elif synthesis_style == "extraction":
            synthesis_prompt = f"""Based on all available research, extract and present the core content relevant to the user's query. Minimal synthesis; focus on clear extraction.

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

Extract and present the relevant content in a clear, structured way. Minimal commentary. Only include information directly relevant to the query.

Your extracted content:"""

        elif synthesis_style == "ingestion":
            synthesis_prompt = f"""Based on all available research, produce structured content suitable for document ingestion (e.g. headings, bullet points, clear sections).

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

Produce well-structured content with clear headings and sections that can be ingested into a knowledge base. Only include information directly relevant to the query.

Your structured content:"""

        elif should_present_options and query_type in ["subjective", "mixed"]:
            synthesis_prompt = f"""Based on all available research, present {num_options} distinct, well-researched approaches to the user's query.

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

REASONING FOR PRESENTING OPTIONS:
{reasoning}

Provide a well-organized response that:
1. Presents {num_options} distinct approaches/options (each with clear title/name)
2. For each option, include key characteristics, advantages, trade-offs, when to use
3. Highlight key differences between the options
4. Cite sources where appropriate

Format as:
## Option 1: [Name]
[Description...]

## Option 2: [Name]
[Description...]

Your response with {num_options} distinct options:"""

        elif query_type == "mixed":
            synthesis_prompt = f"""Based on all available research, provide a comprehensive answer with primary approach synthesized and notable alternatives mentioned.

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

REASONING:
{reasoning}

Provide a well-organized response that synthesizes the primary approach and mentions notable alternatives. Only include information directly relevant to the query.

Your comprehensive response:"""

        else:
            synthesis_prompt = f"""Based on all available research, provide a comprehensive answer to the user's query.

USER QUERY: {query}

RESEARCH FINDINGS:
{full_context}

Provide a well-organized, thorough response that directly answers the query, synthesizes information from all sources, and cites sources where appropriate. Only include information directly relevant to the query. If no relevant information was found, state that clearly.

Your comprehensive response:"""

        synthesis_llm = agent._get_llm(temperature=0.3, state=state)
        datetime_context = agent._get_datetime_context()

        shared_memory = state.get("shared_memory", {})
        handoff_context = shared_memory.get("handoff_context", {})
        handoff_note = ""

        if handoff_context:
            source_agent = handoff_context.get("source_agent", "unknown")
            reference_doc = handoff_context.get("reference_document", {})
            if reference_doc.get("has_content"):
                ref_content = reference_doc.get("content", "")
                ref_filename = reference_doc.get("filename", "unknown")
                handoff_note = f"""

**AGENT HANDOFF CONTEXT**:
- Delegated by: {source_agent}
- User has reference document: {ref_filename}

**USER'S REFERENCE DOCUMENT CONTENT**:
{ref_content}

When synthesizing your answer, integrate information from the user's reference document with your research findings."""
                logger.info(f"Handoff context available for synthesis from {source_agent}")

        synthesis_messages = [
            SystemMessage(content="You are an expert research synthesizer."),
            SystemMessage(content=datetime_context),
        ]

        conversation_messages = state.get("messages", [])
        if conversation_messages:
            synthesis_messages.extend(conversation_messages)
            logger.info(f"Including {len(conversation_messages)} conversation messages for synthesis context")

        full_synthesis_prompt = synthesis_prompt + handoff_note
        synthesis_messages.append(HumanMessage(content=full_synthesis_prompt))

        response = await synthesis_llm.ainvoke(synthesis_messages)

        final_response = response.content

        attachment_analysis = state.get("attachment_analysis")
        if attachment_analysis and not attachment_analysis.get("error"):
            attachment_result = attachment_analysis.get("result", "")
            logger.info(f"Including attachment analysis in response: {len(attachment_result)} characters")
            final_response = f"{final_response}\n\n**Attached Image Analysis:**\n{attachment_result}"

        image_search_results = None
        structured_images = None
        round1_assessment = state.get("round1_assessment", {})
        best_source = round1_assessment.get("best_source", "local")

        if round1_results and best_source == "local":
            image_search_results = round1_results.get("image_search_results")
            structured_images = round1_results.get("structured_images")
            logger.info(f"Local assessment: best_source='{best_source}' - including local image results")
        elif round1_results:
            logger.info(f"Local assessment: best_source='{best_source}' - SKIPPING local image results (not relevant)")

        if image_search_results and not structured_images:
            logger.info(f"Appending image search results to response: {len(image_search_results)} characters")
            final_response = f"{final_response}\n\n{image_search_results}"

        if structured_images:
            logger.info(f"Collected {len(structured_images)} structured image(s) for AgentResponse")

        logger.info(f"Synthesis complete: {len(final_response)} characters")

        formatting_recommendations = await detect_post_processing_needs(query, final_response)

        if formatting_recommendations:
            logger.info(f"Post-processing recommended: table={formatting_recommendations.get('table_recommended')}, chart={formatting_recommendations.get('chart_recommended')}, timeline={formatting_recommendations.get('timeline_recommended')}")

        routing_recommendation = None
        if formatting_recommendations and (formatting_recommendations.get("table_recommended") or formatting_recommendations.get("timeline_recommended")):
            routing_recommendation = "data_formatting"

        sources_used = []
        if round1_results:
            sources_used.append("local_round1")
        if round2_results:
            sources_used.append("local_round2")
        if web_round1_results:
            sources_used.append("web_round1")
        if web_round2_results:
            sources_used.append("web_round2")

        shared_memory = state.get("shared_memory", {}) or {}
        shared_memory["primary_agent_selected"] = "research_agent"
        shared_memory["last_agent"] = "research_agent"

        return {
            "final_response": final_response,
            "research_complete": True,
            "sources_used": sources_used,
            "current_round": ResearchRound.FINAL_SYNTHESIS.value,
            "routing_recommendation": routing_recommendation,
            "formatting_recommendations": formatting_recommendations,
            "citations": [],
            "structured_images": structured_images,
            "shared_memory": shared_memory,
            "skill_config": skill_config,
        }

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        shared_memory = state.get("shared_memory", {}) or {}
        shared_memory["primary_agent_selected"] = "research_agent"
        shared_memory["last_agent"] = "research_agent"
        return {
            "final_response": f"Research completed but synthesis failed: {str(e)}",
            "research_complete": True,
            "error": str(e),
            "shared_memory": shared_memory,
            "skill_config": state.get("skill_config", {}),
        }


def combine_post_processed_results(
    formatted_output: str,
    format_type: Optional[str],
    visualization_data: Optional[str],
    chart_type: Optional[str],
    chart_output_format: Optional[str],
) -> str:
    """Combine formatted output and visualization into final response"""
    combined = formatted_output

    if visualization_data and chart_type:
        combined += "\n\n"
        if chart_output_format == "html":
            combined += "---\n*Interactive {0} chart rendered below.*\n".format(chart_type.title())
        elif chart_output_format == "base64_png":
            combined += f"## Chart: {chart_type.title()}\n\n"
            combined += f"![Chart: {chart_type}]({visualization_data})"
        else:
            combined += f"## Chart: {chart_type.title()}\n\n"
            combined += f"Chart generated successfully (format: {chart_output_format})"

    return combined


async def post_process_results_node(agent: "FullResearchAgent", state: ResearchState) -> Dict[str, Any]:
    """Post-process research results using formatting and/or visualization subgraphs in parallel"""
    try:
        formatting_recommendations = state.get("formatting_recommendations", {})
        query = state.get("query", "")
        final_response = state.get("final_response", "")
        structured_images = state.get("structured_images")

        workflow = await agent._get_workflow()
        checkpointer = workflow.checkpointer
        metadata = state.get("metadata", {})
        messages = state.get("messages", [])
        shared_memory = state.get("shared_memory", {})
        config = agent._get_checkpoint_config(metadata)

        needs_formatting = formatting_recommendations.get("table_recommended") or formatting_recommendations.get("timeline_recommended")
        needs_visualization = formatting_recommendations.get("chart_recommended")

        if needs_visualization:
            query_length = len(query.split())
            response_length = len(final_response)
            if query_length < 5 and response_length < 500:
                logger.info("Skipping visualization for simple query")
                needs_visualization = False

        logger.info(f"Post-processing: formatting={needs_formatting}, visualization={needs_visualization}")

        formatting_task = None
        visualization_task = None

        if needs_formatting:
            formatting_sg = agent._get_data_formatting_subgraph(checkpointer)
            formatting_query = f"Format the following research results into a well-organized structure:\n\n{final_response}"
            formatting_state = {
                "query": formatting_query,
                "messages": messages,
                "shared_memory": shared_memory,
                "user_id": state.get("user_id", shared_memory.get("user_id", "system")),
                "metadata": metadata,
            }
            formatting_task = formatting_sg.ainvoke(formatting_state, config)

        if needs_visualization:
            visualization_sg = agent._get_visualization_subgraph(checkpointer)
            research_data_for_viz = final_response
            if not research_data_for_viz or len(research_data_for_viz) < 200:
                for msg in reversed(messages[-10:]):
                    if hasattr(msg, "content"):
                        content = msg.content
                        is_assistant = (hasattr(msg, "type") and msg.type == "ai") or (hasattr(msg, "role") and msg.role == "assistant")
                        if is_assistant and len(content) > 200:
                            research_data_for_viz = content
                            logger.info(f"Using previous research data from conversation ({len(content)} chars) for visualization")
                            break

            visualization_state = {
                "query": query,
                "messages": messages,
                "research_data": research_data_for_viz,
                "shared_memory": shared_memory,
                "user_id": state.get("user_id", shared_memory.get("user_id", "system")),
                "metadata": metadata,
            }
            logger.info(f"Calling visualization subgraph with {len(research_data_for_viz)} chars of research data")
            visualization_task = visualization_sg.ainvoke(visualization_state, config)

        tasks = []
        task_indices = {}
        if formatting_task:
            task_indices["formatting"] = len(tasks)
            tasks.append(formatting_task)
        if visualization_task:
            task_indices["visualization"] = len(tasks)
            tasks.append(visualization_task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []

        formatting_result = None
        visualization_result = None

        if "formatting" in task_indices:
            idx = task_indices["formatting"]
            if idx < len(results) and not isinstance(results[idx], Exception):
                formatting_result = results[idx]

        if "visualization" in task_indices:
            idx = task_indices["visualization"]
            if idx < len(results) and not isinstance(results[idx], Exception):
                visualization_result = results[idx]

        formatted_output = final_response
        format_type = None
        if formatting_result:
            formatted_output = formatting_result.get("formatted_output", final_response)
            format_type = formatting_result.get("format_type", "structured_text")
            logger.info(f"Data formatting complete: {format_type}")

        visualization_data = None
        chart_type = None
        chart_output_format = None
        static_visualization_data = None
        static_format = None

        if visualization_result:
            viz_result = visualization_result.get("visualization_result", {})
            if viz_result.get("success"):
                visualization_data = viz_result.get("chart_data")
                chart_type = viz_result.get("chart_type")
                chart_output_format = viz_result.get("output_format")
                static_visualization_data = viz_result.get("static_visualization_data")
                static_format = viz_result.get("static_format")
                logger.info(f"Visualization complete: {chart_type}, format: {chart_output_format}, static: {bool(static_visualization_data)}")
            else:
                logger.warning(f"Visualization failed: {viz_result.get('error')}")

        combined_response = combine_post_processed_results(
            formatted_output, format_type, visualization_data, chart_type, chart_output_format
        )

        return {
            "final_response": combined_response,
            "format_type": format_type,
            "formatted": needs_formatting and formatting_result is not None,
            "visualization_results": {
                "success": visualization_result is not None and visualization_result.get("visualization_result", {}).get("success", False),
                "chart_type": chart_type,
                "chart_data": visualization_data,
                "output_format": chart_output_format,
                "static_visualization_data": static_visualization_data,
                "static_format": static_format,
            } if visualization_result else None,
            "static_visualization_data": static_visualization_data,
            "static_format": static_format,
            "structured_images": structured_images,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "skill_config": state.get("skill_config", {}),
        }

    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "final_response": state.get("final_response", ""),
            "formatted": False,
            "post_processing_error": str(e),
            "structured_images": state.get("structured_images"),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "skill_config": state.get("skill_config", {}),
        }
