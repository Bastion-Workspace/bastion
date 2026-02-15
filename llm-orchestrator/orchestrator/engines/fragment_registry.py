"""
Fragment Registry and Invoker - expose subgraphs as plan step targets.

Fragments are reusable subgraph workflows (document retrieval, web research,
visualization, data formatting) that can be invoked from compound plans
alongside skills.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FragmentDef(BaseModel):
    """Definition of a subgraph fragment that can be invoked as a plan step."""

    name: str
    description: str = Field(description="For LLM planner prompt")
    input_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="context_bridge_key -> subgraph_state_key",
    )
    output_key: str = Field(description="Subgraph state key for main text result")
    needs_checkpointer: bool = False


FRAGMENT_REGISTRY: Dict[str, FragmentDef] = {
    "document_retrieval": FragmentDef(
        name="document_retrieval",
        description="Search and retrieve documents from local knowledge base with intelligent chunking",
        input_mapping={"query": "query", "user_id": "user_id"},
        output_key="formatted_context",
        needs_checkpointer=False,
    ),
    "web_research": FragmentDef(
        name="web_research",
        description="Search the web and crawl top results for current information",
        input_mapping={"query": "query"},
        output_key="search_results",
        needs_checkpointer=False,
    ),
    "visualization": FragmentDef(
        name="visualization",
        description="Generate charts and data visualizations from research data",
        input_mapping={"query": "query"},
        output_key="static_visualization_data",
        needs_checkpointer=False,
    ),
    "data_formatting": FragmentDef(
        name="data_formatting",
        description="Transform data into structured formats: tables, lists, timelines, comparisons",
        input_mapping={"query": "query"},
        output_key="formatted_output",
        needs_checkpointer=False,
    ),
    "image_search": FragmentDef(
        name="image_search",
        description="Search local image/comic collection by visual content, series, author, or date",
        input_mapping={"query": "query", "user_id": "user_id"},
        output_key="image_search_results",
        needs_checkpointer=False,
    ),
    "diagramming": FragmentDef(
        name="diagramming",
        description="Generate Mermaid diagrams, ASCII circuit diagrams, or pin tables from a description",
        input_mapping={"query": "query"},
        output_key="diagram_content",
        needs_checkpointer=False,
    ),
    "full_document_analysis": FragmentDef(
        name="full_document_analysis",
        description="Deep-read up to 2 full documents and synthesize answers to specific questions",
        input_mapping={"query": "original_query", "document_ids": "document_ids"},
        output_key="synthesis",
        needs_checkpointer=False,
    ),
    "gap_analysis": FragmentDef(
        name="gap_analysis",
        description="Analyze research results for information gaps, missing entities, and suggest follow-up queries",
        input_mapping={"query": "query", "results": "results"},
        output_key="gap_analysis",
        needs_checkpointer=False,
    ),
    "assessment": FragmentDef(
        name="assessment",
        description="Assess whether research results are sufficient to answer a query, with confidence score",
        input_mapping={"query": "query", "results": "results"},
        output_key="assessment",
        needs_checkpointer=False,
    ),
    "fact_verification": FragmentDef(
        name="fact_verification",
        description="Cross-reference and verify factual claims from research with web sources",
        input_mapping={"query": "query", "research_findings": "research_findings"},
        output_key="consensus_findings",
        needs_checkpointer=False,
    ),
    "entity_relationship": FragmentDef(
        name="entity_relationship",
        description="Discover entity relationships via knowledge graph (Neo4j) from document corpus",
        input_mapping={"query": "query", "user_id": "user_id"},
        output_key="kg_formatted_results",
        needs_checkpointer=False,
    ),
    "knowledge_document": FragmentDef(
        name="knowledge_document",
        description="Assemble verified research into a structured markdown knowledge document with citations",
        input_mapping={
            "query": "query",
            "verified_claims": "verified_claims",
            "contradictions": "contradictions",
            "uncertainties": "uncertainties",
        },
        output_key="markdown_content",
        needs_checkpointer=False,
    ),
}


def get_fragment(name: str) -> Optional[FragmentDef]:
    """Return the fragment definition for the given name, or None."""
    return FRAGMENT_REGISTRY.get(name)


def get_all_fragments() -> List[FragmentDef]:
    """Return all registered fragment definitions."""
    return list(FRAGMENT_REGISTRY.values())


async def invoke_fragment(
    fragment_name: str,
    query: str,
    metadata: Dict[str, Any],
    messages: List[Any],
    prior_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Invoke a subgraph fragment with standard state interface.

    Builds the subgraph, maps context into its state, runs ainvoke, and
    returns a result dict compatible with ContextBridge.store_result
    (response, agent_type).
    """
    frag = get_fragment(fragment_name)
    if not frag:
        logger.warning("Unknown fragment: %s", fragment_name)
        return {"response": "", "agent_type": fragment_name}

    checkpointer = None
    if frag.needs_checkpointer:
        from orchestrator.checkpointer import get_async_postgres_saver
        checkpointer = await get_async_postgres_saver()

    from orchestrator.subgraphs.intelligent_document_retrieval_subgraph import (
        build_intelligent_document_retrieval_subgraph,
    )
    from orchestrator.subgraphs.web_research_subgraph import build_web_research_subgraph
    from orchestrator.subgraphs.visualization_subgraph import build_visualization_subgraph
    from orchestrator.subgraphs.data_formatting_subgraph import build_data_formatting_subgraph
    from orchestrator.subgraphs.image_search_subgraph import build_image_search_subgraph
    from orchestrator.subgraphs.diagramming_subgraph import build_diagramming_subgraph
    from orchestrator.subgraphs.full_document_analysis_subgraph import (
        build_full_document_analysis_subgraph,
    )
    from orchestrator.subgraphs.gap_analysis_subgraph import build_gap_analysis_subgraph
    from orchestrator.subgraphs.assessment_subgraph import build_assessment_subgraph
    from orchestrator.subgraphs.fact_verification_subgraph import (
        build_fact_verification_subgraph,
    )
    from orchestrator.subgraphs.entity_relationship_subgraph import (
        build_entity_relationship_subgraph,
    )
    from orchestrator.subgraphs.knowledge_document_subgraph import (
        build_knowledge_document_subgraph,
    )

    build_fns = {
        "document_retrieval": lambda cp: build_intelligent_document_retrieval_subgraph(),
        "web_research": lambda cp: build_web_research_subgraph(cp),
        "visualization": lambda cp: build_visualization_subgraph(cp),
        "data_formatting": lambda cp: build_data_formatting_subgraph(cp),
        "image_search": lambda cp: build_image_search_subgraph(cp),
        "diagramming": lambda cp: build_diagramming_subgraph(cp),
        "full_document_analysis": lambda cp: build_full_document_analysis_subgraph(cp),
        "gap_analysis": lambda cp: build_gap_analysis_subgraph(cp),
        "assessment": lambda cp: build_assessment_subgraph(cp),
        "fact_verification": lambda cp: build_fact_verification_subgraph(cp),
        "entity_relationship": lambda cp: build_entity_relationship_subgraph(cp),
        "knowledge_document": lambda cp: build_knowledge_document_subgraph(cp),
    }
    build_fn = build_fns.get(fragment_name)
    if not build_fn:
        return {"response": "", "agent_type": fragment_name}

    try:
        compiled = build_fn(checkpointer)
    except Exception as e:
        logger.exception("Failed to build fragment %s: %s", fragment_name, e)
        return {"response": f"Fragment build failed: {e}", "agent_type": fragment_name}

    user_id = metadata.get("user_id", "system")
    shared_memory = dict(metadata.get("shared_memory") or {})
    shared_memory.update(prior_context)

    base_state: Dict[str, Any] = {
        "query": query,
        "metadata": metadata,
        "user_id": user_id,
        "shared_memory": shared_memory,
        "messages": messages,
    }

    for ctx_key, state_key in frag.input_mapping.items():
        if ctx_key in prior_context:
            val = prior_context[ctx_key]
            if state_key == "research_findings" and isinstance(val, str):
                base_state[state_key] = {"combined_results": val}
            else:
                base_state[state_key] = val
        elif ctx_key == "query":
            base_state[state_key] = query
        elif ctx_key == "user_id":
            base_state[state_key] = user_id

    if fragment_name == "document_retrieval":
        base_state.setdefault("retrieval_mode", "fast")
        base_state.setdefault("max_results", 5)
        base_state.setdefault("small_doc_threshold", 5000)
        base_state.setdefault("retrieved_documents", [])
        base_state.setdefault("formatted_context", "")
        base_state.setdefault("retrieval_metadata", {})
        base_state.setdefault("skip_sufficiency_check", False)
        base_state.setdefault("error", "")
    elif fragment_name == "image_search":
        base_state.setdefault("limit", 10)
    elif fragment_name == "full_document_analysis":
        base_state.setdefault("document_ids", base_state.get("document_ids") or [])
        base_state.setdefault("analysis_queries", prior_context.get("analysis_queries") or [query])
        base_state.setdefault("chunk_context", [])

    try:
        result = await compiled.ainvoke(base_state)
    except Exception as e:
        logger.exception("Fragment %s invoke failed: %s", fragment_name, e)
        return {"response": f"Fragment failed: {e}", "agent_type": fragment_name}

    raw = result.get(frag.output_key, "")
    if raw is None:
        raw = ""
    if isinstance(raw, dict):
        text = json.dumps(raw, indent=2, default=str)
    elif isinstance(raw, str):
        text = raw
    else:
        text = str(raw)

    response_dict: Dict[str, Any] = {"response": text, "agent_type": fragment_name}

    # Preserve image data produced by subgraph fragments (used by frontend rendering and downstream plan steps).
    structured_images = result.get("structured_images")
    if structured_images:
        response_dict["structured_images"] = structured_images
    image_search_results = result.get("image_search_results")
    if image_search_results:
        response_dict["image_search_results"] = image_search_results

    return response_dict
