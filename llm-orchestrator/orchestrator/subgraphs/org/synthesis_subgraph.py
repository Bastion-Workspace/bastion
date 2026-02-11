"""
Org Synthesis Subgraph

Builds context and generates LLM-based answers about org file content.
"""

import logging
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class OrgSynthesisState(TypedDict):
    """State for org synthesis subgraph"""
    # Inputs
    query: str
    query_type: str
    matched_headings: List[Dict[str, Any]]
    matched_todos: List[Dict[str, Any]]
    matched_projects: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    org_structure: List[Dict[str, Any]]
    todos: List[Dict[str, Any]]
    projects: List[Dict[str, Any]]
    all_tags: Dict[str, List[str]]
    content: str  # Full org file content
    filename: str
    # Intermediate
    context: str  # Built context string for LLM
    # Outputs
    synthesized_answer: str
    confidence: float
    sources: List[str]
    # Critical 5
    metadata: Dict[str, Any]
    user_id: str
    shared_memory: Dict[str, Any]
    messages: List[Any]


async def build_context_node(state: OrgSynthesisState) -> Dict[str, Any]:
    """Build context string from matched content"""
    try:
        query_type = state.get("query_type", "content")
        matched_headings = state.get("matched_headings", [])
        matched_todos = state.get("matched_todos", [])
        matched_projects = state.get("matched_projects", [])
        statistics = state.get("statistics", {})
        org_structure = state.get("org_structure", [])
        content = state.get("content", "")
        filename = state.get("filename", "")
        
        context_parts = []
        
        # Add full org file content (most reliable approach)
        if content:
            context_parts.append("=== CONTEXT FROM ORG FILE ===\n")
            context_parts.append(f"File: {filename}\n\n")
            # Include full content - LLM can focus on relevant parts
            context_parts.append(content)
            context_parts.append("\n\n")
            logger.info(f"Built context with {len(content)} chars from {filename}")
        else:
            logger.warning(f"No content available for context building (filename: {filename})")
        
        # Add statistics summary (helpful context)
        if statistics:
            context_parts.append("=== ORG FILE STATISTICS ===\n")
            context_parts.append(f"Total headings: {statistics.get('total_headings', 0)}\n")
            context_parts.append(f"Total TODOs: {statistics.get('total_todos', 0)}\n")
            context_parts.append(f"Total projects: {statistics.get('total_projects', 0)}\n")
            if statistics.get('completion_rate') is not None:
                context_parts.append(f"Completion rate: {statistics.get('completion_rate')}%\n")
            context_parts.append("\n")
        
        # Add query-matched headings (most relevant)
        if matched_headings:
            context_parts.append("=== RELEVANT HEADINGS (Query-Matched) ===\n")
            matched_heading_texts = {h.get("heading", "").strip() for h in matched_headings}
            
            for heading_info in matched_headings:
                heading = heading_info.get("heading", "")
                level = heading_info.get("level", 1)
                tags = heading_info.get("tags", [])
                todo_state = heading_info.get("todo_state")
                parent_path = heading_info.get("parent_path", [])
                subtree_content = heading_info.get("subtree_content", "")
                
                # If subtree_content is missing, look it up from org_structure
                if not subtree_content:
                    for org_heading in org_structure:
                        if org_heading.get("heading", "").strip() == heading.strip():
                            subtree_content = org_heading.get("subtree_content", "")
                            break
                
                indent = "  " * (level - 1)
                todo_str = f"{todo_state} " if todo_state else ""
                tags_str = f" :{':'.join(tags)}:" if tags else ""
                path_str = f" ({' > '.join(parent_path)})" if parent_path else ""
                
                context_parts.append(f"{indent}* {todo_str}{heading}{tags_str}{path_str}")
                
                # Include subtree content (all child headings and their content)
                if subtree_content:
                    # Limit to first 2000 chars to avoid token limits
                    content_preview = subtree_content[:2000]
                    context_parts.append(f"\n  Content:\n{content_preview}\n")
                else:
                    logger.warning(f"No subtree_content found for matched heading: {heading}")
            context_parts.append("\n")
        
        # Add matched TODOs if query is about TODOs
        if query_type == "todo" and matched_todos:
            context_parts.append("=== RELEVANT TODOs ===\n")
            for todo in matched_todos[:10]:  # Limit to top 10
                heading = todo.get("heading", "")
                todo_state = todo.get("todo_state", "")
                tags = todo.get("tags", [])
                parent_path = todo.get("parent_path", [])
                
                tags_str = f" :{':'.join(tags)}:" if tags else ""
                path_str = f" ({' > '.join(parent_path)})" if parent_path else ""
                
                context_parts.append(f"* {todo_state} {heading}{tags_str}{path_str}\n")
            context_parts.append("\n")
        
        # Add matched projects if query is about projects
        if query_type == "project" and matched_projects:
            context_parts.append("=== RELEVANT PROJECTS ===\n")
            for project in matched_projects[:10]:  # Limit to top 10
                heading = project.get("heading", "")
                tags = project.get("tags", [])
                parent_path = project.get("parent_path", [])
                subtree_content = project.get("subtree_content", "")[:300]
                
                tags_str = f" :{':'.join(tags)}:" if tags else ""
                path_str = f" ({' > '.join(parent_path)})" if parent_path else ""
                
                context_parts.append(f"* {heading}{tags_str}{path_str}")
                if subtree_content:
                    context_parts.append(f"\n  {subtree_content}\n")
            context_parts.append("\n")
        
        context = "".join(context_parts)
        
        logger.info(f"build_context_node returning context with {len(context)} chars")
        
        return {
            "context": context,
            # ✅ CRITICAL 5 preserved
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # Preserve all state
            "query_type": query_type,
            "matched_headings": matched_headings,
            "matched_todos": matched_todos,
            "matched_projects": matched_projects,
            "statistics": statistics,
            "org_structure": org_structure,
            "todos": state.get("todos", []),
            "projects": state.get("projects", []),
            "all_tags": state.get("all_tags", {}),
            "content": content,
            "filename": filename,
            "synthesized_answer": state.get("synthesized_answer", ""),
            "confidence": state.get("confidence", 0.0),
            "sources": state.get("sources", [])
        }
        
    except Exception as e:
        logger.error(f"Failed to build context: {e}")
        return {
            "context": "",
            "error": str(e),
            # ✅ CRITICAL 5 preserved even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "query_type": state.get("query_type", "content"),
            "matched_headings": state.get("matched_headings", []),
            "matched_todos": state.get("matched_todos", []),
            "matched_projects": state.get("matched_projects", []),
            "statistics": state.get("statistics", {}),
            "org_structure": state.get("org_structure", []),
            "todos": state.get("todos", []),
            "projects": state.get("projects", []),
            "all_tags": state.get("all_tags", {}),
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }


async def generate_answer_node(state: OrgSynthesisState) -> Dict[str, Any]:
    """Use LLM to generate answer from matched content"""
    try:
        query = state.get("query", "")
        query_type = state.get("query_type", "content")
        context = state.get("context", "")
        matched_headings = state.get("matched_headings", [])
        matched_todos = state.get("matched_todos", [])
        matched_projects = state.get("matched_projects", [])
        
        # Get LLM
        from orchestrator.agents.base_agent import BaseAgent
        temp_agent = BaseAgent("temp_org_synthesis")
        llm = temp_agent._get_llm(temperature=0.7, state=state)
        
        # Build system prompt based on query type
        if query_type == "todo":
            system_prompt = """You are an Org Mode TODO assistant. Answer questions about TODO items, their status, and organization.

When answering:
- Be specific about TODO states (TODO, DONE, WAITING, etc.)
- Reference parent headings for context
- Mention tags when relevant
- Provide clear, actionable information"""
        elif query_type == "project":
            system_prompt = """You are an Org Mode project assistant. Answer questions about projects, their status, and organization.

When answering:
- Reference project headings and their content
- Mention tags and organization
- Provide context about project structure"""
        elif query_type == "tag":
            system_prompt = """You are an Org Mode tag assistant. Answer questions about tags and tagged entries.

When answering:
- List entries with specific tags
- Explain tag organization
- Provide context about tagged content"""
        else:
            system_prompt = """You are an Org Mode content assistant. Answer questions about the content of an Org file.

When answering:
- Reference specific headings and their content
- Use the file structure to provide context
- Be specific and cite relevant sections
- Synthesize information from multiple headings when relevant"""
        
        user_prompt = f"""USER QUERY: {query}

CONTEXT FROM ORG FILE:
{context}

Please answer the user's query based on the provided context. Be specific and reference relevant headings, TODOs, or projects when relevant."""

        # Debug logging
        logger.info(f"Context length for LLM: {len(context)} chars")
        logger.info(f"First 200 chars of context: {context[:200]}")
        logger.info(f"User prompt length: {len(user_prompt)} chars")

        # Get datetime context
        datetime_context = temp_agent._get_datetime_context(state)
        
        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=datetime_context),
            HumanMessage(content=user_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Build sources list
        sources = []
        for heading_info in matched_headings[:5]:  # Top 5 sources
            heading = heading_info.get("heading", "")
            if heading:
                sources.append(heading)
        
        logger.info(f"Generated answer ({len(answer)} chars) with {len(sources)} sources")
        
        return {
            "synthesized_answer": answer,
            "sources": sources,
            # ✅ CRITICAL 5 preserved
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": query,
            # Preserve all state
            "query_type": query_type,
            "matched_headings": matched_headings,
            "matched_todos": matched_todos,
            "matched_projects": matched_projects,
            "statistics": state.get("statistics", {}),
            "org_structure": state.get("org_structure", []),
            "todos": state.get("todos", []),
            "projects": state.get("projects", []),
            "all_tags": state.get("all_tags", {}),
            "context": context,
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to generate answer: {e}")
        return {
            "synthesized_answer": f"I encountered an error while generating an answer: {str(e)}",
            "sources": [],
            "error": str(e),
            # ✅ CRITICAL 5 preserved even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "query_type": state.get("query_type", "content"),
            "matched_headings": state.get("matched_headings", []),
            "matched_todos": state.get("matched_todos", []),
            "matched_projects": state.get("matched_projects", []),
            "statistics": state.get("statistics", {}),
            "org_structure": state.get("org_structure", []),
            "todos": state.get("todos", []),
            "projects": state.get("projects", []),
            "all_tags": state.get("all_tags", {}),
            "context": state.get("context", ""),
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }


async def assess_confidence_node(state: OrgSynthesisState) -> Dict[str, Any]:
    """Assess confidence in the generated answer"""
    try:
        matched_headings = state.get("matched_headings", [])
        matched_todos = state.get("matched_todos", [])
        matched_projects = state.get("matched_projects", [])
        synthesized_answer = state.get("synthesized_answer", "")
        
        # Simple confidence assessment based on matched content
        confidence = 0.5  # Base confidence
        
        if matched_headings:
            confidence += 0.2
        if matched_todos or matched_projects:
            confidence += 0.2
        if len(synthesized_answer) > 100:  # Substantial answer
            confidence += 0.1
        
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        logger.info(f"Assessed confidence: {confidence:.2f}")
        
        return {
            "confidence": confidence,
            # ✅ CRITICAL 5 preserved
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            # Preserve all state
            "query_type": state.get("query_type", "content"),
            "matched_headings": matched_headings,
            "matched_todos": matched_todos,
            "matched_projects": matched_projects,
            "statistics": state.get("statistics", {}),
            "org_structure": state.get("org_structure", []),
            "todos": state.get("todos", []),
            "projects": state.get("projects", []),
            "all_tags": state.get("all_tags", {}),
            "synthesized_answer": synthesized_answer,
            "sources": state.get("sources", []),
            "context": state.get("context", ""),
            "content": state.get("content", ""),
            "filename": state.get("filename", "")
        }
        
    except Exception as e:
        logger.error(f"Failed to assess confidence: {e}")
        return {
            "confidence": 0.5,
            "error": str(e),
            # ✅ CRITICAL 5 preserved even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "query_type": state.get("query_type", "content"),
            "matched_headings": state.get("matched_headings", []),
            "matched_todos": state.get("matched_todos", []),
            "matched_projects": state.get("matched_projects", []),
            "statistics": state.get("statistics", {}),
            "org_structure": state.get("org_structure", []),
            "todos": state.get("todos", []),
            "projects": state.get("projects", []),
            "all_tags": state.get("all_tags", {}),
            "synthesized_answer": state.get("synthesized_answer", ""),
            "sources": state.get("sources", []),
            "context": state.get("context", "")
        }


def build_org_synthesis_subgraph(checkpointer) -> StateGraph:
    """Build subgraph for synthesizing answers"""
    subgraph = StateGraph(OrgSynthesisState)
    
    subgraph.add_node("build_context", build_context_node)
    subgraph.add_node("generate_answer", generate_answer_node)
    subgraph.add_node("assess_confidence", assess_confidence_node)
    
    subgraph.set_entry_point("build_context")
    subgraph.add_edge("build_context", "generate_answer")
    subgraph.add_edge("generate_answer", "assess_confidence")
    subgraph.add_edge("assess_confidence", END)
    
    return subgraph.compile(checkpointer=checkpointer)
