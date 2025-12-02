"""
Information Analysis Tools - LLM-powered analysis for understanding information needs and generating targeted queries

These tools help agents understand what information is needed and generate project-aware search queries.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


async def analyze_information_needs_tool(
    query: str,
    query_type: str,
    project_context: Dict[str, Any],
    context_keys: Optional[List[str]] = None,
    domain_name: str = "project",
    user_id: str = "system",
    llm_model: Optional[str] = None,
    get_llm_func: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Analyze what information is needed to answer a query in project context.
    
    Uses LLM to identify information gaps, relevant entities, detail level needed,
    and whether content is for new documentation or updating existing.
    
    **Use Cases:**
    - Research Agent: Identify what information gaps exist before searching
    - Content Analysis Agent: Determine what analysis is needed
    - Fiction Editing Agent: Identify what story elements need development
    - Any Planning Agent: Use for gap analysis
    
    **Workflow:**
    1. Extract project context (components, protocols, characters, plot points, etc.)
    2. Use LLM to analyze what information is needed
    3. Identify gaps, relevant entities, detail level
    4. Determine if content is new or updating existing
    
    Args:
        query: User query to analyze
        query_type: Type of query (e.g., "circuit_design", "character_development", "research")
        project_context: Dict of project context organized by category
            Format: {"components": [...], "protocols": [...], "characters": [...], etc.}
        context_keys: Optional list of context keys to include in analysis
            If None, uses all keys from project_context
        domain_name: Domain name for prompt customization (default: "project")
        user_id: User ID (for logging)
        llm_model: Optional LLM model to use
        get_llm_func: Optional function to get LLM instance (for agents that have this)
        
    Returns:
        Dict with:
        - information_gaps: List of identified information gaps
        - relevant_entities: List of relevant entities from project context
        - content_type: "new" | "update" | "both"
        - detail_level: "overview" | "detailed" | "implementation"
        - related_sections: List of related sections that might need updates
        - search_focus: What to focus search on
        
    Example:
        ```python
        from orchestrator.tools.information_analysis_tools import analyze_information_needs_tool
        
        # Electronics project context
        project_context = {
            "components": ["ESP32", "keyboard matrix"],
            "protocols": ["I2C", "SPI"],
            "architecture": ["microcontroller-based"]
        }
        
        result = await analyze_information_needs_tool(
            query="How to implement keyboard scanning",
            query_type="circuit_design",
            project_context=project_context,
            context_keys=["components", "protocols"],
            domain_name="electronics"
        )
        
        gaps = result["information_gaps"]
        # ["keyboard matrix circuit design", "ESP32 GPIO configuration"]
        ```
    """
    try:
        # Extract context values
        if context_keys is None:
            context_keys = list(project_context.keys())
        
        context_summary = {}
        for key in context_keys:
            values = project_context.get(key, [])
            if isinstance(values, list):
                context_summary[key] = values[:10]  # Limit to top 10
            else:
                context_summary[key] = values
        
        # Build context string for prompt
        context_parts = []
        for key, values in context_summary.items():
            if values:
                if isinstance(values, list):
                    context_parts.append(f"- {key.capitalize()}: {', '.join(str(v) for v in values)}")
                else:
                    context_parts.append(f"- {key.capitalize()}: {values}")
        
        context_string = '\n'.join(context_parts) if context_parts else "None specified"
        
        # Get LLM instance
        if not get_llm_func:
            raise ValueError("get_llm_func is required for LLM-based analysis. Agents should pass their _get_llm method (e.g., lambda: self._get_llm(temperature=0.1, model=model)).")
        
        # Call with appropriate parameters
        # get_llm_func should be a callable that returns an LLM instance
        # Agents typically pass: lambda: self._get_llm(temperature=0.1, model=model, state=state)
        if callable(get_llm_func):
            llm = get_llm_func()
        else:
            raise ValueError("get_llm_func must be a callable that returns an LLM instance")
        
        # Build prompt (domain-agnostic)
        prompt = f"""Analyze what information is needed to answer this {domain_name} query in the context of the project.

**QUERY**: {query}

**QUERY TYPE**: {query_type}

**PROJECT CONTEXT**:
{context_string}

**TASK**: Analyze what information is needed:
1. What specific information gaps exist?
2. What entities from the project context are relevant?
3. Is this for new content or updating existing documentation?
4. What level of detail is needed (overview, detailed specs, implementation)?
5. What related sections might need updates?

Return ONLY valid JSON:
{{
  "information_gaps": ["gap1", "gap2"],
  "relevant_entities": ["entity1", "entity2"],
  "content_type": "new|update|both",
  "detail_level": "overview|detailed|implementation",
  "related_sections": ["section1", "section2"],
  "search_focus": "What to focus search on"
}}"""
        
        try:
            structured_llm = llm.with_structured_output({
                "title": "InformationNeedsAnalysis",
                "description": "Analysis of information gaps and search requirements for a query",
                "type": "object",
                "properties": {
                    "information_gaps": {"type": "array", "items": {"type": "string"}},
                    "relevant_entities": {"type": "array", "items": {"type": "string"}},
                    "content_type": {"type": "string"},
                    "detail_level": {"type": "string"},
                    "related_sections": {"type": "array", "items": {"type": "string"}},
                    "search_focus": {"type": "string"}
                }
            })
            result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
            result_dict = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
        except Exception as e:
            logger.warning(f"⚠️ Structured output failed, trying fallback: {e}")
            response = await llm.ainvoke([{"role": "user", "content": prompt}])
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract JSON
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    result_dict = json.loads(json_match.group(0))
                except:
                    result_dict = {}
            else:
                result_dict = {}
        
        logger.info(f"🔍 Information needs analyzed: {len(result_dict.get('information_gaps', []))} gaps identified")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"❌ Information needs analysis failed: {e}")
        # Return safe defaults
        return {
            "information_gaps": [],
            "relevant_entities": [],
            "content_type": "new",
            "detail_level": "detailed",
            "related_sections": [],
            "search_focus": query
        }


async def generate_project_aware_queries_tool(
    query: str,
    query_type: str,
    information_needs: Dict[str, Any],
    project_context: Dict[str, Any],
    domain_examples: Optional[List[str]] = None,
    user_id: str = "system",
    num_queries: int = 5,
    llm_model: Optional[str] = None,
    get_llm_func: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Generate project-aware search queries using project context and information needs.
    
    Uses LLM to create targeted queries that incorporate project context (components,
    protocols, characters, plot points, etc.) and focus on identified information gaps.
    
    **Use Cases:**
    - Research Agent: Generate project-aware research queries
    - Content Analysis Agent: Create targeted analysis queries
    - Fiction Editing Agent: Generate queries using story context
    - Any Agent with Project Context: Leverage project context for better searches
    
    **Workflow:**
    1. Extract relevant entities from project context
    2. Use information needs to identify gaps
    3. Generate 3-5 targeted queries that incorporate context
    4. Prioritize queries by specificity
    
    **Relationship to Query Expansion:**
    - Query expansion (`expand_query_tool`): Creates semantic variations of a query
    - Project-aware generation: Creates context-specific, targeted queries
    - Can be used together: Expand first, then make project-aware
    
    Args:
        query: Original query
        query_type: Type of query (e.g., "circuit_design", "character_development")
        information_needs: Result from `analyze_information_needs_tool`
        project_context: Dict of project context organized by category
        domain_examples: Optional list of domain-specific examples for prompt
        user_id: User ID (for logging)
        num_queries: Number of queries to generate (default: 5)
        llm_model: Optional LLM model to use
        get_llm_func: Optional function to get LLM instance
        
    Returns:
        Dict with:
        - search_queries: List of query dicts with "query", "priority", "focus"
        - query_expansion_used: Whether query expansion was used (always False for this tool)
        
    Example:
        ```python
        from orchestrator.tools.information_analysis_tools import (
            analyze_information_needs_tool,
            generate_project_aware_queries_tool
        )
        
        # First analyze needs
        information_needs = await analyze_information_needs_tool(
            query="keyboard scanning",
            query_type="circuit_design",
            project_context={"components": ["ESP32"], "protocols": ["I2C"]},
            domain_name="electronics"
        )
        
        # Then generate targeted queries
        result = await generate_project_aware_queries_tool(
            query="keyboard scanning",
            query_type="circuit_design",
            information_needs=information_needs,
            project_context={"components": ["ESP32"], "protocols": ["I2C"]},
            domain_examples=[
                "ESP32 keyboard matrix scanning circuit design",
                "I2C sensor communication protocol implementation"
            ],
            domain_name="electronics"
        )
        
        queries = result["search_queries"]
        # [
        #   {"query": "ESP32 keyboard matrix scanning", "priority": 1, "focus": "Project-specific implementation"},
        #   {"query": "keyboard scanning circuit design I2C", "priority": 2, "focus": "Protocol integration"}
        # ]
        ```
    """
    try:
        # Extract relevant entities from information needs
        relevant_entities = information_needs.get("relevant_entities", [])
        information_gaps = information_needs.get("information_gaps", [])
        search_focus = information_needs.get("search_focus", query)
        
        # Extract project entities from context
        project_entities = []
        for key, values in project_context.items():
            if isinstance(values, list):
                project_entities.extend([str(v) for v in values[:5]])  # Top 5 per category
            else:
                project_entities.append(str(values))
        
        # Combine relevant entities
        all_entities = list(set(relevant_entities + project_entities[:10]))  # Deduplicate
        
        # Build examples string
        examples_string = ""
        if domain_examples:
            examples_string = "\n**EXAMPLES**:\n" + "\n".join(f"- {ex}" for ex in domain_examples[:3])
        
        # Get LLM instance
        if not get_llm_func:
            raise ValueError("get_llm_func is required for LLM-based query generation. Agents should pass their _get_llm method (e.g., lambda: self._get_llm(temperature=0.2, model=model)).")
        
        # Call with appropriate parameters
        # get_llm_func should be a callable that returns an LLM instance
        if callable(get_llm_func):
            llm = get_llm_func()
        else:
            raise ValueError("get_llm_func must be a callable that returns an LLM instance")
        
        # Build prompt
        prompt = f"""Generate targeted search queries for finding relevant information to answer this query.

**ORIGINAL QUERY**: {query}

**QUERY TYPE**: {query_type}

**PROJECT CONTEXT**:
- Relevant Entities: {', '.join(all_entities[:10]) if all_entities else 'None'}

**INFORMATION GAPS**:
{chr(10).join(f"- {gap}" for gap in information_gaps[:5]) if information_gaps else "None identified"}

**SEARCH FOCUS**: {search_focus}

**TASK**: Generate {num_queries} targeted search queries that:
1. Include relevant project entities when applicable
2. Focus on the specific information gaps
3. Are optimized for semantic search (natural language, not keywords)
4. Vary in specificity (some broad, some specific)
5. Prioritize queries that will find segments within project documents

{examples_string}

Return ONLY valid JSON:
{{
  "search_queries": [
    {{"query": "query1", "priority": 1, "focus": "what this query targets"}},
    {{"query": "query2", "priority": 2, "focus": "what this query targets"}}
  ],
  "query_expansion_used": false
}}"""
        
        try:
            structured_llm = llm.with_structured_output({
                "title": "ProjectAwareSearchQueries",
                "description": "Generated search queries optimized for project context",
                "type": "object",
                "properties": {
                    "search_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "priority": {"type": "integer"},
                                "focus": {"type": "string"}
                            }
                        }
                    },
                    "query_expansion_used": {"type": "boolean"}
                }
            })
            result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
            result_dict = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
        except Exception as e:
            logger.warning(f"⚠️ Structured output failed, trying fallback: {e}")
            # Fallback: use query expansion tool
            from orchestrator.tools.enhancement_tools import expand_query_tool
            expansion_result = await expand_query_tool(query, num_variations=3)
            expanded_queries = expansion_result.get('expanded_queries', [query])
            
            # Add project context to expanded queries
            search_queries = []
            for i, eq in enumerate(expanded_queries[:num_queries], 1):
                if all_entities:
                    # Add first entity to query
                    enhanced_query = f"{eq} {all_entities[0]}"
                else:
                    enhanced_query = eq
                search_queries.append({
                    "query": enhanced_query,
                    "priority": i,
                    "focus": "General search with project context"
                })
            
            result_dict = {
                "search_queries": search_queries,
                "query_expansion_used": True
            }
        
        # Sort by priority
        search_queries = result_dict.get("search_queries", [])
        search_queries.sort(key=lambda x: x.get("priority", 999))
        
        logger.info(f"🔍 Generated {len(search_queries)} project-aware search queries")
        for i, sq in enumerate(search_queries[:3], 1):
            logger.info(f"  {i}. [{sq.get('priority', '?')}] {sq.get('query', '')[:80]}")
        
        return {
            "search_queries": search_queries,
            "query_expansion_used": result_dict.get("query_expansion_used", False)
        }
        
    except Exception as e:
        logger.error(f"❌ Project-aware query generation failed: {e}")
        # Fallback to simple query
        return {
            "search_queries": [{"query": query, "priority": 1, "focus": "General search"}],
            "query_expansion_used": False
        }


# Tool registry
INFORMATION_ANALYSIS_TOOLS = {
    'analyze_information_needs': analyze_information_needs_tool,
    'generate_project_aware_queries': generate_project_aware_queries_tool
}

