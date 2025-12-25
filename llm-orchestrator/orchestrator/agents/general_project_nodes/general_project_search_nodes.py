"""
General Project Agent - Search Nodes Module
Handles information needs analysis, query generation, content search, and quality assessment
"""

import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GeneralProjectSearchNodes:
    """Search-related nodes for general project agent"""
    
    def __init__(self, agent):
        """
        Initialize with reference to main agent for access to helper methods
        
        Args:
            agent: GeneralProjectAgent instance (for _get_llm, _get_fast_model, etc.)
        """
        self.agent = agent
    
    async def analyze_information_needs_node(self, state) -> Dict[str, Any]:
        """
        Analyze what information the project needs to answer the query.
        
        Uses project context to identify:
        - What information gaps exist
        - What specifications/design/tasks are relevant
        - What level of detail is needed
        - Whether information is for new content or updating existing
        """
        try:
            query = state.get("query", "")
            query_type = state.get("query_type", "general")
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            referenced_context = state.get("referenced_context", {})
            
            # Extract project context
            shared_memory = metadata.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor", {})
            frontmatter = active_editor.get("frontmatter", {})
            
            # Build project context summary for general projects
            project_specifications = []
            project_design = []
            project_tasks = []
            
            if frontmatter:
                project_specifications = frontmatter.get("specifications", [])
                project_design = frontmatter.get("design", [])
                project_tasks = frontmatter.get("tasks", [])
            
            # Build project context dict for tool
            project_context = {
                "specifications": project_specifications,
                "design": project_design,
                "tasks": project_tasks
            }
            
            # Extract from referenced files
            if referenced_context:
                for category, files in referenced_context.items():
                    if category in ["specifications", "design", "tasks"] and isinstance(files, list):
                        for file_doc in files:
                            if isinstance(file_doc, dict):
                                filename = file_doc.get("filename", "")
                                if filename and filename not in project_context.get(category, []):
                                    project_context.setdefault(category, []).append(filename)
            
            # Use universal information needs analysis tool
            from orchestrator.tools.information_analysis_tools import analyze_information_needs_tool
            
            fast_model = self.agent._get_fast_model(state)
            get_llm_func = lambda: self.agent._get_llm(temperature=0.1, model=fast_model, state=state)
            
            result_dict = await analyze_information_needs_tool(
                query=query,
                query_type=query_type,
                project_context=project_context,
                context_keys=["specifications", "design", "tasks"],
                domain_name="general_project",
                user_id=user_id,
                llm_model=fast_model,
                get_llm_func=get_llm_func
            )
            
            # Map generic "relevant_entities" to general project fields
            relevant_entities = result_dict.get("relevant_entities", [])
            relevant_specifications = []
            relevant_design = []
            relevant_tasks = []
            
            for entity in relevant_entities:
                entity_lower = entity.lower()
                if any(term in entity_lower for term in ["spec", "requirement", "specification"]):
                    relevant_specifications.append(entity)
                elif any(term in entity_lower for term in ["design", "architecture", "approach"]):
                    relevant_design.append(entity)
                elif any(term in entity_lower for term in ["task", "todo", "checklist", "milestone"]):
                    relevant_tasks.append(entity)
                else:
                    # Default to specifications for general entities
                    relevant_specifications.append(entity)
            
            # Merge with existing project context
            result_dict["relevant_specifications"] = list(set(relevant_specifications + project_context.get("specifications", [])))
            result_dict["relevant_design"] = list(set(relevant_design + project_context.get("design", [])))
            result_dict["relevant_tasks"] = list(set(relevant_tasks + project_context.get("tasks", [])))
            
            logger.info(f"Information needs analyzed: {len(result_dict.get('information_gaps', []))} gaps identified")
            
            return {
                "information_needs": result_dict,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {})
            }
            
        except Exception as e:
            logger.error(f"Information needs analysis failed: {e}")
            return {
                "information_needs": {
                    "information_gaps": [],
                    "relevant_specifications": [],
                    "relevant_design": [],
                    "relevant_tasks": [],
                    "content_type": "new",
                    "detail_level": "detailed",
                    "related_sections": [],
                    "search_focus": query
                },
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {})
            }
    
    async def generate_project_aware_queries_node(self, state) -> Dict[str, Any]:
        """
        Generate project-aware search queries using project context and information needs.
        
        Creates targeted queries that:
        - Include project specifications/design/tasks
        - Focus on identified information gaps
        - Are optimized for semantic search
        """
        try:
            query = state.get("query", "")
            query_type = state.get("query_type", "general")
            information_needs = state.get("information_needs", {})
            referenced_context = state.get("referenced_context", {})
            metadata = state.get("metadata", {})
            user_id = state.get("user_id", "")
            
            # Check if this is a retry (re-search) - increment counter if so
            search_retry_count = state.get("search_retry_count", 0)
            has_previous_quality_assessment = bool(state.get("search_quality_assessment"))
            
            if has_previous_quality_assessment:
                # This is a retry - increment counter
                search_retry_count += 1
                logger.info(f"Re-search attempt {search_retry_count} - refining queries based on previous results")
            
            # Extract project context
            shared_memory = metadata.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor", {})
            frontmatter = active_editor.get("frontmatter", {})
            
            # Build project context dict for tool
            project_specifications = information_needs.get("relevant_specifications", [])
            project_design = information_needs.get("relevant_design", [])
            
            if not project_specifications and frontmatter:
                project_specifications = frontmatter.get("specifications", [])[:5]
            
            project_context = {
                "specifications": project_specifications,
                "design": project_design
            }
            
            # General project examples for prompt
            domain_examples = [
                "HVAC system sizing and installation requirements",
                "Landscaping design and plant selection",
                "Garden bed layout and soil preparation",
                "Home improvement project timeline and materials"
            ]
            
            # Use universal project-aware query generation tool
            from orchestrator.tools.information_analysis_tools import generate_project_aware_queries_tool
            
            fast_model = self.agent._get_fast_model(state)
            get_llm_func = lambda: self.agent._get_llm(temperature=0.2, model=fast_model, state=state)
            
            result_dict = await generate_project_aware_queries_tool(
                query=query,
                query_type=query_type,
                information_needs=information_needs,
                project_context=project_context,
                domain_examples=domain_examples,
                user_id=user_id,
                num_queries=3,
                llm_model=fast_model,
                get_llm_func=get_llm_func
            )
            
            search_queries = result_dict.get("search_queries", [])
            
            logger.info(f"Generated {len(search_queries)} project-aware search queries")
            for i, sq in enumerate(search_queries[:3], 1):
                logger.info(f"  {i}. [{sq.get('priority', '?')}] {sq.get('query', '')[:80]}")
            
            return {
                "search_queries": search_queries,
                "query_expansion_used": result_dict.get("query_expansion_used", False),
                "search_retry_count": search_retry_count,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "information_needs": state.get("information_needs", {})
            }
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Fallback to simple query
            return {
                "search_queries": [{"query": query, "priority": 1, "focus": "General search"}],
                "query_expansion_used": False,
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "information_needs": state.get("information_needs", {})
            }
    
    async def search_content_node(self, state) -> Dict[str, Any]:
        """
        Semantic segment search for general project content.
        
        Uses project-aware queries to find relevant SEGMENTS within documents,
        not just documents. This enables precise updates to specific sections.
        """
        try:
            user_id = state["user_id"]
            metadata = state.get("metadata", {})
            referenced_context = state.get("referenced_context", {})
            search_queries = state.get("search_queries", [])
            
            logger.info(f"Performing semantic segment search with {len(search_queries)} queries")
            
            # Check if user explicitly requested web search
            query = state.get("query", "")
            query_lower = query.lower()
            explicit_web_search_keywords = [
                "search the web", "search online", "look up online", "search internet",
                "find online", "web search", "search for", "look up", "google",
                "search web for", "search online for", "find on the web"
            ]
            is_explicit_web_request = any(keyword in query_lower for keyword in explicit_web_search_keywords)
            
            # Use the universal segment search tool
            from orchestrator.tools.segment_search_tools import search_segments_across_documents_tool
            
            # Extract queries from search_queries (can be dicts or strings)
            query_list = []
            for sq in search_queries[:3]:  # Top 3 queries
                if isinstance(sq, dict):
                    query_list.append(sq.get("query", ""))
                else:
                    query_list.append(str(sq))
            query_list = [q for q in query_list if q.strip()]
            
            # General project domain keywords for content extraction
            general_project_keywords = [
                'requirement', 'specification', 'design', 'plan', 'timeline',
                'task', 'milestone', 'budget', 'material', 'equipment',
                'installation', 'maintenance', 'approach', 'method', 'process',
                'project', 'scope', 'goal', 'objective', 'deliverable'
            ]
            
            # Perform segment search
            segment_result = await search_segments_across_documents_tool(
                queries=query_list,
                project_documents=referenced_context,
                user_id=user_id,
                limit_per_query=5,
                max_queries=3,
                prioritize_project_docs=True,
                context_window=350,
                domain_keywords=general_project_keywords
            )
            
            segments_list = segment_result.get("segments", [])
            documents_found_count = segment_result.get("documents_found_count", 0)
            
            logger.info(f"Found {len(segments_list)} relevant segments across {documents_found_count} documents")
            
            # Convert segments to document format for compatibility
            documents = []
            for segment in segments_list[:15]:  # Top 15 segments
                documents.append({
                    "document_id": segment.get("document_id"),
                    "title": segment.get("section_name") or segment.get("filename", ""),
                    "filename": segment.get("filename", ""),
                    "content": segment.get("content", ""),
                    "tags": ["project", "segment"],
                    "category": "project",
                    "source": segment.get("source", "library_document"),
                    "relevance_score": segment.get("relevance_score", 0.5),
                    "section_name": segment.get("section_name"),
                    "section_start": segment.get("section_start"),
                    "section_end": segment.get("section_end")
                })
            
            return {
                "documents": documents,
                "segments": segments_list,
                "web_search_explicit": is_explicit_web_request,
                "documents_found_count": len(documents),
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "search_queries": state.get("search_queries", [])
            }
        except Exception as e:
            logger.error(f"Search content failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "documents": [],
                "segments": [],
                "web_search_explicit": False,
                "documents_found_count": 0,
                "error": str(e),
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "search_queries": state.get("search_queries", [])
            }
    
    async def assess_result_quality_node(self, state) -> Dict[str, Any]:
        """
        Assess the quality of search results to determine if they answer the query.
        
        Uses LLM to evaluate:
        - Do the results answer the query?
        - Are results relevant to project context?
        - Is additional information needed (web search)?
        - Should we re-search with different queries?
        """
        try:
            query = state.get("query", "")
            query_type = state.get("query_type", "general")
            documents = state.get("documents", [])
            segments = state.get("segments", [])
            information_needs = state.get("information_needs", {})
            
            # Check for verification keywords
            query_lower = query.lower()
            verification_keywords = [
                "double check", "verify", "confirm", "check if", "is this correct",
                "is this right", "validate", "fact check", "cross reference",
                "look up", "research", "find out", "investigate"
            ]
            is_verification_query = any(keyword in query_lower for keyword in verification_keywords)
            
            # Build summary of search results
            result_summary = []
            for i, seg in enumerate(segments[:10], 1):
                result_summary.append({
                    "index": i,
                    "source": seg.get("source", "unknown"),
                    "section": seg.get("section_name", "Unknown"),
                    "filename": seg.get("filename", ""),
                    "content_preview": seg.get("content", "")[:200],
                    "relevance_score": seg.get("relevance_score", 0.0)
                })
            
            # Count results by source
            project_doc_count = sum(1 for seg in segments if seg.get("source") == "project_document")
            library_doc_count = sum(1 for seg in segments if seg.get("source") == "library_document")
            
            # Use LLM to assess quality
            fast_model = self.agent._get_fast_model(state)
            llm = self.agent._get_llm(temperature=0.1, model=fast_model, state=state)
            
            prompt = f"""Assess whether the search results adequately answer this query.

**QUERY**: {query}

**QUERY TYPE**: {query_type}

**INFORMATION NEEDS**:
- Gaps: {', '.join(information_needs.get('information_gaps', [])[:5])}
- Detail Level: {information_needs.get('detail_level', 'detailed')}
- Content Type: {information_needs.get('content_type', 'new')}

**SEARCH RESULTS**:
- Total segments found: {len(segments)}
- Project documents: {project_doc_count}
- Library documents: {library_doc_count}

**RESULT SUMMARY**:
{json.dumps(result_summary, indent=2) if result_summary else "No results found"}

**TASK**: Evaluate:
1. Do these results answer the query? (quality_score: 0.0-1.0)
2. Are results relevant to the project context?
3. Is additional information needed? (needs_web_search: true/false)
4. Should we re-search with different queries? (should_re_search: true/false)

**IMPORTANT**: If the query asks to "verify", "double check", "confirm", or "validate" information, you MUST set needs_web_search=true to get external verification.

**QUALITY CRITERIA**:
- 0.8-1.0: Results fully answer the query with project-relevant information
- 0.5-0.7: Results partially answer, RECOMMEND web search for verification/latest info/details
- 0.3-0.5: Results are somewhat relevant but incomplete - NEED web search
- 0.0-0.3: Results don't answer the query, need re-search or web search

Return ONLY valid JSON:
{{
  "quality_score": 0.75,
  "adequately_answers_query": true,
  "relevant_to_project": true,
  "needs_web_search": false,
  "should_re_search": false,
  "reasoning": "Brief explanation of assessment",
  "missing_information": ["what's missing if any"]
}}"""
            
            try:
                structured_llm = llm.with_structured_output({
                    "type": "object",
                    "properties": {
                        "quality_score": {"type": "number"},
                        "adequately_answers_query": {"type": "boolean"},
                        "relevant_to_project": {"type": "boolean"},
                        "needs_web_search": {"type": "boolean"},
                        "should_re_search": {"type": "boolean"},
                        "reasoning": {"type": "string"},
                        "missing_information": {"type": "array", "items": {"type": "string"}}
                    }
                })
                result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
                result_dict = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
            except Exception:
                # Fallback assessment
                if len(segments) == 0:
                    result_dict = {
                        "quality_score": 0.0,
                        "adequately_answers_query": False,
                        "relevant_to_project": False,
                        "needs_web_search": True,
                        "should_re_search": True,
                        "reasoning": "No results found",
                        "missing_information": ["All information"]
                    }
                elif project_doc_count > 0:
                    result_dict = {
                        "quality_score": 0.7,
                        "adequately_answers_query": True,
                        "relevant_to_project": True,
                        "needs_web_search": False,
                        "should_re_search": False,
                        "reasoning": "Found relevant project documents",
                        "missing_information": []
                    }
                else:
                    result_dict = {
                        "quality_score": 0.5,
                        "adequately_answers_query": True,
                        "relevant_to_project": False,
                        "needs_web_search": query_type in ["design", "planning"],
                        "should_re_search": False,
                        "reasoning": "Found library documents but may need latest info",
                        "missing_information": []
                    }
            
            quality_score = result_dict.get("quality_score", 0.5)
            
            # Override needs_web_search if this is a verification query
            if is_verification_query and not result_dict.get("needs_web_search", False):
                logger.info(f"Overriding needs_web_search=True due to verification keywords in query")
                result_dict["needs_web_search"] = True
                result_dict["reasoning"] = result_dict.get("reasoning", "") + " (Web search required for verification)"
            
            if is_verification_query and 0.5 <= quality_score <= 0.7:
                if not result_dict.get("needs_web_search", False):
                    logger.info(f"Overriding needs_web_search=True for moderate quality verification query")
                    result_dict["needs_web_search"] = True
            
            logger.info(f"Search quality assessment: score={quality_score:.2f}, needs_web={result_dict.get('needs_web_search')}, re_search={result_dict.get('should_re_search')}")
            
            return {
                "search_quality_assessment": result_dict,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "documents": state.get("documents", []),
                "segments": state.get("segments", []),
                "search_queries": state.get("search_queries", []),
                "search_retry_count": state.get("search_retry_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                "search_quality_assessment": {
                    "quality_score": 0.5,
                    "adequately_answers_query": len(segments) > 0,
                    "relevant_to_project": False,
                    "needs_web_search": True,
                    "should_re_search": False,
                    "reasoning": "Assessment failed, defaulting to web search",
                    "missing_information": []
                },
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "documents": state.get("documents", []),
                "segments": state.get("segments", []),
                "search_queries": state.get("search_queries", []),
                "search_retry_count": state.get("search_retry_count", 0)
            }


