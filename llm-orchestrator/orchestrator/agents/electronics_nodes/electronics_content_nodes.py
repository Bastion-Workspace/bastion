"""
Electronics Agent - Content Extraction and Routing Nodes Module
Handles content extraction and routing to project files
"""

import logging
import json
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ElectronicsContentNodes:
    """Content routing and saving nodes for electronics agent"""
    
    def __init__(self, agent):
        """
        Initialize with reference to main agent for access to helper methods
        
        Args:
            agent: ElectronicsAgent instance (for _get_llm, _get_fast_model, etc.)
        """
        self.agent = agent
    
    async def extract_and_route_content_node(self, state) -> Dict[str, Any]:
        """
        Merged node: extract_content_structure + route_and_save_content
        Unified content extraction and routing in single LLM call.
        """
        try:
            response = state.get("response", {})
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            referenced_context = state.get("referenced_context", {})
            # Check both metadata.active_editor and shared_memory.active_editor
            active_editor = metadata.get("active_editor") or metadata.get("shared_memory", {}).get("active_editor", {})
            
            if not response_text or len(response_text.strip()) < 50:
                logger.info("🔌 No substantial content to extract")
                return {"save_plan": None}
            
            # Get available files from referenced context
            # referenced_context is structured as: {"components": [files], "protocols": [files], ...}
            available_files = []
            file_content_previews = {}  # Store content for section detection
            
            # Also include the project plan from active_editor
            if active_editor and active_editor.get("document_id"):
                project_plan_content = active_editor.get("content", "")
                # Extract section names from project plan
                sections = re.findall(r'^##+\s+(.+)$', project_plan_content, re.MULTILINE)
                project_plan_doc = {
                    "document_id": active_editor.get("document_id"),
                    "filename": active_editor.get("filename", "project_plan.md"),
                    "summary": f"Main project plan document. Sections: {', '.join(sections[:10])}" if sections else "Main project plan document"
                }
                available_files.append(project_plan_doc)
                file_content_previews[active_editor.get("document_id")] = project_plan_content[:1000]
            
            # Flatten referenced_context (it's organized by category)
            for category, file_list in referenced_context.items():
                if isinstance(file_list, list):
                    for file_doc in file_list:
                        if isinstance(file_doc, dict):
                            doc_id = file_doc.get("document_id")
                            filename = file_doc.get("filename", f"document_{doc_id}")
                            content = file_doc.get("content", "")
                            
                            # Extract frontmatter for file description
                            frontmatter = file_doc.get("frontmatter", {})
                            file_description = frontmatter.get("description", "")
                            file_title = frontmatter.get("title", "")
                            
                            # Extract section names from content
                            sections = []
                            if content:
                                sections = re.findall(r'^##+\s+(.+)$', content, re.MULTILINE)
                                file_content_previews[doc_id] = content[:1000]  # Store preview for section matching
                            
                            # Build comprehensive summary with file purpose
                            summary_parts = []
                            if file_title:
                                summary_parts.append(f"Title: {file_title}")
                            if file_description:
                                summary_parts.append(f"Purpose: {file_description}")
                            
                            # Add existing summary if available
                            existing_summary = file_doc.get("summary", "")
                            if existing_summary and existing_summary not in summary_parts:
                                summary_parts.append(existing_summary)
                            
                            # Add content preview if no description
                            if not file_description and content:
                                content_preview = content[:200].replace('\n', ' ').strip()
                                if content_preview:
                                    summary_parts.append(f"Content: {content_preview}")
                            
                            # Add section info
                            if sections:
                                summary_parts.append(f"Sections: {', '.join(sections[:8])}")
                            
                            summary = " | ".join(summary_parts) if summary_parts else filename
                            
                            available_files.append({
                                "document_id": doc_id,
                                "filename": filename,
                                "summary": summary[:400] if summary else filename  # Increased length for better routing context
                            })
            
            logger.info(f"🔌 Found {len(available_files)} available files for routing")
            
            # Unified prompt for extraction + routing
            llm = self.agent._get_llm(temperature=0.1, state=state)
            
            # Get conversation history for better context
            messages = state.get("messages", [])
            conversation_context = ""
            if len(messages) > 1:
                # Include last few messages for context
                recent_messages = messages[-3:]
                conversation_context = "\n".join([
                    f"{'User' if hasattr(msg, 'type') and msg.type == 'human' else 'Assistant'}: {msg.content if hasattr(msg, 'content') else str(msg)}"
                    for msg in recent_messages
                ])
            
            # Extract existing section content for sections that might be updated
            # This is CRITICAL - the LLM needs to see what's currently there to make informed decisions
            existing_sections_content = {}
            
            # Always extract existing sections - the LLM needs to see what's there to make good decisions
            # This is especially important for updates, replacements, and corrections
            for category, file_list in referenced_context.items():
                if isinstance(file_list, list):
                    for file_doc in file_list:
                        if isinstance(file_doc, dict):
                            content = file_doc.get("content", "")
                            filename = file_doc.get("filename", "")
                            
                            if not content:
                                continue
                            
                            # Extract all sections from this file
                            section_pattern = r'^(##+\s+[^\n]+)$'
                            sections = []
                            for match in re.finditer(section_pattern, content, re.MULTILINE):
                                section_header = match.group(1)
                                section_start = match.start()
                                
                                # Find section end
                                section_end_match = re.search(r'\n##+\s+', content[section_start + 1:], re.MULTILINE)
                                if section_end_match:
                                    section_end = section_start + 1 + section_end_match.start()
                                else:
                                    section_end = len(content)
                                
                                section_name = re.sub(r'^##+\s+', '', section_header).strip()
                                section_content = content[section_start:section_end]
                                
                                # ALWAYS include ALL sections - the LLM needs to see everything to make informed decisions
                                # Don't filter - let the LLM decide what's relevant based on the full context
                                # Store section content keyed by filename and section name
                                if filename not in existing_sections_content:
                                    existing_sections_content[filename] = {}
                                existing_sections_content[filename][section_name] = section_content
            
            logger.info(f"🔍 Extracted {sum(len(sections) for sections in existing_sections_content.values())} existing sections for review")
            
            # Build existing content context for prompt
            existing_content_context = ""
            if existing_sections_content:
                existing_content_context = "\n\n**EXISTING SECTION CONTENT** (review this before making updates):\n"
                for filename, sections in existing_sections_content.items():
                    existing_content_context += f"\n**File: {filename}**\n"
                    for section_name, section_content in sections.items():
                        # Truncate very long sections but keep enough context
                        content_preview = section_content[:2000] if len(section_content) > 2000 else section_content
                        if len(section_content) > 2000:
                            content_preview += "\n[... content truncated ...]"
                        existing_content_context += f"\n### Section: {section_name}\n{content_preview}\n"
            
            prompt = f"""Extract structured content from this response and determine where to save it.

**USER QUERY**: {state.get('query', '')}

**AGENT RESPONSE**: {response_text[:3000]}

**CONVERSATION CONTEXT**:
{conversation_context[:500] if conversation_context else "No previous context"}

**AVAILABLE FILES** (use exact filename from this list):
{json.dumps(available_files, indent=2) if available_files else "No files available - use 'project_plan' for main document"}

{existing_content_context}

**CRITICAL INSTRUCTIONS**:
1. **EXTRACT ALL TECHNICAL DETAILS**: 
   - Voltage/current values, resistance, power ratings
   - Component specifications and part numbers
   - Calculations, formulas, and design equations
   - Design decisions and trade-offs
   - Protocol specifications and communication details
   - Code snippets, firmware logic, and software architecture
   - Testing procedures and validation criteria

2. **MAINTAIN DOCUMENT COHERENCE** (CRITICAL):
   - **Read existing sections** - Understand the document's narrative flow and structure
   - **Write cohesive content** - New content should read naturally with existing sections
   - **Update related sections together** - If updating a component, update all references (specs, code, schematics)
   - **Maintain consistency** - Use same terminology, formatting, and level of detail throughout
   - **Preserve document structure** - Don't break existing section hierarchies or organization
   - **Write for humans** - Content should read like a well-written technical document, not data dumps
   - **Connect ideas** - Link new content to existing sections with transitions and context
   - **Group related updates** - If multiple sections need updates, group them logically

3. **ROUTE TO APPROPRIATE FILES** (use AVAILABLE FILES list above):
   - **Match content type to file purpose**: Read the file summaries/descriptions in AVAILABLE FILES to understand each file's purpose
   - **Component specs/values**: Route to files with "component", "spec", or "hardware" in name/description
   - **Hardware design**: Route to files with "design", "architecture", "hardware", or "system" in name/description
   - **Protocols/communication**: Route to files with "protocol", "communication", or "interface" in name/description
   - **Code/firmware**: Route to files with "software", "code", "firmware", or "implementation" in name/description
   - **General project info**: Route to "project_plan" or files with "plan", "overview", or "requirements" in name/description
   - **If no clear match**: Use the file whose description/summary best matches the content type
   - **Multiple matches**: Split content logically across multiple files if appropriate

4. **DETECT REVISIONS AND CORRECTIONS**: 
   - **CRITICAL: REVIEW EXISTING SECTION CONTENT ABOVE** - The "EXISTING SECTION CONTENT" section shows you what's currently in the files
   - **Compare new information with existing content** - Look at what's there now and decide what needs to change
   - **Revisions**: 'instead of X, use Y' or 'change X to Y' means REPLACE existing content
   - **Corrections**: 'Actually, it is X not Y' or 'That is wrong, it should be X' means CORRECT existing content
   - **Removals**: 'Remove X' or 'Delete Y' or 'We are not using Z anymore' means REMOVE content (replace section with corrected version or note)
   - **Error Detection**: If user says something contradicts existing content, DETECT the conflict and CORRECT it
   - **Bad Concepts**: If user indicates a previous assumption/plan was wrong, UPDATE or REMOVE that content
   - **Always check existing content** in files before adding - if it conflicts, CORRECT it rather than creating duplicates
   - **Update all related sections** - When correcting/updating, find ALL sections that reference the changed content and update them together
   
   **CRITICAL: COMPONENT REPLACEMENTS**:
   - **REVIEW EXISTING SECTIONS ABOVE** - Look at ALL sections that mention the old component
   - **THOROUGH SEARCH REQUIRED**: Search through ALL existing sections shown above - don't miss any references
   - When user says "use Y instead of X" or "replace X with Y":
     * **FIND ALL SECTIONS** that mention X (the old component) - check headers AND content
     * **INCLUDE DELETION OPERATIONS** for sections that are primarily about X (entire sections mentioning X)
     * Create NEW content that ONLY mentions Y (the new component)
     * Do NOT include any references to X in the new content
     * Write content as if X never existed - only document Y
   - Example: If user says "use ADW221S instead of AQV252G":
     * Search ALL sections in EXISTING SECTION CONTENT for "AQV252G" (case-insensitive)
     * For sections like "PhotoMOS SSR Driver Circuit (AQV252G)" - mark for DELETION
     * For sections that mention AQV252G in the body - either DELETE the section or REMOVE those references
     * Create replacement content that ONLY mentions "ADW221S"
     * Do NOT include "AQV252G" anywhere in the new content
   - **BE THOROUGH**: Check circuit diagrams, code comments, specifications - remove ALL references
   - The old component references will be automatically deleted - your job is to write clean replacement content with ONLY the new component AND identify sections that need deletion

5. **MATCH EXISTING SECTIONS**: Use exact section names that exist in files (check file summaries for section lists)

6. **FORMAT AS MARKDOWN**: All content must be markdown (not JSON, not raw text)
   - Use proper headings, lists, code blocks, and tables
   - Maintain consistent formatting with existing document style
   - Write complete sentences and paragraphs, not bullet dumps

7. **BE GRANULAR BUT COHESIVE**: 
   - Break content into logical pieces - don't dump everything in one place
   - BUT ensure each piece reads well on its own and connects to related sections
   - Group related technical details together in coherent paragraphs

8. **ALWAYS EXTRACT**: Even if content seems small, extract it! Technical details like 12VDC and 40-44 ohms are valuable and should be saved.

9. **NEW REFERENCE FILE CREATION** (when content is too detailed for project plan):
   - **When to create**: Content is substantial (>1500 chars) AND doesn't fit existing files well
   - **Criteria**: Content is about a distinct, specific topic that warrants its own file
   - **Set flags**: "create_new_file": true, "suggested_filename": "descriptive-name.md", "file_summary": "Brief purpose"
   - **File naming**: Use lowercase-with-dashes.md format (e.g., "motor-driver-specs.md", "i2c-protocol.md")
   - **Automatic handling**: System will create file, add to project plan frontmatter, save content
   - **Conservative approach**: Only create if NO existing file is a good match (<20% relevance)
   - **Examples of when to create**:
     * Complete component specifications >1500 chars (e.g., detailed microcontroller specs)
     * Protocol documentation with multiple sections (e.g., SPI communication protocol)
     * Comprehensive circuit design documents
     * Detailed code implementations or firmware modules
   - **Examples of when NOT to create**:
     * Small component values or specifications (<1500 chars)
     * Content that fits well in existing component/protocol files
     * General design notes (belongs in project_plan)

10. **USE EXACT FILENAMES**: For existing files, always use the exact filename from AVAILABLE FILES

11. **WRITE COHESIVE CONTENT**:
    - Read the existing section content to understand context and style
    - Write new content that flows naturally from existing content
    - Use transitions and context to connect ideas
    - Maintain consistent voice and technical level
    - Write complete, readable paragraphs - not just data points

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "routing": [
    {{
      "content_type": "components|code|calculations|general",
      "target_file": "filename.md (exact filename from AVAILABLE FILES) or 'project_plan'",
      "action": "append|replace|remove",
      "create_new_file": false,  // Set to true if creating a new reference file
      "suggested_filename": "",  // Required if create_new_file is true (e.g., "motor-driver-specs.md")
      "file_summary": "",  // Required if create_new_file is true (brief description of file purpose)
      "section": "Exact section name (match existing sections if updating/correcting)",
      "content": "Markdown formatted content (lists, tables, code blocks, etc.)",
      "action": "append|replace|remove"
    }}
  ]
}}

**ACTION FIELD**:
- **append**: Add new content to a new or existing section (default if section doesn't exist)
- **replace**: Replace existing section content (use when correcting errors, updating values, or revising)
  - **IMPORTANT**: When replacing due to component change, the new content must NOT include the old component name
  - Example: If replacing "AQV252G" with "ADW221S", new content should only mention ADW221S
- **remove**: Remove section content entirely (use when user explicitly requests removal or content is obsolete)
"""
            
            try:
                from orchestrator.models.electronics_models import UnifiedContentPlan
                structured_llm = llm.with_structured_output(UnifiedContentPlan)
                result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
                save_plan = result.dict() if hasattr(result, 'dict') else result.model_dump()
            except Exception:
                # Fallback
                response_obj = await llm.ainvoke([{"role": "user", "content": prompt}])
                content = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
                parsed = self.agent._extract_json_from_response(content) or {}
                save_plan = {"routing": parsed.get("routing", [])}
            
            # UNIVERSAL SECTION ANALYSIS: Analyze existing sections for ANY type of change
            # This handles component replacements, corrections, direction changes, specification updates, etc.
            user_query = state.get('query', '')
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            
            section_operations = await self._analyze_existing_sections_for_updates(
                user_query, response_text, save_plan.get('routing', []), referenced_context, state
            )
            
            # Prepend operations (deletions and edits) before new content
            if section_operations:
                logger.info(f"🔌 Identified {len(section_operations)} section operations: {sum(1 for op in section_operations if op.get('action') == 'remove')} deletions, {sum(1 for op in section_operations if op.get('action') == 'replace')} edits")
                save_plan['routing'] = section_operations + save_plan.get('routing', [])
            
            routing_count = len(save_plan.get('routing', []))
            logger.info(f"🔌 Extracted and routed {routing_count} content items")
            
            if routing_count == 0:
                logger.warning("⚠️ No content items extracted - response may not have saveable content")
            else:
                # Log what's being routed
                for i, route in enumerate(save_plan.get('routing', [])[:5], 1):
                    logger.info(f"🔌 Route {i}: {route.get('content_type')} -> {route.get('target_file')} ({route.get('section')})")
            
            return {
                "save_plan": save_plan
            }
            
        except Exception as e:
            logger.error(f"❌ Content extraction and routing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"save_plan": None}
    
    async def _analyze_existing_sections_for_updates(
        self,
        user_query: str,
        agent_response: str,
        routing_items: List[Dict[str, Any]],
        referenced_context: Dict[str, Any],
        state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        UNIVERSAL section analysis - intelligently analyzes existing sections for ANY type of change.
        
        This method handles:
        - Component replacements (e.g., "use ADW221S instead of AQV252G")
        - Corrections (e.g., "actually, use 5V not 12V")
        - Direction changes (e.g., "we're not doing that anymore")
        - Specification updates (e.g., "change resistor to 10k")
        - Any other type of update that requires analyzing existing content
        
        Uses LLM to intelligently determine what needs to change, rather than brittle pattern matching.
        """
        try:
            # Use shared method from save_nodes for consistent component replacement detection
            from orchestrator.agents.electronics_nodes.electronics_save_nodes import ElectronicsSaveNodes
            save_nodes = ElectronicsSaveNodes(self.agent)
            
            # Extract old components from query for component replacement detection
            old_components = save_nodes._extract_old_components_from_query(user_query)
            
            query_lower = user_query.lower()
            response_lower = agent_response.lower() if agent_response else ""
            
            # Check for change indicators
            change_indicators = [
                'instead of', 'replace', 'switch from', 'change', 'actually',
                'wrong', 'should be', 'not anymore', 'remove', 'delete',
                'update', 'correct', 'no longer'
            ]
            has_change_indicators = any(indicator in query_lower for indicator in change_indicators)
            
            # If no change indicators and no old components, skip analysis
            if not has_change_indicators and not old_components:
                return []
            
            user_id = state.get("user_id", "")
            
            # Extract query terms for relevance filtering
            query_terms = set(re.findall(r'\b[A-Z][A-Z0-9]+\b', user_query))  # Component names
            query_terms.update(re.findall(r'\b\d+[Vv]|\d+[Aa]h?|\d+[ΩΩ]|\d+[Ww]\b', user_query))  # Specs
            query_terms.update(query_lower.split())
            
            sections_to_analyze = []
            
            # Search through referenced files for relevant sections
            from orchestrator.tools.document_tools import search_within_document_tool, get_document_content_tool
            
            for category, file_list in referenced_context.items():
                if isinstance(file_list, list):
                    for file_doc in file_list:
                        if isinstance(file_doc, dict):
                            filename = file_doc.get("filename", "")
                            doc_id = file_doc.get("document_id")
                            
                            if not doc_id:
                                # Fallback: if no document_id, skip search and load full content
                                content = file_doc.get("content", "")
                                if not content:
                                    continue
                                # Use old method for files without document_id
                                logger.debug(f"⚠️ No document_id for {filename} - using fallback method")
                            else:
                                # PHASE 2: Use in-document search to pre-filter sections
                                # Search for key terms within this document
                                search_queries = []
                                
                                # Add old component names (highest priority)
                                for old_comp in old_components:
                                    search_queries.append(old_comp)
                                
                                # Add component names from query
                                for term in query_terms:
                                    if len(term) > 3 and term.isupper():  # Component names
                                        search_queries.append(term)
                                
                                # Add key words from query (if change indicators present)
                                change_indicators_list = [
                                    'instead of', 'replace', 'switch from', 'change', 'actually',
                                    'wrong', 'should be', 'not anymore', 'remove', 'delete',
                                    'update', 'correct', 'no longer'
                                ]
                                has_change_indicators_local = any(indicator in query_lower for indicator in change_indicators_list)
                                
                                if has_change_indicators_local:
                                    # For change queries, search for all terms together
                                    if search_queries:
                                        # Search for all terms (AND logic)
                                        combined_query = " ".join(search_queries[:5])  # Limit to 5 terms
                                        search_result = await search_within_document_tool(
                                            document_id=doc_id,
                                            query=combined_query,
                                            search_type="all_terms",
                                            context_window=100,
                                            case_sensitive=False,
                                            user_id=user_id
                                        )
                                        
                                        if search_result.get("total_matches", 0) > 0:
                                            # Found matches - get sections from matches
                                            matches = search_result.get("matches", [])
                                            seen_sections = set()
                                            
                                            for match in matches:
                                                section_start = match.get("section_start")
                                                section_end = match.get("section_end")
                                                section_name = match.get("section_name")
                                                
                                                if section_start is not None and section_end is not None:
                                                    section_key = (doc_id, section_start, section_end)
                                                    if section_key not in seen_sections:
                                                        seen_sections.add(section_key)
                                                        
                                                        # Get full section content
                                                        content = await get_document_content_tool(doc_id, user_id)
                                                        if not content.startswith("Error") and section_start < len(content) and section_end <= len(content):
                                                            section_content = content[section_start:section_end]
                                                            
                                                            sections_to_analyze.append({
                                                                "filename": filename,
                                                                "section_name": section_name or "Unknown",
                                                                "section_content": section_content,
                                                                "start": section_start,
                                                                "end": section_end,
                                                                "document_id": doc_id
                                                            })
                                            
                                            logger.info(f"🔍 Found {len(seen_sections)} relevant sections in {filename} via in-document search")
                                            continue  # Skip to next file - we found matches
                                
                                # If no matches found or no change indicators, fall back to loading full content
                                # but only if we have specific terms to search for
                                if not search_queries:
                                    # No specific terms - load full content (fallback)
                                    content = file_doc.get("content", "")
                                    if not content:
                                        continue
                                else:
                                    # We have search terms but no matches - skip this file
                                    logger.debug(f"🔍 No matches found in {filename} for search terms - skipping")
                                    continue
                            
                            # Fallback: Load full content and use old method (for files without document_id or when search fails)
                            if "content" not in locals() or not content:
                                content = file_doc.get("content", "")
                            
                            if not content:
                                continue
                            
                            # Find all sections
                            section_pattern = r'^(##+\s+[^\n]+)$'
                            for section_match in re.finditer(section_pattern, content, re.MULTILINE):
                                section_header = section_match.group(1)
                                section_start = section_match.start()
                                
                                section_end_match = re.search(r'\n##+\s+', content[section_start + 1:], re.MULTILINE)
                                if section_end_match:
                                    section_end = section_start + 1 + section_end_match.start()
                                else:
                                    section_end = len(content)
                                
                                section_name = re.sub(r'^##+\s+', '', section_header).strip()
                                section_content = content[section_start:section_end]
                                section_content_lower = section_content.lower()
                                
                                # Check if section is potentially relevant
                                # Include if:
                                # 1. Section mentions terms from query/response
                                # 2. Query indicates changes/corrections (always check all sections)
                                # 3. Response contains component names, specifications, or technical terms
                                
                                # Extract component names, specifications, and key terms
                                query_terms_local = set(re.findall(r'\b[A-Z][A-Z0-9]+\b', user_query))  # Component names like AQV252G
                                query_terms_local.update(re.findall(r'\b\d+[Vv]|\d+[Aa]h?|\d+[ΩΩ]|\d+[Ww]\b', user_query))  # Specs like 12V, 390Ω
                                query_terms_local.update(query_lower.split())  # All words
                                
                                response_terms = set()
                                if agent_response:
                                    response_terms.update(re.findall(r'\b[A-Z][A-Z0-9]+\b', agent_response[:1000]))
                                    response_terms.update(response_lower.split()[:20])  # First 20 words
                                
                                all_terms = query_terms_local | response_terms
                                
                                # Check for change indicators
                                change_indicators_local = [
                                    'instead of', 'replace', 'switch from', 'change', 'actually',
                                    'wrong', 'should be', 'not anymore', 'remove', 'delete',
                                    'update', 'correct', 'no longer'
                                ]
                                has_change_indicators_local = any(indicator in query_lower for indicator in change_indicators_local)
                                
                                # Determine relevance
                                is_relevant = False
                                
                                if has_change_indicators_local:
                                    # If query indicates changes, still filter by relevance
                                    # Check if section mentions terms from query/response (component names, etc.)
                                    section_mentions_terms = any(
                                        term.lower() in section_content_lower or 
                                        term.lower() in section_name.lower()
                                        for term in all_terms
                                        if len(term) > 2  # Skip very short terms
                                    )
                                    
                                    # For component replacements, prioritize sections mentioning old/new components
                                    old_components_local = save_nodes._extract_old_components_from_query(user_query)
                                    mentions_old_component = any(
                                        old_comp.lower() in section_content_lower or 
                                        old_comp.lower() in section_name.lower()
                                        for old_comp in old_components_local
                                    )
                                    
                                    # Include if mentions query terms OR old components OR component-related keywords
                                    if section_mentions_terms or mentions_old_component:
                                        is_relevant = True
                                    elif query_terms_local and any(len(t) > 3 and t.isupper() for t in query_terms_local):
                                        # Query mentions component names - check component-related sections
                                        component_keywords = ['component', 'relay', 'ssr', 'mosfet', 'driver', 'circuit', 'specification']
                                        if any(kw in section_content_lower or kw in section_name.lower() for kw in component_keywords):
                                            is_relevant = True
                                else:
                                    # Otherwise, check if section mentions query/response terms
                                    section_mentions_terms = any(
                                        term.lower() in section_content_lower or 
                                        term.lower() in section_name.lower()
                                        for term in all_terms
                                        if len(term) > 2  # Skip very short terms
                                    )
                                    
                                    # Also check for component-related sections if query mentions components
                                    if query_terms_local and any(len(t) > 3 and t.isupper() for t in query_terms_local):
                                        # Query mentions component names - check component-related sections
                                        component_keywords = ['component', 'relay', 'ssr', 'mosfet', 'driver', 'circuit', 'specification']
                                        if any(kw in section_content_lower or kw in section_name.lower() for kw in component_keywords):
                                            is_relevant = True
                                    elif section_mentions_terms:
                                        is_relevant = True
                                
                                if is_relevant:
                                    sections_to_analyze.append({
                                        "filename": filename,
                                        "section_name": section_name,
                                        "section_content": section_content,
                                        "start": section_start,
                                        "end": section_end
                                    })
            
            if not sections_to_analyze:
                return []
            
            # Limit sections to analyze to avoid excessive processing time
            # If too many sections, prioritize the most relevant ones
            MAX_SECTIONS_TO_ANALYZE = 20  # Reasonable limit for performance
            if len(sections_to_analyze) > MAX_SECTIONS_TO_ANALYZE:
                logger.info(f"🔍 Found {len(sections_to_analyze)} sections to analyze - limiting to {MAX_SECTIONS_TO_ANALYZE} most relevant")
                # Prioritize sections that mention old component names or change-related terms
                old_components_local = save_nodes._extract_old_components_from_query(user_query)
                
                def section_relevance_score(section_info):
                    score = 0
                    section_content_lower = section_info.get("section_content", "").lower()
                    section_name_lower = section_info.get("section_name", "").lower()
                    
                    # Higher score for sections mentioning old components
                    for old_comp in old_components_local:
                        if old_comp.lower() in section_content_lower or old_comp.lower() in section_name_lower:
                            score += 10
                    
                    # Score for change-related terms
                    change_terms = ['replace', 'old', 'new', 'update', 'change', 'remove', 'delete']
                    for term in change_terms:
                        if term in section_content_lower or term in section_name_lower:
                            score += 2
                    
                    # Score for query terms
                    query_words = set(query_lower.split())
                    section_words = set(section_content_lower.split())
                    overlap = len(query_words & section_words)
                    score += overlap
                    
                    return score
                
                # Sort by relevance and take top N
                sections_to_analyze.sort(key=section_relevance_score, reverse=True)
                sections_to_analyze = sections_to_analyze[:MAX_SECTIONS_TO_ANALYZE]
                logger.info(f"🔍 Selected top {len(sections_to_analyze)} most relevant sections for analysis")
            
            # Use LLM to intelligently analyze sections
            # Batch sections into groups for efficiency (analyze multiple sections per LLM call)
            llm = self.agent._get_llm(temperature=0.1, state=state)
            
            operations = []
            
            # Batch sections: analyze 5 sections per LLM call to balance speed and quality
            BATCH_SIZE = 5
            section_batches = [
                sections_to_analyze[i:i + BATCH_SIZE] 
                for i in range(0, len(sections_to_analyze), BATCH_SIZE)
            ]
            
            logger.info(f"🔍 Analyzing {len(sections_to_analyze)} sections in {len(section_batches)} batch(es)")
            
            # Process batches
            for batch_idx, section_batch in enumerate(section_batches):
                # Build batch prompt with multiple sections
                sections_text = []
                for idx, section_info in enumerate(section_batch, 1):
                    section_text = f"""
**SECTION {idx}**:
- **File**: {section_info['filename']}
- **Section Name**: {section_info['section_name']}
- **Content**: {section_info['section_content'][:1500]}
"""
                    sections_text.append(section_text)
                
                batch_prompt = f"""Analyze these existing sections in light of new information/instructions.

**USER QUERY**: {user_query}

**AGENT RESPONSE** (new information):
{agent_response[:1500] if agent_response else "No specific response provided"}

**SECTIONS TO ANALYZE**:
{''.join(sections_text)}

**ANALYSIS REQUIRED**:
For EACH section, compare it with the user query and agent response. Determine if the section needs updates:

1. **DELETE**: Section is obsolete, incorrect, or contradicts new information
   - Example: Section describes old approach that's been replaced
   - Example: Section is exclusively about something being removed
   → Action: DELETE entire section
   
2. **EDIT**: Section has useful content but needs updates
   - Example: Section mentions old component/approach that needs replacement
   - Example: Section has incorrect specifications that need correction
   - Example: Section describes old direction that needs updating
   → Action: EDIT section - update incorrect/outdated parts, preserve useful content
   
3. **NO CHANGE**: Section is still accurate and doesn't conflict with new information
   → Action: SKIP (don't include in operations)

**OUTPUT FORMAT** (JSON only - array of analyses, one per section):
[
  {{
    "section_index": 1,
    "section_name": "Section Name",
    "filename": "filename.md",
    "action": "delete|edit|skip",
    "reasoning": "Why this action (1-2 sentences)",
    "edited_content": "If action is 'edit', provide the FULL edited section content. If 'delete' or 'skip', this field is empty."
  }},
  {{
    "section_index": 2,
    ...
  }}
]

**CRITICAL RULES**:
- DELETE only if section is obsolete or exclusively about something being removed
- EDIT if section has useful content but needs updates (replace old with new, correct errors, update specifications)
- When editing, preserve ALL useful content, formatting, diagrams, code, structure
- Only update what needs to change - don't rewrite unnecessarily
- Keep section structure and organization intact
- Be granular - update specific parts, not entire sections unless necessary
- Analyze ALL sections in the batch - return one analysis per section
"""
                
                try:
                    response = await llm.ainvoke([{"role": "user", "content": batch_prompt}])
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    # Extract JSON array from response
                    json_match = re.search(r'\[[^\]]*\{[^}]*"section_index"[^}]*\}[^\]]*\]', content, re.DOTALL)
                    if json_match:
                        analyses = json.loads(json_match.group(0))
                    else:
                        # Try to find single object and wrap in array
                        single_match = re.search(r'\{[^{}]*"section_index"[^{}]*\}', content, re.DOTALL)
                        if single_match:
                            analyses = [json.loads(single_match.group(0))]
                        else:
                            # Fallback: create skip analyses for all sections
                            analyses = [{"action": "skip", "section_index": i+1} for i in range(len(section_batch))]
                    
                    # Process each analysis in the batch
                    for analysis in analyses:
                        section_idx = analysis.get("section_index", 1) - 1  # Convert to 0-based
                        if section_idx < 0 or section_idx >= len(section_batch):
                            logger.warning(f"⚠️ Invalid section_index {analysis.get('section_index')} in batch analysis - skipping")
                            continue
                        
                        section_info = section_batch[section_idx]
                        action = analysis.get("action", "skip")
                        edited_content = analysis.get("edited_content", "")
                        reasoning = analysis.get("reasoning", "")
                        
                        if action == "delete":
                            logger.info(f"🗑️ LLM decided to DELETE section '{section_info['section_name']}' in {section_info['filename']}: {reasoning}")
                            operations.append({
                                "content_type": "general",
                                "target_file": section_info["filename"],
                                "section": section_info["section_name"],
                                "content": "",
                                "action": "remove",
                                "delete_range": {
                                    "start": section_info["start"],
                                    "end": section_info["end"],
                                    "original_text": section_info["section_content"]
                                }
                            })
                        elif action == "edit" and edited_content:
                            logger.info(f"✏️ LLM decided to EDIT section '{section_info['section_name']}' in {section_info['filename']}: {reasoning}")
                            operations.append({
                                "content_type": "general",
                                "target_file": section_info["filename"],
                                "section": section_info["section_name"],
                                "content": edited_content,
                                "action": "replace"
                            })
                        elif action == "skip":
                            # Section is still accurate - no change needed
                            logger.debug(f"✓ Section '{section_info['section_name']}' in {section_info['filename']} is still accurate - no changes needed")
                        else:
                            # Invalid action or missing edited_content
                            logger.debug(f"⚠️ Invalid analysis result for section '{section_info['section_name']}' - skipping")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to analyze batch {batch_idx + 1}: {e}")
                    # Don't add fallback operations - if analysis fails, skip the batch
                    # Better to skip than make incorrect changes
            
            logger.info(f"✅ Analyzed {len(sections_to_analyze)} sections: {sum(1 for op in operations if op.get('action') == 'remove')} to delete, {sum(1 for op in operations if op.get('action') == 'replace')} to edit, {len(sections_to_analyze) - len(operations)} no change needed")
            return operations
            
        except Exception as e:
            logger.warning(f"⚠️ Error analyzing sections for component replacement: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []
    

