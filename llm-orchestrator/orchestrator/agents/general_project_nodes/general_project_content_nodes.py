"""
General Project Agent - Content Extraction and Routing Nodes Module
Handles content extraction and routing to project files
"""

import logging
import json
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class GeneralProjectContentNodes:
    """Content routing and saving nodes for general project agent"""
    
    def __init__(self, agent):
        """
        Initialize with reference to main agent for access to helper methods
        
        Args:
            agent: GeneralProjectAgent instance (for _get_llm, _get_fast_model, etc.)
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
            active_editor = metadata.get("active_editor") or metadata.get("shared_memory", {}).get("active_editor", {})
            
            if not response_text or len(response_text.strip()) < 50:
                logger.info("No substantial content to extract")
                return {
                    "save_plan": None,
                    # ✅ CRITICAL: Preserve critical state keys
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "referenced_context": state.get("referenced_context", {}),
                    "response": state.get("response", {})
                }
            
            # Get available files from referenced context
            available_files = []
            file_content_previews = {}
            
            # Include the project plan from active_editor
            if active_editor and active_editor.get("document_id"):
                project_plan_content = active_editor.get("content", "")
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
                            
                            frontmatter = file_doc.get("frontmatter", {})
                            file_description = frontmatter.get("description", "")
                            file_title = frontmatter.get("title", "")
                            
                            sections = []
                            if content:
                                sections = re.findall(r'^##+\s+(.+)$', content, re.MULTILINE)
                                file_content_previews[doc_id] = content[:1000]
                            
                            summary_parts = []
                            if file_title:
                                summary_parts.append(f"Title: {file_title}")
                            if file_description:
                                summary_parts.append(f"Purpose: {file_description}")
                            
                            existing_summary = file_doc.get("summary", "")
                            if existing_summary and existing_summary not in summary_parts:
                                summary_parts.append(existing_summary)
                            
                            if not file_description and content:
                                content_preview = content[:200].replace('\n', ' ').strip()
                                if content_preview:
                                    summary_parts.append(f"Content: {content_preview}")
                            
                            if sections:
                                summary_parts.append(f"Sections: {', '.join(sections[:8])}")
                            
                            summary = " | ".join(summary_parts) if summary_parts else filename
                            
                            available_files.append({
                                "document_id": doc_id,
                                "filename": filename,
                                "summary": summary[:400] if summary else filename
                            })
            
            logger.info(f"Found {len(available_files)} available files for routing")
            
            # Unified prompt for extraction + routing
            llm = self.agent._get_llm(temperature=0.1, state=state)
            
            # Get conversation history for better context
            messages = state.get("messages", [])
            conversation_context = ""
            if len(messages) > 1:
                recent_messages = messages[-3:]
                conversation_context = "\n".join([
                    f"{'User' if hasattr(msg, 'type') and msg.type == 'human' else 'Assistant'}: {msg.content if hasattr(msg, 'content') else str(msg)}"
                    for msg in recent_messages
                ])
            
            # Extract existing section content from ALL files (including project_plan.md from active_editor)
            existing_sections_content = {}
            
            # CRITICAL: First, extract sections from project_plan.md (active_editor content)
            metadata = state.get("metadata", {})
            active_editor = metadata.get("active_editor") or metadata.get("shared_memory", {}).get("active_editor", {})
            if active_editor and active_editor.get("content"):
                project_plan_content = active_editor.get("content", "")
                project_plan_filename = active_editor.get("filename", "project_plan.md")
                
                if project_plan_content:
                    section_pattern = r'^(##+\s+[^\n]+)$'
                    for match in re.finditer(section_pattern, project_plan_content, re.MULTILINE):
                        section_header = match.group(1)
                        section_start = match.start()
                        
                        section_end_match = re.search(r'\n##+\s+', project_plan_content[section_start + 1:], re.MULTILINE)
                        if section_end_match:
                            section_end = section_start + 1 + section_end_match.start()
                        else:
                            section_end = len(project_plan_content)
                        
                        section_name = re.sub(r'^##+\s+', '', section_header).strip()
                        section_content = project_plan_content[section_start:section_end]
                        
                        if project_plan_filename not in existing_sections_content:
                            existing_sections_content[project_plan_filename] = {}
                        existing_sections_content[project_plan_filename][section_name] = section_content
                    
                    logger.info(f"Extracted {len(existing_sections_content.get(project_plan_filename, {}))} sections from project_plan.md (active_editor)")
            
            # Then extract sections from other referenced files
            for category, file_list in referenced_context.items():
                if isinstance(file_list, list):
                    for file_doc in file_list:
                        if isinstance(file_doc, dict):
                            content = file_doc.get("content", "")
                            filename = file_doc.get("filename", "")
                            
                            if not content:
                                continue
                            
                            section_pattern = r'^(##+\s+[^\n]+)$'
                            for match in re.finditer(section_pattern, content, re.MULTILINE):
                                section_header = match.group(1)
                                section_start = match.start()
                                
                                section_end_match = re.search(r'\n##+\s+', content[section_start + 1:], re.MULTILINE)
                                if section_end_match:
                                    section_end = section_start + 1 + section_end_match.start()
                                else:
                                    section_end = len(content)
                                
                                section_name = re.sub(r'^##+\s+', '', section_header).strip()
                                section_content = content[section_start:section_end]
                                
                                if filename not in existing_sections_content:
                                    existing_sections_content[filename] = {}
                                existing_sections_content[filename][section_name] = section_content
            
            logger.info(f"Extracted {sum(len(sections) for sections in existing_sections_content.values())} existing sections for review")
            
            # Build existing content context for prompt
            existing_content_context = ""
            if existing_sections_content:
                existing_content_context = "\n\n**EXISTING SECTION CONTENT** (review this before making updates):\n"
                for filename, sections in existing_sections_content.items():
                    existing_content_context += f"\n**File: {filename}**\n"
                    for section_name, section_content in sections.items():
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
1. **FRONTMATTER UPDATES** (SPECIAL HANDLING):
   - If updating project metadata (title, status, goals, scope, etc.), use section="frontmatter"
   - Format frontmatter content as RAW YAML fields (not markdown, no code blocks)
   - Only include fields that are being updated - existing fields will be preserved
   - Example: If updating title and adding goals:
     ```
     "section": "frontmatter",
     "content": "title: New Project Title\nstatus: active\ngoals:\n  - Goal 1\n  - Goal 2"
     ```
   - Common frontmatter fields: title, status, goals, scope, files, references, budget, timeline

2. **EXTRACT ALL PROJECT DETAILS**: 
   - Requirements and specifications
   - Design decisions and approaches
   - Task lists and milestones
   - Project notes and documentation
   - Budget and resource information
   - Timeline and scheduling details

3. **MAINTAIN DOCUMENT COHERENCE**:
   - Read existing sections - understand the document's flow and structure
   - Write cohesive content - new content should read naturally with existing sections
   - Update related sections together - if updating a requirement, update all references
   - Maintain consistency - use same terminology, formatting, and level of detail
   - Preserve document structure - don't break existing section hierarchies
   - Write for humans - content should read like well-written project documentation
   - Connect ideas - link new content to existing sections with transitions

3. **ROUTE TO APPROPRIATE FILES** (use AVAILABLE FILES list above):
   - **Frontmatter/metadata updates**: Use section="frontmatter" for project_plan when updating title, status, goals, scope, etc.
   - **Requirements/specifications**: Route to files with "spec", "requirement", or "specification" in name/description
   - **Design/architecture**: Route to files with "design", "architecture", or "approach" in name/description
   - **Tasks/milestones**: Route to files with "task", "todo", "checklist", or "milestone" in name/description
   - **General project info**: Route to "project_plan" or files with "plan", "overview", or "summary" in name/description
   - **Notes/documentation**: Route to files with "note", "documentation", or "docs" in name/description
   - **If no clear match**: Use the file whose description/summary best matches the content type
   - **Multiple matches**: Split content logically across multiple files if appropriate

4. **DETECT REVISIONS AND CORRECTIONS**: 
   - **CRITICAL: REVIEW EXISTING SECTION CONTENT ABOVE**
   - **Compare new information with existing content** - decide what needs to change
   - **Revisions**: 'instead of X, use Y' or 'change X to Y' means REPLACE existing content
   - **Corrections**: 'Actually, it is X not Y' or 'That is wrong, it should be X' means CORRECT existing content
   - **Removals**: 'Remove X' or 'Delete Y' or 'We are not using Z anymore' means REMOVE content
   - **Error Detection**: If user says something contradicts existing content, DETECT the conflict and CORRECT it
   - **Always check existing content** in files before adding - if it conflicts, CORRECT it rather than creating duplicates
   - **Update all related sections** - When correcting/updating, find ALL sections that reference the changed content

5. **MATCH EXISTING SECTIONS**: Use exact section names that exist in files (check file summaries for section lists)

6. **FORMAT AS MARKDOWN**: All content must be markdown
   - Use proper headings, lists, code blocks, and tables
   - Maintain consistent formatting with existing document style
   - Write complete sentences and paragraphs, not bullet dumps

7. **BE GRANULAR BUT COHESIVE**: 
   - Break content into logical pieces - don't dump everything in one place
   - BUT ensure each piece reads well on its own and connects to related sections
   - Group related details together in coherent paragraphs

8. **ALWAYS EXTRACT**: Even if content seems small, extract it! Project details are valuable and should be saved.

9. **NEW REFERENCE FILE CREATION** (when content is too detailed for project plan):
   - **When to create**: Content is substantial (>1500 chars) AND doesn't fit existing files well
   - **Criteria**: Content is about a distinct, specific topic that warrants its own file
   - **Set flags**: "create_new_file": true, "suggested_filename": "descriptive-name.md", "file_summary": "Brief purpose"
   - **File naming**: Use lowercase-with-dashes.md format (e.g., "hvac-specifications.md", "planting-schedule.md")
   - **Automatic handling**: System will create file, add to project plan frontmatter, save content
   - **Conservative approach**: Only create if NO existing file is a good match (<20% relevance)
   - **Examples of when to create**:
     * Detailed specifications >1500 chars that don't fit in existing spec files
     * Complete task lists/checklists that are too long for project plan
     * Detailed design documents with multiple sections
     * Comprehensive notes on a specific subtopic
   - **Examples of when NOT to create**:
     * Small updates or additions (<1500 chars)
     * Content that fits well in existing files
     * General project information (belongs in project_plan)

10. **USE EXACT FILENAMES**: For existing files, always use the exact filename from AVAILABLE FILES

11. **WRITE COHESIVE CONTENT**:
    - Read the existing section content to understand context and style
    - Write new content that flows naturally from existing content
    - Use transitions and context to connect ideas
    - Maintain consistent voice and level
    - Write complete, readable paragraphs

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "routing": [
    {{
      "content_type": "requirements|design|specifications|tasks|notes|general",
      "target_file": "filename.md (exact filename from AVAILABLE FILES) or 'project_plan'",
      "section": "Exact section name (use 'frontmatter' for metadata updates, otherwise match existing sections)",
      "content": "Markdown formatted content (or raw YAML for frontmatter updates)",
      "action": "append|replace|remove",
      "create_new_file": false,  // Set to true if creating a new reference file
      "suggested_filename": "",  // Required if create_new_file is true (e.g., "hvac-specifications.md")
      "file_summary": ""  // Required if create_new_file is true (brief description of file purpose)
    }}
  ]
}}

**ACTION FIELD**:
- **append**: Add new content to a new or existing section (default if section doesn't exist)
- **replace**: Replace existing section content (use when correcting errors, updating values, or revising)
- **remove**: Remove section content entirely (use when user explicitly requests removal or content is obsolete)
"""
            
            try:
                from orchestrator.models.general_project_models import GeneralProjectUnifiedContentPlan
                structured_llm = llm.with_structured_output(GeneralProjectUnifiedContentPlan)
                result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
                save_plan = result.dict() if hasattr(result, 'dict') else result.model_dump()
            except Exception:
                # Fallback
                response_obj = await llm.ainvoke([{"role": "user", "content": prompt}])
                content = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
                parsed = self.agent._extract_json_from_response(content) or {}
                save_plan = {"routing": parsed.get("routing", [])}
            
            routing_count = len(save_plan.get('routing', []))
            logger.info(f"Extracted and routed {routing_count} content items")
            
            if routing_count == 0:
                logger.warning("No content items extracted - response may not have saveable content")
            else:
                for i, route in enumerate(save_plan.get('routing', [])[:5], 1):
                    logger.info(f"Route {i}: {route.get('content_type')} -> {route.get('target_file')} ({route.get('section')})")
            
            return {
                "save_plan": save_plan,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "response": state.get("response", {})
            }
            
        except Exception as e:
            logger.error(f"Content extraction and routing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "save_plan": None,
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "response": state.get("response", {})
            }


