"""
General Project Agent - Project Plan Nodes Module
Handles proactive project plan updates with high-level summaries and decisions
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GeneralProjectPlanNodes:
    """Project plan management nodes for general project agent"""
    
    def __init__(self, agent):
        """
        Initialize with reference to main agent for access to helper methods
        
        Args:
            agent: GeneralProjectAgent instance (for _get_llm, _get_fast_model, etc.)
        """
        self.agent = agent
    
    async def update_project_plan_node(self, state) -> Dict[str, Any]:
        """
        Proactively update project plan with high-level summaries and decisions.
        Uses header-based sections (##) for single-document structure.
        
        This node updates the project plan with:
        - High-level conversation summaries
        - Decision summaries
        - Project state overview
        - Early project definition when nothing is defined yet
        """
        try:
            query = state.get("query", "")
            response = state.get("response", {})
            response_text = response.get("response", "") if isinstance(response, dict) else str(response)
            messages = state.get("messages", [])
            project_decisions = state.get("project_decisions", [])
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            active_editor = metadata.get("active_editor") or metadata.get("shared_memory", {}).get("active_editor", {})
            
            # Resolve project document ID (explicit or from active editor)
            project_plan_doc_id = state.get("project_document_id") or active_editor.get("document_id")
            existing_sections = state.get("existing_sections", [])
            
            if not project_plan_doc_id:
                logger.info("No project plan document - skipping update")
                return {
                    # ✅ CRITICAL: Preserve critical state keys
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "referenced_context": state.get("referenced_context", {}),
                    "project_decisions": state.get("project_decisions", []),
                    "existing_sections": existing_sections,
                    "project_document_id": project_plan_doc_id
                }
            
            # Get current project plan content
            from orchestrator.tools.document_tools import get_document_content_tool
            current_content = await get_document_content_tool(project_plan_doc_id, user_id)
            if current_content.startswith("Error"):
                logger.warning("Could not read project plan content - skipping update")
                return {
                    # ✅ CRITICAL: Preserve critical state keys
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "referenced_context": state.get("referenced_context", {}),
                    "project_decisions": state.get("project_decisions", []),
                    "existing_sections": existing_sections,
                    "project_document_id": project_plan_doc_id
                }
            
            # Build conversation summary (last 5-10 messages)
            conversation_summary = ""
            if messages:
                recent_messages = messages[-10:]
                for msg in recent_messages:
                    if hasattr(msg, 'content') and isinstance(msg.content, str):
                        role = "User" if hasattr(msg, 'type') and msg.type == "human" else "Agent"
                        content_preview = msg.content[:200] if len(msg.content) > 200 else msg.content
                        conversation_summary += f"{role}: {content_preview}\n"
            
            # Build decision summary
            decision_summary = ""
            if project_decisions:
                recent_decisions = project_decisions[-5:]
                for decision in recent_decisions:
                    decision_type = decision.get("decision_type", "decision")
                    decision_text = decision.get("decision_summary", "")
                    if decision_text:
                        decision_summary += f"- **{decision_type.replace('_', ' ').title()}**: {decision_text}\n"
            
            # Standard header sections for single-document structure
            standard_sections = [
                "Project Overview",
                "Supplies Needed",
                "Tasks and Milestones",
                "Design Decisions",
                "Notes and Details"
            ]
            
            # Use LLM to generate project plan update
            fast_model = self.agent._get_fast_model(state)
            llm = self.agent._get_llm(temperature=0.2, model=fast_model, state=state)
            
            sections_info = ""
            if existing_sections:
                sections_info = f"\n**EXISTING SECTIONS**: {', '.join(existing_sections)}\nUse these existing sections when possible. If a section doesn't exist, use the standard section names below."
            else:
                sections_info = "\n**NO EXISTING SECTIONS FOUND** - Use the standard section names below."
            
            prompt = f"""You are a project plan maintenance expert. Update the project plan using standardized header-based sections (##).

**CURRENT PROJECT PLAN**:
{current_content[:3000]}

{sections_info}

**STANDARD SECTIONS** (use these header names):
- ## Project Overview: Goals, scope, and high-level summary
- ## Supplies Needed: Tools, materials, parts, and hardware
- ## Tasks and Milestones: TODO items, schedule, and progress
- ## Design Decisions: Rationale for choices made during the project
- ## Notes and Details: Any other relevant information

**RECENT CONVERSATION**:
{conversation_summary[:1500]}

**RECENT DECISIONS**:
{decision_summary if decision_summary else "None"}

**CURRENT QUERY**: {query}

**TASK**: Generate updates to the project plan that:
1. **Target specific headers** - Use the standard section names above (e.g., "Supplies Needed", "Tasks and Milestones")
2. **Summarize high-level decisions** made in recent conversation (goes in "Design Decisions")
3. **Update project overview** with key information discussed (goes in "Project Overview")
4. **Maintain project state** - what's been decided, what's in progress, what's planned
5. **Early project definition** - if project plan is mostly empty/placeholder, fill it with what's been discussed
6. **Keep it high-level** - organize information under appropriate headers

**UPDATE STRATEGY**:
- Match content to the most appropriate standard section header
- If a section exists, update it; if not, create it
- **Large-scale operations** (section-level):
  - Use "replace" or "update" to replace entire section content
  - Use "append" to add new content to end of section
  - Use when making broad changes or adding substantial new information
- **Granular operations** (text-level within section):
  - Use "granular_replace" with original_text to replace specific text within a section
  - Use "granular_insert" with anchor_text to insert text after specific content
  - Use when making precise edits to existing content (e.g., updating a single item in a list, fixing a typo, changing a specific detail)
- Focus on: Project goals, design decisions, approach overview, key constraints, timeline
- Keep summaries concise and high-level

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "updates": [
    {{
      "section": "Supplies Needed",
      "content": "- Item 1\\n- Item 2",
      "action": "update|append|replace|granular_replace",
      "original_text": "Optional: specific text to replace (for granular edits within section)",
      "anchor_text": "Optional: text to insert after (for granular inserts)"
    }}
  ],
  "should_update": true
}}

**ACTION TYPES**:
- "replace": Replace entire section (large-scale)
- "append": Append to end of section (large-scale)
- "update": Update entire section (same as replace)
- "granular_replace": Replace specific text within section (granular - requires original_text)
- "granular_insert": Insert text at specific point (granular - requires anchor_text)

For granular edits, provide original_text or anchor_text to target specific content within the section.

Return ONLY the JSON object, no markdown, no code blocks."""
            
            try:
                structured_llm = llm.with_structured_output({
                    "type": "object",
                    "properties": {
                        "updates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "section": {"type": "string"},
                                    "content": {"type": "string"},
                                    "action": {"type": "string"},
                                    "original_text": {"type": "string", "description": "For granular_replace: specific text to replace within section"},
                                    "anchor_text": {"type": "string", "description": "For granular_insert: text to insert after within section"}
                                },
                                "required": ["section", "content", "action"]
                            }
                        },
                        "should_update": {"type": "boolean"}
                    }
                })
                result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
                result_dict = result if isinstance(result, dict) else result.dict() if hasattr(result, 'dict') else result.model_dump()
            except Exception as e:
                logger.warning(f"Structured output failed: {e}")
                if project_decisions:
                    result_dict = {
                        "updates": [{
                            "section": "Recent Decisions",
                            "content": decision_summary,
                            "action": "append"
                        }],
                        "should_update": True
                    }
                else:
                    result_dict = {"updates": [], "should_update": False}
            
            if not result_dict.get("should_update", False):
                logger.info("LLM determined project plan update not needed")
                return {
                    # ✅ CRITICAL: Preserve critical state keys
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "referenced_context": state.get("referenced_context", {}),
                    "project_decisions": state.get("project_decisions", [])
                }
            
            updates = result_dict.get("updates", [])
            if not updates:
                return {
                    # ✅ CRITICAL: Preserve critical state keys
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "referenced_context": state.get("referenced_context", {}),
                    "project_decisions": state.get("project_decisions", [])
                }
            
            # Apply updates to project plan
            from orchestrator.tools.project_content_tools import propose_section_update, append_project_content
            
            updated_sections = []
            for update in updates:
                section = update.get("section", "")
                content = update.get("content", "")
                action = update.get("action", "update")
                
                if not section or not content:
                    continue
                
                try:
                    if action == "replace" or action == "update":
                        await propose_section_update(
                            project_plan_doc_id, current_content, section, content, user_id,
                            active_editor=active_editor,
                            auto_apply_if_closed=True,
                            add_timestamp=False
                        )
                    else:  # append
                        await append_project_content(
                            project_plan_doc_id, section, content, user_id,
                            active_editor=active_editor,
                            auto_apply_if_closed=True
                        )
                    updated_sections.append(section)
                    logger.info(f"Updated project plan section: {section}")
                except Exception as e:
                    logger.warning(f"Failed to update project plan section '{section}': {e}")
            
            if updated_sections:
                logger.info(f"Project plan updated with {len(updated_sections)} section(s): {', '.join(updated_sections)}")
            
            return {
                "project_plan_updated": True,
                "updated_sections": updated_sections,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "project_decisions": state.get("project_decisions", []),
                "existing_sections": existing_sections,
                "project_document_id": project_plan_doc_id
            }
            
        except Exception as e:
            logger.error(f"Project plan update failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "project_decisions": state.get("project_decisions", []),
                "existing_sections": state.get("existing_sections", []),
                "project_document_id": state.get("project_document_id")
            }


