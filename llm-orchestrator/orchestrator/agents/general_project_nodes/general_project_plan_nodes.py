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
            
            project_plan_doc_id = active_editor.get("document_id")
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
                    "project_decisions": state.get("project_decisions", [])
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
                    "project_decisions": state.get("project_decisions", [])
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
            
            # Use LLM to generate project plan update
            fast_model = self.agent._get_fast_model(state)
            llm = self.agent._get_llm(temperature=0.2, model=fast_model, state=state)
            
            prompt = f"""You are a project plan maintenance expert. Update the project plan with high-level summaries and decisions.

**CURRENT PROJECT PLAN**:
{current_content[:3000]}

**RECENT CONVERSATION**:
{conversation_summary[:1500]}

**RECENT DECISIONS**:
{decision_summary if decision_summary else "None"}

**CURRENT QUERY**: {query}

**TASK**: Generate updates to the project plan that:
1. **Summarize high-level decisions** made in recent conversation
2. **Update project overview** with key information discussed
3. **Maintain project state** - what's been decided, what's in progress, what's planned
4. **Early project definition** - if project plan is mostly empty/placeholder, fill it with what's been discussed
5. **Keep it high-level** - don't include detailed specs (those go in specific files)

**UPDATE STRATEGY**:
- If project plan has placeholder sections, replace them with actual information
- If project plan has existing sections, update them with new information
- Focus on: Project goals, design decisions, approach overview, key constraints, timeline
- Keep summaries concise and high-level

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "updates": [
    {{
      "section": "Section name (e.g., 'Project Overview', 'Key Decisions', 'Design Approach')",
      "content": "Markdown formatted content to update this section",
      "action": "update|append|replace"
    }}
  ],
  "should_update": true
}}

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
                                    "action": {"type": "string"}
                                }
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
                "project_decisions": state.get("project_decisions", [])
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
                "project_decisions": state.get("project_decisions", [])
            }


