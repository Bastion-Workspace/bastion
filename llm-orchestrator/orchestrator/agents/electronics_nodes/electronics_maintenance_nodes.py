"""
Electronics Agent - Maintenance Nodes Module
Handles documentation maintenance operations based on maintenance plans and verification results
"""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ElectronicsMaintenanceNodes:
    """Documentation maintenance nodes for electronics agent"""
    
    def __init__(self, agent):
        """
        Initialize with reference to main agent for access to helper methods
        
        Args:
            agent: ElectronicsAgent instance (for _get_llm, _get_fast_model, etc.)
        """
        self.agent = agent
    
    async def execute_maintenance_node(self, state) -> Dict[str, Any]:
        """
        Execute documentation maintenance operations based on maintenance plan and verification results.
        
        Uses LLM to determine the best way to execute maintenance operations:
        - Remove obsolete sections
        - Update outdated content
        - Archive superseded information
        """
        try:
            maintenance_plan = state.get("documentation_maintenance_plan", {})
            verification_result = state.get("documentation_verification_result", {})
            user_id = state.get("user_id", "")
            
            maintenance_items = maintenance_plan.get("maintenance_items", [])
            inconsistencies = verification_result.get("inconsistencies", [])
            
            if not maintenance_items and not inconsistencies:
                logger.info("🔌 No maintenance operations needed")
                return {}
            
            # Combine maintenance items from plan and inconsistencies from verification
            all_maintenance_ops = []
            
            # Add items from maintenance plan
            for item in maintenance_items:
                all_maintenance_ops.append({
                    "file": item.get("file", ""),
                    "section": item.get("section", ""),
                    "action": item.get("action", "update"),
                    "reason": item.get("reason", ""),
                    "suggested_content": item.get("suggested_content")
                })
            
            # Add items from inconsistencies
            for inconsistency in inconsistencies:
                all_maintenance_ops.append({
                    "file": inconsistency.get("file", ""),
                    "section": inconsistency.get("section", ""),
                    "action": "update",  # Inconsistencies need updates
                    "reason": inconsistency.get("description", ""),
                    "suggested_fix": inconsistency.get("suggested_fix", "")
                })
            
            logger.info(f"🔌 Executing {len(all_maintenance_ops)} maintenance operation(s)")
            
            # Execute maintenance operations
            from orchestrator.tools.document_tools import get_document_content_tool
            from orchestrator.tools.document_editing_tools import update_document_content_tool
            from orchestrator.tools.project_content_tools import propose_section_update
            
            executed_ops = []
            
            for op in all_maintenance_ops:
                try:
                    file_path = op.get("file", "")
                    section = op.get("section", "")
                    action = op.get("action", "update")
                    
                    if not file_path or not section:
                        continue
                    
                    # Resolve document_id from file path
                    referenced_context = state.get("referenced_context", {})
                    doc_id = None
                    
                    # Try to find document_id from referenced_context
                    for ref_path, ref_info in referenced_context.items():
                        if file_path in ref_path or ref_path in file_path:
                            if isinstance(ref_info, dict):
                                doc_id = ref_info.get("document_id")
                            break
                    
                    if not doc_id:
                        logger.warning(f"⚠️ Could not resolve document_id for {file_path} - skipping")
                        continue
                    
                    # Get existing content
                    existing_content = await get_document_content_tool(doc_id, user_id)
                    if existing_content.startswith("Error"):
                        logger.warning(f"⚠️ Could not read content for {file_path}")
                        continue
                    
                    # Execute action based on type
                    if action == "remove":
                        # Remove section - replace with removal note
                        from orchestrator.tools.project_content_tools import propose_section_update
                        removal_note = f"*This section was removed on {datetime.now().strftime('%Y-%m-%d')}. Reason: {op.get('reason', 'Obsolete')}.*"
                        await propose_section_update(
                            doc_id, existing_content, section, removal_note, user_id,
                            active_editor=state.get("metadata", {}).get("active_editor"),
                            auto_apply_if_closed=True,
                            add_timestamp=False
                        )
                        executed_ops.append(f"Removed section '{section}' from {file_path}")
                        logger.info(f"✅ Removed section '{section}' from {file_path}")
                    
                    elif action == "update":
                        # Update section with suggested content
                        suggested_content = op.get("suggested_content") or op.get("suggested_fix", "")
                        if suggested_content:
                            await propose_section_update(
                                doc_id, existing_content, section, suggested_content, user_id,
                                active_editor=state.get("metadata", {}).get("active_editor"),
                                auto_apply_if_closed=True,
                                add_timestamp=True
                            )
                            executed_ops.append(f"Updated section '{section}' in {file_path}")
                            logger.info(f"✅ Updated section '{section}' in {file_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to execute maintenance operation: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            if executed_ops:
                logger.info(f"✅ Executed {len(executed_ops)} maintenance operation(s)")
                return {
                    "maintenance_executed": True,
                    "operations": executed_ops
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Maintenance execution failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

