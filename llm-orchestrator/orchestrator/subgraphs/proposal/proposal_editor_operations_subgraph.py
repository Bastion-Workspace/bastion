"""
Proposal Editor Operations Subgraph

For editing existing proposals, generates targeted edit operations that
map changes back to the requirements they address.
"""

import logging
import re
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


async def compare_sections_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compare existing proposal sections with newly generated sections"""
    try:
        logger.info("Comparing proposal sections...")
        
        existing_content = state.get("editor_content", "")
        new_sections = state.get("sections", {})
        
        if not existing_content:
            logger.warning("No existing proposal content to compare")
            return {
                "section_differences": {},
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        differences = {}
        
        for section_name, new_section in new_sections.items():
            section_header = f"## {section_name.replace('_', ' ').title()}"
            
            if section_header in existing_content:
                pattern = rf"{re.escape(section_header)}\n(.*?)(?=\n##|\Z)"
                match = re.search(pattern, existing_content, re.DOTALL)
                
                if match:
                    existing_section_text = match.group(1).strip()
                    new_text = new_section.content if hasattr(new_section, 'content') else str(new_section)
                    
                    if existing_section_text != new_text:
                        differences[section_name] = {
                            "exists": True,
                            "changed": True,
                            "existing_length": len(existing_section_text),
                            "new_length": len(new_text)
                        }
                    else:
                        differences[section_name] = {"exists": True, "changed": False}
                else:
                    differences[section_name] = {"exists": True, "changed": True}
            else:
                differences[section_name] = {"exists": False, "changed": True}
        
        logger.info(f"Identified differences in {len([d for d in differences.values() if d.get('changed')])} sections")
        
        return {
            "section_differences": differences,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to compare sections: {e}")
        return {
            "section_differences": {},
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def generate_edit_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate editor operations for proposal updates"""
    try:
        logger.info("Generating editor operations...")
        
        section_differences = state.get("section_differences", {})
        new_sections = state.get("sections", {})
        
        if not section_differences:
            return {
                "editor_operations": [],
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        operations = []
        
        for section_name, diff_info in section_differences.items():
            if not diff_info.get("changed"):
                continue
            
            new_section = new_sections.get(section_name)
            if not new_section:
                continue
            
            section_header = section_name.replace('_', ' ').title()
            content = new_section.content if hasattr(new_section, 'content') else str(new_section)
            
            if diff_info.get("exists"):
                operation = {
                    "op_type": "replace_range",
                    "text": f"## {section_header}\n\n{content}",
                    "original_text": f"## {section_header}",
                    "occurrence_index": 0,
                    "section_name": section_name
                }
            else:
                operation = {
                    "op_type": "insert_after_heading",
                    "text": f"## {section_header}\n\n{content}",
                    "anchor_text": "## Terms and Conditions",
                    "section_name": section_name
                }
            
            operations.append(operation)
        
        logger.info(f"Generated {len(operations)} editor operations")
        
        return {
            "editor_operations": operations,
            "operations_count": len(operations),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to generate edit operations: {e}")
        return {
            "editor_operations": [],
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


async def map_operations_to_requirements_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Map editor operations back to the requirements they address"""
    try:
        logger.info("Mapping operations to requirements...")
        
        editor_operations = state.get("editor_operations", [])
        requirement_index = state.get("requirement_index", {})
        
        if not editor_operations:
            return {
                "operations_requirement_map": {},
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
        
        operations_map = {}
        
        section_to_reqs = {}
        for req_id, section_name in requirement_index.items():
            if section_name not in section_to_reqs:
                section_to_reqs[section_name] = []
            section_to_reqs[section_name].append(req_id)
        
        for i, op in enumerate(editor_operations):
            section_name = op.get("section_name", "")
            reqs = section_to_reqs.get(section_name, [])
            
            operations_map[f"operation_{i}"] = {
                "operation_index": i,
                "section_name": section_name,
                "requirement_ids": reqs
            }
        
        logger.info(f"Mapped {len(editor_operations)} operations to requirements")
        
        return {
            "operations_requirement_map": operations_map,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }
    
    except Exception as e:
        logger.error(f"Failed to map operations to requirements: {e}")
        return {
            "operations_requirement_map": {},
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", "")
        }


def build_proposal_editor_operations_subgraph(checkpointer, llm_factory=None, get_datetime_context=None) -> StateGraph:
    """Build editor operations subgraph"""
    workflow = StateGraph(dict)
    
    workflow.add_node("compare_sections", compare_sections_node)
    workflow.add_node("generate_edit_operations", generate_edit_operations_node)
    workflow.add_node("map_operations_to_requirements", map_operations_to_requirements_node)
    
    workflow.set_entry_point("compare_sections")
    
    workflow.add_edge("compare_sections", "generate_edit_operations")
    workflow.add_edge("generate_edit_operations", "map_operations_to_requirements")
    workflow.add_edge("map_operations_to_requirements", END)
    
    return workflow.compile(checkpointer=checkpointer)
