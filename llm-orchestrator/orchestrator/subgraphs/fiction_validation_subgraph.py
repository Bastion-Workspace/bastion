"""
Validation Subgraph for Fiction Agents

Reusable subgraph that handles:
- Outline sync detection
- Continuity state loading
- Consistency validation
- Continuity validation
- Continuity state updates

Can be used by any agent needing continuity validation.
"""

import logging
import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from orchestrator.models.continuity_models import ContinuityState
from orchestrator.models.editor_models import ManuscriptEdit
from orchestrator.services.fiction_continuity_tracker import FictionContinuityTracker
from orchestrator.utils.fiction_utilities import unwrap_json_response

logger = logging.getLogger(__name__)


# ============================================
# Pydantic Models for Structured Outputs
# ============================================

class OutlineDiscrepancy(BaseModel):
    """A single discrepancy between outline and manuscript"""
    type: str = Field(description="Type of discrepancy: missing_beat, changed_beat, character_action_mismatch, story_progression_issue")
    outline_expectation: str = Field(description="What the outline says should happen")
    manuscript_current: str = Field(description="What the manuscript currently has (or 'missing')")
    severity: str = Field(description="Severity: critical, major, minor")
    suggestion: str = Field(description="Suggestion for how to resolve the discrepancy")


class OutlineSyncAnalysis(BaseModel):
    """Analysis of outline vs manuscript alignment"""
    needs_sync: bool = Field(description="Whether the manuscript needs updates to match the outline")
    discrepancies: List[OutlineDiscrepancy] = Field(default_factory=list, description="List of discrepancies found")
    summary: str = Field(description="Summary of the alignment analysis")


# ============================================
# State Schema
# ============================================

# Use Dict[str, Any] for compatibility with main agent state
FictionValidationState = Dict[str, Any]


# ============================================
# Helper Functions
# ============================================

def _get_structured_edit(state: Dict[str, Any]) -> Optional[ManuscriptEdit]:
    """Safely extract and validate structured_edit from state"""
    edit_dict = state.get("structured_edit")
    if not edit_dict:
        return None
    
    if isinstance(edit_dict, ManuscriptEdit):
        return edit_dict
    
    if not isinstance(edit_dict, dict):
        return None
    
    try:
        return ManuscriptEdit(**edit_dict)
    except Exception:
        return None


def _get_continuity_state(state: Dict[str, Any]) -> Optional[ContinuityState]:
    """Safely extract and validate continuity_state from state"""
    continuity_dict = state.get("continuity_state")
    if not continuity_dict:
        return None
    
    if isinstance(continuity_dict, ContinuityState):
        return continuity_dict
    
    if not isinstance(continuity_dict, dict):
        return None
    
    try:
        return ContinuityState(**continuity_dict)
    except Exception:
        return None


# _unwrap_json_response is now imported from orchestrator.utils.fiction_utilities


# ============================================
# Node Functions
# ============================================

async def detect_outline_changes_node(state: Dict[str, Any], llm_factory, get_datetime_context) -> Dict[str, Any]:
    """Detect if outline has changed and manuscript needs updates"""
    try:
        logger.info("Detecting outline changes...")
        
        current_chapter_text = state.get("current_chapter_text", "")
        outline_current_chapter_text = state.get("outline_current_chapter_text")
        # Support both chapter_number and current_chapter_number
        current_chapter_number = state.get("chapter_number") or state.get("current_chapter_number")
        current_request = state.get("current_request", "").lower()
        
        # Skip if no outline or no existing chapter text
        if not outline_current_chapter_text or len(current_chapter_text.strip()) < 100:
            return {
                "outline_sync_analysis": None,
                "outline_needs_sync": False,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        # Outline sync is OPT-IN only
        sync_keywords = [
            "sync with outline", "check outline", "align with outline", "outline sync",
            "match outline", "follow outline", "outline alignment", "compare to outline",
            "outline discrepancies", "outline changes", "update to match outline"
        ]
        has_sync_request = any(kw in current_request for kw in sync_keywords)
        
        if not has_sync_request:
            logger.info("No explicit outline sync request - skipping outline sync detection")
            return {
                "outline_sync_analysis": None,
                "outline_needs_sync": False,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        logger.info("User requested outline sync - checking alignment")
        
        # Use LLM to compare outline to manuscript
        llm = llm_factory(temperature=0.2)
        
        logger.info(f"📖 Outline sync: Comparing FULL Chapter {current_chapter_number} ({len(current_chapter_text)} chars) with outline")
        
        comparison_prompt = f"""Compare the current outline for Chapter {current_chapter_number} with the existing manuscript chapter.

**CURRENT OUTLINE FOR CHAPTER {current_chapter_number}**:
{outline_current_chapter_text}

**EXISTING MANUSCRIPT CHAPTER {current_chapter_number} (FULL CHAPTER - NO TRUNCATION)**:
{current_chapter_text}

**YOUR TASK**: Compare the outline to the manuscript and identify ONLY MAJOR discrepancies.

**OUTPUT FORMAT**: Return ONLY valid JSON:
{{
  "needs_sync": true|false,
  "discrepancies": [
    {{
      "type": "missing_beat|changed_beat|character_action_mismatch|story_progression_issue",
      "outline_expectation": "What the outline says should happen",
      "manuscript_current": "What the manuscript currently has (or 'missing')",
      "severity": "high|medium|low",
      "suggestion": "Specific revision needed"
    }}
  ],
  "summary": "Brief summary of what needs updating"
}}

Return ONLY the JSON object, no markdown, no code blocks."""
        
        datetime_context = get_datetime_context()
        messages = [
            SystemMessage(content="You are an outline-manuscript comparison tool. Compare outlines to manuscripts and identify ONLY major discrepancies."),
            SystemMessage(content=datetime_context),
            HumanMessage(content=comparison_prompt)
        ]
        
        try:
            structured_llm = llm.with_structured_output(OutlineSyncAnalysis)
            result = await structured_llm.ainvoke(messages)
            
            if isinstance(result, OutlineSyncAnalysis):
                sync_analysis = result.model_dump()
            elif isinstance(result, dict):
                sync_analysis = result
            else:
                sync_analysis = result.dict() if hasattr(result, 'dict') else result.model_dump()
                
        except Exception as e:
            logger.warning(f"Structured output failed, using fallback: {e}")
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            content = unwrap_json_response(content)
            try:
                sync_analysis_dict = json.loads(content)
                sync_analysis_obj = OutlineSyncAnalysis(**sync_analysis_dict)
                sync_analysis = sync_analysis_obj.model_dump()
            except Exception as parse_error:
                logger.error(f"Failed to parse outline sync analysis: {parse_error}")
                return {
                    "outline_sync_analysis": None,
                    "outline_needs_sync": False,
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "manuscript": state.get("manuscript", ""),
                    "filename": state.get("filename", ""),
                    "current_chapter_text": state.get("current_chapter_text", ""),
                    "current_chapter_number": state.get("current_chapter_number"),
                    "chapter_ranges": state.get("chapter_ranges", []),
                    "current_request": state.get("current_request", ""),
                    "active_editor": state.get("active_editor", {}),
                    "frontmatter": state.get("frontmatter", {}),
                    "structured_edit": state.get("structured_edit"),
                }
        
        needs_sync = sync_analysis.get("needs_sync", False)
        discrepancies = sync_analysis.get("discrepancies", [])
        
        if needs_sync and discrepancies:
            logger.info(f"⚠️ Outline sync needed: {len(discrepancies)} discrepancy(ies) detected")
        else:
            logger.info("✅ Manuscript appears in sync with outline")
        
        return {
            "outline_sync_analysis": sync_analysis,
            "outline_needs_sync": needs_sync,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }
        
    except Exception as e:
        logger.error(f"Failed to detect outline changes: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "outline_sync_analysis": None,
            "outline_needs_sync": False,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }


async def load_continuity_node(state: Dict[str, Any], continuity_tracker: FictionContinuityTracker) -> Dict[str, Any]:
    """Load continuity state from companion document or create if missing"""
    try:
        logger.info("Loading continuity state...")
        
        # Skip continuity for simple requests
        is_simple = state.get("is_simple_request", False)
        if is_simple:
            logger.info("⚡ Simple request - skipping continuity loading")
            return {
                "continuity_state": None,
                "continuity_document_id": None,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        frontmatter = state.get("frontmatter", {})
        active_editor = state.get("active_editor", {})
        # Get user_id from state, with fallback to metadata
        user_id = state.get("user_id", "system")
        metadata = state.get("metadata", {})
        # If user_id is "system" (fallback), try to get it from metadata
        if user_id == "system" and metadata:
            user_id = metadata.get("user_id", "system")
        # Log warning if still "system" when creating user document
        if user_id == "system":
            logger.warning("⚠️ user_id is 'system' when creating continuity document - RLS may not work correctly")
        filename = state.get("filename", "")
        # Support both manuscript_content and manuscript
        manuscript = state.get("manuscript_content") or state.get("manuscript", "")
        
        # Check if continuity reference exists in frontmatter
        continuity_ref = frontmatter.get("continuity")
        continuity_doc = None
        
        if continuity_ref:
            logger.info(f"Found continuity reference in frontmatter: {continuity_ref}")
            
            from orchestrator.tools.reference_file_loader import load_file_by_path
            
            continuity_doc = await load_file_by_path(
                ref_path=continuity_ref if isinstance(continuity_ref, str) else continuity_ref[0],
                user_id=user_id,
                active_editor=active_editor
            )
            
            if continuity_doc and continuity_doc.get("content"):
                try:
                    continuity_data = json.loads(continuity_doc["content"])
                    continuity_state = ContinuityState(**continuity_data)
                    logger.info(f"Loaded continuity state (last analyzed: Chapter {continuity_state.last_analyzed_chapter})")
                    
                    return {
                        "continuity_state": continuity_state.dict(),
                        "continuity_document_id": continuity_doc.get("document_id"),
                        "metadata": state.get("metadata", {}),
                        "user_id": state.get("user_id", "system"),
                        "shared_memory": state.get("shared_memory", {}),
                        "messages": state.get("messages", []),
                        "query": state.get("query", ""),
                        "manuscript": state.get("manuscript", ""),
                        "filename": state.get("filename", ""),
                        "current_chapter_text": state.get("current_chapter_text", ""),
                        "current_chapter_number": state.get("current_chapter_number"),
                        "chapter_ranges": state.get("chapter_ranges", []),
                        "current_request": state.get("current_request", ""),
                        "active_editor": state.get("active_editor", {}),
                        "frontmatter": state.get("frontmatter", {}),
                        "structured_edit": state.get("structured_edit"),
                    }
                except Exception as e:
                    logger.error(f"Failed to parse continuity state: {e}")
        
        # Try to find .continuity.json file in same directory
        if not continuity_doc:
            from orchestrator.tools.reference_file_loader import load_file_by_path
            from pathlib import Path
            
            continuity_filename = filename.replace(".md", ".continuity.json") if filename.endswith(".md") else f"{filename}.continuity.json"
            
            logger.info(f"Trying to find continuity file in same directory: {continuity_filename}")
            
            continuity_doc = await load_file_by_path(
                ref_path=f"./{continuity_filename}",
                user_id=user_id,
                active_editor=active_editor
            )
            
            if continuity_doc and continuity_doc.get("content"):
                try:
                    continuity_data = json.loads(continuity_doc["content"])
                    continuity_state = ContinuityState(**continuity_data)
                    logger.info(f"✅ Found and loaded continuity file: {continuity_filename} (last analyzed: Chapter {continuity_state.last_analyzed_chapter})")
                    
                    return {
                        "continuity_state": continuity_state.dict(),
                        "continuity_document_id": continuity_doc.get("document_id"),
                        "metadata": state.get("metadata", {}),
                        "user_id": state.get("user_id", "system"),
                        "shared_memory": state.get("shared_memory", {}),
                        "messages": state.get("messages", []),
                        "query": state.get("query", ""),
                        "manuscript": state.get("manuscript", ""),
                        "filename": state.get("filename", ""),
                        "current_chapter_text": state.get("current_chapter_text", ""),
                        "current_chapter_number": state.get("current_chapter_number"),
                        "chapter_ranges": state.get("chapter_ranges", []),
                        "current_request": state.get("current_request", ""),
                        "active_editor": state.get("active_editor", {}),
                        "frontmatter": state.get("frontmatter", {}),
                        "structured_edit": state.get("structured_edit"),
                    }
                except Exception as e:
                    logger.error(f"Failed to parse continuity state from auto-discovered file: {e}")
        
        # No continuity file exists - analyze entire manuscript
        logger.info("No continuity file found - analyzing manuscript for continuity tracking...")
        
        characters_bodies = state.get("characters_bodies", [])
        outline_body = state.get("outline_body")
        
        # Extract continuity from entire manuscript
        continuity_state = await continuity_tracker.extract_continuity_from_manuscript(
            manuscript_text=manuscript,
            character_profiles=characters_bodies,
            outline_body=outline_body,
            agent_state=state
        )
        
        continuity_state.manuscript_filename = filename
        continuity_state.user_id = user_id
        
        # Create continuity document
        continuity_filename = filename.replace(".md", ".continuity.json") if filename.endswith(".md") else f"{filename}.continuity.json"
        
        from orchestrator.tools.file_creation_tools import create_user_file_tool
        from orchestrator.utils.frontmatter_utils import add_to_frontmatter_list
        
        folder_id = active_editor.get("folder_id") if active_editor else None
        folder_path = None
        
        if not folder_id and active_editor and active_editor.get("canonical_path"):
            from pathlib import Path
            canonical_path = active_editor.get("canonical_path")
            try:
                path_parts = Path(canonical_path).parts
                if "Users" in path_parts:
                    users_idx = path_parts.index("Users")
                    if users_idx + 2 < len(path_parts) - 1:
                        folder_parts = path_parts[users_idx + 2:-1]
                        if folder_parts:
                            folder_path = "/".join(folder_parts)
            except Exception:
                pass
        
        if not folder_id and not folder_path:
            logger.warning("⚠️ Could not determine folder_id or folder_path - skipping continuity file creation")
            return {
                "continuity_state": continuity_state.dict(),
                "continuity_document_id": None,
                "task_status": "complete",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        continuity_json = json.dumps(continuity_state.dict(), indent=2)
        create_result = await create_user_file_tool(
            filename=continuity_filename,
            content=continuity_json,
            folder_id=folder_id,
            folder_path=folder_path,
            title=f"Continuity tracking for {filename}",
            user_id=user_id
        )
        
        if create_result.get("success"):
            continuity_doc_id = create_result.get("document_id")
            logger.info(f"Created continuity file: {continuity_filename} (document_id: {continuity_doc_id})")
            
            # Update manuscript frontmatter to reference continuity file
            manuscript_doc_id = active_editor.get("document_id")
            if manuscript_doc_id:
                from orchestrator.tools.document_tools import get_document_content_tool
                manuscript_content = await get_document_content_tool(manuscript_doc_id, user_id)
                
                updated_content, success = await add_to_frontmatter_list(
                    content=manuscript_content,
                    list_key="continuity",
                    new_items=[f"./{continuity_filename}"]
                )
                
                if success:
                    from orchestrator.tools.document_editing_tools import update_document_content_tool
                    await update_document_content_tool(
                        document_id=manuscript_doc_id,
                        content=updated_content,
                        user_id=user_id,
                        append=False
                    )
                    logger.info("Updated manuscript frontmatter with continuity reference")
            
            return {
                "continuity_state": continuity_state.dict(),
                "continuity_document_id": continuity_doc_id,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        else:
            logger.error(f"Failed to create continuity file: {create_result.get('error')}")
            return {
                "continuity_state": continuity_state.dict(),
                "continuity_document_id": None,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
    except Exception as e:
        logger.error(f"Failed to load continuity state: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "continuity_state": None,
            "continuity_document_id": None,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }


async def validate_consistency_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate generated content for potential consistency issues"""
    try:
        logger.info("Validating consistency...")
        
        structured_edit = _get_structured_edit(state)
        if not structured_edit:
            return {
                "consistency_warnings": [],
                "continuity_state": state.get("continuity_state"),
                "continuity_document_id": state.get("continuity_document_id"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        operations = structured_edit.operations
        if not operations:
            return {
                "consistency_warnings": [],
                "continuity_state": state.get("continuity_state"),
                "continuity_document_id": state.get("continuity_document_id"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        # Extract generated text
        generated_texts = [op.text for op in operations if op.text]
        if not generated_texts:
            return {
                "consistency_warnings": [],
                "continuity_state": state.get("continuity_state"),
                "continuity_document_id": state.get("continuity_document_id"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        combined_text = "\n\n".join(generated_texts)
        
        warnings = []
        
        # Check 1: Manuscript continuity - look for potential contradictions
        manuscript = state.get("manuscript_content", "")
        if manuscript and combined_text:
            if "## Chapter" in combined_text and "## Chapter" in manuscript:
                existing_chapters = set(re.findall(r'## Chapter (\d+)', manuscript))
                new_chapters = set(re.findall(r'## Chapter (\d+)', combined_text))
                duplicates = existing_chapters & new_chapters
                if duplicates:
                    warnings.append(
                        f"⚠️ Chapter number collision: Chapter(s) {', '.join(duplicates)} "
                        f"already exist in manuscript"
                    )
        
        # Check 2: Style consistency
        style_body = state.get("style_body")
        if style_body:
            if "present tense" in style_body.lower() and " had " in combined_text.lower():
                warnings.append("⚠️ Possible tense inconsistency: Style guide specifies present tense")
            elif "past tense" in style_body.lower() and any(
                combined_text.count(f" {verb} ") > 3 
                for verb in ["am", "is", "are"]
            ):
                warnings.append("⚠️ Possible tense inconsistency: Style guide specifies past tense")
        
        # Check 3: Universe rules
        rules_body = state.get("rules_body")
        if rules_body and combined_text:
            if "no magic" in rules_body.lower() and any(
                word in combined_text.lower() 
                for word in ["spell", "magic", "enchant", "wizard"]
            ):
                warnings.append("⚠️ Possible universe rule violation: Magic use detected but rules forbid it")
        
        return {
            "consistency_warnings": warnings,
            "continuity_state": state.get("continuity_state"),
            "continuity_document_id": state.get("continuity_document_id"),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }
        
    except Exception as e:
        logger.error(f"Consistency validation failed: {e}")
        return {
            "consistency_warnings": [],
            "continuity_state": state.get("continuity_state"),
            "continuity_document_id": state.get("continuity_document_id"),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }


async def validate_continuity_node(state: Dict[str, Any], continuity_tracker: FictionContinuityTracker) -> Dict[str, Any]:
    """Validate generated content against continuity"""
    try:
        logger.info("Validating content against continuity...")
        
        continuity_state = _get_continuity_state(state)
        if not continuity_state:
            logger.info("No continuity state - skipping validation")
            return {
                "continuity_violations": [],
                "continuity_state": state.get("continuity_state"),
                "continuity_document_id": state.get("continuity_document_id"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        structured_edit = _get_structured_edit(state)
        if not structured_edit:
            logger.info("No structured_edit - skipping continuity validation")
            return {
                "continuity_violations": [],
                "continuity_state": state.get("continuity_state"),
                "continuity_document_id": state.get("continuity_document_id"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        operations = structured_edit.operations
        if not operations:
            logger.info("No operations - skipping continuity validation")
            return {
                "continuity_violations": [],
                "continuity_state": state.get("continuity_state"),
                "continuity_document_id": state.get("continuity_document_id"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        # Combine generated text
        new_content = "\n\n".join([op.text for op in operations if op.text])
        # Support both chapter_number and current_chapter_number
        target_chapter = state.get("chapter_number") or state.get("current_chapter_number")
        
        if not new_content or not new_content.strip():
            logger.info("No new content to validate - skipping continuity validation")
            return {
                "continuity_violations": [],
                "continuity_state": state.get("continuity_state"),
                "continuity_document_id": state.get("continuity_document_id"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        if target_chapter is None:
            logger.warning("Cannot validate continuity - chapter number not available")
            return {
                "continuity_violations": [],
                "continuity_state": state.get("continuity_state"),
                "continuity_document_id": state.get("continuity_document_id"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        # Validate
        validation_result = await continuity_tracker.validate_new_content(
            new_content=new_content,
            target_chapter_number=target_chapter,
            continuity_state=continuity_state,
            character_profiles=state.get("characters_bodies", []),
            agent_state=state
        )
        
        # Log violations
        if validation_result.violations:
            logger.warning(f"⚠️ Continuity violations detected: {len(validation_result.violations)}")
            for violation in validation_result.violations:
                logger.warning(f"  - {violation.severity.upper()}: {violation.description}")
        
        return {
            "continuity_violations": [v.dict() for v in validation_result.violations],
            "continuity_state": state.get("continuity_state"),
            "continuity_document_id": state.get("continuity_document_id"),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }
        
    except Exception as e:
        logger.error(f"Failed to validate continuity: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "continuity_violations": [],
            "continuity_state": state.get("continuity_state"),
            "continuity_document_id": state.get("continuity_document_id"),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }


async def update_continuity_node(state: Dict[str, Any], continuity_tracker: FictionContinuityTracker) -> Dict[str, Any]:
    """Update continuity state with new chapter"""
    try:
        logger.info("Updating continuity state...")
        
        # Skip continuity updates for simple requests
        is_simple = state.get("is_simple_request", False)
        if is_simple:
            logger.info("⚡ Simple request - skipping continuity update")
            return {
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        task_status = state.get("task_status", "")
        if task_status == "error":
            return {
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        current_chapter_text = state.get("current_chapter_text", "")
        # Support both chapter_number and current_chapter_number
        current_chapter_number = state.get("chapter_number") or state.get("current_chapter_number")
        continuity_doc_id = state.get("continuity_document_id")
        
        if not current_chapter_text or not current_chapter_number:
            return {
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        if not continuity_doc_id:
            logger.warning("No continuity document ID - cannot update")
            return {
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
                "active_editor": state.get("active_editor", {}),
                "frontmatter": state.get("frontmatter", {}),
                "structured_edit": state.get("structured_edit"),
            }
        
        # Load existing state
        existing_state = _get_continuity_state(state)
        
        # Extract continuity from current chapter
        updated_state = await continuity_tracker.extract_continuity_from_chapter(
            chapter_text=current_chapter_text,
            chapter_number=current_chapter_number,
            existing_state=existing_state,
            character_profiles=state.get("characters_bodies", []),
            outline_body=state.get("outline_body"),
            agent_state=state
        )
        
        updated_state.manuscript_filename = state.get("filename", "")
        updated_state.user_id = state.get("user_id", "")
        
        logger.info(f"Updated continuity state (now tracking {len(updated_state.character_states)} characters, {len(updated_state.plot_threads)} threads)")
        
        # Update continuity document
        user_id = state.get("user_id", "system")
        continuity_json = json.dumps(updated_state.dict(), indent=2)
        
        from orchestrator.tools.document_editing_tools import update_document_content_tool
        update_result = await update_document_content_tool(
            document_id=continuity_doc_id,
            content=continuity_json,
            user_id=user_id,
            append=False
        )
        
        if update_result.get("success"):
            logger.info("Updated continuity document")
        else:
            logger.warning(f"Failed to update continuity document: {update_result.get('error')}")
        
        return {
            "updated_continuity": updated_state.dict(),
            "continuity_state": updated_state.dict(),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }
        
    except Exception as e:
        logger.error(f"Failed to update continuity: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
            "active_editor": state.get("active_editor", {}),
            "frontmatter": state.get("frontmatter", {}),
            "structured_edit": state.get("structured_edit"),
        }


# ============================================
# Subgraph Builder
# ============================================

def build_validation_subgraph(checkpointer, llm_factory, get_datetime_context, continuity_tracker: FictionContinuityTracker) -> StateGraph:
    """Build validation subgraph for fiction agents"""
    # Use Dict[str, Any] for state compatibility
    from typing import Dict, Any
    subgraph = StateGraph(Dict[str, Any])
    
    # Create node functions with dependencies
    async def detect_outline_node(state):
        return await detect_outline_changes_node(state, llm_factory, get_datetime_context)
    
    async def load_continuity_node_wrapper(state):
        return await load_continuity_node(state, continuity_tracker)
    
    async def validate_continuity_node_wrapper(state):
        return await validate_continuity_node(state, continuity_tracker)
    
    async def update_continuity_node_wrapper(state):
        return await update_continuity_node(state, continuity_tracker)
    
    # Add nodes
    subgraph.add_node("detect_outline_changes", detect_outline_node)
    subgraph.add_node("load_continuity", load_continuity_node_wrapper)
    subgraph.add_node("validate_consistency", validate_consistency_node)
    subgraph.add_node("validate_continuity", validate_continuity_node_wrapper)
    subgraph.add_node("update_continuity", update_continuity_node_wrapper)
    
    # Set entry point
    subgraph.set_entry_point("detect_outline_changes")
    
    # Flow
    subgraph.add_edge("detect_outline_changes", "load_continuity")
    subgraph.add_edge("load_continuity", "validate_consistency")
    subgraph.add_edge("validate_consistency", "validate_continuity")
    subgraph.add_edge("validate_continuity", "update_continuity")
    subgraph.add_edge("update_continuity", END)
    
    return subgraph.compile(checkpointer=checkpointer)

