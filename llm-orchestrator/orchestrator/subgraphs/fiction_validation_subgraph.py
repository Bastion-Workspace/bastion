"""
Validation Subgraph for Fiction Agents

Reusable subgraph that handles:
- Outline sync detection
- Consistency validation

Can be used by any agent needing validation.
"""

import logging
import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from orchestrator.models.editor_models import ManuscriptEdit
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


async def validate_consistency_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate generated content for potential consistency issues"""
    try:
        logger.info("Validating consistency...")
        
        structured_edit = _get_structured_edit(state)
        if not structured_edit:
            return {
                "consistency_warnings": [],
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

def build_validation_subgraph(checkpointer, llm_factory, get_datetime_context) -> StateGraph:
    """Build validation subgraph for fiction agents"""
    # Use dict for state compatibility (Dict[str, Any] is a type alias, not a valid StateGraph schema)
    from typing import Dict, Any
    subgraph = StateGraph(dict)
    
    # Create node functions with dependencies
    async def detect_outline_node(state):
        return await detect_outline_changes_node(state, llm_factory, get_datetime_context)
    
    # Add nodes
    subgraph.add_node("detect_outline_changes", detect_outline_node)
    subgraph.add_node("validate_consistency", validate_consistency_node)
    
    # Set entry point
    subgraph.set_entry_point("detect_outline_changes")
    
    # Flow
    subgraph.add_edge("detect_outline_changes", "validate_consistency")
    subgraph.add_edge("validate_consistency", END)
    
    return subgraph.compile(checkpointer=checkpointer)

