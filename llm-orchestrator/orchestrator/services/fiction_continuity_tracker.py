"""
Fiction Plot Continuity Tracker

Tracks character states, plot threads, timeline, and world state across chapters
to ensure narrative consistency. Uses LLM-based extraction and validation.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.models.continuity_models import (
    ContinuityState,
    CharacterState,
    PlotThread,
    TimeMarker,
    WorldStateChange,
    UnresolvedTension,
    ContinuityViolation,
    ContinuityValidationResult
)
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Chapter pattern for finding chapters in manuscript
CHAPTER_PATTERN = re.compile(r"^##\s+Chapter\s+(\d+)\b.*$", re.MULTILINE)


class FictionContinuityTracker:
    """
    Tracks plot continuity across fiction manuscript chapters.
    
    Uses LLM-based extraction to maintain state of:
    - Character locations and emotional states
    - Active plot threads
    - Timeline progression
    - World state changes
    - Unresolved tensions
    
    Validates new content against tracked state to catch continuity errors.
    """
    
    def __init__(self, llm_factory):
        """
        Initialize tracker with LLM factory function.
        
        Args:
            llm_factory: Function that returns configured LLM instance
                        (typically agent._get_llm method)
        """
        self._llm_factory = llm_factory
        logger.info("Fiction Continuity Tracker initialized")
    
    async def extract_continuity_from_chapter(
        self,
        chapter_text: str,
        chapter_number: int,
        existing_state: Optional[ContinuityState] = None,
        character_profiles: Optional[List[str]] = None,
        outline_body: Optional[str] = None,
        agent_state: Optional[Dict[str, Any]] = None
    ) -> ContinuityState:
        """
        Extract continuity state from a chapter using LLM analysis.
        
        Args:
            chapter_text: Full text of the chapter
            chapter_number: Chapter number being analyzed
            existing_state: Previous continuity state to update (if any)
            character_profiles: Character profile documents for reference
            outline_body: Story outline for plot thread context
            
        Returns:
            Updated ContinuityState
        """
        logger.info(f"Extracting continuity from Chapter {chapter_number}...")
        
        # Build extraction prompt
        prompt = self._build_extraction_prompt(
            chapter_text=chapter_text,
            chapter_number=chapter_number,
            existing_state=existing_state,
            character_profiles=character_profiles,
            outline_body=outline_body
        )
        
        # Use LLM to extract state (pass agent_state to get user_chat_model if available)
        llm = self._llm_factory(temperature=0.2, state=agent_state)  # Lower temp for factual extraction
        
        system_prompt = (
            "You are a continuity tracking assistant for fiction manuscripts. "
            "You MUST respond with ONLY valid JSON - no prose, no explanations, no markdown code fences. "
            "Your response must be parseable JSON that matches the exact schema provided. "
            "Double-check your JSON for syntax errors (trailing commas, unescaped quotes, etc.) before responding."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = await llm.ainvoke(messages)
        content = response.content
        
        # Log first part of response for debugging JSON issues
        logger.debug(f"Continuity extraction response (first 300 chars): {content[:300]}")
        
        # Parse JSON response with robust error handling
        try:
            json_text = self._extract_json_from_response(content)
            extracted = json.loads(json_text)
            
            # Validate and fix data before creating Pydantic models
            extracted = self._validate_and_fix_continuity_data(extracted, chapter_number)
            
            # Merge with existing state
            if existing_state:
                return self._merge_continuity_states(existing_state, extracted, chapter_number)
            else:
                # Create new state
                return ContinuityState(
                    manuscript_filename="",  # Will be set by caller
                    user_id="",  # Will be set by caller
                    last_analyzed_chapter=chapter_number,
                    **extracted
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from continuity extraction: {e}")
            if 'json_text' in locals():
                logger.error(f"JSON text length: {len(json_text)} chars")
                logger.error(f"JSON text (first 1000 chars): {json_text[:1000]}")
                # Log the problematic area around the error
                if hasattr(e, 'pos') and e.pos:
                    error_pos = e.pos
                    start = max(0, error_pos - 200)
                    end = min(len(json_text), error_pos + 200)
                    logger.error(f"JSON text around error position {error_pos}: {json_text[start:end]}")
            # Try to extract partial data or return minimal state
            try:
                # Try to fix common JSON issues
                if 'json_text' in locals():
                    # Remove trailing commas, fix common issues
                    fixed_json = self._fix_json_common_issues(json_text)
                    extracted = json.loads(fixed_json)
                    logger.info(f"Successfully fixed JSON after initial parse failure")
                    if existing_state:
                        return self._merge_continuity_states(existing_state, extracted, chapter_number)
                    else:
                        return ContinuityState(
                            manuscript_filename="",
                            user_id="",
                            last_analyzed_chapter=chapter_number,
                            **extracted
                        )
            except Exception as e2:
                logger.error(f"Failed to fix JSON: {e2}")
                if 'fixed_json' in locals():
                    logger.error(f"Fixed JSON (first 500 chars): {fixed_json[:500]}")
                
        except ValidationError as ve:
            logger.error(f"Pydantic validation error in continuity extraction: {ve}")
            # Try to fix validation errors and retry
            try:
                if 'extracted' in locals():
                    extracted = self._validate_and_fix_continuity_data(extracted, chapter_number)
                    if existing_state:
                        return self._merge_continuity_states(existing_state, extracted, chapter_number)
                    else:
                        return ContinuityState(
                            manuscript_filename="",
                            user_id="",
                            last_analyzed_chapter=chapter_number,
                            **extracted
                        )
            except Exception as e2:
                logger.error(f"Failed to fix validation errors: {e2}")
                
        except Exception as e:
            logger.error(f"Failed to extract continuity: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
        # Return existing state or empty state on failure
        logger.warning(f"Returning minimal continuity state for Chapter {chapter_number} due to extraction failure")
        return existing_state or ContinuityState(
            manuscript_filename="",
            user_id="",
            last_analyzed_chapter=chapter_number
        )
    
    async def extract_continuity_from_manuscript(
        self,
        manuscript_text: str,
        character_profiles: Optional[List[str]] = None,
        outline_body: Optional[str] = None,
        agent_state: Optional[Dict[str, Any]] = None
    ) -> ContinuityState:
        """
        Extract continuity state from entire manuscript (first-time analysis).
        
        Analyzes all chapters sequentially to build complete continuity state.
        
        Args:
            manuscript_text: Full manuscript text
            character_profiles: Character profile documents for reference
            outline_body: Story outline for plot thread context
            
        Returns:
            Complete ContinuityState built from all chapters
        """
        logger.info("Extracting continuity from full manuscript...")
        
        # Find all chapters
        matches = list(CHAPTER_PATTERN.finditer(manuscript_text))
        if not matches:
            logger.info("No chapters found in manuscript - returning empty state")
            return ContinuityState(
                manuscript_filename="",
                user_id="",
                last_analyzed_chapter=0
            )
        
        chapter_count = len(matches)
        logger.info(f"Found {chapter_count} chapters - analyzing sequentially...")
        
        # Build state incrementally by analyzing each chapter
        continuity_state: Optional[ContinuityState] = None
        
        for idx, match in enumerate(matches):
            chapter_num: Optional[int] = None
            try:
                chapter_num = int(match.group(1))
            except Exception:
                logger.warning(f"Could not parse chapter number from: {match.group(0)}")
                continue
            
            # Extract chapter text
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(manuscript_text)
            chapter_text = manuscript_text[start:end]
            
            logger.info(f"Analyzing Chapter {chapter_num} ({idx + 1}/{chapter_count})...")
            
            # Extract continuity from this chapter (with error handling to continue on failure)
            try:
                continuity_state = await self.extract_continuity_from_chapter(
                    chapter_text=chapter_text,
                    chapter_number=chapter_num,
                    existing_state=continuity_state,
                    character_profiles=character_profiles,
                    outline_body=outline_body,
                    agent_state=agent_state
                )
            except Exception as e:
                logger.error(f"Failed to extract continuity from Chapter {chapter_num}: {e}")
                logger.warning(f"Continuing with remaining chapters...")
                # Continue with existing state (or create minimal state if none exists)
                if continuity_state is None:
                    continuity_state = ContinuityState(
                        manuscript_filename="",
                        user_id="",
                        last_analyzed_chapter=chapter_num
            )
                continue
        
        if continuity_state:
            logger.info(f"Completed manuscript analysis - tracking {len(continuity_state.character_states)} characters, {len(continuity_state.plot_threads)} plot threads")
        else:
            logger.warning("Failed to extract continuity from manuscript")
            continuity_state = ContinuityState(
                manuscript_filename="",
                user_id="",
                last_analyzed_chapter=0
            )
        
        return continuity_state
    
    async def validate_new_content(
        self,
        new_content: str,
        target_chapter_number: int,
        continuity_state: ContinuityState,
        character_profiles: Optional[List[str]] = None,
        agent_state: Optional[Dict[str, Any]] = None
    ) -> ContinuityValidationResult:
        """
        Validate new content against continuity state.
        
        Args:
            new_content: New chapter content or edited content
            target_chapter_number: Chapter number for this content
            continuity_state: Current continuity state
            character_profiles: Character profiles for reference
            
        Returns:
            ContinuityValidationResult with violations and warnings
        """
        logger.info(f"Validating new content for Chapter {target_chapter_number}...")
        
        # Build validation prompt
        prompt = self._build_validation_prompt(
            new_content=new_content,
            chapter_number=target_chapter_number,
            continuity_state=continuity_state,
            character_profiles=character_profiles
        )
        
        # Use LLM to validate (pass agent_state to get user_chat_model if available)
        llm = self._llm_factory(temperature=0.1, state=agent_state)  # Very low temp for validation
        
        system_prompt = (
            "You are a continuity validation assistant for fiction manuscripts. "
            "You MUST respond with ONLY valid JSON - no prose, no explanations, no markdown code fences. "
            "Your response must be parseable JSON that matches the exact schema provided."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = await llm.ainvoke(messages)
        content = response.content
        
        # Parse validation result with robust error handling
        try:
            json_text = self._extract_json_from_response(content)
            result_dict = json.loads(json_text)
            return ContinuityValidationResult(**result_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from validation result: {e}")
            logger.debug(f"JSON text (first 300 chars): {json_text[:300] if 'json_text' in locals() else 'N/A'}")
            # Try to fix common JSON issues
            try:
                if 'json_text' in locals():
                    fixed_json = self._fix_json_common_issues(json_text)
                    result_dict = json.loads(fixed_json)
                    return ContinuityValidationResult(**result_dict)
            except Exception as e2:
                logger.error(f"Failed to fix validation JSON: {e2}")
        except Exception as e:
            logger.error(f"Failed to parse validation result: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Return safe fallback
        return ContinuityValidationResult(
            is_valid=True,  # Assume valid if we can't parse
            violations=[],
            warnings=["Failed to fully validate continuity - proceeding with caution"],
            confidence=0.3
        )
    
    def _build_extraction_prompt(
        self,
        chapter_text: str,
        chapter_number: int,
        existing_state: Optional[ContinuityState],
        character_profiles: Optional[List[str]],
        outline_body: Optional[str]
    ) -> str:
        """Build prompt for extracting continuity from chapter."""
        
        prompt_parts = [
            "=== CONTINUITY EXTRACTION TASK ===\n",
            f"Extract plot continuity information from Chapter {chapter_number}.\n\n",
            f"=== CHAPTER {chapter_number} TEXT ===\n{chapter_text}\n\n"
        ]
        
        if character_profiles:
            prompt_parts.append("=== CHARACTER PROFILES (for reference) ===\n")
            prompt_parts.append("\n---\n".join(character_profiles))
            prompt_parts.append("\n\n")
        
        if outline_body:
            prompt_parts.append(f"=== STORY OUTLINE (for plot thread context) ===\n{outline_body}\n\n")
        
        if existing_state:
            prompt_parts.append("=== EXISTING CONTINUITY STATE ===\n")
            prompt_parts.append(f"Characters currently tracked: {list(existing_state.character_states.keys())}\n")
            prompt_parts.append(f"Active plot threads: {len(existing_state.plot_threads)}\n")
            prompt_parts.append(f"Timeline events: {len(existing_state.timeline)}\n")
            prompt_parts.append(f"Last analyzed: Chapter {existing_state.last_analyzed_chapter}\n\n")
        
        prompt_parts.append(
            "=== YOUR TASK ===\n"
            "Analyze this chapter and extract:\n\n"
            "1. **Character States** - For EACH character that appears:\n"
            "   - Current location (where are they?)\n"
            "   - Emotional state (how do they feel?)\n"
            "   - What they know (new information they learn)\n"
            "   - Relationships (changes in how they relate to others)\n"
            "   - Physical state (injuries, exhaustion, etc.)\n"
            "   - Items they possess (important objects)\n\n"
            "2. **Plot Threads** - Active storylines:\n"
            "   - Identify ongoing plot threads (mysteries, conflicts, goals)\n"
            "   - Note which threads progress in this chapter\n"
            "   - Track unresolved questions\n"
            "   - Mark threads as active, resolved, or background\n\n"
            "3. **Timeline** - Time markers:\n"
            "   - Specific times mentioned ('morning', 'July 15th') → use time_type: 'specific_time'\n"
            "   - Time passage indicators ('three days later', 'the next morning') → use time_type: 'time_passage'\n"
            "   - Flashbacks → use time_type: 'flashback'\n"
            "   - Flashforwards → use time_type: 'flashforward'\n"
            "   - Time of day only ('morning', 'evening') → use time_type: 'time_of_day'\n"
            "   - Relative time ('later', 'earlier') → use time_type: 'relative_time'\n\n"
            "4. **World State Changes** - Lasting changes:\n"
            "   - Major events that change the world (battles, revelations, etc.)\n"
            "   - Weather/environmental changes that persist\n"
            "   - Political or social changes\n"
            "   - Magic or technology that alters things permanently\n"
            "   - Relationship changes between characters that affect the story\n"
            "   **Valid change_type values:** location, weather, political, magical, technological, social, location_status, character_inventory, character_possession, relationship\n\n"
            "5. **Unresolved Tensions** - Active conflicts:\n"
            "   - Character conflicts → use tension_type: 'conflict' or 'character_conflict'\n"
            "   - Mysteries or questions → use tension_type: 'mystery'\n"
            "   - Internal character struggles → use tension_type: 'internal'\n"
            "   - External threats or challenges → use tension_type: 'external' or 'external_threat'\n"
            "   - Relationship tensions → use tension_type: 'relationship'\n\n"
            "**OUTPUT FORMAT**: Return ONLY valid JSON matching this structure:\n"
            "{\n"
            '  "character_states": {\n'
            '    "CharacterName": {\n'
            '      "character_name": "CharacterName",\n'
            f'      "chapter_number": {chapter_number},\n'
            '      "location": "Current location",\n'
            '      "emotional_state": "Current emotion",\n'
            '      "knows_about": ["fact1", "fact2"],\n'
            '      "relationships": {"OtherCharacter": "relationship status"},\n'
            '      "injuries_or_conditions": ["condition1"],\n'
            '      "has_items": ["item1", "item2"]\n'
            '    }\n'
            '  },\n'
            '  "plot_threads": {\n'
            '    "thread_id_1": {\n'
            '      "thread_id": "thread_id_1",\n'
            '      "thread_name": "Thread name",\n'
            '      "description": "What this thread is about",\n'
            f'      "introduced_chapter": {chapter_number},\n'
            f'      "last_mentioned_chapter": {chapter_number},\n'
            '      "status": "active",\n'
            '      "key_events": ["event1"],\n'
            '      "unresolved_questions": ["question1"]\n'
            '    }\n'
            '  },\n'
            '  "timeline": [\n'
            '    {\n'
            f'      "chapter_number": {chapter_number},\n'
            '      "time_type": "specific_time",\n'
            '      "description": "Morning of the third day",\n'
            '      "time_of_day": "morning"\n'
            '    }\n'
            '  ],\n'
            '  "world_state_changes": [\n'
            '    {\n'
            f'      "chapter_number": {chapter_number},\n'
            '      "change_type": "location",\n'
            '      "description": "The bridge collapsed",\n'
            '      "affects": ["Village", "MainCharacter"],\n'
            '      "is_permanent": true\n'
            '    }\n'
            '  ],\n'
            '  **change_type must be one of:** location, weather, political, magical, technological, social, location_status, character_inventory, character_possession, relationship\n'
            '  "unresolved_tensions": {\n'
            '    "tension_id_1": {\n'
            '      "tension_id": "tension_id_1",\n'
            '      "description": "Mystery of the missing artifact",\n'
            f'      "introduced_chapter": {chapter_number},\n'
            f'      "last_escalated_chapter": {chapter_number},\n'
            '      "tension_type": "mystery",\n'
            '      "involves_characters": ["Detective", "Suspect"],\n'
            '      "stakes": "If not found, magic will fail"\n'
            '    }\n'
            '  },\n'
            '  "current_chapter_summary": "Brief summary of story state after this chapter"\n'
            "}\n\n"
            "**CRITICAL JSON FORMAT REQUIREMENTS**:\n"
            "- You MUST return ONLY valid JSON - no prose, no explanations, no markdown code fences\n"
            "- All strings must be properly escaped (use \\\" for quotes inside strings)\n"
            "- All arrays and objects must be properly closed\n"
            "- No trailing commas in arrays or objects\n"
            "- Ensure all field names are in double quotes\n"
            "- Test your JSON before returning it - it must parse correctly\n"
            "- If a field is empty, use an empty array [] or empty object {} or null, not undefined\n\n"
            "**IMPORTANT**:\n"
            "- Focus on FACTUAL information from the chapter\n"
            "- Don't infer things not present in the text\n"
            "- For character states, only include characters who appear in this chapter\n"
            "- Use consistent character names (as they appear in character profiles if provided)\n"
            "- Create unique, descriptive IDs for threads and tensions (e.g., 'fleet_investigation', 'missing_artifact_mystery')\n"
            "- **SELECTIVITY**: Only track information that's CURRENTLY RELEVANT for continuity:\n"
            "  * Character knowledge: Focus on NEW information learned in this chapter, not everything they've ever known\n"
            "  * Character items: Only track SIGNIFICANT items (weapons, keys, important documents), not everyday items\n"
            "  * Plot events: Focus on MAJOR events that affect the story, not minor details\n"
            "  * Questions: Only include questions that are STILL UNRESOLVED and relevant\n"
            "- Be concise - summarize rather than listing every detail\n"
        )
        
        return "".join(prompt_parts)
    
    def _build_validation_prompt(
        self,
        new_content: str,
        chapter_number: int,
        continuity_state: ContinuityState,
        character_profiles: Optional[List[str]]
    ) -> str:
        """Build prompt for validating new content against continuity."""
        
        prompt_parts = [
            "=== CONTINUITY VALIDATION TASK ===\n",
            f"Validate new content for Chapter {chapter_number} against established continuity.\n\n",
            f"=== NEW CONTENT TO VALIDATE ===\n{new_content}\n\n"
        ]
        
        # Add relevant continuity state
        prompt_parts.append("=== ESTABLISHED CONTINUITY (what must remain consistent) ===\n\n")
        
        if continuity_state.character_states:
            prompt_parts.append("**Character States:**\n")
            for char_name, char_state in continuity_state.character_states.items():
                prompt_parts.append(f"- {char_name}:\n")
                if char_state.location:
                    prompt_parts.append(f"  Location: {char_state.location}\n")
                if char_state.emotional_state:
                    prompt_parts.append(f"  Emotional state: {char_state.emotional_state}\n")
                if char_state.knows_about:
                    prompt_parts.append(f"  Knows: {', '.join(char_state.knows_about[:3])}\n")
                if char_state.injuries_or_conditions:
                    prompt_parts.append(f"  Conditions: {', '.join(char_state.injuries_or_conditions)}\n")
            prompt_parts.append("\n")
        
        if continuity_state.plot_threads:
            prompt_parts.append("**Active Plot Threads:**\n")
            for thread in continuity_state.plot_threads.values():
                if thread.status == "active":
                    prompt_parts.append(f"- {thread.thread_name}: {thread.description}\n")
                    if thread.unresolved_questions:
                        prompt_parts.append(f"  Questions: {', '.join(thread.unresolved_questions[:2])}\n")
            prompt_parts.append("\n")
        
        if continuity_state.timeline and chapter_number is not None:
            recent_timeline = [t for t in continuity_state.timeline if t.chapter_number >= chapter_number - 2]
            if recent_timeline:
                prompt_parts.append("**Recent Timeline:**\n")
                for marker in recent_timeline[-3:]:
                    prompt_parts.append(f"- Chapter {marker.chapter_number}: {marker.description}\n")
                prompt_parts.append("\n")
        
        if continuity_state.world_state_changes:
            permanent_changes = [c for c in continuity_state.world_state_changes if c.is_permanent]
            if permanent_changes:
                prompt_parts.append("**Permanent World Changes:**\n")
                for change in permanent_changes[-5:]:
                    prompt_parts.append(f"- {change.description} (affects: {', '.join(change.affects)})\n")
                prompt_parts.append("\n")
        
        if character_profiles:
            prompt_parts.append("=== CHARACTER PROFILES (for reference) ===\n")
            prompt_parts.append("\n---\n".join(character_profiles))
            prompt_parts.append("\n\n")
        
        prompt_parts.append(
            "=== YOUR TASK ===\n"
            "Check if the new content violates any established continuity.\n\n"
            "**Check for violations in:**\n"
            "1. **Character Locations** - Are characters where they should be?\n"
            "2. **Character Knowledge** - Do characters know things they shouldn't (or not know things they should)?\n"
            "3. **Timeline** - Does timing make sense with previous chapters?\n"
            "4. **Character State** - Are emotional states, injuries, conditions consistent?\n"
            "5. **Plot Threads** - Does content contradict active plot threads?\n"
            "6. **World State** - Does content ignore permanent world changes?\n"
            "7. **Relationships** - Are character relationships consistent?\n\n"
            "**OUTPUT FORMAT**: Return ONLY valid JSON:\n"
            "{\n"
            '  "is_valid": true/false,\n'
            '  "violations": [\n'
            '    {\n'
            '      "violation_type": "character_location",\n'
            '      "severity": "high",\n'
            '      "description": "Character X is in Location A, but continuity says they should be in Location B",\n'
            '      "expected": "Character at Location B",\n'
            '      "found": "Character at Location A",\n'
            '      "affected_character": "CharacterName",\n'
            '      "suggestion": "Change location to Location B or add transition scene"\n'
            '    }\n'
            '  ],\n'
            '  "warnings": ["Warning message if something seems off but might be intentional"],\n'
            '  "confidence": 0.9\n'
            "}\n\n"
            "**Severity levels:**\n"
            "- **critical**: Major plot hole or character impossibility\n"
            "- **high**: Clear continuity error that readers will notice\n"
            "- **medium**: Inconsistency that some readers might catch\n"
            "- **low**: Minor issue that's unlikely to be noticed\n\n"
            "**IMPORTANT**:\n"
            "- Only flag CLEAR violations, not stylistic choices\n"
            "- Consider that time may have passed or circumstances changed\n"
            "- If new content EXPLAINS a seeming violation, that's not a violation\n"
            "- Be specific about what's wrong and how to fix it\n"
        )
        
        return "".join(prompt_parts)
    
    def _merge_continuity_states(
        self,
        existing: ContinuityState,
        new_data: Dict[str, Any],
        chapter_number: int
    ) -> ContinuityState:
        """
        Merge new continuity data with existing state.
        
        Strategy:
        - Character states: Update with new info, keep historical data
        - Plot threads: Update status, append events, merge questions
        - Timeline: Append new markers
        - World changes: Append new changes
        - Tensions: Update existing, add new ones
        """
        
        # Update character states
        for char_name, new_char_data in new_data.get("character_states", {}).items():
            if char_name in existing.character_states:
                # Merge with existing
                old_char = existing.character_states[char_name]
                
                # Update with new data, keeping accumulated knowledge
                merged_knows = list(set(old_char.knows_about + new_char_data.get("knows_about", [])))
                merged_items = list(set(old_char.has_items + new_char_data.get("has_items", [])))
                
                existing.character_states[char_name] = CharacterState(
                    character_name=char_name,
                    chapter_number=chapter_number,
                    location=new_char_data.get("location") or old_char.location,
                    emotional_state=new_char_data.get("emotional_state") or old_char.emotional_state,
                    knows_about=merged_knows,
                    relationships={**old_char.relationships, **new_char_data.get("relationships", {})},
                    injuries_or_conditions=new_char_data.get("injuries_or_conditions", old_char.injuries_or_conditions),
                    has_items=merged_items
                )
            else:
                # New character
                existing.character_states[char_name] = CharacterState(**new_char_data)
        
        # Update plot threads
        for thread_id, new_thread_data in new_data.get("plot_threads", {}).items():
            if thread_id in existing.plot_threads:
                old_thread = existing.plot_threads[thread_id]
                
                # Merge events and questions
                merged_events = old_thread.key_events + new_thread_data.get("key_events", [])
                
                # Merge questions, but prioritize new questions (they may indicate resolution)
                new_questions = new_thread_data.get("unresolved_questions", [])
                # If thread is resolved, clear old questions
                if new_thread_data.get("status") == "resolved":
                    merged_questions = new_questions  # Only keep new questions if any
                else:
                    # Merge but remove duplicates
                    merged_questions = list(set(
                        old_thread.unresolved_questions + new_questions
                    ))
                
                existing.plot_threads[thread_id] = PlotThread(
                    thread_id=thread_id,
                    thread_name=new_thread_data.get("thread_name", old_thread.thread_name),
                    description=new_thread_data.get("description", old_thread.description),
                    introduced_chapter=old_thread.introduced_chapter,
                    last_mentioned_chapter=chapter_number,
                    status=new_thread_data.get("status", old_thread.status),
                    key_events=merged_events,
                    unresolved_questions=merged_questions,
                    expected_resolution_chapter=new_thread_data.get("expected_resolution_chapter")
                )
            else:
                existing.plot_threads[thread_id] = PlotThread(**new_thread_data)
        
        # Append timeline markers
        for marker_data in new_data.get("timeline", []):
            existing.timeline.append(TimeMarker(**marker_data))
        
        # Append world state changes
        for change_data in new_data.get("world_state_changes", []):
            existing.world_state_changes.append(WorldStateChange(**change_data))
        
        # Update tensions
        for tension_id, new_tension_data in new_data.get("unresolved_tensions", {}).items():
            if tension_id in existing.unresolved_tensions:
                old_tension = existing.unresolved_tensions[tension_id]
                existing.unresolved_tensions[tension_id] = UnresolvedTension(
                    **{**old_tension.dict(), **new_tension_data, "last_escalated_chapter": chapter_number}
                )
            else:
                existing.unresolved_tensions[tension_id] = UnresolvedTension(**new_tension_data)
        
        # Update metadata
        existing.last_analyzed_chapter = chapter_number
        existing.last_updated = datetime.utcnow().isoformat()
        existing.current_chapter_summary = new_data.get("current_chapter_summary", existing.current_chapter_summary)
        
        # Prune state to keep it lean
        self._prune_continuity_state(existing, chapter_number)
        
        return existing
    
    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        # Try to extract from code blocks first
        json_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', content)
        if json_match:
            return json_match.group(1).strip()
        
        # Try to find JSON object (look for opening brace to closing brace)
        # Use a more robust pattern that handles nested braces
        brace_count = 0
        start_idx = content.find('{')
        if start_idx != -1:
            for i in range(start_idx, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return content[start_idx:i+1].strip()
        
        # Fallback: return content as-is
        return content.strip()
    
    def _validate_and_fix_continuity_data(self, data: Dict[str, Any], chapter_number: int) -> Dict[str, Any]:
        """
        Validate and fix continuity data before Pydantic model creation.
        
        Fixes common issues like invalid enum values, missing required fields, etc.
        """
        # Fix world_state_changes with invalid change_type
        if "world_state_changes" in data and isinstance(data["world_state_changes"], list):
            valid_change_types = [
                "location", "weather", "political", "magical", "technological", 
                "social", "location_status", "character_inventory", 
                "character_possession", "relationship"
            ]
            fixed_changes = []
            for change in data["world_state_changes"]:
                if isinstance(change, dict):
                    change_type = change.get("change_type", "")
                    if change_type not in valid_change_types:
                        # Map invalid types to closest valid type
                        if change_type in ["relationship", "character_relationship"]:
                            change["change_type"] = "relationship"
                        elif change_type in ["inventory", "items"]:
                            change["change_type"] = "character_inventory"
                        elif change_type in ["possession", "ownership"]:
                            change["change_type"] = "character_possession"
                        else:
                            # Default to "social" for unrecognized types
                            logger.warning(f"Invalid change_type '{change_type}', defaulting to 'social'")
                            change["change_type"] = "social"
                    # Ensure required fields
                    if "chapter_number" not in change:
                        change["chapter_number"] = chapter_number
                    if "is_permanent" not in change:
                        change["is_permanent"] = True
                fixed_changes.append(change)
            data["world_state_changes"] = fixed_changes
        
        # Fix plot thread status values
        if "plot_threads" in data and isinstance(data["plot_threads"], dict):
            valid_statuses = ["active", "resolved", "abandoned", "background"]
            for thread_id, thread_data in data["plot_threads"].items():
                if isinstance(thread_data, dict):
                    status = thread_data.get("status", "active")
                    if status not in valid_statuses:
                        logger.warning(f"Invalid thread status '{status}', defaulting to 'active'")
                        thread_data["status"] = "active"
        
        # Fix tension type values
        if "unresolved_tensions" in data and isinstance(data["unresolved_tensions"], dict):
            valid_tension_types = [
                "conflict", "mystery", "relationship", "internal", 
                "external", "external_threat", "character_conflict"
            ]
            for tension_id, tension_data in data["unresolved_tensions"].items():
                if isinstance(tension_data, dict):
                    tension_type = tension_data.get("tension_type", "mystery")
                    if tension_type not in valid_tension_types:
                        # Map to closest valid type
                        if "conflict" in tension_type.lower():
                            tension_data["tension_type"] = "conflict"
                        elif "threat" in tension_type.lower():
                            tension_data["tension_type"] = "external_threat"
                        else:
                            logger.warning(f"Invalid tension_type '{tension_type}', defaulting to 'mystery'")
                            tension_data["tension_type"] = "mystery"
        
        return data
    
    def _fix_json_common_issues(self, json_text: str) -> str:
        """Fix common JSON issues like trailing commas."""
        # Remove trailing commas before closing braces/brackets (most common issue)
        # Use MULTILINE flag to handle line-by-line
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text, flags=re.MULTILINE)
        
        # Remove trailing commas at end of lines before closing braces/brackets
        json_text = re.sub(r',\s*\n\s*([}\]])', r'\n\1', json_text)
        
        return json_text
    
    def _prune_continuity_state(self, state: ContinuityState, current_chapter: int) -> None:
        """
        Prune continuity state to keep it lean and manageable.
        
        Strategy:
        - Character knowledge: Keep only most recent/relevant items (max 20)
        - Character items: Keep only significant items (max 15)
        - Plot thread events: Keep only last 12 events per thread
        - Plot thread questions: Remove resolved questions, keep max 8 active
        - Timeline: Keep only markers from last 25 chapters or last 30 markers
        - Resolved threads: Remove threads resolved >5 chapters ago
        - Resolved tensions: Remove tensions resolved >5 chapters ago
        - Injuries: Only keep current/recent conditions
        """
        # Prune character states
        for char_name, char_state in state.character_states.items():
            # Limit knows_about to most recent 20 items
            if len(char_state.knows_about) > 20:
                # Keep the most recent items (assume they're added in order)
                char_state.knows_about = char_state.knows_about[-20:]
                logger.debug(f"Pruned {char_name}'s knows_about to 20 items")
            
            # Limit has_items to most recent 15 items
            if len(char_state.has_items) > 15:
                char_state.has_items = char_state.has_items[-15:]
                logger.debug(f"Pruned {char_name}'s has_items to 15 items")
            
            # Only keep current/recent injuries (not historical ones)
            # If injuries are from old chapters, they're likely resolved
            if len(char_state.injuries_or_conditions) > 5:
                # Keep only the most recent conditions
                char_state.injuries_or_conditions = char_state.injuries_or_conditions[-5:]
                logger.debug(f"Pruned {char_name}'s injuries_or_conditions to 5 items")
        
        # Prune plot threads
        threads_to_remove = []
        for thread_id, thread in state.plot_threads.items():
            # Remove threads resolved more than 5 chapters ago
            if thread.status == "resolved":
                if current_chapter - thread.last_mentioned_chapter > 5:
                    threads_to_remove.append(thread_id)
                    logger.debug(f"Removing resolved thread {thread_id} (resolved {current_chapter - thread.last_mentioned_chapter} chapters ago)")
                    continue
            
            # Limit key_events to last 12 events
            if len(thread.key_events) > 12:
                thread.key_events = thread.key_events[-12:]
                logger.debug(f"Pruned thread {thread_id} key_events to 12 items")
            
            # Limit unresolved_questions to 8 most important
            if len(thread.unresolved_questions) > 8:
                thread.unresolved_questions = thread.unresolved_questions[-8:]
                logger.debug(f"Pruned thread {thread_id} unresolved_questions to 8 items")
        
        # Remove old resolved threads
        for thread_id in threads_to_remove:
            del state.plot_threads[thread_id]
        
        # Prune timeline - keep only markers from last 25 chapters or last 30 markers
        if len(state.timeline) > 30:
            # Keep markers from recent chapters
            recent_chapters = max(1, current_chapter - 25)
            state.timeline = [
                marker for marker in state.timeline 
                if marker.chapter_number >= recent_chapters
            ]
            # If still too many, keep only last 30
            if len(state.timeline) > 30:
                state.timeline = state.timeline[-30:]
            logger.debug(f"Pruned timeline to {len(state.timeline)} markers")
        
        # Prune unresolved tensions - remove old resolved ones
        tensions_to_remove = []
        for tension_id, tension in state.unresolved_tensions.items():
            # If tension hasn't been mentioned in 10+ chapters, consider it stale
            # (This is a heuristic - could be improved with explicit resolution tracking)
            if current_chapter - tension.last_escalated_chapter > 10:
                # Only remove if it's been a long time and seems inactive
                tensions_to_remove.append(tension_id)
                logger.debug(f"Removing stale tension {tension_id} (last escalated {current_chapter - tension.last_escalated_chapter} chapters ago)")
        
        for tension_id in tensions_to_remove:
            del state.unresolved_tensions[tension_id]
        
        # Prune world state changes - keep only permanent ones and recent temporary ones
        if len(state.world_state_changes) > 50:
            # Keep all permanent changes, but limit temporary ones
            permanent_changes = [c for c in state.world_state_changes if c.is_permanent]
            temporary_changes = [c for c in state.world_state_changes if not c.is_permanent]
            
            # Keep only recent temporary changes (last 20 chapters)
            recent_temporary = [
                c for c in temporary_changes 
                if c.chapter_number >= max(1, current_chapter - 20)
            ]
            
            state.world_state_changes = permanent_changes + recent_temporary
            logger.debug(f"Pruned world_state_changes to {len(state.world_state_changes)} items")
        
        logger.info(f"Pruned continuity state: {len(state.character_states)} characters, {len(state.plot_threads)} threads, {len(state.timeline)} timeline markers, {len(state.unresolved_tensions)} tensions")
    
    def get_continuity_summary(self, state: ContinuityState) -> str:
        """Generate human-readable summary of continuity state."""
        parts = [
            f"=== CONTINUITY STATE SUMMARY ===",
            f"Last analyzed: Chapter {state.last_analyzed_chapter}",
            f"Last updated: {state.last_updated}\n",
            f"**Characters tracked:** {len(state.character_states)}",
        ]
        
        if state.character_states:
            parts.append("Character locations:")
            for char_name, char_state in list(state.character_states.items())[:5]:
                if char_state.location:
                    parts.append(f"  - {char_name}: {char_state.location}")
        
        parts.append(f"\n**Active plot threads:** {len([t for t in state.plot_threads.values() if t.status == 'active'])}")
        parts.append(f"**Timeline events:** {len(state.timeline)}")
        parts.append(f"**Unresolved tensions:** {len(state.unresolved_tensions)}")
        
        if state.current_chapter_summary:
            parts.append(f"\n**Current story state:**\n{state.current_chapter_summary}")
        
        return "\n".join(parts)

