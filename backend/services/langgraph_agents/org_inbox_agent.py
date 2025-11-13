"""
Org Inbox Agent - Roosevelt's "PIM Rough Rider"
Strictly manages items in the user's org-mode inbox.org: add, list, toggle, update.
"""

import logging
from typing import Dict, Any, Optional, List

from services.langgraph_agents.base_agent import BaseAgent
from models.agent_response_models import OrgInboxResult, TaskStatus, OrgInboxInterpretation
from services.langgraph_tools.org_inbox_tools import (
    org_inbox_path,
    org_inbox_list_items,
    org_inbox_add_item,
    org_inbox_toggle_done,
    org_inbox_update_line,
    org_inbox_append_text,
    org_inbox_index_tags,
    org_inbox_apply_tags,
    org_inbox_set_state,
    org_inbox_promote_state,
    org_inbox_demote_state,
    org_inbox_set_schedule_and_repeater,
    org_inbox_archive_done,
)

logger = logging.getLogger(__name__)


class OrgInboxAgent(BaseAgent):
    """
    Roosevelt's Org Inbox Agent: Adds, reviews, and adjusts entries in `inbox.org`.
    """

    def __init__(self):
        super().__init__("org_inbox_agent")
        logger.info("🗂️ BULLY! Org Inbox Agent saddled and ready to ride!")

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            user_message = state.get("messages", [])[-1].content if state.get("messages") else ""
            user_id = state.get("user_id")
            intent = (state.get("intent_classification") or {}).get("intent_type", "").lower()

            # Dispatch based on explicit operation requested in state or infer rudimentarily
            op: Optional[str] = (state.get("org_inbox_operation") or "").lower()
            payload = state.get("org_inbox_payload") or {}

            if not op:
                # Minimal inference: look for verbs; upstream classifier should set op ideally
                lowered = user_message.lower()
                if any(k in lowered for k in ["add ", "capture ", "note ", "todo "]):
                    op = "add"
                elif any(k in lowered for k in ["list", "show", "review", "inbox"]):
                    op = "list"
                elif any(k in lowered for k in ["done", "complete", "toggle"]):
                    op = "toggle"
                elif any(k in lowered for k in ["edit", "update", "change"]):
                    op = "update"
                else:
                    # Default to list for safety
                    op = "list"

            if op == "add":
                # Always use LLM interpreter for org-mode add with conversation context and settings
                interpretation = await self._llm_interpret_add(state, user_message, payload)

                if interpretation.clarification_needed:
                    path = await org_inbox_path(user_id)
                    structured = OrgInboxResult(
                        task_status=TaskStatus.INCOMPLETE,
                        action="add",
                        message=interpretation.clarification_question or "Clarification needed",
                        path=path,
                    )
                    latest = interpretation.clarification_question or "I need a bit more detail before adding this."
                    return {"agent_results": {"org_inbox": structured.dict(), "interpretation": interpretation.dict()}, "latest_response": latest}

                title = interpretation.title.strip()
                scheduled = interpretation.schedule

                # Decide how to write the entry
                if interpretation.entry_kind == "contact":
                    # Build contact entry with PROPERTIES drawer
                    headline = f"* {title}"
                    org_entry = f"{headline}\n"
                    
                    # Add PROPERTIES drawer if contact_properties exist
                    if interpretation.contact_properties:
                        org_entry += ":PROPERTIES:\n"
                        for key, value in interpretation.contact_properties.items():
                            if value:  # Only add non-empty properties
                                org_entry += f":{key}: {value}\n"
                        org_entry += ":END:\n"
                    
                    result = await org_inbox_append_text(org_entry, user_id)
                elif scheduled or interpretation.entry_kind == "event":
                    # Build a proper org-mode entry; set schedule/repeater via tool for correctness
                    org_type = "TODO" if interpretation.entry_kind == "todo" else ""
                    headline = f"* {org_type} {title}".strip()
                    org_entry = f"{headline}\n"
                    result = await org_inbox_append_text(org_entry, user_id)
                    # Determine new headline index
                    listing = await org_inbox_list_items(user_id)
                    items = listing.get("items", [])
                    line_index = items[-1].get("line_index") if items else None
                    if line_index is not None and scheduled:
                        await org_inbox_set_schedule_and_repeater(
                            line_index=line_index,
                            scheduled=scheduled,
                            repeater=getattr(interpretation, "repeater", None),
                            user_id=user_id,
                        )
                else:
                    # Respect entry_kind from interpreter for non-scheduled entries
                    kind = "todo" if interpretation.entry_kind != "checkbox" else "checkbox"
                    result = await org_inbox_add_item(text=title, kind=kind, user_id=user_id)
                    # Capture index if available
                    line_index = result.get("line_index")

                path = await org_inbox_path(user_id)
                structured = OrgInboxResult(
                    task_status=TaskStatus.COMPLETE,
                    action="add",
                    message=f"Added '{title}' to inbox.org",
                    path=path,
                )
                latest = f"Added TODO '{title}' to inbox.org" if interpretation.entry_kind != "event" else f"Added event '{title}' to inbox.org"
                if scheduled:
                    latest += f" (scheduled {scheduled}"
                    if getattr(interpretation, "repeater", None):
                        latest += f", {interpretation.repeater}"
                    latest += ")"
                # Apply tags if requested or infer tags
                desired_tags = payload.get("tags") or []
                # Merge interpreter-suggested tags
                desired_tags = list({*desired_tags, *getattr(interpretation, "suggested_tags", [])})
                auto_tags_applied = []
                if line_index is None:
                    # Best effort: list items and pick the last headline/checkbox index
                    listing = await org_inbox_list_items(user_id)
                    items = listing.get("items", [])
                    if items:
                        line_index = items[-1].get("line_index")

                if line_index is not None:
                    final_tags = list({*(desired_tags or [])})
                    if final_tags:
                        tag_apply_res = await org_inbox_apply_tags(line_index=line_index, tags=final_tags, user_id=user_id)
                        if not tag_apply_res.get("error"):
                            auto_tags_applied = final_tags
                            latest += f" | tags: {':'.join(final_tags)}"

                # Use persona-driven assistant confirmation when provided
                if getattr(interpretation, "assistant_confirmation", None):
                    latest = interpretation.assistant_confirmation
                return {"agent_results": {"org_inbox": structured.dict(), "raw": result, "tags_applied": auto_tags_applied}, "latest_response": latest}

            if op == "list":
                listing = await org_inbox_list_items(user_id)
                path = listing.get("path") or await org_inbox_path(user_id)
                structured = OrgInboxResult(
                    task_status=TaskStatus.COMPLETE,
                    action="list",
                    message="Listed inbox.org items",
                    path=path,
                    items=listing.get("items", []),
                )
                items = listing.get("items", [])
                preview = ", ".join([i.get("text", "").strip() for i in items[:3]])
                latest = f"Inbox has {len(items)} items" + (f": {preview}" if preview else "")
                return {"agent_results": {"org_inbox": structured.dict(), "raw": listing}, "latest_response": latest}

            if op == "toggle":
                line_index = int(payload.get("line_index", -1))
                toggle_res = await org_inbox_toggle_done(line_index=line_index, user_id=user_id)
                path = toggle_res.get("path") or await org_inbox_path(user_id)
                if toggle_res.get("error"):
                    structured = OrgInboxResult(
                        task_status=TaskStatus.ERROR,
                        action="toggle",
                        message=toggle_res.get("error", "Toggle failed"),
                        path=path,
                    )
                    latest = f"Toggle failed: {structured.message}"
                else:
                    structured = OrgInboxResult(
                        task_status=TaskStatus.COMPLETE,
                        action="toggle",
                        message="Toggled item state",
                        path=path,
                        updated_index=toggle_res.get("updated_index"),
                        new_line=toggle_res.get("new_line"),
                    )
                    latest = f"Toggled item at line {structured.updated_index}"
                return {"agent_results": {"org_inbox": structured.dict(), "raw": toggle_res}, "latest_response": latest}

            if op == "update":
                line_index = int(payload.get("line_index", -1))
                new_text = payload.get("new_text", "")
                upd = await org_inbox_update_line(line_index=line_index, new_text=new_text, user_id=user_id)
                path = upd.get("path") or await org_inbox_path(user_id)
                if upd.get("error"):
                    structured = OrgInboxResult(
                        task_status=TaskStatus.ERROR,
                        action="update",
                        message=upd.get("error", "Update failed"),
                        path=path,
                    )
                    latest = f"Update failed: {structured.message}"
                else:
                    structured = OrgInboxResult(
                        task_status=TaskStatus.COMPLETE,
                        action="update",
                        message="Updated inbox line",
                        path=path,
                        updated_index=upd.get("updated_index"),
                        new_line=upd.get("new_line"),
                    )
                    latest = f"Updated line {structured.updated_index}: {new_text.strip()}"
                return {"agent_results": {"org_inbox": structured.dict(), "raw": upd}, "latest_response": latest}

            if op == "promote":
                # Promote a headline along the configured TODO sequence
                sequence = self._get_todo_sequence()
                line_index = int(payload.get("line_index", -1))
                res = await org_inbox_promote_state(line_index=line_index, sequence=sequence, user_id=user_id)
                path = res.get("path") or await org_inbox_path(user_id)
                if res.get("error"):
                    structured = OrgInboxResult(
                        task_status=TaskStatus.ERROR,
                        action="update",
                        message=res.get("error", "Promote failed"),
                        path=path,
                    )
                    latest = f"Promote failed: {structured.message}"
                else:
                    structured = OrgInboxResult(
                        task_status=TaskStatus.COMPLETE,
                        action="update",
                        message="Promoted task state",
                        path=path,
                        updated_index=res.get("updated_index"),
                        new_line=res.get("new_line"),
                    )
                    latest = "Promoted task state"
                return {"agent_results": {"org_inbox": structured.dict(), "raw": res}, "latest_response": latest}

            if op == "schedule":
                # Set schedule and optional repeater
                line_index = int(payload.get("line_index", -1))
                scheduled = payload.get("scheduled")  # org timestamp <YYYY-MM-DD Dow>
                repeater = payload.get("repeater")  # e.g., +1w
                res = await org_inbox_set_schedule_and_repeater(line_index=line_index, scheduled=scheduled, repeater=repeater, user_id=user_id)
                path = res.get("path") or await org_inbox_path(user_id)
                if res.get("error"):
                    structured = OrgInboxResult(
                        task_status=TaskStatus.ERROR,
                        action="update",
                        message=res.get("error", "Schedule failed"),
                        path=path,
                    )
                    latest = f"Schedule failed: {structured.message}"
                else:
                    structured = OrgInboxResult(
                        task_status=TaskStatus.COMPLETE,
                        action="update",
                        message="Updated schedule/repeater",
                        path=path,
                        updated_index=res.get("updated_index"),
                        new_line=res.get("scheduled_line"),
                    )
                    latest = "Updated schedule/repeater"
                return {"agent_results": {"org_inbox": structured.dict(), "raw": res}, "latest_response": latest}

            if op == "archive_done":
                if not user_id:
                    raise ValueError("Missing user_id for archive operation")
                res = await org_inbox_archive_done(user_id)
                if res.get("error"):
                    structured = OrgInboxResult(
                        task_status=TaskStatus.ERROR,
                        action="archive_done",
                        message=res.get("error", "Archive failed"),
                        path=res.get("path", ""),
                    )
                    latest = f"Archive failed: {structured.message}"
                else:
                    structured = OrgInboxResult(
                        task_status=TaskStatus.COMPLETE,
                        action="archive_done",
                        message=f"Archived {res.get('archived_count', 0)} DONE items",
                        path=res.get("path", ""),
                    )
                    latest = structured.message
                return {"agent_results": {"org_inbox": structured.dict(), "raw": res}, "latest_response": latest}

            # Fallback to list
            listing = await org_inbox_list_items(user_id)
            path = listing.get("path") or await org_inbox_path(user_id)
            structured = OrgInboxResult(
                task_status=TaskStatus.COMPLETE,
                action="list",
                message="Listed inbox.org items",
                path=path,
                items=listing.get("items", []),
            )
            items = listing.get("items", [])
            preview = ", ".join([i.get("text", "").strip() for i in items[:3]])
            latest = f"Inbox has {len(items)} items" + (f": {preview}" if preview else "")
            return {"agent_results": {"org_inbox": structured.dict(), "raw": listing}, "latest_response": latest}

        except Exception as e:
            logger.error(f"❌ OrgInboxAgent failure: {e}")
            structured = OrgInboxResult(
                task_status=TaskStatus.ERROR,
                action="list",
                message=str(e),
                path=None,
            )
            return {"agent_results": {"org_inbox": structured.dict()}, "latest_response": f"Inbox error: {str(e)}"}

    async def _generate_smart_org_entry(self, text: str, scheduled: str) -> str:
        """Use LLM intelligence to generate appropriate org-mode entry with smart title and type."""
        try:
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            prompt = f"""You are Roosevelt's Org-Mode Intelligence Agent. Create a proper org-mode entry from the user's request.

USER REQUEST: "{text}"
DETECTED DATE: {scheduled}

TASK: Generate an appropriate org-mode entry with:
1. Correct entry type (TODO for tasks, plain headline for events/birthdays/appointments)
2. Concise, descriptive title
3. Proper org-mode formatting

ORG-MODE ENTRY TYPES:
- **TODO**: For tasks to complete ("TODO Finish report", "TODO Call doctor")
- **Plain Headline**: For events, birthdays, appointments, meetings ("* My Birthday", "* Team Meeting", "* Wedding Anniversary")

EXAMPLES:
- "my birthday" → "* My Birthday"
- "doctor appointment" → "* Doctor Appointment" 
- "finish the project" → "* TODO Finish the project"
- "mom's anniversary" → "* Mom's Anniversary"
- "team meeting" → "* Team Meeting"

STRUCTURED OUTPUT REQUIREMENT:
You MUST respond with valid JSON matching this schema:
{{
    "entry_type": "TODO" or "",
    "title": "Concise, appropriate title",
    "additional_content": "Optional additional lines (empty string if none)"
}}

**BULLY!** Create an entry that captures the user's intent with proper org-mode formatting!"""

            response = await chat_service.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0.3
            )
            
            # Parse structured response
            import json
            try:
                result = json.loads(response.choices[0].message.content)
                entry_type = result.get("entry_type", "")
                title = result.get("title", "New Entry")
                additional_content = result.get("additional_content", "")
                
                # Build org-mode entry
                headline = f"* {entry_type} {title}".strip() if entry_type else f"* {title}"
                org_entry = f"{headline}\nSCHEDULED: {scheduled}\n"
                
                if additional_content:
                    org_entry += f"{additional_content}\n"
                
                logger.info(f"🎯 SMART ORG ENTRY: Generated '{headline}' with LLM intelligence")
                return org_entry
                
            except json.JSONDecodeError:
                logger.warning("⚠️ Failed to parse LLM org entry response, falling back to simple format")
                # Fallback to previous logic
                title = self._extract_title(text)
                entry_type = self._determine_entry_type(text)
                headline = f"* {entry_type} {title}".strip() if entry_type else f"* {title}"
                return f"{headline}\nSCHEDULED: {scheduled}\n"
                
        except Exception as e:
            logger.error(f"❌ Smart org entry generation failed: {e}")
            # Fallback to previous logic
            title = self._extract_title(text)
            entry_type = self._determine_entry_type(text)
            headline = f"* {entry_type} {title}".strip() if entry_type else f"* {title}"
            return f"{headline}\nSCHEDULED: {scheduled}\n"

    async def _llm_interpret_add(self, state: Dict[str, Any], user_message: str, payload: Dict[str, Any]) -> OrgInboxInterpretation:
        """Use LLM to interpret add requests: resolve pronouns, choose entry kind, schedule, tags, and repeater."""
        chat_service = await self._get_chat_service()
        model_name = await self._get_model_name()
        text = (payload.get("text") or user_message or "").strip()
        # Build conversation context (last 5 messages max)
        messages = state.get("messages", [])[-5:]
        context_lines: List[str] = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", "user")
            content = getattr(m, "content", "")
            context_lines.append(f"- {role}: {content}")
        context_block = "\n".join(context_lines) if context_lines else ""
        # Include org settings (TODO sequence and default tags)
        from config import settings
        todo_sequence = (settings.ORG_TODO_SEQUENCE or "TODO|DONE").split("|")
        default_tags = [t.strip() for t in (settings.ORG_DEFAULT_TAGS or "").split(",") if t.strip()]
        # Include top existing tags from index (best-effort)
        try:
            tag_index = await org_inbox_index_tags()
        except Exception:
            tag_index = {}
        top_existing_tags = sorted(tag_index.items(), key=lambda kv: kv[1], reverse=True)[:15]
        top_tag_list = [k for k, _ in top_existing_tags]

        available_tags = list(set(default_tags + top_tag_list))
        prompt_lines: List[str] = []
        persona = (state.get("persona") or {}).get("persona_style") or "professional"
        prompt_lines.append("You are Roosevelt's Org-Mode Personal Assistant. Analyze the user's request, act like a helpful assistant, and produce both structured JSON for execution AND a natural confirmation.")
        prompt_lines.append("")
        prompt_lines.append("USER MESSAGE:")
        prompt_lines.append(text)
        prompt_lines.append("")
        prompt_lines.append("CONTEXT (recent conversation messages):")
        prompt_lines.append(context_block)
        prompt_lines.append("")
        prompt_lines.append(f"AVAILABLE TODO STATES (in order): {todo_sequence}")
        prompt_lines.append(f"AVAILABLE TAGS (defaults + top existing): {available_tags}")
        prompt_lines.append("")
        prompt_lines.append("REQUIREMENTS:")
        prompt_lines.append("- Resolve pronouns like \"it/this/that\" to the most likely actionable phrase from the message.")
        prompt_lines.append("- Choose entry_kind: \"todo\" for tasks, \"event\" for appointments/meetings/birthdays, \"contact\" for people/contacts, \"checkbox\" for quick lists.")
        prompt_lines.append("- For contacts, extract properties into contact_properties object:")
        prompt_lines.append("  * Name fields: FIRST_NAME, MIDDLE_NAME, LAST_NAME")
        prompt_lines.append("  * Contact info: EMAIL_HOME, EMAIL_WORK, PHONE_MOBILE, PHONE_WORK, PHONE_HOME")
        prompt_lines.append("  * Organization: COMPANY, TITLE")
        prompt_lines.append("  * Location: ADDRESS_HOME, ADDRESS_WORK")
        prompt_lines.append("  * Web: WEBSITE_PERSONAL, WEBSITE_BUSINESS")
        prompt_lines.append("  * Social: SOCIAL_LINKEDIN, SOCIAL_TWITTER, SOCIAL_FACEBOOK, etc.")
        prompt_lines.append("  * Personal: BIRTHDAY (YYYY-MM-DD), ANNIVERSARY (YYYY-MM-DD), RELATIONSHIP, SPOUSE, CHILD_1, CHILD_2, etc.")
        prompt_lines.append("  * Notes: Any additional info goes in contact_properties[\"NOTES\"]")
        prompt_lines.append("- Extract schedule as an org timestamp like <YYYY-MM-DD Dow> if a date/time window is present; else null.")
        prompt_lines.append("- If repeating cadence is implied (weekly, daily, monthly), return a repeater like +1w, .+1m; else null.")
        prompt_lines.append("- Suggest up to 3 tags as simple lowercase slugs.")
        prompt_lines.append("- If the request is ambiguous, set clarification_needed true and propose a brief clarification_question.")
        prompt_lines.append("- Prefer choosing TODO states and tags from the provided AVAILABLE lists when applicable.")
        prompt_lines.append(f"- Generate a concise assistant_confirmation in a {persona} tone (one sentence) describing what was added/scheduled and what tags/links were included. If clarification_needed, phrase a single clear question instead.")
        prompt_lines.append("")
        prompt_lines.append("OUTPUT JSON SCHEMA:")
        prompt_lines.append("{")
        prompt_lines.append("  \"title\": \"string\",")
        prompt_lines.append("  \"entry_kind\": \"todo\" | \"event\" | \"contact\" | \"checkbox\",")
        prompt_lines.append("  \"schedule\": \"<YYYY-MM-DD Dow>\" | null,")
        prompt_lines.append("  \"repeater\": \"+1w\" | \".+1w\" | \"+1m\" | null,")
        prompt_lines.append("  \"suggested_tags\": [\"string\"],")
        prompt_lines.append("  \"contact_properties\": {\"EMAIL\": \"string\", \"PHONE\": \"string\", \"BIRTHDAY\": \"YYYY-MM-DD\", \"COMPANY\": \"string\"} | null,")
        prompt_lines.append("  \"clarification_needed\": true|false,")
        prompt_lines.append("  \"clarification_question\": \"string|null\"")
        prompt_lines.append(",  \"assistant_confirmation\": \"string|null\"")
        prompt_lines.append("}")
        prompt_lines.append("")
        prompt_lines.append("Respond with ONLY the JSON.")
        prompt = "\n".join(prompt_lines)

        response = await chat_service.openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.2,
        )
        import json
        try:
            data = json.loads(response.choices[0].message.content)
        except Exception:
            # Minimal safe fallback to ensure non-heuristic flow still returns a sane object
            data = {
                "title": text.strip().rstrip("?.!"),
                "entry_kind": "todo",
                "schedule": None,
                "repeater": None,
                "suggested_tags": [],
                "clarification_needed": False,
                "clarification_question": None,
            }
        return OrgInboxInterpretation(
            title=data.get("title", "New entry"),
            entry_kind=data.get("entry_kind", "todo"),
            schedule=data.get("schedule"),
            deadline=None,
            repeater=data.get("repeater"),
            suggested_tags=data.get("suggested_tags", []),
            clarification_needed=bool(data.get("clarification_needed", False)),
            clarification_question=data.get("clarification_question"),
            contact_properties=data.get("contact_properties"),
        )

    def _resolve_pronoun_reference(self, full_text: str, add_span_start: Optional[int] = None) -> Optional[str]:
        """Heuristically resolve pronouns like 'it/this/that' to the preceding actionable phrase.
        Strategy: take the sentence before the 'add ... to my inbox' segment (if provided),
        strip helper prefixes ("I need to", "please"), remove temporal phrases ("this week"), and return.
        """
        import re
        text = full_text.strip()
        cutoff = add_span_start if add_span_start is not None else len(text)
        before = text[:cutoff].strip()
        if not before:
            before = text
        # Split into sentences; take the last non-empty before cutoff
        sentences = re.split(r"(?<=[.!?])\s+", before)
        candidate = ""
        for s in reversed(sentences):
            s = s.strip()
            if s:
                candidate = s
                break
        if not candidate:
            candidate = before
        candidate_lower = candidate.lower()
        # Remove trailing question marks or periods
        candidate = candidate.rstrip("?.! ")
        # Remove common helper prefixes
        helper_prefixes = [
            "i need to ", "i should ", "i have to ", "need to ", "have to ",
            "please ", "can you ", "could you ", "would you ", "add ", "schedule ",
            "let's ", "lets ", "we need to ",
        ]
        for prefix in helper_prefixes:
            if candidate_lower.startswith(prefix):
                candidate = candidate[len(prefix):]
                candidate_lower = candidate.lower()
                break
        # Remove simple temporal phrases
        temporal_patterns = [
            r"\bthis week\b", r"\btoday\b", r"\btomorrow\b", r"\btonight\b",
            r"\bthis weekend\b", r"\bnext week\b", r"\bthis afternoon\b",
            r"\bthis evening\b", r"\bthis morning\b",
        ]
        for pat in temporal_patterns:
            candidate = re.sub(pat, "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        # Title minimal normalization: capitalize first char if starts with a letter
        if candidate:
            # Prefer removing possessive "my" only if it improves brevity without losing meaning
            cand_no_my = re.sub(r"^my\s+", "", candidate, flags=re.IGNORECASE).strip()
            # Keep original if removing 'my' empties or worsens clarity
            candidate = cand_no_my or candidate
            # Capitalize first letter
            candidate = candidate[0].upper() + candidate[1:]
        return candidate or None

    def _extract_scheduled_timestamp(self, text: str) -> Optional[str]:
        try:
            import re
            from datetime import datetime
            # Very lightweight detection for patterns like 'March 30th, 2026' or 'Mar 30, 2026'
            # Normalize suffixes st/nd/rd/th
            cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)
            # Try multiple strptime patterns
            patterns = [
                "%B %d, %Y",  # March 30, 2026
                "%b %d, %Y",  # Mar 30, 2026
                "%m/%d/%Y",
                "%Y-%m-%d",
            ]
            for pat in patterns:
                try:
                    dt = datetime.strptime(cleaned.strip(), pat)
                    return f"<{dt.strftime('%Y-%m-%d %a')}>"
                except Exception:
                    continue
            # Fallback: extract a date-like token from the string
            date_match = re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})", cleaned)
            if date_match:
                try:
                    dt = datetime.strptime(date_match.group(1), "%B %d, %Y")
                    return f"<{dt.strftime('%Y-%m-%d %a')}>"
                except Exception:
                    pass
            return None
        except Exception:
            return None

    def _get_todo_sequence(self) -> List[str]:
        from config import settings
        seq = settings.ORG_TODO_SEQUENCE.split("|") if getattr(settings, "ORG_TODO_SEQUENCE", None) else ["TODO", "DONE"]
        return [s.strip() for s in seq if s.strip()]

    def _determine_entry_type(self, text: str) -> str:
        """Determine if this should be a TODO, event, or other org-mode entry type."""
        text_lower = text.lower()
        
        # Birthday and anniversary patterns
        birthday_keywords = ["birthday", "birth day", "bday", "anniversary", "born"]
        if any(keyword in text_lower for keyword in birthday_keywords):
            return ""  # Plain headline for birthdays/events
        
        # Holiday and celebration patterns  
        holiday_keywords = ["holiday", "celebration", "party", "wedding", "graduation"]
        if any(keyword in text_lower for keyword in holiday_keywords):
            return ""  # Plain headline for events
        
        # Meeting and appointment patterns
        meeting_keywords = ["meeting", "appointment", "call", "interview", "conference"]
        if any(keyword in text_lower for keyword in meeting_keywords):
            return ""  # Plain headline for appointments
        
        # Task patterns (things to do)
        task_keywords = ["todo", "task", "complete", "finish", "work on", "need to"]
        if any(keyword in text_lower for keyword in task_keywords):
            return "TODO"
        
        # Default to plain headline for scheduled items
        return ""

    def _extract_title(self, text: str) -> str:
        # If the text is a full sentence like "Can you add my birthday (March 30th, 2026) to my org inbox?"
        # produce a concise title
        base = text.strip()
        text_lower = base.lower()
        
        # Special handling for birthdays
        if any(keyword in text_lower for keyword in ["birthday", "birth day", "bday"]):
            # Extract whose birthday it is
            if "my birthday" in text_lower:
                return "My Birthday"
            else:
                # Try to extract a name before "birthday"
                import re
                name_match = re.search(r"(\w+(?:\s+\w+)?)'s\s+birthday", text_lower)
                if name_match:
                    return f"{name_match.group(1).title()}'s Birthday"
                return "Birthday"
        
        # Special handling for anniversaries
        if "anniversary" in text_lower:
            return "Anniversary"
        
        # Simple heuristics for general entries
        for prefix in ["can you ", "please ", "add ", "schedule "]:
            if base.lower().startswith(prefix):
                base = base[len(prefix):]
                break
        
        # Remove question mark and parentheses content
        import re
        base = re.sub(r"\([^)]*\)", "", base)
        base = base.replace("?", "").strip()
        if len(base) > 80:
            base = base[:77] + "..."
        if len(base) == 0:
            base = "New entry"
        return base


