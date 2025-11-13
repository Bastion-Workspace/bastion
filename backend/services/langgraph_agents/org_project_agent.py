"""
Org Project Agent - Roosevelt's "Project Rough Rider"
Captures projects into Org inbox with preview-and-confirm UX.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from services.langgraph_agents.base_agent import BaseAgent, TaskStatus
from models.agent_response_models import OrgProjectCaptureIntent, OrgProjectCaptureResult


logger = logging.getLogger(__name__)


class OrgProjectAgent(BaseAgent):
    """Agent to capture a project into inbox.org with a concise HITL flow."""

    def __init__(self):
        super().__init__("org_project_agent")
        logger.info("🗂️ BULLY! Org Project Agent mounted and ready for a cavalry charge!")

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            user_message = self._extract_current_user_query(state)
            user_id = state.get("user_id")

            shared_memory = state.get("shared_memory", {}) or {}
            pending: Dict[str, Any] = shared_memory.get("pending_project_capture", {}) or {}

            # 0) If we have a pending capture awaiting more details, merge and advance
            if pending and not pending.get("awaiting_confirmation"):
                # Merge new information
                pending = self._merge_user_details_into_pending(pending, user_message)
                # Recompute missing fields
                remaining_missing = self._compute_missing_fields(
                    OrgProjectCaptureIntent(
                        title=pending.get("title") or "",
                        description=pending.get("description"),
                        target_date=pending.get("target_date"),
                        tags=pending.get("tags") or ["project"],
                        initial_tasks=pending.get("initial_tasks") or [],
                    )
                )
                if not remaining_missing:
                    # Build preview and ask for confirmation
                    preview = self._build_project_block_preview(pending)
                    pending["preview_block"] = preview
                    pending["awaiting_confirmation"] = True
                    shared_memory["pending_project_capture"] = pending
                    response = self._build_preview_message(preview)
                    result = OrgProjectCaptureResult(
                        task_status=TaskStatus.PERMISSION_REQUIRED,
                        message="Awaiting confirmation to write project to inbox.org",
                        path=None,
                        preview_block=preview,
                        written_block=None,
                        line_start_index=None,
                        line_end_index=None,
                    )
                    return {
                        "agent_results": {
                            "structured_response": result.dict(),
                            **self._create_agent_result(response=response, task_status=TaskStatus.PERMISSION_REQUIRED)
                        },
                        "latest_response": response,
                        "shared_memory": shared_memory,
                        "requires_user_input": True,
                        "is_complete": False,
                    }
                else:
                    # Ask once more for the specific remaining fields
                    question = self._build_clarification_question(
                        OrgProjectCaptureIntent(
                            title=pending.get("title") or "",
                            description=pending.get("description"),
                            target_date=pending.get("target_date"),
                            tags=pending.get("tags") or ["project"],
                            initial_tasks=pending.get("initial_tasks") or [],
                        ),
                        remaining_missing,
                    )
                    pending["missing_fields"] = remaining_missing
                    shared_memory["pending_project_capture"] = pending
                    response = f"BULLY! To capture this project, please provide: {', '.join(remaining_missing)}.\n{question}"
                    result = OrgProjectCaptureResult(
                        task_status=TaskStatus.INCOMPLETE,
                        message="Awaiting project details",
                        path=None,
                        preview_block=None,
                        written_block=None,
                        line_start_index=None,
                        line_end_index=None,
                    )
                    return {
                        "agent_results": {
                            "structured_response": result.dict(),
                            **self._create_agent_result(response=response, task_status=TaskStatus.INCOMPLETE)
                        },
                        "latest_response": response,
                        "shared_memory": shared_memory,
                        "requires_user_input": True,
                        "is_complete": False,
                    }

            # 1) If awaiting confirmation, act on user response
            if pending.get("awaiting_confirmation"):
                if self._is_confirmation(user_message):
                    return await self._commit_project_block(state, pending, user_id)
                elif self._is_cancellation(user_message):
                    shared_memory.pop("pending_project_capture", None)
                    response = "Project capture cancelled."
                    return {
                        "agent_results": self._create_agent_result(response=response, task_status=TaskStatus.COMPLETE),
                        "latest_response": response,
                        "shared_memory": shared_memory,
                        "is_complete": True,
                    }
                else:
                    # Treat as edits to fields (e.g., user provided description/tasks/date)
                    pending = self._merge_user_details_into_pending(pending, user_message)
                    preview = self._build_project_block_preview(pending)
                    pending["preview_block"] = preview
                    shared_memory["pending_project_capture"] = pending
                    response = self._build_preview_message(preview)
                    return {
                        "agent_results": self._create_agent_result(response=response, task_status=TaskStatus.PERMISSION_REQUIRED),
                        "latest_response": response,
                        "shared_memory": shared_memory,
                        "requires_user_input": True,
                        "is_complete": False,
                    }

            # 2) Otherwise, initialize capture intent with SMART extraction
            intent = self._derive_initial_intent(user_message)

            # Use LLM to extract description and initial tasks from the very first message
            # so the user doesn't have to repeat themselves
            try:
                intent = await self._smart_enrich_intent_from_message(intent, user_message, state)
            except Exception as enrich_err:
                logger.warning(f"⚠️ Smart enrichment skipped due to error: {enrich_err}")

            # Request missing fields in one crisp volley
            missing = self._compute_missing_fields(intent)
            if missing:
                question = self._build_clarification_question(intent, missing)
                # Save pending
                shared_memory["pending_project_capture"] = {
                    "title": intent.title,
                    "description": intent.description,
                    "target_date": intent.target_date,
                    "tags": intent.tags or ["project"],
                    "initial_tasks": intent.initial_tasks,
                    "missing_fields": missing,
                    "awaiting_confirmation": False,
                }
                result = OrgProjectCaptureResult(
                    task_status=TaskStatus.INCOMPLETE,
                    message="Awaiting project details",
                    path=None,
                    preview_block=None,
                    written_block=None,
                    line_start_index=None,
                    line_end_index=None,
                )
                response = f"BULLY! To capture this project, please provide: {', '.join(missing)}.\n{question}"
                return {
                    "agent_results": {
                        "structured_response": result.dict(),
                        **self._create_agent_result(response=response, task_status=TaskStatus.INCOMPLETE)
                    },
                    "latest_response": response,
                    "shared_memory": shared_memory,
                    "requires_user_input": True,
                    "is_complete": False,
                }

            # 3) Build preview and ask for confirmation
            preview = self._build_project_block_preview(intent.dict())
            pending = {
                "title": intent.title,
                "description": intent.description,
                "target_date": intent.target_date,
                "tags": intent.tags or ["project"],
                "initial_tasks": intent.initial_tasks,
                "preview_block": preview,
                "awaiting_confirmation": True,
            }
            shared_memory["pending_project_capture"] = pending
            response = self._build_preview_message(preview)
            result = OrgProjectCaptureResult(
                task_status=TaskStatus.PERMISSION_REQUIRED,
                message="Awaiting confirmation to write project to inbox.org",
                path=None,
                preview_block=preview,
                written_block=None,
                line_start_index=None,
                line_end_index=None,
            )
            return {
                "agent_results": {
                    "structured_response": result.dict(),
                    **self._create_agent_result(response=response, task_status=TaskStatus.PERMISSION_REQUIRED)
                },
                "latest_response": response,
                "shared_memory": shared_memory,
                "requires_user_input": True,
                "is_complete": False,
            }

        except Exception as e:
            logger.error(f"❌ OrgProjectAgent failure: {e}")
            response = f"Project capture error: {str(e)}"
            return {
                "agent_results": self._create_agent_result(response=response, task_status=TaskStatus.ERROR),
                "latest_response": response,
                "is_complete": True,
            }

    # ---------------------
    # Helpers
    # ---------------------
    def _derive_initial_intent(self, user_message: str) -> OrgProjectCaptureIntent:
        title = user_message.strip()
        # Heuristic cleanup
        lowered = title.lower()
        for lead in ["start project", "create project", "new project", "project:", "project "]:
            if lowered.startswith(lead):
                title = title[len(lead):].strip(" -:–—")
                break
        # Default tags include :project:
        return OrgProjectCaptureIntent(title=title, tags=["project"])  # description/tasks/date missing initially

    async def _smart_enrich_intent_from_message(
        self,
        intent: OrgProjectCaptureIntent,
        user_message: str,
        state: Dict[str, Any]
    ) -> OrgProjectCaptureIntent:
        """Use LLM to extract a short description and up to 5 starter tasks from the initial message.

        Structured output enforced to avoid brittle string matching.
        """
        # If both description and tasks already present (e.g., from pending merge), skip
        if intent.description and intent.initial_tasks:
            return intent

        chat_service = await self._get_chat_service()
        model_name = await self._get_model_name()

        system_prompt = (
            "You are Roosevelt's Org Project Capture Intelligence. "
            "Given a single user message, extract a concise project description (1–2 sentences) "
            "and up to 5 concrete starter tasks if reasonably inferable. If insufficient detail exists, "
            "leave fields empty rather than guessing wildly. Respond with VALID JSON only."
        )

        schema_instructions = (
            "STRUCTURED OUTPUT REQUIRED:\n"
            "You MUST respond with valid JSON matching this schema:\n"
            "{\n"
            "  \"description\": \"string or empty if unclear\",\n"
            "  \"initial_tasks\": [\"task 1\", \"task 2\"]  // up to 5, may be empty\n"
            "}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": schema_instructions},
            {"role": "user", "content": f"USER MESSAGE: {user_message}"},
        ]

        response = await chat_service.openai_client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.2
        )

        content = (response.choices[0].message.content or "").strip()
        import json
        try:
            data = json.loads(content)
            description = (data.get("description") or "").strip()
            tasks_raw = data.get("initial_tasks") or []
            # Defensive normalization
            if not isinstance(tasks_raw, list):
                tasks_raw = []
            tasks = []
            for t in tasks_raw:
                if isinstance(t, str):
                    t_clean = t.strip()
                    if t_clean:
                        tasks.append(t_clean)
                if len(tasks) >= 5:
                    break

            # Only apply if we actually got value
            if description and not intent.description:
                intent.description = description
            if tasks and not intent.initial_tasks:
                intent.initial_tasks = tasks
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse smart enrichment JSON: {e}")
            # Fall back to non-LLM heuristics already present
            pass

        return intent

    def _compute_missing_fields(self, intent: OrgProjectCaptureIntent) -> List[str]:
        missing = []
        if not intent.description:
            missing.append("description")
        if not intent.initial_tasks:
            missing.append("initial_tasks")
        return missing

    def _build_clarification_question(self, intent: OrgProjectCaptureIntent, missing: List[str]) -> str:
        return (
            "Please reply with a short description (1–2 sentences), up to 5 starter tasks "
            "(bulleted or comma-separated), and an optional target date as an Org timestamp like <YYYY-MM-DD Dow>."
        )

    def _merge_user_details_into_pending(self, pending: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        text = user_message.strip()
        # 1) Parse labeled fields first
        labeled_desc, labeled_tasks = self._parse_labeled_fields(text)
        if labeled_desc and not pending.get("description"):
            pending["description"] = labeled_desc
        if labeled_tasks:
            merged = list(dict.fromkeys((pending.get("initial_tasks") or []) + labeled_tasks))
            pending["initial_tasks"] = merged[:5]

        # 2) Fallback parsing: bullets or comma-separated become tasks; other lines enrich description
        tasks: List[str] = []
        description_lines: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith(("description:", "desc:")):
                continue
            if stripped.lower().startswith("tasks:"):
                # Already handled; skip here
                continue
            if stripped.startswith(("- ", "* ")):
                tasks.append(stripped[2:].strip())
            else:
                description_lines.append(stripped)
        if not labeled_tasks and not tasks and "," in text:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) >= 2:
                tasks = parts[:5]
                description_lines = []
        if description_lines and not pending.get("description"):
            pending["description"] = " ".join(description_lines).strip()
        if tasks:
            merged = list(dict.fromkeys((pending.get("initial_tasks") or []) + tasks))
            pending["initial_tasks"] = merged[:5]

        # 3) Target date from org timestamp
        import re
        m = re.search(r"<\d{4}-\d{2}-\d{2}[^>]*>", text)
        if m and not pending.get("target_date"):
            pending["target_date"] = m.group(0)
        return pending

    def _parse_labeled_fields(self, text: str) -> tuple[str, List[str]]:
        """Parse Description:/Desc: and Tasks: labels. Returns (description, tasks)."""
        description: List[str] = []
        tasks: List[str] = []
        current_section: Optional[str] = None
        for raw in text.splitlines():
            line = raw.strip()
            lower = line.lower()
            if lower.startswith("description:") or lower.startswith("desc:"):
                current_section = "description"
                rest = line.split(":", 1)[1].strip()
                if rest:
                    description.append(rest)
                continue
            if lower.startswith("tasks:"):
                current_section = "tasks"
                rest = line.split(":", 1)[1].strip()
                if rest:
                    # Single item allowed; also split by ';' if present
                    if ";" in rest:
                        tasks.extend([p.strip() for p in rest.split(";") if p.strip()])
                    else:
                        tasks.append(rest)
                continue
            # Accumulate lines under current section
            if current_section == "description" and line:
                description.append(line)
            elif current_section == "tasks" and line:
                if line.startswith(("- ", "* ")):
                    tasks.append(line[2:].strip())
                else:
                    tasks.append(line)
        desc_text = " ".join(description).strip() if description else ""
        # De-dup and cap tasks
        tasks_dedup = list(dict.fromkeys([t for t in (t or "").strip() for t in tasks if (t or "").strip()]))[:5]
        return desc_text, tasks_dedup

    def _build_project_block_preview(self, data: Dict[str, Any]) -> str:
        title = (data.get("title") or "Untitled Project").strip()
        tags = data.get("tags") or ["project"]
        tag_suffix = ":" + ":".join(sorted({t.strip(': ') for t in tags if t})) + ":"
        created = datetime.now().strftime("[%Y-%m-%d %a %H:%M]")
        lines: List[str] = []
        lines.append(f"* {title} {tag_suffix}")
        lines.append(":PROPERTIES:")
        lines.append(f":ID:       {datetime.now().strftime('%Y%m%d%H%M%S')}")
        lines.append(f":CREATED:  {created}")
        lines.append(":END:")
        desc = (data.get("description") or "").strip()
        if desc:
            lines.append(desc)
        # Optional scheduled date on parent
        td = (data.get("target_date") or "").strip()
        if td:
            lines.append(f"SCHEDULED: {td}")
        # Child TODOs
        for t in (data.get("initial_tasks") or []):
            t_clean = t.strip()
            if t_clean:
                lines.append(f"** TODO {t_clean}")
        return "\n".join(lines) + "\n"

    def _build_preview_message(self, preview_block: str) -> str:
        return (
            "By George! Here's the project preview. Shall I add it to inbox.org?\n\n" +
            "```org\n" + preview_block.rstrip("\n") + "\n```\n" +
            "Reply 'yes' to proceed, or edit details (description, tasks, date)."
        )

    def _is_confirmation(self, user_message: str) -> bool:
        lm = (user_message or "").strip().lower()
        return any(w in lm for w in ["yes", "y", "ok", "okay", "proceed", "do it", "confirm"])

    def _is_cancellation(self, user_message: str) -> bool:
        lm = (user_message or "").strip().lower()
        return any(w in lm for w in ["no", "cancel", "stop", "abort"])

    async def _commit_project_block(self, state: Dict[str, Any], pending: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
        try:
            # Ensure preview exists
            preview = pending.get("preview_block") or self._build_project_block_preview(pending)
            # Write block
            append_block = await self._get_tool_function("org_inbox_append_block")
            if not append_block:
                raise Exception("Tool org_inbox_append_block unavailable")
            tool_result = await append_block(block=preview, user_id=user_id)
            if tool_result.get("error"):
                raise Exception(tool_result.get("error"))

            path = tool_result.get("path")
            line_start = tool_result.get("line_start_index")
            line_end = tool_result.get("line_end_index")

            # Clear pending from shared memory
            shared_memory = state.get("shared_memory", {}) or {}
            shared_memory.pop("pending_project_capture", None)

            result = OrgProjectCaptureResult(
                task_status=TaskStatus.COMPLETE,
                message="Project captured to inbox.org",
                path=path,
                preview_block=preview,
                written_block=preview,
                line_start_index=line_start,
                line_end_index=line_end,
            )
            response = f"BULLY! Project added to inbox: {path} (lines {line_start}–{line_end})."
            return {
                "agent_results": {
                    "structured_response": result.dict(),
                    **self._create_agent_result(response=response, task_status=TaskStatus.COMPLETE)
                },
                "latest_response": response,
                "shared_memory": shared_memory,
                "is_complete": True,
            }
        except Exception as e:
            logger.error(f"❌ Commit project failed: {e}")
            response = f"Failed to write project to inbox.org: {e}"
            return {
                "agent_results": self._create_agent_result(response=response, task_status=TaskStatus.ERROR),
                "latest_response": response,
                "is_complete": True,
            }


