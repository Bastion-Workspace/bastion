"""
Org Project Agent
LangGraph agent for capturing projects into inbox.org with HITL preview-confirm flow
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from enum import Enum

from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"


class OrgProjectCaptureIntent(BaseModel):
    """Structured intent for capturing a project into Org inbox"""
    title: str = Field(description="Project title")
    description: Optional[str] = Field(default=None, description="Short project description/goal")
    target_date: Optional[str] = Field(default=None, description="Org timestamp like <YYYY-MM-DD Dow>")
    tags: List[str] = Field(default_factory=lambda: ["project"], description="Tag list for the project")
    initial_tasks: List[str] = Field(default_factory=list, description="Up to 5 starter TODO items")


class OrgProjectAgent(BaseAgent):
    """
    Org Project Agent for capturing structured projects into inbox.org
    
    Uses HITL (preview-confirm) workflow for user validation before writing
    """
    
    def __init__(self):
        super().__init__("org_project_agent")
        self._grpc_client = None
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            from orchestrator.clients.backend_tool_client import get_backend_tool_client
            self._grpc_client = await get_backend_tool_client()
        return self._grpc_client
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process project capture request with HITL flow"""
        try:
            logger.info("🗂️ Org Project Agent: Starting project capture")
            
            # Extract state components
            messages = state.get("messages", [])
            user_id = state.get("user_id")
            shared_memory = state.get("shared_memory", {})
            pending = shared_memory.get("pending_project_capture", {})
            
            # Get latest user message
            latest_message = messages[-1] if messages else None
            user_message = latest_message.content if hasattr(latest_message, 'content') else ""
            
            # Case 0: Pending capture awaiting more details (not confirmation)
            if pending and not pending.get("awaiting_confirmation"):
                pending = self._merge_user_details_into_pending(pending, user_message)
                
                intent = OrgProjectCaptureIntent(
                    title=pending.get("title", ""),
                    description=pending.get("description"),
                    target_date=pending.get("target_date"),
                    tags=pending.get("tags", ["project"]),
                    initial_tasks=pending.get("initial_tasks", [])
                )
                
                remaining_missing = self._compute_missing_fields(intent)
                
                if not remaining_missing:
                    # Build preview and request confirmation
                    preview = self._build_project_block_preview(pending)
                    pending["preview_block"] = preview
                    pending["awaiting_confirmation"] = True
                    shared_memory["pending_project_capture"] = pending
                    response = self._build_preview_message(preview)
                    
                    return {
                        "messages": [AIMessage(content=response)],
                        "shared_memory": shared_memory,
                        "agent_results": {
                            "agent_type": "org_project_agent",
                            "task_status": "permission_required",
                            "preview": preview
                        },
                        "requires_user_input": True,
                        "is_complete": False
                    }
                else:
                    # Request missing fields
                    question = self._build_clarification_question(intent, remaining_missing)
                    pending["missing_fields"] = remaining_missing
                    shared_memory["pending_project_capture"] = pending
                    response = f"To capture this project, please provide: {', '.join(remaining_missing)}.\n{question}"
                    
                    return {
                        "messages": [AIMessage(content=response)],
                        "shared_memory": shared_memory,
                        "agent_results": {
                            "agent_type": "org_project_agent",
                            "task_status": "incomplete",
                            "missing_fields": remaining_missing
                        },
                        "requires_user_input": True,
                        "is_complete": False
                    }
            
            # Case 1: Awaiting confirmation
            if pending.get("awaiting_confirmation"):
                if self._is_confirmation(user_message):
                    return await self._commit_project_block(state, pending, user_id)
                elif self._is_cancellation(user_message):
                    shared_memory.pop("pending_project_capture", None)
                    response = "Project capture cancelled."
                    return {
                        "messages": [AIMessage(content=response)],
                        "shared_memory": shared_memory,
                        "agent_results": {
                            "agent_type": "org_project_agent",
                            "task_status": "complete",
                            "cancelled": True
                        },
                        "is_complete": True
                    }
                else:
                    # Treat as edits
                    pending = self._merge_user_details_into_pending(pending, user_message)
                    preview = self._build_project_block_preview(pending)
                    pending["preview_block"] = preview
                    shared_memory["pending_project_capture"] = pending
                    response = self._build_preview_message(preview)
                    
                    return {
                        "messages": [AIMessage(content=response)],
                        "shared_memory": shared_memory,
                        "agent_results": {
                            "agent_type": "org_project_agent",
                            "task_status": "permission_required",
                            "preview": preview
                        },
                        "requires_user_input": True,
                        "is_complete": False
                    }
            
            # Case 2: Initialize new capture with smart enrichment
            intent = self._derive_initial_intent(user_message)
            
            # Use LLM to extract description and tasks
            try:
                intent = await self._smart_enrich_intent_from_message(intent, user_message, state)
            except Exception as enrich_err:
                logger.warning(f"⚠️ Smart enrichment skipped: {enrich_err}")
            
            # Check for missing fields
            missing = self._compute_missing_fields(intent)
            
            if missing:
                question = self._build_clarification_question(intent, missing)
                shared_memory["pending_project_capture"] = {
                    "title": intent.title,
                    "description": intent.description,
                    "target_date": intent.target_date,
                    "tags": intent.tags or ["project"],
                    "initial_tasks": intent.initial_tasks,
                    "missing_fields": missing,
                    "awaiting_confirmation": False
                }
                response = f"To capture this project, please provide: {', '.join(missing)}.\n{question}"
                
                return {
                    "messages": [AIMessage(content=response)],
                    "shared_memory": shared_memory,
                    "agent_results": {
                        "agent_type": "org_project_agent",
                        "task_status": "incomplete",
                        "missing_fields": missing
                    },
                    "requires_user_input": True,
                    "is_complete": False
                }
            
            # Build preview and request confirmation
            preview = self._build_project_block_preview(intent.dict())
            pending = {
                "title": intent.title,
                "description": intent.description,
                "target_date": intent.target_date,
                "tags": intent.tags or ["project"],
                "initial_tasks": intent.initial_tasks,
                "preview_block": preview,
                "awaiting_confirmation": True
            }
            shared_memory["pending_project_capture"] = pending
            response = self._build_preview_message(preview)
            
            return {
                "messages": [AIMessage(content=response)],
                "shared_memory": shared_memory,
                "agent_results": {
                    "agent_type": "org_project_agent",
                    "task_status": "permission_required",
                    "preview": preview
                },
                "requires_user_input": True,
                "is_complete": False
            }
            
        except Exception as e:
            logger.error(f"❌ Org Project Agent ERROR: {e}")
            return self._create_error_result(f"Project capture error: {str(e)}")
    
    def _derive_initial_intent(self, user_message: str) -> OrgProjectCaptureIntent:
        """Extract initial project intent from user message"""
        title = user_message.strip()
        
        # Cleanup common prefixes
        lowered = title.lower()
        for lead in ["start project", "create project", "new project", "project:", "project "]:
            if lowered.startswith(lead):
                title = title[len(lead):].strip(" -:–—")
                break
        
        return OrgProjectCaptureIntent(title=title, tags=["project"])
    
    async def _smart_enrich_intent_from_message(
        self,
        intent: OrgProjectCaptureIntent,
        user_message: str,
        state: Dict[str, Any]
    ) -> OrgProjectCaptureIntent:
        """Use LLM to extract description and starter tasks from message"""
        if intent.description and intent.initial_tasks:
            return intent
        
        chat_service = await self._get_chat_service()
        
        system_prompt = (
            "You are an Org Project Capture Assistant. "
            "Extract a concise project description (1-2 sentences) "
            "and up to 5 concrete starter tasks from the user message. "
            "If insufficient detail exists, leave fields empty. "
            "Respond with VALID JSON only."
        )
        
        schema_instructions = (
            "STRUCTURED OUTPUT REQUIRED:\n"
            "{\n"
            '  "description": "string or empty",\n'
            '  "initial_tasks": ["task 1", "task 2"]  // up to 5, may be empty\n'
            "}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": schema_instructions},
            {"role": "user", "content": f"USER MESSAGE: {user_message}"}
        ]
        
        response = await chat_service.openai_client.chat.completions.create(
            messages=messages,
            model=chat_service.model,
            temperature=0.2
        )
        
        content = (response.choices[0].message.content or "").strip()
        
        try:
            data = json.loads(content)
            description = (data.get("description") or "").strip()
            tasks_raw = data.get("initial_tasks") or []
            
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
            
            if description and not intent.description:
                intent.description = description
            if tasks and not intent.initial_tasks:
                intent.initial_tasks = tasks
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse smart enrichment JSON: {e}")
        
        return intent
    
    def _compute_missing_fields(self, intent: OrgProjectCaptureIntent) -> List[str]:
        """Determine which required fields are missing"""
        missing = []
        if not intent.description:
            missing.append("description")
        if not intent.initial_tasks:
            missing.append("initial_tasks")
        return missing
    
    def _build_clarification_question(self, intent: OrgProjectCaptureIntent, missing: List[str]) -> str:
        """Build question requesting missing information"""
        return (
            "Please reply with a short description (1-2 sentences), up to 5 starter tasks "
            "(bulleted or comma-separated), and an optional target date as <YYYY-MM-DD Dow>."
        )
    
    def _merge_user_details_into_pending(self, pending: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Merge user-provided details into pending capture"""
        text = user_message.strip()
        
        # Parse labeled fields first
        labeled_desc, labeled_tasks = self._parse_labeled_fields(text)
        if labeled_desc and not pending.get("description"):
            pending["description"] = labeled_desc
        if labeled_tasks:
            merged = list(dict.fromkeys((pending.get("initial_tasks") or []) + labeled_tasks))
            pending["initial_tasks"] = merged[:5]
        
        # Fallback parsing
        tasks = []
        description_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith(("description:", "desc:", "tasks:")):
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
        
        # Target date from org timestamp
        m = re.search(r"<\d{4}-\d{2}-\d{2}[^>]*>", text)
        if m and not pending.get("target_date"):
            pending["target_date"] = m.group(0)
        
        return pending
    
    def _parse_labeled_fields(self, text: str) -> tuple:
        """Parse Description: and Tasks: labels"""
        description = []
        tasks = []
        current_section = None
        
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
                    if ";" in rest:
                        tasks.extend([p.strip() for p in rest.split(";") if p.strip()])
                    else:
                        tasks.append(rest)
                continue
            
            if current_section == "description" and line:
                description.append(line)
            elif current_section == "tasks" and line:
                if line.startswith(("- ", "* ")):
                    tasks.append(line[2:].strip())
                else:
                    tasks.append(line)
        
        desc_text = " ".join(description).strip() if description else ""
        tasks_dedup = list(dict.fromkeys([t.strip() for t in tasks if t and t.strip()]))[:5]
        
        return desc_text, tasks_dedup
    
    def _build_project_block_preview(self, data: Dict[str, Any]) -> str:
        """Build org-mode formatted project block"""
        title = (data.get("title") or "Untitled Project").strip()
        tags = data.get("tags") or ["project"]
        tag_suffix = ":" + ":".join(sorted({t.strip(': ') for t in tags if t})) + ":"
        created = datetime.now().strftime("[%Y-%m-%d %a %H:%M]")
        
        lines = []
        lines.append(f"* {title} {tag_suffix}")
        lines.append(":PROPERTIES:")
        lines.append(f":ID:       {datetime.now().strftime('%Y%m%d%H%M%S')}")
        lines.append(f":CREATED:  {created}")
        lines.append(":END:")
        
        desc = (data.get("description") or "").strip()
        if desc:
            lines.append(desc)
        
        td = (data.get("target_date") or "").strip()
        if td:
            lines.append(f"SCHEDULED: {td}")
        
        for t in (data.get("initial_tasks") or []):
            t_clean = t.strip()
            if t_clean:
                lines.append(f"** TODO {t_clean}")
        
        return "\n".join(lines) + "\n"
    
    def _build_preview_message(self, preview_block: str) -> str:
        """Build preview message for user confirmation"""
        return (
            "Here's the project preview. Shall I add it to inbox.org?\n\n" +
            "```org\n" + preview_block.rstrip("\n") + "\n```\n" +
            "Reply 'yes' to proceed, or edit details (description, tasks, date)."
        )
    
    def _is_confirmation(self, user_message: str) -> bool:
        """Check if user message is a confirmation"""
        lm = (user_message or "").strip().lower()
        return any(w in lm for w in ["yes", "y", "ok", "okay", "proceed", "do it", "confirm"])
    
    def _is_cancellation(self, user_message: str) -> bool:
        """Check if user message is a cancellation"""
        lm = (user_message or "").strip().lower()
        return any(w in lm for w in ["no", "cancel", "stop", "abort"])
    
    async def _commit_project_block(self, state: Dict[str, Any], pending: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
        """Write project block to inbox.org via gRPC"""
        try:
            preview = pending.get("preview_block") or self._build_project_block_preview(pending)
            
            grpc_client = await self._get_grpc_client()
            result = await grpc_client.append_org_inbox_text(
                text=preview,
                user_id=user_id
            )
            
            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))
            
            path = result.get("path")
            line_start = result.get("line_start_index")
            line_end = result.get("line_end_index")
            
            # Clear pending
            shared_memory = state.get("shared_memory", {})
            shared_memory.pop("pending_project_capture", None)
            
            response = f"Project added to inbox: {path} (lines {line_start}-{line_end})."
            
            return {
                "messages": [AIMessage(content=response)],
                "shared_memory": shared_memory,
                "agent_results": {
                    "agent_type": "org_project_agent",
                    "task_status": "complete",
                    "path": path,
                    "line_start": line_start,
                    "line_end": line_end
                },
                "is_complete": True
            }
            
        except Exception as e:
            logger.error(f"❌ Commit project failed: {e}")
            response = f"Failed to write project to inbox.org: {e}"
            return self._create_error_result(response)
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.error(f"❌ Org Project Agent error: {error_message}")
        return {
            "messages": [AIMessage(content=f"Project capture failed: {error_message}")],
            "agent_results": {
                "agent_type": "org_project_agent",
                "task_status": "error",
                "error": error_message
            },
            "is_complete": True
        }


# Singleton instance
_org_project_agent_instance = None


def get_org_project_agent() -> OrgProjectAgent:
    """Get global org project agent instance"""
    global _org_project_agent_instance
    if _org_project_agent_instance is None:
        _org_project_agent_instance = OrgProjectAgent()
    return _org_project_agent_instance







