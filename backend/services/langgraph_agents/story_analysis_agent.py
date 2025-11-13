"""
Story Analysis Agent

Provides focused analysis for fiction manuscripts in the active editor.
Consumes `shared_memory.active_editor` and returns a concise analysis message
plus a structured payload in `agent_results`.
"""

import logging
import json
import re
from datetime import datetime
from typing import Any, Dict

from .base_agent import BaseAgent


logger = logging.getLogger(__name__)


class StoryAnalysisAgent(BaseAgent):
    def __init__(self):
        # Reuse chat agent tool set; this agent does not need special tools
        super().__init__("chat_agent")
        logger.info("📖 BULLY! Story Analysis Agent ready to critique fiction with a big stick!")

    def _build_system_prompt(self) -> str:
        return (
            "You are a professional story analysis expert. You will receive the FULL manuscript "
            "for context, but you must FOCUS YOUR ANALYSIS on exactly what the user requested.\n\n"
            "**CRITICAL SCOPE RULES:**\n"
            "- If user asks about a SPECIFIC CHAPTER (e.g., 'Review Chapter 2'), analyze ONLY that chapter\n"
            "- If user asks a SPECIFIC QUESTION (e.g., 'Does Vivian have enough screen time?'), answer that question using evidence from the full manuscript\n"
            "- If user asks about MULTIPLE CHAPTERS (e.g., 'Analyze chapters 3-5'), focus on those chapters\n"
            "- If user asks for GENERAL REVIEW (e.g., 'Review the manuscript', 'Is this ready for publication?'), provide comprehensive analysis of the whole work\n\n"
            "**ALWAYS:**\n"
            "- Start by acknowledging what you're analyzing (e.g., 'Analysis of Chapter 2:' or 'Assessment of Vivian's presence throughout the manuscript:')\n"
            "- Provide specific, actionable recommendations\n"
            "- Reference specific passages when helpful\n"
            "- Keep tone supportive and direct\n"
            "- Use the full manuscript context to provide deeper insights, but stay focused on the user's specific request"
        )

    def _unwrap_json_response(self, content: str) -> str:
        """Unwrap accidental JSON/code-fence envelopes and return plain text."""
        try:
            txt = content.strip()
            if '```json' in txt:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', txt)
                if m:
                    txt = m.group(1).strip()
            elif '```' in txt:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', txt)
                if m:
                    txt = m.group(1).strip()
            # If JSON envelope like {"message": "..."}
            if txt.startswith('{') and txt.endswith('}'):
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    return obj.get('message') or obj.get('text') or content
            return content
        except Exception:
            return content

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}

            manuscript = active_editor.get("content", "") or ""
            filename = active_editor.get("filename") or "document.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}

            doc_type = str((frontmatter.get("type") or "")).strip().lower()
            if doc_type != "fiction":
                # Not a fiction manuscript; this agent is not applicable
                state["latest_response"] = "Story analysis is only available for fiction manuscripts."
                state["is_complete"] = True
                return state

            title = str(frontmatter.get("title") or "").strip()

            # Extract recent user request (for optional focus hints)
            user_request = ""
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    user_request = str(msg.content).lower()
                    break
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_request = str(msg.get("content") or "").lower()
                    break

            header_lines = []
            
            if title:
                header_lines.append(f"STORY TITLE: {title}")
                header_lines.append("")
            
            # **ROOSEVELT: PASS USER REQUEST EXPLICITLY TO FOCUS ANALYSIS**
            header_lines.append("=== USER REQUEST ===")
            # Get the actual user message (not just lowercase version)
            actual_user_request = ""
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    actual_user_request = str(msg.content)
                    break
                if isinstance(msg, dict) and msg.get("role") == "user":
                    actual_user_request = str(msg.get("content") or "")
                    break
            
            if actual_user_request:
                header_lines.append(f'"{actual_user_request}"')
            else:
                header_lines.append("Please provide a comprehensive analysis of this manuscript.")
            
            header_lines.append("")
            header_lines.append("INSTRUCTIONS:")
            header_lines.append("- The FULL MANUSCRIPT is provided below for context")
            header_lines.append("- FOCUS your analysis on exactly what the user requested above")
            header_lines.append("- If analyzing a specific chapter, START with that chapter's heading in your response")
            header_lines.append("- If answering a specific question, START with a direct answer")
            header_lines.append("- Be specific and constructive, with actionable recommendations")
            header_lines.append("- Reference specific passages when relevant")
            header_lines.append("")
            header_lines.append("=== FULL MANUSCRIPT (FOR CONTEXT) ===")
            header_lines.append(manuscript)

            system_prompt = self._build_system_prompt()

            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()

            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "\n".join(header_lines)},
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content or ""
            content = self._unwrap_json_response(content)

            state["agent_results"] = {
                "structured_response": {
                    "task_status": "complete",
                    "analysis_text": content,
                    "mode": "story_analysis",
                    "filename": filename,
                },
                "timestamp": datetime.now().isoformat(),
                "mode": "story_analysis",
            }
            state["latest_response"] = content
            try:
                from langchain_core.messages import AIMessage
                if content and content.strip():
                    state.setdefault("messages", []).append(AIMessage(content=content))
            except Exception:
                pass
            state["is_complete"] = True
            return state
        except Exception as e:
            logger.error(f"❌ Story analysis failed: {e}")
            state["latest_response"] = "Story analysis failed."
            state["is_complete"] = True
            return state


