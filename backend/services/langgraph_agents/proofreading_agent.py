"""
Proofreading Agent - Roosevelt's MASTER COPY EDITOR

Scopes the active editor content (chapter or whole doc under 7,500 words),
loads referenced style guide if present, and returns structured corrections.
Supports modes: clarity, compliance (style guide), and accuracy (may request
permission for fact checks via research agent flow).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, TaskStatus
from langchain_core.messages import AIMessage
from models.proofreading_models import (
    ProofreadingResult,
    CorrectionEntry,
    get_proofreading_structured_output,
)
from utils.chapter_scope import find_chapter_ranges, locate_chapter_index, paragraph_bounds
from services.file_context_loader import FileContextLoader


logger = logging.getLogger(__name__)


def _strip_frontmatter_block(text: str) -> str:
    try:
        import re
        return re.sub(r'^---\s*\n[\s\S]*?\n---\s*\n', '', text, flags=re.MULTILINE)
    except Exception:
        return text


class ProofreadingAgent(BaseAgent):
    def __init__(self):
        super().__init__("proofreading_agent")
        logger.info("📝 BULLY! Proofreading Agent mounted and ready to bust errors!")

    def _build_system_prompt(self) -> str:
        # Identity + rules derived from user's desktop prompt builder
        return (
            "You are a MASTER COPY EDITOR and PROFESSIONAL PROOFREADER with expertise in grammar, style, "
            "consistency, and technical accuracy. Identify and correct errors while maintaining the author's voice.\n\n"
            "=== RESPONSE FORMAT ===\n"
            "Return ONLY valid JSON matching the provided schema. For each specific correction, produce entries with:\n"
            "- original_text: exact, verbatim span from the source including surrounding punctuation\n"
            "- changed_to: your corrected version\n"
            "- explanation: short reason only when unclear\n\n"
            "CRITICAL SCOPE ANALYSIS: When a change affects surrounding grammar, select the full sentence or clause."
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            shared_memory = state.get("shared_memory", {}) or {}
            active_editor = shared_memory.get("active_editor", {}) or {}

            manuscript = active_editor.get("content", "") or ""
            filename = active_editor.get("filename") or "document.md"
            frontmatter = active_editor.get("frontmatter", {}) or {}
            cursor_offset = int(active_editor.get("cursor_offset", -1))
            canonical_path = active_editor.get("canonical_path") or None

            # Robust type detection: trim and fallback to parsing manuscript frontmatter
            doc_type_raw = frontmatter.get("type", "")
            doc_type = str(doc_type_raw).strip().lower()
            if doc_type not in ("fiction", "non-fiction", "nonfiction", "article"):
                try:
                    from utils.frontmatter_utils import parse_frontmatter as _pf
                    parsed_fm, _ = _pf(manuscript)
                    if isinstance(parsed_fm, dict):
                        doc_type_parsed = str(parsed_fm.get("type", "")).strip().lower()
                        if doc_type_parsed in ("fiction", "non-fiction", "nonfiction", "article"):
                            frontmatter = parsed_fm
                            doc_type = doc_type_parsed
                except Exception:
                    pass
            if doc_type not in ("fiction", "non-fiction", "nonfiction", "article"):
                return self._create_success_result(
                    response="Active editor is not fiction/non-fiction/article; proofreading agent skipping.",
                    tools_used=[],
                    processing_time=0.0,
                    additional_data={"skipped": True},
                )

            # Load referenced style guide if present
            style_text = None
            try:
                fm_for_loader = dict(frontmatter)
                if canonical_path:
                    fm_for_loader["__canonical_path__"] = canonical_path
                loader = FileContextLoader()
                loaded = loader.load_referenced_context(filename, fm_for_loader)
                if loaded.style:
                    style_text = loaded.style.content
            except Exception:
                style_text = None

            # Determine content scope: chapter around cursor, or entire doc if < 7,500 words
            body = _strip_frontmatter_block(manuscript)
            word_count = len((body or "").split())
            scope_text = body
            if word_count >= 7500:
                try:
                    chapter_ranges = find_chapter_ranges(manuscript)
                    if cursor_offset >= 0 and chapter_ranges:
                        idx = locate_chapter_index(chapter_ranges, cursor_offset)
                        if idx != -1:
                            rng = chapter_ranges[idx]
                            scope_text = _strip_frontmatter_block(manuscript[rng.start:rng.end])
                        else:
                            # fallback to paragraph
                            p0, p1 = paragraph_bounds(manuscript, max(cursor_offset, 0))
                            scope_text = _strip_frontmatter_block(manuscript[p0:p1])
                    else:
                        # fallback to paragraph around 0
                        p0, p1 = paragraph_bounds(manuscript, 0)
                        scope_text = _strip_frontmatter_block(manuscript[p0:p1])
                except Exception:
                    p0, p1 = paragraph_bounds(manuscript, max(cursor_offset, 0))
                    scope_text = _strip_frontmatter_block(manuscript[p0:p1])

            # Determine mode from latest user message intent
            requested_mode = self._infer_mode_from_request(state)

            system_prompt = self._build_system_prompt()
            now_line = f"Current Date/Time: {datetime.now().isoformat()}"

            style_block = (
                "=== AUTHOR'S STYLE GUIDE ===\n"
                "The following style guide defines the author's preferred conventions.\n"
                "CRITICAL: When style conflicts with general grammar, FOLLOW THE STYLE GUIDE.\n\n"
                f"{style_text}\n"
            ) if style_text else ""

            # Build messages with strict structured output requirement
            required_schema_hint = (
                "STRUCTURED OUTPUT REQUIRED: Respond ONLY with valid JSON for ProofreadingResult. Fields: "
                "task_status, mode, summary, corrections[] (original_text, changed_to, explanation, scope), "
                "style_guide_used, consistency_checks, permission_request, metadata. "
                "SCOPE must be one of: 'word', 'phrase', 'clause', 'sentence', 'paragraph', 'duplicate'. "
                "Do not use any other scope values. "
                "CRITICAL: Provide exact, verbatim original_text that can be found in the source document. "
                "The system will automatically generate editor operations from your corrections."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": now_line},
                {"role": "user", "content": (
                    "=== DOCUMENT METADATA ===\n"
                    f"Filename: {filename}\nType: {doc_type}\n\n"
                    + (style_block if style_block else "")
                    + "=== SCOPE TEXT (frontmatter stripped) ===\n"
                    f"{scope_text}\n\n"
                    "=== RESPONSE DIRECTIVES ===\n"
                    "- Respect intentional choices and genre conventions.\n"
                    "- Focus on actual errors; maintain voice.\n"
                    "- Use complete, verbatim original_text with natural boundaries.\n"
                    f"- Mode: {requested_mode or 'clarity'} (clarity|compliance|accuracy).\n\n"
                    + required_schema_hint
                )},
            ]

            # Execute LLM call (no external tools needed)
            # Prefer text-completion model if configured
            chat_service = await self._get_chat_service()
            try:
                from services.settings_service import settings_service
                tc_model = await settings_service.get_text_completion_model()
                model_name = tc_model or await self._get_model_name()
            except Exception:
                model_name = await self._get_model_name()
            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
            )

            content = response.choices[0].message.content or "{}"
            structured = self._parse_structured_proofreading(content)

            # If accuracy mode and model indicates permission_required, surface it
            agent_results = {
                "structured_response": structured.dict(),
                "timestamp": datetime.now().isoformat(),
                "mode": structured.mode or requested_mode or "clarity",
            }

            # ROOSEVELT'S EDITOR INTEGRATION: Generate editor operations from corrections
            editor_ops = self._generate_editor_operations(structured, scope_text, filename)
            structured.editor_operations = editor_ops
            
            state["agent_results"] = agent_results
            state["latest_response"] = self._format_user_message(structured)
            
            # Add editor operations to state for frontend integration
            if editor_ops:
                state["editor_operations"] = editor_ops
                state["manuscript_edit"] = {
                    "target_filename": filename,
                    "operations": editor_ops,
                    "scope": "paragraph",
                    "summary": f"Proofreading corrections ({len(editor_ops)} operations)",
                    "safety": "low"
                }
            
            # Append to LangGraph messages for downstream nodes
            try:
                if state["latest_response"] and str(state["latest_response"]).strip():
                    state.setdefault("messages", []).append(AIMessage(content=state["latest_response"]))
            except Exception:
                pass
            state["is_complete"] = True
            return state

        except Exception as e:
            logger.error(f"❌ Proofreading agent failed: {e}")
            # Graceful error
            error = ProofreadingResult(
                task_status="error",
                mode="clarity",
                summary=f"Proofreading failed: {str(e)}",
                corrections=[],
                metadata={"error": str(e)},
            )
            state["agent_results"] = {"structured_response": error.dict(), "error": True}
            state["latest_response"] = "Proofreading encountered an error."
            state["is_complete"] = True
            return state

    def _parse_structured_proofreading(self, content: str) -> ProofreadingResult:
        try:
            import json
            import re
            text = content.strip()
            
            # Handle various JSON formats
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            
            # Find JSON object if not at start
            if not text.startswith('{'):
                m = re.search(r'\{[\s\S]*\}', text)
                if m:
                    text = m.group(0)
            
            # Parse JSON
            data = json.loads(text)
            
            # Normalize fields
            data.setdefault("task_status", "complete")
            data.setdefault("mode", "clarity")
            data.setdefault("corrections", [])
            data.setdefault("summary", "Proofreading completed")
            data.setdefault("style_guide_used", False)
            data.setdefault("metadata", {})
            
            return ProofreadingResult(**data)
        except Exception as e:
            logger.error(f"❌ Failed to parse proofreading JSON: {e}")
            # Fallback: create a simple result with the raw content
            # Create a more user-friendly fallback
            return ProofreadingResult(
                task_status="complete",
                mode="clarity",
                summary="Proofreading completed with some formatting issues",
                corrections=[CorrectionEntry(
                    original_text="[Formatting issue]", 
                    changed_to="Please review the text manually for corrections", 
                    explanation="LLM response format needs adjustment", 
                    scope="sentence"
                )],
                metadata={"parse_error": str(e), "raw_content_length": len(content)},
            )

    def _format_user_message(self, result: ProofreadingResult) -> str:
        # Render full corrections for chat visibility with requested code blocks
        corrections = list(result.corrections or [])
        mode = result.mode or "clarity"
        header = f"## Proofreading ({mode})\n"
        if result.style_guide_used:
            header += f"Using style guide: {result.style_guide_used}\n\n"
        if not corrections:
            return header + "No corrections suggested."
        lines = [header, f"{len(corrections)} correction(s) suggested:\n"]
        for idx, c in enumerate(corrections, 1):
            try:
                original = c.original_text or ""
                changed = c.changed_to or ""
                explanation = c.explanation or ""
            except Exception:
                original = getattr(c, "original_text", "")
                changed = getattr(c, "changed_to", "")
                explanation = getattr(c, "explanation", "")
            lines.append(f"### {idx}.")
            lines.append("Original text:")
            lines.append("```")
            lines.append(original)
            lines.append("```")
            lines.append("")
            lines.append("Changed to:")
            lines.append("```")
            lines.append(changed)
            lines.append("```")
            if explanation:
                lines.append("")
                lines.append(f"Reason: {explanation}")
            lines.append("")
        return "\n".join(lines)

    def _generate_editor_operations(self, result: ProofreadingResult, scope_text: str, filename: str) -> List[Dict[str, Any]]:
        """Generate editor operations from proofreading corrections."""
        operations = []
        
        if not result.corrections:
            return operations
            
        # Simple hash function for pre_hash (matches frontend implementation)
        def simple_hash(text: str) -> str:
            h = 0
            for char in text:
                h = (h * 31 + ord(char)) & 0xFFFFFFFF
            return hex(h)[2:]
        
        for correction in result.corrections:
            try:
                original_text = correction.original_text or ""
                changed_to = correction.changed_to or ""
                explanation = correction.explanation or ""
                
                # Skip if no actual change
                if original_text == changed_to:
                    continue
                
                # Find the original text in the scope
                start_pos = scope_text.find(original_text)
                if start_pos == -1:
                    # Try to find a partial match or similar text
                    # This is a simplified approach - in production, you might want more sophisticated matching
                    continue
                
                end_pos = start_pos + len(original_text)
                
                # Create editor operation
                operation = {
                    "op_type": "replace_range",
                    "start": start_pos,
                    "end": end_pos,
                    "text": changed_to,
                    "pre_hash": simple_hash(original_text),
                    "note": f"Proofreading: {explanation}" if explanation else "Proofreading correction"
                }
                
                operations.append(operation)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate editor operation for correction: {e}")
                continue
        
        return operations

    def _infer_mode_from_request(self, state: Dict[str, Any]) -> str:
        try:
            latest = ""
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    latest = (msg.get("content") or "").lower()
                    break
                if hasattr(msg, "type") and msg.type == "human":
                    latest = (msg.content or "").lower()
                    break
            if "style" in latest or "compliance" in latest:
                return "compliance"
            if "accuracy" in latest or "fact" in latest or "fact-check" in latest or "fact check" in latest:
                return "accuracy"
            return "clarity"
        except Exception:
            return "clarity"


