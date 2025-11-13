"""
Editor Suggestion Service
Generates inline continuation suggestions for the markdown editor using OpenRouter via LazyChatService.
"""

import logging
import re
from typing import Optional, Dict, Any

from config import settings
from models.editor_models import EditorSuggestionRequest, EditorSuggestionResponse

logger = logging.getLogger(__name__)


class EditorSuggestionService:
    def __init__(self):
        self._lazy = None

    async def _get_client(self):
        if self._lazy is None:
            from services.lazy_chat_service import LazyChatService
            self._lazy = LazyChatService()
            await self._lazy.initialize_minimal()
        return self._lazy

    async def suggest(self, req: EditorSuggestionRequest) -> EditorSuggestionResponse:
        """Return a short continuation suggestion for current cursor position.

        Model selection strategy:
        - Prefer enabled model from LazyChatService (Settings panel managed)
        - If not available, fall back to settings.FAST_MODEL
        """
        try:
            client = await self._get_client()

            # Build concise, grounded prompt
            system_parts = [
                "You complete the user's current sentence or bullet as a subtle inline suggestion.",
                "OUTPUT FORMAT REQUIREMENT: Return ONLY the continuation wrapped EXACTLY in <suggestion>...</suggestion> with no other text.",
                "Do not include quotes, code fences, role markers, or explanations.",
                "No leading newlines; avoid starting with whitespace.",
                "Keep tone consistent with the surrounding content and frontmatter context.",
                "End your continuation at a natural sentence boundary (., !, or ?).",
                "Avoid returning partial words at the end. If you cannot finish a sentence within the limit, stop before the last complete word.",
                "Hard limit to the requested max characters; prefer ending at a natural boundary.",
            ]

            if req.language:
                system_parts.append(f"Language/mode: {req.language}.")

            if isinstance(req.frontmatter, dict) and req.frontmatter:
                title = str(req.frontmatter.get('title', '')).strip()
                audience = str(req.frontmatter.get('audience', '')).strip()
                ftype = str(req.frontmatter.get('type', '')).strip()
                hints = []
                if title:
                    hints.append(f"title='{title}'")
                if audience:
                    hints.append(f"audience='{audience}'")
                if ftype:
                    hints.append(f"type='{ftype}'")
                if hints:
                    system_parts.append("Frontmatter hints: " + ", ".join(hints))

            system_prompt = "\n".join(system_parts)

            # Provide tight local context: last ~750 words for prefix, ~600 chars for suffix
            def _last_n_words(text: str, n: int) -> str:
                if not text:
                    return ""
                words = re.findall(r"\S+", text)
                if len(words) <= n:
                    return text
                # Join the last n words, but preserve original trailing spacing by slicing from end
                # Find start index of the last n-th word in the original string
                target = " ".join(words[-n:])
                # Fallback: if join loses spacing fidelity, slice approximately by length
                approx = text[-max(len(target) + 50, 0):]
                # Try to find the target in the approx slice to keep punctuation/spacing
                idx = approx.rfind(words[-n])
                return approx[idx:] if idx != -1 else target

            prefix_full = req.prefix or ""
            prefix = _last_n_words(prefix_full, 750)
            suffix = (req.suffix or "")[:600]

            user_prompt = (
                "Continue the text at the cursor as inline ghost text.\n\n"
                "--- PREFIX (before cursor) ---\n" + prefix + "\n"
                "--- SUFFIX (after cursor) ---\n" + suffix + "\n\n"
                f"Max characters: {int(req.max_chars)}\n"
                "Return your continuation wrapped EXACTLY as: <suggestion>...your text...</suggestion> and nothing else."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Choose a compact completion (no streaming needed server-side)
            temperature = float(req.temperature or 0.25)
            # Force the dedicated text-completion model for editor suggestions if configured
            try:
                from services.settings_service import settings_service
                tc_model = await settings_service.get_text_completion_model()
                if tc_model:
                    client.current_model = tc_model
                elif not client.current_model:
                    client.current_model = settings.FAST_MODEL
            except Exception:
                # Fallback to FAST_MODEL if anything goes awry
                try:
                    if not client.current_model:
                        client.current_model = settings.FAST_MODEL
                except Exception:
                    pass

            logger.info(
                "📝 EditorSuggest: requesting model=%s max_chars=%s prefix_len=%s suffix_len=%s",
                client.current_model, int(req.max_chars), len(prefix), len(suffix)
            )

            resp = await client.simple_llm_call(
                messages,
                stream=False,
                temperature=temperature,
                max_tokens=200,
            )

            def _extract_content(resp_obj) -> str:
                try:
                    # OpenAI-style chat response
                    msg = getattr(resp_obj.choices[0], "message", None)
                    if msg is not None:
                        c = getattr(msg, "content", None)
                        if isinstance(c, str) and c.strip():
                            return c.strip()
                        # Some providers return list-of-parts
                        if isinstance(c, list):
                            parts = []
                            for p in c:
                                if isinstance(p, dict) and isinstance(p.get("text"), str):
                                    parts.append(p["text"]) 
                            if parts:
                                return "".join(parts).strip()
                    # Legacy/text-completion style
                    txt = getattr(resp_obj.choices[0], "text", None)
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
                    # Fallback to dict-style indexing if present
                    ch0 = getattr(resp_obj, "choices", [{}])[0]
                    if isinstance(ch0, dict):
                        c = ch0.get("message", {}).get("content") if isinstance(ch0.get("message"), dict) else None
                        if isinstance(c, str) and c.strip():
                            return c.strip()
                        if isinstance(ch0.get("text"), str) and ch0["text"].strip():
                            return ch0["text"].strip()
                except Exception:
                    pass
                return ""

            content = _extract_content(resp)
            # Prefer extracting from <suggestion> tags if provided
            if content:
                try:
                    m = re.search(r"<suggestion>([\s\S]*?)</suggestion>", content, re.IGNORECASE)
                    if m:
                        content = m.group(1).strip()
                except Exception:
                    pass

            # Sanitize output to avoid leading newline when inserting inline
            content = content.lstrip("\n\r")

            def _trim_to_sentence_boundary(text: str, max_chars: int) -> str:
                s = (text or "").strip()
                if not s:
                    return ""
                # If within limit, keep as is
                if len(s) <= max_chars:
                    return s
                # Soft trim to limit
                trimmed = s[: max_chars]
                # Avoid cutting through a word: backtrack to whitespace/punct
                i = len(trimmed)
                while i > 0 and trimmed[i - 1].isalnum():
                    i -= 1
                if i > 0:
                    trimmed = trimmed[:i].rstrip()
                # Prefer last sentence terminator before cutoff
                last_end = None
                for m in re.finditer(r"[.!?](?=[)\]\}\"\s]|$)", trimmed):
                    last_end = m.end()
                if last_end:
                    return trimmed[:last_end].rstrip()
                return trimmed

            # Truncate gently to requested max and prefer ending on a sentence boundary
            content = _trim_to_sentence_boundary(content, int(req.max_chars))

            # Try to capture finish reason if present for diagnostics
            finish_reason = None
            try:
                finish_reason = getattr(resp.choices[0], "finish_reason", None)
                if finish_reason is None and isinstance(getattr(resp, "choices", None), list):
                    ch0 = resp.choices[0]
                    if isinstance(ch0, dict):
                        finish_reason = ch0.get("finish_reason")
            except Exception:
                pass

            logger.info(
                "📝 EditorSuggest: model=%s finish_reason=%s suggestion_len=%s",
                client.current_model, finish_reason, len(content or "")
            )

            return EditorSuggestionResponse(
                suggestion=content,
                confidence=0.7,
                model_used=(client.current_model or None),
            )
        except Exception as e:
            logger.error(f"❌ Editor suggestion failed: {e}")
            return EditorSuggestionResponse(suggestion="", confidence=0.0, model_used=None)


editor_suggestion_service = EditorSuggestionService()


