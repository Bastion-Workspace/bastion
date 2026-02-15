"""
Paragraph-numbered two-phase editing for fiction.

Phase 1: LLM identifies which numbered paragraphs need changes (rewrite/delete).
Phase 2: Batched LLM rewrites for those paragraphs; system constructs operations
with guaranteed-exact original_text from paragraph map.
"""

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.utils.fiction_utilities import unwrap_json_response as _unwrap_json_response

logger = logging.getLogger(__name__)


def _paragraph_by_id(paragraph_map: List[Any], pid: str) -> Any:
    """Get paragraph by id from list of ParagraphInfo or dicts."""
    for p in paragraph_map or []:
        pid_val = getattr(p, "id", None) or (p.get("id") if isinstance(p, dict) else None)
        if pid_val == pid:
            return p
    return None


def _paragraph_text(p: Any) -> str:
    """Get text from ParagraphInfo or dict."""
    if hasattr(p, "text"):
        return p.text
    if isinstance(p, dict):
        return p.get("text", "")
    return ""


async def identify_paragraph_edits_node(
    state: Dict[str, Any], llm_factory
) -> Dict[str, Any]:
    """
    Phase 1: LLM reads numbered chapter and returns which paragraph IDs need
    changes (rewrite/delete) and a brief instruction per edit.
    """
    try:
        generation_context_parts = state.get("generation_context_parts", [])
        current_request = state.get("current_request", "")
        paragraph_map = state.get("paragraph_map", [])
        valid_ids = {getattr(p, "id", None) or (p.get("id") if isinstance(p, dict) else None) for p in paragraph_map}
        valid_ids.discard(None)

        if not paragraph_map:
            logger.warning("Paragraph edit path: paragraph_map empty, skipping identify")
            return {
                "paragraph_edits": [],
                "structured_edit": None,
                "task_status": "error",
                "error": "No paragraph map (empty or unnumbered chapter)",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }

        context_text = "".join(generation_context_parts) if generation_context_parts else ""
        prompt = f"""You are an editor analyzing a chapter that has been numbered by paragraph ([P1], [P2], ...).

**USER REQUEST**: {current_request}

**NUMBERED CHAPTER TEXT** (use paragraph IDs in your response):
{context_text[:120000]}

**YOUR TASK**: Identify which paragraphs need to be changed to fulfill the user's request.
- For each paragraph that needs a change, specify: paragraph_id (e.g. P14), action ("rewrite" or "delete"), and a brief instruction for the rewriter.
- Use "rewrite" when the paragraph should be modified (e.g. remove redundancy, tighten, fix tone).
- Use "delete" when the entire paragraph should be removed (e.g. redundant restatement).
- Only reference paragraph IDs that actually appear in the numbered text above.
- Do not invent paragraph IDs. Valid IDs are P1 through P{len(paragraph_map)}.

**OUTPUT**: Return ONLY valid JSON in this exact format:
{{
  "summary": "One or two sentence human-readable summary of what will change.",
  "edits": [
    {{ "paragraph_id": "P14", "action": "rewrite", "instruction": "Remove the redundant climate system hum reference while keeping atmosphere." }},
    {{ "paragraph_id": "P5", "action": "delete", "instruction": "Redundant restatement of Fleet's refusal already covered in P3." }}
  ]
}}

Return only the JSON object, no markdown fences."""

        llm = llm_factory(temperature=0.3, state=state)
        messages = [
            SystemMessage(content="You are an analytical editor. Return only valid JSON."),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        content = _unwrap_json_response(content)

        data = json.loads(content)
        summary = data.get("summary", "Paragraph-level edits applied.")
        raw_edits = data.get("edits", [])

        # Validate paragraph IDs and filter invalid
        edits = []
        for e in raw_edits:
            pid = (e.get("paragraph_id") or "").strip()
            if not pid:
                continue
            if pid not in valid_ids:
                logger.warning("Identify phase referenced invalid paragraph_id=%s, skipping", pid)
                continue
            action = (e.get("action") or "rewrite").strip().lower()
            if action not in ("rewrite", "delete"):
                action = "rewrite"
            instruction = (e.get("instruction") or "").strip() or "Apply the user's request."
            edits.append({"paragraph_id": pid, "action": action, "instruction": instruction})

        logger.info("Identify phase: %d edits (summary: %s)", len(edits), summary[:80])

        return {
            "paragraph_edits": edits,
            "paragraph_edit_summary": summary,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "paragraph_map": paragraph_map,
            "generation_context_parts": generation_context_parts,
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
        }
    except Exception as e:
        logger.error("Identify paragraph edits failed: %s", e, exc_info=True)
        return {
            "paragraph_edits": [],
            "task_status": "error",
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def rewrite_paragraphs_node(
    state: Dict[str, Any], llm_factory
) -> Dict[str, Any]:
    """
    Phase 2: For each edit with action=rewrite, send paragraph + context to LLM
    in one batched call; collect rewrites. For action=delete, no LLM call.
    """
    try:
        paragraph_map = state.get("paragraph_map", [])
        paragraph_edits = state.get("paragraph_edits", [])
        rewrite_edits = [e for e in paragraph_edits if (e.get("action") or "rewrite") == "rewrite"]
        delete_edits = [e for e in paragraph_edits if (e.get("action") or "") == "delete"]

        paragraph_rewrites: Dict[str, str] = {}

        if not rewrite_edits:
            logger.info("Rewrite phase: no rewrites (only deletes or empty)")
            return {
                "paragraph_rewrites": paragraph_rewrites,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "paragraph_map": paragraph_map,
                "paragraph_edits": paragraph_edits,
                "paragraph_edit_summary": state.get("paragraph_edit_summary", ""),
                "manuscript": state.get("manuscript", ""),
                "filename": state.get("filename", ""),
                "frontmatter": state.get("frontmatter", {}),
                "current_chapter_text": state.get("current_chapter_text", ""),
                "current_chapter_number": state.get("current_chapter_number"),
                "chapter_ranges": state.get("chapter_ranges", []),
                "current_request": state.get("current_request", ""),
            }

        # Build lookup by id and prev/next for context
        by_id = {}
        for i, p in enumerate(paragraph_map or []):
            pid = getattr(p, "id", None) or (p.get("id") if isinstance(p, dict) else None)
            if pid:
                by_id[pid] = {"index": i, "p": p}

        prompt_parts = [
            "You are a fiction editor. Rewrite each paragraph below according to the instruction. "
            "Preserve voice, style, and narrative flow. Return ONLY valid JSON.\n\n",
        ]
        for e in rewrite_edits:
            pid = e.get("paragraph_id", "")
            instruction = e.get("instruction", "Apply the user's request.")
            p = _paragraph_by_id(paragraph_map, pid)
            if not p:
                continue
            text = _paragraph_text(p)
            if not text:
                continue
            idx = by_id.get(pid, {}).get("index", -1)
            prev_text = ""
            next_text = ""
            if idx >= 0 and paragraph_map:
                if idx > 0:
                    prev_p = paragraph_map[idx - 1]
                    prev_text = _paragraph_text(prev_p)
                if idx + 1 < len(paragraph_map):
                    next_p = paragraph_map[idx + 1]
                    next_text = _paragraph_text(next_p)
            prompt_parts.append(f"=== REWRITE {pid} ===\n")
            prompt_parts.append(f"Instruction: {instruction}\n")
            if prev_text:
                prompt_parts.append(f"Context before: \"{prev_text[:200]}...\"\n" if len(prev_text) > 200 else f"Context before: \"{prev_text}\"\n")
            prompt_parts.append(f"ORIGINAL TEXT:\n{text}\n")
            if next_text:
                prompt_parts.append(f"Context after: \"{next_text[:200]}...\"\n" if len(next_text) > 200 else f"Context after: \"{next_text}\"\n")
            prompt_parts.append("\n")

        prompt_parts.append(
            "**OUTPUT**: Return ONLY a JSON object mapping paragraph_id to rewritten text. Example:\n"
            '{"P14": "The rewritten paragraph text here.", "P24": "Another rewritten paragraph."}\n'
            "Keys must match the paragraph IDs above. No markdown fences."
        )

        frontmatter = state.get("frontmatter", {})
        temperature = float(frontmatter.get("temperature", 0.4))
        llm = llm_factory(temperature=temperature, state=state)
        messages = [
            SystemMessage(content="You are a fiction editor. Return only valid JSON: an object with paragraph IDs as keys and rewritten text as values."),
            HumanMessage(content="\n".join(prompt_parts)),
        ]
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        content = _unwrap_json_response(content)
        data = json.loads(content)
        for pid, rewritten in (data or {}).items():
            if isinstance(rewritten, str) and pid in by_id:
                paragraph_rewrites[pid] = rewritten.strip()

        logger.info("Rewrite phase: %d rewrites collected", len(paragraph_rewrites))

        return {
            "paragraph_rewrites": paragraph_rewrites,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "paragraph_map": paragraph_map,
            "paragraph_edits": paragraph_edits,
            "paragraph_edit_summary": state.get("paragraph_edit_summary", ""),
            "manuscript": state.get("manuscript", ""),
            "filename": state.get("filename", ""),
            "frontmatter": state.get("frontmatter", {}),
            "current_chapter_text": state.get("current_chapter_text", ""),
            "current_chapter_number": state.get("current_chapter_number"),
            "chapter_ranges": state.get("chapter_ranges", []),
            "current_request": state.get("current_request", ""),
        }
    except Exception as e:
        logger.error("Rewrite paragraphs failed: %s", e, exc_info=True)
        return {
            "paragraph_rewrites": {},
            "task_status": "error",
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


def construct_operations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ManuscriptEdit from paragraph_map, paragraph_edits, and paragraph_rewrites.
    No LLM call; original_text comes from paragraph map (guaranteed exact).
    """
    try:
        paragraph_map = state.get("paragraph_map", [])
        paragraph_edits = state.get("paragraph_edits", [])
        paragraph_rewrites = state.get("paragraph_rewrites", {})
        summary = state.get("paragraph_edit_summary", "Paragraph-level edits applied.")
        filename = state.get("filename", "manuscript.md")

        operations: List[Dict[str, Any]] = []
        for e in paragraph_edits:
            pid = e.get("paragraph_id", "")
            action = (e.get("action") or "rewrite").strip().lower()
            p = _paragraph_by_id(paragraph_map, pid)
            if not p:
                continue
            original_text = _paragraph_text(p)
            if not original_text:
                continue
            if action == "delete":
                op = {
                    "op_type": "delete_range",
                    "original_text": original_text,
                    "text": "",
                    "note": e.get("instruction", ""),
                }
                operations.append(op)
            else:
                rewritten = paragraph_rewrites.get(pid)
                if rewritten is None:
                    continue
                op = {
                    "op_type": "replace_range",
                    "original_text": original_text,
                    "text": rewritten,
                    "note": e.get("instruction", ""),
                }
                operations.append(op)

        if not operations:
            logger.warning("Construct operations: no valid operations built")
            return {
                "structured_edit": None,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }

        # Build ManuscriptEdit (dict form for state; resolution expects same shape)
        structured_edit = {
            "target_filename": filename,
            "scope": "chapter",
            "summary": summary,
            "safety": "medium",
            "operations": operations,
            "chapter_index": state.get("current_chapter_number"),
        }
        logger.info("Construct operations: %d operations -> structured_edit", len(operations))

        return {
            "structured_edit": structured_edit,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
    except Exception as e:
        logger.error("Construct operations failed: %s", e, exc_info=True)
        return {
            "structured_edit": None,
            "task_status": "error",
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
