"""
Unified file editing tools for Agent Factory.

patch_file: batched semantic edits (insert_after_heading, replace, delete, append).
Edits are sent to the backend; resolution to positions happens just-in-time there.
append_to_file: sugar for appending content; delegates to patch_file.
Both always create DB proposals for user approval.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.frontmatter_utils import frontmatter_end_index

logger = logging.getLogger(__name__)


# ── I/O models ─────────────────────────────────────────────────────────────────

class PatchFileInputs(BaseModel):
    """Required inputs for patch_file."""
    document_id: str = Field(description="Document ID to edit")
    edits: List[Dict[str, Any]] = Field(
        description=(
            "List of edits. Each edit is an object with: "
            "operation: 'replace' | 'delete' | 'insert_after_heading' | 'append'; "
            "target: For replace/delete, the EXACT text to find (include 2-3 surrounding context lines for reliable matching). "
            "For insert_after_heading, the heading text (e.g. '## Section Name'). Not needed for append. "
            "content: The new text for replace/insert_after_heading/append; not needed for delete. "
            "IMPORTANT for replace/delete: 'target' must be VERBATIM text from the document with enough context (2-3 lines) so the match is unambiguous."
        )
    )


class PatchFileParams(BaseModel):
    """Optional parameters for patch_file."""
    summary: str = Field(default="", description="Human-readable summary of changes")
    agent_name: str = Field(default="unknown", description="Agent making the proposal")


class PatchFileOutputs(BaseModel):
    """Outputs for patch_file."""
    success: bool = Field(description="Whether the proposal was created")
    proposal_id: Optional[str] = Field(default=None, description="Proposal ID if created")
    document_id: str = Field(description="Document ID")
    operations_applied: int = Field(description="Number of edits resolved and proposed")
    skipped_edits: List[str] = Field(default_factory=list, description="Reasons for edits that failed resolution")
    message: Optional[str] = Field(default=None, description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class AppendToFileInputs(BaseModel):
    """Required inputs for append_to_file."""
    document_id: str = Field(description="Document ID to edit")
    content: str = Field(description="Content to append to the end of the document")


class AppendToFileParams(BaseModel):
    """Optional parameters for append_to_file."""
    heading: Optional[str] = Field(default=None, description="Optional heading to prepend (e.g. '## New Section')")
    summary: str = Field(default="", description="Human-readable summary of the change")
    agent_name: str = Field(default="unknown", description="Agent making the proposal")


class AppendToFileOutputs(BaseModel):
    """Outputs for append_to_file (same shape as patch_file)."""
    success: bool = Field(description="Whether the proposal was created")
    proposal_id: Optional[str] = Field(default=None, description="Proposal ID if created")
    document_id: str = Field(description="Document ID")
    operations_applied: int = Field(description="Number of edits applied (0 or 1)")
    skipped_edits: List[str] = Field(default_factory=list, description="Reasons for skipped edits")
    message: Optional[str] = Field(default=None, description="Status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── Map Agent Factory edit to internal EditorOperation ───────────────────────────

def _normalize_edit(edit: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common LLM field variants so different models (e.g. Grok) can pass edits."""
    out = dict(edit)
    # operation: accept op_type, type, operation
    if "operation" not in out or not (out.get("operation") or "").strip():
        out["operation"] = (out.get("op_type") or out.get("type") or "").strip() or ""
    # target / text to replace: accept original_text, old_text, target
    if "target" not in out or out.get("target") is None:
        out["target"] = out.get("original_text") or out.get("old_text")
    # content / new text: accept text, new_content, content
    if "content" not in out or out.get("content") is None:
        out["content"] = out.get("text") or out.get("new_content")
    return out


def _edit_to_op_dict(edit: Dict[str, Any], doc_content: str, fm_end: int) -> Optional[Dict[str, Any]]:
    """
    Map a single simplified edit to internal EditorOperation format.
    For append we need doc_content to compute anchor (last line or empty body).
    Returns None if the edit is invalid.

    For replace/delete: target is preserved exactly (no strip) so resolved span matches verbatim.
    For insert_after_heading/append: target is trimmed for anchor matching.
    """
    edit = _normalize_edit(edit)
    op_name = (edit.get("operation") or "").strip().lower()
    # Map common model variants to our operation names
    if op_name in ("replace_range",):
        op_name = "replace"
    elif op_name in ("delete_range",):
        op_name = "delete"
    raw_target = edit.get("target")
    content = edit.get("content")
    if content is None:
        content = ""

    if op_name == "replace":
        # search_text (with context) for robust matching; original_text kept as alias
        target = raw_target if raw_target is not None else ""
        if not target or not target.strip():
            return None
        return {
            "op_type": "replace_range",
            "search_text": target,
            "original_text": target,
            "text": content,
        }
    if op_name == "delete":
        target = raw_target if raw_target is not None else ""
        if not target or not target.strip():
            return None
        return {
            "op_type": "delete_range",
            "search_text": target,
            "original_text": target,
            "text": "",
        }
    if op_name == "insert_after_heading":
        # target is heading text (with or without # prefix); trim ok for anchor
        target = (raw_target or "").strip() if raw_target is not None else ""
        anchor = target if target else ""
        # Pass content verbatim; spacing is normalized by the backend resolver
        return {
            "op_type": "insert_after_heading",
            "anchor_text": anchor,
            "text": content,
        }
    if op_name == "append":
        target = (raw_target or "").strip() if raw_target is not None else ""
        body = doc_content[fm_end:].strip()
        if not body:
            return {
                "op_type": "insert_after_heading",
                "anchor_text": "",
                "text": content,
            }
        last_line = body.split("\n")[-1].strip()
        return {
            "op_type": "insert_after",
            "anchor_text": last_line,
            "text": content,
        }
    return None


def _reason_for_invalid_edit(edit: Dict[str, Any]) -> str:
    """Return a short reason why an edit was rejected (for skip message and LLM feedback)."""
    edit = _normalize_edit(edit)
    op = (edit.get("operation") or "").strip().lower()
    target = edit.get("target")
    target_ok = target is not None and (target if isinstance(target, str) else "").strip()
    if op in ("replace", "replace_range", "delete", "delete_range") and not target_ok:
        return "replace/delete require 'target' (exact text to find); missing or empty."
    if not op:
        return "missing 'operation' (use: replace, delete, insert_after_heading, or append)."
    if op not in ("replace", "replace_range", "delete", "delete_range", "insert_after_heading", "append"):
        return f"unknown operation '{op}'; use replace, delete, insert_after_heading, or append."
    return "invalid or missing target."


def _normalize_edits_arg(edits: Any) -> List[Dict[str, Any]]:
    """Ensure edits is a list of dicts. Parse JSON string if needed; skip non-dicts."""
    if edits is None:
        return []
    if isinstance(edits, str):
        try:
            parsed = json.loads(edits)
            if isinstance(parsed, list):
                edits = parsed
            else:
                return []
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(edits, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(edits):
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            try:
                obj = json.loads(item)
                if isinstance(obj, dict):
                    out.append(obj)
            except (json.JSONDecodeError, TypeError):
                pass
    return out


# ── patch_file_tool ────────────────────────────────────────────────────────────

async def patch_file_tool(
    document_id: str,
    edits: List[Dict[str, Any]],
    summary: str = "",
    agent_name: str = "unknown",
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Propose a batch of edits to a document. Creates a single proposal for user approval.
    Edits are resolved in order; all resolved operations are submitted as one proposal.
    """
    edits = _normalize_edits_arg(edits)
    if not edits:
        err = (
            "No edits provided or edits could not be parsed. "
            "Provide a list of objects; each must have 'operation' (replace, delete, insert_after_heading, or append) "
            "and for replace/delete include 'target' (exact text to find in the document)."
        )
        return {
            "success": False,
            "proposal_id": None,
            "document_id": document_id,
            "operations_applied": 0,
            "skipped_edits": [err],
            "message": err,
            "error": err,
            "formatted": err,
        }

    client = await get_backend_tool_client()
    raw = await client.get_document_content(document_id=document_id, user_id=user_id)
    if raw is None:
        err = f"Document {document_id} not found."
        return {
            "success": False,
            "proposal_id": None,
            "document_id": document_id,
            "operations_applied": 0,
            "skipped_edits": [],
            "message": err,
            "error": err,
            "formatted": err,
        }

    doc_content = raw if isinstance(raw, str) else str(raw)
    fm_end = frontmatter_end_index(doc_content)

    semantic_ops: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for i, edit in enumerate(edits):
        op_dict = _edit_to_op_dict(edit, doc_content, fm_end)
        if op_dict is None:
            reason = _reason_for_invalid_edit(edit)
            skipped.append(f"Edit {i + 1} ({edit.get('operation', '?')}): {reason}")
            continue
        semantic_ops.append(op_dict)

    if not semantic_ops:
        err = "No valid edits. " + "; ".join(skipped)
        err += " Each edit must have 'operation' and for replace/delete include 'target' (exact text with context)."
        logger.info(
            "patch_file: no valid edits for doc=%s edits=%s skipped=%s",
            document_id,
            len(edits),
            skipped,
        )
        return {
            "success": False,
            "proposal_id": None,
            "document_id": document_id,
            "operations_applied": 0,
            "skipped_edits": skipped,
            "message": err,
            "error": err,
            "formatted": err,
        }

    effective_agent = (_pipeline_metadata or {}).get("agent_profile_name") or agent_name
    summary_text = summary or f"Patch file: {len(semantic_ops)} edit(s)"

    response = await client.propose_document_edit(
        document_id=document_id,
        edit_type="operations",
        operations=semantic_ops,
        agent_name=effective_agent,
        summary=summary_text,
        requires_preview=True,
        user_id=user_id,
    )

    if response.get("success"):
        logger.info(
            "patch_file: proposal created doc=%s proposal_id=%s operations=%s",
            document_id,
            response.get("proposal_id"),
            len(semantic_ops),
        )
        fmt = (
            f"Proposal created for {document_id}. Proposal ID: {response.get('proposal_id')}. "
            f"Proposed {len(semantic_ops)} edit(s)."
        )
        if skipped:
            fmt += f" Skipped: {'; '.join(skipped)}"
        return {
            "success": True,
            "proposal_id": response.get("proposal_id"),
            "document_id": document_id,
            "operations_applied": len(semantic_ops),
            "skipped_edits": skipped,
            "message": response.get("message"),
            "error": None,
            "formatted": fmt,
        }

    return {
        "success": False,
        "proposal_id": response.get("proposal_id"),
        "document_id": document_id,
        "operations_applied": 0,
        "skipped_edits": skipped,
        "message": response.get("message"),
        "error": response.get("error"),
        "formatted": response.get("error", response.get("message", "Proposal failed.")),
    }


# ── append_to_file_tool ────────────────────────────────────────────────────────

async def append_to_file_tool(
    document_id: str,
    content: str,
    heading: Optional[str] = None,
    summary: str = "",
    agent_name: str = "unknown",
    user_id: str = "system",
    _pipeline_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Propose appending content to the end of a document. Optional heading prepended to the content.
    Delegates to patch_file with a single append edit.
    """
    full_content = f"{heading}\n\n{content}" if heading else content
    return await patch_file_tool(
        document_id=document_id,
        edits=[{"operation": "append", "content": full_content}],
        summary=summary or "Append to document",
        agent_name=agent_name,
        user_id=user_id,
        _pipeline_metadata=_pipeline_metadata,
    )


# ── Registry ───────────────────────────────────────────────────────────────────

register_action(
    name="patch_file",
    category="document",
    description=(
        "Propose batched edits to a document for user approval. Edits are matched semantically against "
        "the current document content at display and apply time — do NOT provide character positions. "
        "For replace/delete, provide generous verbatim context in 'target' to ensure unique matching. "
        "For org-mode files: do not use for TODO state, tags, or priority — use list_todos and update_todo (or toggle_todo) so changes apply directly."
    ),
    short_description="Propose batched edits to a document for user approval",
    inputs_model=PatchFileInputs,
    params_model=PatchFileParams,
    outputs_model=PatchFileOutputs,
    tool_function=patch_file_tool,
    retriable=False,
)
register_action(
    name="append_to_file",
    category="document",
    description="Propose appending content to a document; optional heading. Uses same semantic proposal system as patch_file.",
    inputs_model=AppendToFileInputs,
    params_model=AppendToFileParams,
    outputs_model=AppendToFileOutputs,
    tool_function=append_to_file_tool,
    retriable=False,
)
