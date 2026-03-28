"""
Load agent line reference_config into prompt-ready text for CustomAgentRunner.

Context injection only; does not change tool permissions.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from orchestrator.tools.document_tools import get_document_content_tool
from orchestrator.utils.frontmatter_utils import strip_frontmatter_block

logger = logging.getLogger(__name__)

# Rough caps to avoid blowing context windows
MAX_TOTAL_CHARS = 400_000
MAX_PER_FILE_CHARS = 80_000


def line_ref_safe_key(name: str) -> str:
    """Stable key for inputs['line_ref_<key>'] template variables."""
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or "entry").lower()).strip("_")
    return s or "entry"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[Truncated: {len(text)} chars total, showing first {max_chars}]"


async def _load_one_document(
    document_id: str,
    user_id: str,
    budget: int,
    load_strategy: str,
) -> Optional[str]:
    if budget <= 0:
        return None
    result = await get_document_content_tool(document_id=document_id, user_id=user_id)
    if not isinstance(result, dict):
        return None
    raw = result.get("content") or ""
    if not raw:
        return None
    t = raw.lstrip()
    text = strip_frontmatter_block(raw) if t.startswith("---") else raw
    if load_strategy == "metadata_first":
        preview = text[:1200]
        return _truncate(preview, min(MAX_PER_FILE_CHARS, budget))
    return _truncate(text, min(MAX_PER_FILE_CHARS, budget))


async def load_line_references(ref_config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Load folders and documents from line reference_config.

    Returns:
        combined: markdown blob for {line_refs}
        file_count: number of document bodies successfully loaded
        by_entry: display_name -> content for {line_ref_<safe_name>}
        ref_ids: list of {document_id, title, access} for playbook {line_ref_ids}
        skipped_count: documents that were listed but could not be loaded
    """
    if not ref_config or not isinstance(ref_config, dict):
        return {
            "combined": "",
            "file_count": 0,
            "by_entry": {},
            "ref_ids": [],
            "skipped_count": 0,
        }

    load_strategy = (ref_config.get("load_strategy") or "full").strip().lower()
    if load_strategy not in ("full", "metadata_first"):
        load_strategy = "full"

    from orchestrator.backend_tool_client import get_backend_tool_client

    client = await get_backend_tool_client()

    combined_parts: List[str] = ["=== LINE REFERENCE FILES ==="]
    by_entry: Dict[str, str] = {}
    seen_doc_ids: Set[str] = set()
    total_chars = 0
    file_count = 0
    skipped_count = 0
    ref_ids: List[Dict[str, str]] = []

    folders = ref_config.get("folders") or []
    if isinstance(folders, list):
        for folder in folders:
            if not isinstance(folder, dict):
                continue
            fid = (folder.get("folder_id") or "").strip()
            label = (folder.get("name") or fid or "folder")[:500]
            if not fid:
                continue
            folder_access = (folder.get("access") or "read").strip().lower()
            if folder_access not in ("read", "read_write"):
                folder_access = "read"

            list_res = await client.list_folder_documents(fid, user_id=user_id, limit=500, offset=0)
            if not list_res.get("success"):
                combined_parts.append(f"--- Folder: {label} (error: {list_res.get('error', 'unknown')}) ---\n")
                continue
            docs = list_res.get("documents") or []
            if load_strategy == "metadata_first":
                lines = [f"- {d.get('filename', '')} (id: {d.get('document_id', '')})" for d in docs]
                block = f"--- Folder: {label} (file list) ---\n" + "\n".join(lines)
                if total_chars + len(block) > MAX_TOTAL_CHARS:
                    block = _truncate(block, MAX_TOTAL_CHARS - total_chars)
                combined_parts.append(block)
                by_entry[label] = block
                total_chars += len(block)
                file_count += len(docs)
                for d in docs:
                    did = (d.get("document_id") or "").strip()
                    if not did:
                        continue
                    ref_ids.append(
                        {
                            "document_id": did,
                            "title": (d.get("filename") or did)[:500],
                            "access": folder_access,
                        }
                    )
                continue

            folder_chunks: List[str] = [f"--- Folder: {label} ---"]
            for d in docs:
                did = (d.get("document_id") or "").strip()
                if not did or did in seen_doc_ids:
                    continue
                remaining = MAX_TOTAL_CHARS - total_chars
                if remaining < 500:
                    folder_chunks.append("[Additional files omitted: context budget exhausted]")
                    break
                fname = d.get("filename") or did
                body = await _load_one_document(did, user_id, remaining, load_strategy)
                if not body:
                    skipped_count += 1
                    folder_chunks.append(
                        f"[{fname} (document_id: {did}) - not found or inaccessible]"
                    )
                    continue
                seen_doc_ids.add(did)
                file_count += 1
                ref_ids.append(
                    {
                        "document_id": did,
                        "title": fname[:500],
                        "access": folder_access,
                    }
                )
                chunk = f"### {fname} (document_id: {did})\n{body}"
                total_chars += len(chunk)
                folder_chunks.append(chunk)
            block = "\n\n".join(folder_chunks)
            combined_parts.append(block)
            by_entry[label] = block

    documents = ref_config.get("documents") or []
    if isinstance(documents, list):
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            did = (doc.get("document_id") or "").strip()
            label = (doc.get("title") or did)[:500]
            if not did or did in seen_doc_ids:
                continue
            doc_access = (doc.get("access") or "read").strip().lower()
            if doc_access not in ("read", "read_write"):
                doc_access = "read"

            remaining = MAX_TOTAL_CHARS - total_chars
            if remaining < 500:
                combined_parts.append("[Additional references omitted: context budget exhausted]")
                break
            body = await _load_one_document(did, user_id, remaining, load_strategy)
            if not body:
                skipped_count += 1
                note = f"--- Document: {label} (document_id: {did}) [NOT FOUND / INACCESSIBLE] ---"
                total_chars += len(note)
                combined_parts.append(note)
                by_entry[label] = note
                continue
            seen_doc_ids.add(did)
            file_count += 1
            ref_ids.append(
                {
                    "document_id": did,
                    "title": label,
                    "access": doc_access,
                }
            )
            block = f"--- Document: {label} (document_id: {did}) ---\n{body}"
            total_chars += len(block)
            combined_parts.append(block)
            by_entry[label] = block

    combined_parts.append("=== END LINE REFERENCE FILES ===")
    has_content = file_count > 0 or skipped_count > 0 or len(combined_parts) > 2
    combined = "\n\n".join(combined_parts) if has_content else ""

    return {
        "combined": combined,
        "file_count": file_count,
        "by_entry": by_entry,
        "ref_ids": ref_ids,
        "skipped_count": skipped_count,
    }
