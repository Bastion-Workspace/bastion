"""
Universal typed document creation.

Creates documents with correct frontmatter and body template from the document type registry.
Supports hub-and-spoke (hub_child types get back-reference and hub frontmatter update) and
shared-library types (optional initial_references only).
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.tools.document_tools import get_document_content_tool
from orchestrator.tools.document_editing_tools import update_document_content_tool
from orchestrator.tools.file_creation_tools import create_user_file_tool
from orchestrator.utils.document_type_registry import get_type_spec
from orchestrator.utils.frontmatter_utils import add_to_frontmatter_list
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Folder resolution (extracted from create_new_project_file for reuse)
# ---------------------------------------------------------------------------

async def resolve_folder_from_context(
    active_editor: Optional[Dict[str, Any]],
    document_id: Optional[str],
    user_id: str,
) -> tuple:
    """
    Resolve folder_id and/or folder_path from active_editor or document metadata.

    Priority: 1) active_editor.folder_id, 2) document metadata folder_id, 3) folder_path from canonical_path.

    Returns:
        (folder_id, folder_path) - either or both may be set, or (None, None).
    """
    folder_id = None
    folder_path = None

    if active_editor:
        folder_id = active_editor.get("folder_id")
        if folder_id:
            logger.debug("Using folder_id from active_editor: %s", folder_id)

    if not folder_id and document_id:
        try:
            client = await get_backend_tool_client()
            doc_info = await client.get_document(document_id, user_id)
            if doc_info and doc_info.get("metadata"):
                folder_id = doc_info["metadata"].get("folder_id")
                if folder_id:
                    logger.debug("Got folder_id from document metadata: %s", folder_id)
        except Exception as e:
            logger.warning("Could not get folder_id from document: %s", e)

    if not folder_id and active_editor:
        canonical_path = active_editor.get("canonical_path", "")
        if not canonical_path and document_id:
            try:
                client = await get_backend_tool_client()
                doc_info = await client.get_document(document_id, user_id)
                if doc_info and doc_info.get("metadata"):
                    canonical_path = doc_info["metadata"].get("canonical_path", "")
            except Exception:
                pass
        if canonical_path:
            try:
                path_parts = Path(canonical_path).parts
                if "Users" in path_parts:
                    users_idx = path_parts.index("Users")
                    if users_idx + 2 < len(path_parts) - 1:
                        folder_parts = path_parts[users_idx + 2 : -1]
                        if folder_parts:
                            folder_path = "/".join(folder_parts)
                            logger.debug("Extracted folder_path from canonical_path: %s", folder_path)
            except Exception as e:
                logger.warning("Failed to extract folder_path from canonical_path: %s", e)

    return (folder_id, folder_path)


def _slugify_title(title: str) -> str:
    """Produce a safe filename stem from a title."""
    s = (title or "").strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_.-]", "", s) or "document"


def _render_body_template(template: str, title: str, query: str = "") -> str:
    """Replace {{title}}, {{creation_date}}, {{query}} in body template."""
    creation_date = datetime.utcnow().strftime("%Y-%m-%d")
    return (
        template.replace("{{title}}", title or "")
        .replace("{{creation_date}}", creation_date)
        .replace("{{query}}", query or "")
    )


# ---------------------------------------------------------------------------
# I/O models for create_typed_document
# ---------------------------------------------------------------------------

class CreateTypedDocumentInputs(BaseModel):
    """Required inputs for create_typed_document."""
    doc_type: str = Field(description="Document type: project, electronics, outline, fiction, project_child, electronics_child, character, rules, style, reference")
    title: str = Field(description="Document title")


class CreateTypedDocumentParams(BaseModel):
    """Optional parameters."""
    folder_id: Optional[str] = Field(default=None, description="Folder ID to create document in")
    folder_path: Optional[str] = Field(default=None, description="Folder path if folder_id not set")
    hub_document_id: Optional[str] = Field(default=None, description="For hub_child types: parent hub document ID")
    initial_references: Optional[Dict[str, str]] = Field(default=None, description="Explicit refs e.g. style: ./style.md")


class CreateTypedDocumentOutputs(BaseModel):
    """Outputs for create_typed_document."""
    success: bool = Field(description="Whether the document was created")
    document_id: Optional[str] = Field(default=None, description="Created document ID")
    filename: Optional[str] = Field(default=None, description="Created filename")
    type_key: Optional[str] = Field(default=None, description="Document type used")
    hub_updated: bool = Field(default=False, description="Whether hub frontmatter was updated")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def create_typed_document_tool(
    doc_type: str,
    title: str,
    folder_id: Optional[str] = None,
    folder_path: Optional[str] = None,
    hub_document_id: Optional[str] = None,
    initial_references: Optional[Dict[str, str]] = None,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Create a typed document with correct frontmatter and body template.

    Behavior depends on the type's model (hub, hub_child, shared_library).
    For hub_child types, pass hub_document_id to link the new file to the hub
    and update the hub's frontmatter.
    """
    spec = get_type_spec(doc_type)
    if not spec:
        err = f"Unknown document type: {doc_type}"
        logger.warning(err)
        return {
            "success": False,
            "document_id": None,
            "filename": None,
            "type_key": doc_type,
            "hub_updated": False,
            "error": err,
            "formatted": err,
        }

    initial_references = initial_references or {}
    filename = _slugify_title(title) + ".md"
    if not filename or filename == ".md":
        filename = "document.md"

    # Build default frontmatter from spec, apply title and initial_references
    fm = dict(spec.default_frontmatter)
    fm["title"] = title
    for k, v in initial_references.items():
        if v:
            fm[k] = v

    # Body
    body = _render_body_template(spec.body_template, title)

    # Hub child: add back-reference to hub and optionally update hub's frontmatter
    hub_filename_for_ref = None
    if spec.model == "hub_child" and spec.hub_key and hub_document_id:
        try:
            client = await get_backend_tool_client()
            hub_doc = await client.get_document(hub_document_id, user_id)
            if hub_doc:
                hub_filename_for_ref = hub_doc.get("filename")
                if hub_filename_for_ref:
                    fm[spec.hub_key] = f"./{hub_filename_for_ref}"
        except Exception as e:
            logger.warning("Could not resolve hub filename for back-reference: %s", e)

    # Build content
    try:
        import yaml
        fm_yaml = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
        content = f"---\n{fm_yaml.strip()}\n---\n\n{body}"
    except Exception as e:
        logger.warning("YAML dump failed, using simple frontmatter: %s", e)
        parts = ["---"]
        for k, v in fm.items():
            if v is not None and v != "":
                if isinstance(v, list):
                    parts.append(f"{k}:")
                    for i in v:
                        parts.append(f"  - {i}")
                else:
                    parts.append(f"{k}: {v}")
        parts.append("---")
        content = "\n".join(parts) + "\n\n" + body

    # Create file
    result = await create_user_file_tool(
        filename=filename,
        content=content,
        folder_id=folder_id,
        folder_path=folder_path,
        title=title,
        user_id=user_id,
    )

    if not result.get("success"):
        err = result.get("error", "Unknown error")
        return {
            "success": False,
            "document_id": None,
            "filename": filename,
            "type_key": doc_type,
            "hub_updated": False,
            "error": err,
            "formatted": f"Failed to create document: {err}",
        }

    new_doc_id = result.get("document_id")
    hub_updated = False

    # Hub child: update hub's frontmatter to include new file
    if spec.model == "hub_child" and spec.parent_frontmatter_list and hub_document_id and new_doc_id:
        try:
            _r = await get_document_content_tool(hub_document_id, user_id)
            hub_content = _r.get("content", _r) if isinstance(_r, dict) else _r
            if hub_content and not hub_content.startswith("Error"):
                new_entry = f"./{filename}"
                updated_content, ok = await add_to_frontmatter_list(
                    content=hub_content,
                    list_key=spec.parent_frontmatter_list,
                    new_items=[new_entry],
                    also_update_files=(spec.parent_frontmatter_list != "files"),
                )
                if ok:
                    await update_document_content_tool(
                        document_id=hub_document_id,
                        content=updated_content,
                        user_id=user_id,
                        append=False,
                    )
                    hub_updated = True
                    logger.info("Updated hub frontmatter with new file: %s", filename)
        except Exception as e:
            logger.warning("Could not update hub frontmatter: %s", e)

    msg = f"Created {doc_type} document '{title}' ({filename}, ID: {new_doc_id})."
    if hub_updated:
        msg += " Hub frontmatter updated."
    return {
        "success": True,
        "document_id": new_doc_id,
        "filename": filename,
        "type_key": doc_type,
        "hub_updated": hub_updated,
        "error": None,
        "formatted": msg,
    }


register_action(
    name="create_typed_document",
    category="file",
    description="Create a typed document (project, electronics, outline, fiction, character, rules, style, reference, etc.) with correct frontmatter and template. For hub_child types pass hub_document_id to link to parent.",
    inputs_model=CreateTypedDocumentInputs,
    params_model=CreateTypedDocumentParams,
    outputs_model=CreateTypedDocumentOutputs,
    tool_function=create_typed_document_tool,
    retriable=False,
)
