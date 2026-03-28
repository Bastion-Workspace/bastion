"""
Unified Reference File Loader - Works across all agents

This module provides a consistent mechanism for loading referenced files
from frontmatter, supporting different reference patterns and cascading.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from orchestrator.tools.document_tools import search_documents_tool, get_document_content_tool
from orchestrator.utils.document_type_registry import get_type_spec, get_hub_child_spec_for_frontmatter
from orchestrator.utils.frontmatter_utils import strip_frontmatter_block

logger = logging.getLogger(__name__)


# ── I/O models for load_file_by_path ───────────────────────────────────────

class LoadFileByPathInputs(BaseModel):
    """Required inputs for load_file_by_path."""
    ref_path: str = Field(description="Reference path (e.g., ./component_list.md, ../file.md)")
    user_id: str = Field(default="system", description="User ID for access control")


class LoadFileByPathOutputs(BaseModel):
    """Outputs for load_file_by_path."""
    document_id: Optional[str] = Field(default=None, description="Document ID if found")
    filename: Optional[str] = Field(default=None, description="Filename")
    content: Optional[str] = Field(default=None, description="Full content")
    path: Optional[str] = Field(default=None, description="Reference path")
    found: bool = Field(description="Whether the file was found")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── I/O models for load_referenced_files ────────────────────────────────────

class LoadReferencedFilesInputs(BaseModel):
    """Required inputs for load_referenced_files."""
    user_id: str = Field(default="system", description="User ID for access control")


class LoadReferencedFilesOutputs(BaseModel):
    """Outputs for load_referenced_files."""
    loaded_files: Dict[str, Any] = Field(default_factory=dict, description="Loaded files by category")
    category_count: int = Field(description="Number of categories with loaded files")
    total_files: int = Field(description="Total number of files loaded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def load_file_by_path(
    ref_path: str,
    user_id: str = "system",
    base_filename: Optional[str] = None,
    active_editor: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load a file by its reference path using TRUE FILESYSTEM PATH RESOLUTION.

    Resolves relative paths from the active editor's canonical_path, then finds
    the document by actual filesystem path. Deterministic and fast.

    Args:
        ref_path: Reference path (e.g., "./component_list.md", "../file.md", "file.md")
        user_id: User ID for access control
        base_filename: Optional base filename (deprecated - use active_editor)
        active_editor: Active editor dict with canonical_path for base directory

    Returns:
        Dict with document_id, filename, content, path, found, formatted
    """
    try:
        from orchestrator.backend_tool_client import get_backend_tool_client
        
        logger.info(f"📄 Loading referenced file via path resolution: {ref_path}")
        
        # Get base path from active editor's canonical_path
        base_path = None
        if active_editor:
            canonical_path = active_editor.get("canonical_path") or active_editor.get("file_path")
            if canonical_path:
                try:
                    from pathlib import Path
                    base_path = str(Path(canonical_path).parent)
                    logger.info(f"📄 Base path from active editor: {base_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to extract base path from canonical_path: {e}")

            # If we only have a filename but no canonical path, we CANNOT resolve it without searching
            # **ROOSEVELT FIX:** We TRUST the user's explicit path references. NEVER search for files!
            elif active_editor.get("filename") and not canonical_path:
                logger.warning(f"⚠️ Cannot resolve path for filename '{active_editor['filename']}' without searching. Skipping.")
                return None

        # If no base path from active editor, try base_filename
        if not base_path and base_filename:
            try:
                from pathlib import Path
                base_path = str(Path(base_filename).parent)
                logger.info(f"📄 Base path from base_filename: {base_path}")
            except Exception:
                pass

        if not base_path:
            logger.warning(f"⚠️ No base path available - cannot resolve relative path: {ref_path}")
            return None
        
        # Normalize ref_path: if it's a bare filename (no ./ or ../), treat as same directory
        normalized_ref = ref_path
        if not ref_path.startswith('./') and not ref_path.startswith('../') and '/' not in ref_path and '\\' not in ref_path:
            # Bare filename - assume same directory
            normalized_ref = f"./{ref_path}"
            logger.info(f"📄 Normalized bare filename to relative path: {ref_path} -> {normalized_ref}")
        
        # Use backend tool client to find document by path
        client = await get_backend_tool_client()
        doc_info = await client.find_document_by_path(
            file_path=normalized_ref,
            user_id=user_id,
            base_path=base_path
        )
        
        if not doc_info:
            logger.warning(f"⚠️ Could not find document by path: {ref_path} (base: {base_path})")
            return {
                "document_id": None,
                "filename": None,
                "content": None,
                "path": ref_path,
                "found": False,
                "formatted": f"File not found: {ref_path}",
            }

        # Security check: ensure we aren't loading system files or logs
        resolved_path_lower = doc_info.get("resolved_path", "").lower()
        system_dirs = ['/logs/', '\\logs\\', '/processed/', '\\processed\\', '/node_modules/', '\\node_modules\\']
        if any(sys_dir in resolved_path_lower for sys_dir in system_dirs):
            logger.error(f"SECURITY: Attempted to load log/system file as reference: {resolved_path_lower}")
            return {
                "document_id": None,
                "filename": None,
                "content": None,
                "path": ref_path,
                "found": False,
                "formatted": f"Access denied for path: {ref_path}",
            }

        document_id = doc_info.get("document_id")
        resolved_path = doc_info.get("resolved_path")
        filename = doc_info.get("filename", Path(ref_path).name)

        logger.info(f"Found document {document_id} at {resolved_path}")

        # Get full content
        content_result = await get_document_content_tool(document_id, user_id)
        content = content_result.get("content", content_result) if isinstance(content_result, dict) else content_result

        # Check for specific error messages from get_document_content_tool
        if not content or content.startswith("Document not found:") or content.startswith("Error getting document content:"):
            logger.warning(f"Could not load content for document: {document_id}")
            return {
                "document_id": document_id,
                "filename": filename,
                "content": None,
                "path": ref_path,
                "found": False,
                "formatted": f"Found {filename} but could not load content.",
            }

        # Diagnostic: log content fingerprint so we can verify correct document (e.g. outline for book 7 vs 4)
        _preview_for_log = (content.strip() or "")[:200].replace("\n", " ")
        logger.info(
            "REFERENCE LOADED: ref_path=%s document_id=%s resolved_path=%s len=%s preview=%s",
            ref_path, document_id, resolved_path, len(content), _preview_for_log,
        )

        # Strip YAML frontmatter so refs inject only body content (e.g. into editor_refs_*)
        if isinstance(content, str) and content.strip().startswith("---"):
            content = strip_frontmatter_block(content)

        preview = content[:200] + "..." if len(content) > 200 else content
        return {
            "document_id": document_id,
            "filename": filename,
            "content": content,
            "path": ref_path,
            "found": True,
            "formatted": f"Loaded {filename} ({len(content)} chars). Preview: {preview}",
        }

    except Exception as e:
        logger.error(f"Error loading file by path '{ref_path}': {e}")
        return {
            "document_id": None,
            "filename": None,
            "content": None,
            "path": ref_path,
            "found": False,
            "formatted": f"Error loading {ref_path}: {e}",
        }


async def extract_reference_paths(
    frontmatter: Dict[str, Any],
    reference_config: Dict[str, List[str]]
) -> List[Tuple[str, str]]:
    """
    Extract reference paths from frontmatter based on configuration.
    
    Args:
        frontmatter: Document frontmatter dict
        reference_config: Dict mapping category names to list of frontmatter keys
                         e.g., {"outline": ["outline"], "rules": ["rules"], "characters": ["characters", "character_*"]}
    
    Returns:
        List of (path, category) tuples
    """
    referenced_paths = []
    
    for category, keys in reference_config.items():
        for key in keys:
            # Handle wildcard keys (e.g., "character_*")
            if key.endswith("*"):
                prefix = key[:-1]
                for fm_key, fm_value in frontmatter.items():
                    if str(fm_key).startswith(prefix) and fm_value:
                        # Handle single values and lists
                        if isinstance(fm_value, list):
                            referenced_paths.extend([(path, category) for path in fm_value if path])
                        elif isinstance(fm_value, str):
                            referenced_paths.append((fm_value, category))
            else:
                ref_value = frontmatter.get(key)
                if ref_value:
                    # Handle both single values and lists
                    if isinstance(ref_value, list):
                        referenced_paths.extend([(path, category) for path in ref_value if path])
                    elif isinstance(ref_value, str):
                        # Handle comma-separated values
                        paths = [p.strip() for p in ref_value.split(",") if p.strip()]
                        referenced_paths.extend([(path, category) for path in paths])
    
    return referenced_paths


async def load_referenced_files(
    active_editor: Optional[Dict[str, Any]],
    user_id: str,
    reference_config: Dict[str, List[str]],
    doc_type_filter: Optional[str] = None,
    cascade_config: Optional[Dict[str, Dict[str, List[str]]]] = None
) -> Dict[str, Any]:
    """
    Unified reference file loader - works for all agents.
    
    Loads referenced files from active editor frontmatter based on configuration.
    Supports cascading references (e.g., outline → rules/style/characters).
    
    Args:
        active_editor: Active editor dict with frontmatter (from shared_memory)
        user_id: User ID for access control
        reference_config: Dict mapping category names to list of frontmatter keys
                         e.g., {
                             "outline": ["outline"],
                             "components": ["components", "component"],
                             "characters": ["characters", "character_*"]
                         }
        doc_type_filter: Only load references if document type matches (None = load any type)
        cascade_config: Optional cascading config
                        e.g., {
                            "outline": {
                                "rules": ["rules"],
                                "style": ["style"],
                                "characters": ["characters", "character_*"]
                            }
                        }
                        If provided, loads the primary reference (e.g., outline),
                        then extracts its frontmatter and loads cascaded references.
    
    Returns:
        Dict with:
            - loaded_files: Dict of loaded files by category
            - error: str (if failed)
    """
    try:
        loaded_files = {}
        
        # Debug logging BEFORE the check
        logger.info("="*80)
        logger.info("🔍 REFERENCE FILE LOADER DEBUG:")
        logger.info(f"   active_editor type: {type(active_editor)}")
        logger.info(f"   active_editor is None: {active_editor is None}")
        
        if active_editor:
            try:
                logger.info(f"   active_editor keys: {list(active_editor.keys())}")
                logger.info(f"   has 'content': {bool(active_editor.get('content'))}")
                content_val = active_editor.get('content', '')
                logger.info(f"   content type: {type(content_val)}")
                logger.info(f"   content length: {len(content_val) if content_val else 0}")
                logger.info(f"   has 'filename': {bool(active_editor.get('filename'))}")
                logger.info(f"   filename value: {active_editor.get('filename')}")
                logger.info(f"   has 'frontmatter': {bool(active_editor.get('frontmatter'))}")
                frontmatter_debug = active_editor.get('frontmatter', {})
                if frontmatter_debug:
                    logger.info(f"   frontmatter type: {type(frontmatter_debug)}")
                    logger.info(f"   frontmatter keys: {list(frontmatter_debug.keys())}")
                    logger.info(f"   frontmatter has 'outline': {bool(frontmatter_debug.get('outline'))}")
                    logger.info(f"   frontmatter['outline'] value: {frontmatter_debug.get('outline')}")
                else:
                    logger.info(f"   frontmatter is empty or None")
            except Exception as debug_err:
                logger.error(f"   DEBUG LOGGING ERROR: {debug_err}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.info(f"   active_editor is falsy (empty dict or None)")
        
        logger.info("="*80)
        
        # Check if we have an active editor with actual content
        if not active_editor or (not active_editor.get("content") and not active_editor.get("filename") and not active_editor.get("frontmatter")):
            logger.info("No active editor - skipping referenced file loading")
            total = sum(len(v) if isinstance(v, list) else 0 for v in loaded_files.values())
            return {
                "loaded_files": loaded_files,
                "category_count": 0,
                "total_files": 0,
                "error": None,
                "formatted": "No active editor - no referenced files loaded.",
            }
        
        frontmatter = active_editor.get("frontmatter", {})
        doc_type = frontmatter.get("type", "").lower()
        
        # Debug: Log frontmatter keys to see what's available
        logger.info(f"📄 Frontmatter keys: {list(frontmatter.keys())}")
        for key in ["files", "components", "protocols", "schematics", "specifications"]:
            if key in frontmatter:
                value = frontmatter[key]
                logger.info(f"📄 Frontmatter['{key}'] = {value} (type: {type(value).__name__})")
        
        # Only load references if document type matches filter
        if doc_type_filter and doc_type != doc_type_filter:
            logger.info(f"Active editor type is '{doc_type}', not '{doc_type_filter}' - skipping referenced files")
            return {
                "loaded_files": loaded_files,
                "category_count": 0,
                "total_files": 0,
                "error": None,
                "formatted": f"Document type '{doc_type}' does not match filter - no files loaded.",
            }
        
        # Child-to-hub cascade: if this is a hub_child (e.g. project/electronics child with project_plan),
        # load the hub and then all siblings from the hub's frontmatter.
        hub_child_spec = get_hub_child_spec_for_frontmatter(frontmatter)
        if hub_child_spec and hub_child_spec.hub_key:
            hub_path = frontmatter.get(hub_child_spec.hub_key)
            if isinstance(hub_path, list):
                hub_path = hub_path[0] if hub_path else None
            if hub_path:
                try:
                    hub_doc = await load_file_by_path(
                        ref_path=hub_path,
                        user_id=user_id,
                        base_filename=active_editor.get("filename"),
                        active_editor=active_editor,
                    )
                    if hub_doc and hub_doc.get("found"):
                        hub_content = hub_doc.get("content", "")
                        import yaml
                        import re
                        fm_match = re.match(r"^---\s*\n([\s\S]*?)\n---\s*\n", hub_content or "")
                        if fm_match:
                            hub_frontmatter = yaml.safe_load(fm_match.group(1)) or {}
                            hub_type = (hub_frontmatter.get("type") or "").lower()
                            hub_spec = get_type_spec(hub_type)
                            if hub_spec and hub_spec.reference_categories:
                                referenced_paths_from_hub = await extract_reference_paths(
                                    hub_frontmatter, hub_spec.reference_categories
                                )
                                loaded_files["hub"] = [hub_doc]
                                if hub_child_spec.hub_key:
                                    loaded_files[hub_child_spec.hub_key] = [hub_doc]
                                import asyncio
                                for ref_path, category in referenced_paths_from_hub:
                                    try:
                                        doc = await load_file_by_path(
                                            ref_path=ref_path,
                                            user_id=user_id,
                                            base_filename=hub_doc.get("filename"),
                                            active_editor=active_editor,
                                        )
                                        if doc and doc.get("found"):
                                            if category not in loaded_files:
                                                loaded_files[category] = []
                                            loaded_files[category].append(doc)
                                    except Exception as e:
                                        logger.warning("Failed to load sibling %s: %s", ref_path, e)
                                category_count = len(loaded_files)
                                total_files = sum(
                                    len(v) if isinstance(v, list) else 0 for v in loaded_files.values()
                                )
                                return {
                                    "loaded_files": loaded_files,
                                    "category_count": category_count,
                                    "total_files": total_files,
                                    "error": None,
                                    "formatted": f"Loaded hub and {total_files - 1} sibling(s) from hub.",
                                }
                except Exception as e:
                    logger.warning("Child-to-hub cascade failed: %s", e)
        
        logger.info(f"📄 Loading referenced files from frontmatter (type: {doc_type})")
        
        # Extract reference paths from frontmatter
        referenced_paths = await extract_reference_paths(frontmatter, reference_config)
        
        logger.info(f"📄 Extracted {len(referenced_paths)} reference path(s) from frontmatter")
        if referenced_paths:
            for path, category in referenced_paths[:5]:  # Log first 5
                logger.info(f"📄 Reference: {category} -> {path}")
        
        if not referenced_paths:
            logger.info("No referenced files found in frontmatter")
            return {
                "loaded_files": loaded_files,
                "category_count": 0,
                "total_files": 0,
                "error": None,
                "formatted": "No referenced files found in frontmatter.",
            }
        
        logger.info(f"📄 Found {len(referenced_paths)} referenced file(s) to load")
        
        # Load referenced files in parallel for better performance
        import asyncio
        
        async def load_single_file(ref_path: str, category: str) -> tuple:
            """Load a single file and return (category, loaded_doc or None, ref_path)"""
            try:
                loaded_doc = await load_file_by_path(
                    ref_path=ref_path,
                    user_id=user_id,
                    base_filename=active_editor.get("filename"),
                    active_editor=active_editor
                )
                
                if loaded_doc and loaded_doc.get("found"):
                    logger.info(f"Loaded {category} file: {loaded_doc.get('filename')}")
                    return (category, loaded_doc, None)
                else:
                    logger.warning(f"Failed to load {category} file: {ref_path}")
                    return (category, None, ref_path)
                    
            except Exception as e:
                logger.error(f"❌ Error loading {category} file '{ref_path}': {e}")
                return (category, None, ref_path)
        
        # Load all files in parallel
        load_tasks = [
            load_single_file(ref_path, category)
            for ref_path, category in referenced_paths
        ]
        results = await asyncio.gather(*load_tasks)
        
        # Organize loaded files by category
        for category, loaded_doc, ref_path in results:
            if loaded_doc:
                if category not in loaded_files:
                    loaded_files[category] = []
                loaded_files[category].append(loaded_doc)
        
        # Handle cascading references (e.g., outline → rules/style/characters)
        if cascade_config:
            for primary_category, cascade_refs in cascade_config.items():
                if primary_category in loaded_files and loaded_files[primary_category]:
                    # Get the first primary file (e.g., outline)
                    primary_file = loaded_files[primary_category][0]
                    primary_content = primary_file.get("content", "")
                    
                    # Parse frontmatter from primary file content
                    try:
                        import re
                        import yaml
                        
                        # Extract YAML frontmatter block
                        fm_match = re.match(r'^---\s*\n([\s\S]*?)\n---\s*\n', primary_content)
                        if fm_match:
                            primary_frontmatter = yaml.safe_load(fm_match.group(1)) or {}
                            
                            # Extract cascaded reference paths
                            cascade_paths = await extract_reference_paths(primary_frontmatter, cascade_refs)
                            
                            # Load cascaded files
                            for cascade_path, cascade_category in cascade_paths:
                                try:
                                    cascade_doc = await load_file_by_path(
                                        ref_path=cascade_path,
                                        user_id=user_id,
                                        base_filename=primary_file.get("filename"),
                                        active_editor=active_editor  # Pass through for project context
                                    )
                                    
                                    if cascade_doc and cascade_doc.get("found"):
                                        if cascade_category not in loaded_files:
                                            loaded_files[cascade_category] = []
                                        loaded_files[cascade_category].append(cascade_doc)
                                        logger.info(f"Loaded cascaded {cascade_category} file: {cascade_doc.get('filename')}")
                                except Exception as e:
                                    logger.error(f"❌ Error loading cascaded {cascade_category} file '{cascade_path}': {e}")
                    except Exception as e:
                        logger.warning(f"Could not parse frontmatter from primary file for cascading: {e}")

        category_count = len(loaded_files)
        total_files = sum(len(v) if isinstance(v, list) else 0 for v in loaded_files.values())
        summary = f"Loaded {total_files} file(s) in {category_count} categor(ies)."
        if category_count:
            cats = ", ".join(f"{k}: {len(v)}" for k, v in loaded_files.items() if isinstance(v, list))
            summary = f"Loaded {total_files} file(s): {cats}."
        return {
            "loaded_files": loaded_files,
            "category_count": category_count,
            "total_files": total_files,
            "error": None,
            "formatted": summary,
        }

    except Exception as e:
        logger.error(f"Error in load_referenced_files: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "loaded_files": {},
            "category_count": 0,
            "total_files": 0,
            "error": str(e),
            "formatted": f"Error loading referenced files: {e}",
        }


def extract_ref_prefix_paths(
    frontmatter: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """
    Scan frontmatter for ref_* keys. Returns (path, category) tuples
    where category = key with 'ref_' stripped.
    Single strings and lists of strings are both supported.
    """
    results = []
    for key, value in frontmatter.items():
        if not key.startswith("ref_"):
            continue
        category = key[4:]
        if isinstance(value, str) and value:
            results.append((value, category))
        elif isinstance(value, list):
            results.extend((v, category) for v in value if isinstance(v, str) and v)
    return results


# ── Registry ───────────────────────────────────────────────────────────────
# Not registered: used internally by built-in agents (electronics, general_project)
# for context loading before editing. Not exposed in Agent Factory tool list.
