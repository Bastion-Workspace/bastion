"""
Unified Reference File Loader - Works across all agents

This module provides a consistent mechanism for loading referenced files
from frontmatter, supporting different reference patterns and cascading.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from orchestrator.tools.document_tools import search_documents_structured, get_document_content_tool

logger = logging.getLogger(__name__)


async def load_file_by_path(
    ref_path: str,
    user_id: str,
    base_filename: Optional[str] = None,
    active_editor: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
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
        Dict with document info and content, or None if not found
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

            # If we only have a filename but no canonical path, try to resolve it
            elif active_editor.get("filename") and not canonical_path:
                logger.info(f"📄 Attempting to resolve canonical path for filename: {active_editor['filename']}")
                try:
                    # Try to find the document and get its canonical path
                    client = await get_backend_tool_client()
                    # Search for document by filename
                    search_results = await client.search_documents_structured(
                        query=active_editor['filename'],
                        user_id=user_id,
                        limit=5
                    )

                    # Find the exact match
                    for result in search_results:
                        doc_metadata = result.get('document', {})
                        if doc_metadata.get('filename') == active_editor['filename']:
                            # Get full document info to get canonical path
                            doc_content = await client.get_document_content(
                                document_id=doc_metadata.get('document_id'),
                                user_id=user_id
                            )
                            if doc_content and 'canonical_path' in doc_content:
                                resolved_path = doc_content['canonical_path']
                                try:
                                    from pathlib import Path
                                    base_path = str(Path(resolved_path).parent)
                                    logger.info(f"✅ Resolved base path for active editor: {base_path}")
                                except Exception as e:
                                    logger.warning(f"⚠️ Failed to extract base path from resolved path: {e}")
                                break

                except Exception as e:
                    logger.warning(f"⚠️ Failed to resolve canonical path for active editor: {e}")

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
            return None
        
        document_id = doc_info.get("document_id")
        resolved_path = doc_info.get("resolved_path")
        
        logger.info(f"✅ Found document {document_id} at {resolved_path}")
        
        # Get full content
        content = await get_document_content_tool(document_id, user_id)
        
        # Check for SPECIFIC error messages from get_document_content_tool, not arbitrary "Error" in content
        if not content or content.startswith("Document not found:") or content.startswith("Error getting document content:"):
            logger.warning(f"⚠️ Could not load content for document: {document_id}")
            return None
        
        return {
            "document_id": document_id,
            "filename": doc_info.get("filename", Path(ref_path).name),
            "title": doc_info.get("filename", Path(ref_path).name),  # Use filename as title
            "content": content,
            "path": ref_path,
            "resolved_path": resolved_path
        }
        
    except Exception as e:
        logger.error(f"❌ Error loading file by path '{ref_path}': {e}")
        return None


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
        
        # Check if we have an active editor with actual content
        if not active_editor or (not active_editor.get("content") and not active_editor.get("filename") and not active_editor.get("frontmatter")):
            logger.info("📄 No active editor - skipping referenced file loading")
            return {"loaded_files": loaded_files}
        
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
            logger.info(f"📄 Active editor type is '{doc_type}', not '{doc_type_filter}' - skipping referenced files")
            return {"loaded_files": loaded_files}
        
        logger.info(f"📄 Loading referenced files from frontmatter (type: {doc_type})")
        
        # Extract reference paths from frontmatter
        referenced_paths = await extract_reference_paths(frontmatter, reference_config)
        
        logger.info(f"📄 Extracted {len(referenced_paths)} reference path(s) from frontmatter")
        if referenced_paths:
            for path, category in referenced_paths[:5]:  # Log first 5
                logger.info(f"📄 Reference: {category} -> {path}")
        
        if not referenced_paths:
            logger.info("📄 No referenced files found in frontmatter")
            return {"loaded_files": loaded_files}
        
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
                
                if loaded_doc:
                    logger.info(f"✅ Loaded {category} file: {loaded_doc.get('filename')}")
                    return (category, loaded_doc, None)
                else:
                    logger.warning(f"⚠️ Failed to load {category} file: {ref_path}")
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
                                    
                                    if cascade_doc:
                                        if cascade_category not in loaded_files:
                                            loaded_files[cascade_category] = []
                                        loaded_files[cascade_category].append(cascade_doc)
                                        logger.info(f"✅ Loaded cascaded {cascade_category} file: {cascade_doc.get('filename')}")
                                except Exception as e:
                                    logger.error(f"❌ Error loading cascaded {cascade_category} file '{cascade_path}': {e}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not parse frontmatter from primary file for cascading: {e}")
        
        return {"loaded_files": loaded_files}
        
    except Exception as e:
        logger.error(f"❌ Error in load_referenced_files: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "loaded_files": {},
            "error": str(e)
        }

