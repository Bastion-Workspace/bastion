"""
General Project Agent - Content Saving Nodes Module
Handles saving content to project files and verification
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class GeneralProjectSaveNodes:
    """Content saving nodes for general project agent"""
    
    def __init__(self, agent):
        """
        Initialize with reference to main agent for access to helper methods
        
        Args:
            agent: GeneralProjectAgent instance (for _get_llm, _get_fast_model, etc.)
        """
        self.agent = agent
    
    async def _resolve_document_id(
        self,
        target_file: str,
        active_editor: Dict[str, Any],
        referenced_context: Dict[str, Any],
        user_id: str,
        target_document_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Resolve document_id from target_file name.
        
        Args:
            target_file: Name of the file to resolve
            active_editor: Active editor context
            referenced_context: Referenced files context
            user_id: User ID
            target_document_id: Locked document ID from request start (prevents race conditions)
        
        Returns:
            document_id if found, None otherwise
        """
        doc_id = None
        
        # Check if target_file matches project plan (handle both "project_plan" and "project_plan.md")
        is_project_plan = (
            target_file == "project_plan" or
            target_file == "project_plan.md" or
            target_file.endswith("/project_plan.md") or
            target_file.endswith("\\project_plan.md")
        )
        
        if is_project_plan:
            # 🔒 Use locked target_document_id to prevent race conditions during tab switches
            doc_id = target_document_id or active_editor.get("document_id")
            logger.info(f"Resolving project_plan: target_document_id={target_document_id}, active_editor.document_id={active_editor.get('document_id')}, canonical_path={active_editor.get('canonical_path')}")
            
            if not doc_id and active_editor.get("canonical_path"):
                logger.info(f"Attempting canonical_path lookup for: {active_editor.get('canonical_path')}")
                try:
                    from orchestrator.backend_tool_client import get_backend_tool_client
                    client = await get_backend_tool_client()
                    find_result = await client.find_document_by_path(
                        file_path=active_editor.get("canonical_path"),
                        user_id=user_id
                    )
                    logger.info(f"canonical_path lookup result: {find_result}")
                    if find_result and find_result.get("document_id"):
                        doc_id = find_result.get("document_id")
                        logger.info(f"✅ Found project_plan document_id via canonical_path lookup: {doc_id}")
                except Exception as e:
                    logger.warning(f"Failed to lookup document_id from canonical_path: {e}")
            
            # Also check if filename matches
            if not doc_id:
                editor_filename = active_editor.get("filename", "")
                if editor_filename == "project_plan.md" or editor_filename.endswith("/project_plan.md"):
                    # 🔒 Use locked target_document_id first, fallback to active_editor
                    doc_id = target_document_id or active_editor.get("document_id")
                    if doc_id:
                        logger.info(f"✅ Found project_plan document_id from filename match: {doc_id}")
            
            # Fallback: search for project_plan.md document - DISABLED
            # **ROOSEVELT FIX:** We TRUST the user's explicit path references. NEVER search for files!
            if not doc_id:
                logger.warning("Active editor missing document_id - cannot resolve project_plan.md without searching. Skipping.")
            
            logger.info(f"Routing to project_plan (document_id: {doc_id})")
        else:
            # Find in referenced context
            for category, file_list in referenced_context.items():
                if isinstance(file_list, list):
                    for file_doc in file_list:
                        if isinstance(file_doc, dict):
                            file_filename = file_doc.get("filename", "")
                            file_doc_id = file_doc.get("document_id")
                            if file_filename == target_file or file_filename.endswith(target_file) or target_file in file_filename:
                                doc_id = file_doc_id
                                logger.info(f"Found {target_file} in {category} category (document_id: {doc_id})")
                                break
                    if doc_id:
                        break
        
        return doc_id
    
    async def save_content_node(self, state) -> Dict[str, Any]:
        """
        Save node using batch editor - groups edits by document and applies atomically.
        """
        try:
            # Check if pending_save_plan should be restored
            pending_save_plan = state.get("pending_save_plan")
            if pending_save_plan:
                logger.info("Restoring pending save plan from approval")
                state["save_plan"] = pending_save_plan
            
            save_plan = state.get("save_plan", {})
            if not save_plan or not save_plan.get("routing"):
                logger.info("No content to save")
                return {
                    # ✅ CRITICAL: Preserve critical state keys
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "referenced_context": state.get("referenced_context", {}),
                    "save_plan": state.get("save_plan")
                }
            
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            active_editor = metadata.get("active_editor") or metadata.get("shared_memory", {}).get("active_editor", {})
            referenced_context = state.get("referenced_context", {})
            
            # Debug: Log active_editor details
            if active_editor:
                logger.info(f"Active editor found: filename={active_editor.get('filename')}, canonical_path={active_editor.get('canonical_path')}, has_content={bool(active_editor.get('content'))}")
            else:
                logger.info("No active editor found in metadata")
            
            routing_items = save_plan.get("routing", [])
            
            # **PHASE 1: CREATE NEW REFERENCE FILES**
            # Process any items that need new files created first
            project_plan_doc_id = active_editor.get("document_id") if active_editor else None
            new_files_created = []
            
            for item in routing_items[:]:  # Iterate over copy so we can modify routing_items
                create_new = item.get("create_new_file", False)
                if not create_new:
                    continue
                
                suggested_filename = item.get("suggested_filename", "")
                file_summary = item.get("file_summary", "")
                
                if not suggested_filename:
                    logger.warning("create_new_file=true but no suggested_filename provided - skipping")
                    continue
                
                logger.info(f"Creating new reference file: {suggested_filename} ({file_summary})")
                
                try:
                    # Determine folder from active_editor
                    # PRIORITY 1: Use folder_id if available (most reliable)
                    folder_id = None
                    folder_path = None
                    
                    if active_editor:
                        folder_id = active_editor.get("folder_id")
                        if folder_id:
                            logger.info(f"Using folder_id from active_editor: {folder_id}")
                    
                    # PRIORITY 2: Extract relative folder_path from canonical_path
                    if not folder_id and active_editor and active_editor.get("canonical_path"):
                        from pathlib import Path
                        canonical_path = active_editor.get("canonical_path")
                        try:
                            # Parse canonical_path to get folder hierarchy
                            # Format: /app/uploads/Users/{username}/Projects/NGen Oscillators/project_plan.md
                            path_parts = Path(canonical_path).parts
                            
                            # Find "Users" to start folder path
                            if "Users" in path_parts:
                                users_idx = path_parts.index("Users")
                                if users_idx + 2 < len(path_parts) - 1:  # username + at least one folder + filename
                                    # Get folder parts (skip username and filename)
                                    folder_parts = path_parts[users_idx + 2:-1]
                                    if folder_parts:
                                        folder_path = "/".join(folder_parts)
                                        logger.info(f"Extracted folder_path from canonical_path: {folder_path}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to extract folder_path from canonical_path: {e}")
                    
                    if not folder_id and not folder_path:
                        logger.warning("⚠️ Could not determine folder_id or folder_path - file will be created in My Documents root")
                    
                    # Create the new file with initial frontmatter
                    from orchestrator.tools.file_creation_tools import create_user_file_tool
                    initial_content = f"""---
type: general
summary: {file_summary}
---

# {suggested_filename.replace('.md', '').replace('-', ' ').title()}

{item.get('content', '')}
"""
                    
                    new_doc_id = await create_user_file_tool(
                        filename=suggested_filename,
                        content=initial_content,
                        user_id=user_id,
                        folder_id=folder_id,
                        folder_path=folder_path
                    )
                    
                    logger.info(f"✅ Created new reference file: {suggested_filename} (doc_id: {new_doc_id})")
                    
                    # Add to project plan frontmatter if we have a project plan doc_id
                    if project_plan_doc_id:
                        from orchestrator.utils.document_batch_editor import DocumentEditBatch
                        plan_batch = DocumentEditBatch(project_plan_doc_id, user_id, "general_project_agent")
                        await plan_batch.initialize()
                        plan_batch.add_frontmatter_update({}, {"files": [f"./{suggested_filename}"]})
                        await plan_batch.apply()
                        logger.info(f"✅ Added {suggested_filename} reference to project plan frontmatter")
                    
                    new_files_created.append(suggested_filename)
                    
                    # Remove this item from routing_items since we already saved the content
                    routing_items.remove(item)
                    
                except Exception as e:
                    logger.error(f"Failed to create new reference file {suggested_filename}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # **PHASE 2: SPLIT EDITS - PLAN VS REFERENCED FILES**
            editing_mode = state.get("editing_mode", False)
            plan_edits = []
            referenced_edits = []
            
            for item in routing_items:
                target_file = item.get("target_file", "")
                if not target_file:
                    continue
                
                # Check if this edit targets the project plan (active editor)
                is_plan_edit = (
                    target_file == "project_plan" or
                    target_file == "project_plan.md" or
                    target_file == active_editor.get("filename", "") or
                    target_file == active_editor.get("canonical_path", "")
                )
                
                # Use inline editing if file is open (has document_id) - works in both editing and generation mode
                has_active_editor = bool(active_editor and active_editor.get("document_id"))
                
                if is_plan_edit and has_active_editor:
                    # Store for inline editing (skip frontmatter updates - apply directly)
                    if item.get("section", "").lower() != "frontmatter":
                        plan_edits.append(item)
                else:
                    # Apply directly (referenced files or no active editor)
                    referenced_edits.append(item)
            
            # **PHASE 2B: GROUP REFERENCED EDITS BY DOCUMENT**
            edits_by_doc: Dict[str, List[Dict[str, Any]]] = {}
            
            for item in referenced_edits:
                target_file = item.get("target_file", "")
                if not target_file:
                    continue
                
                # Resolve document_id (pass locked target_document_id to prevent race conditions)
                doc_id = await self._resolve_document_id(
                    target_file, active_editor, referenced_context, user_id,
                    target_document_id=project_plan_doc_id
                )
                
                if not doc_id:
                    logger.warning(f"Could not resolve document_id for {target_file} - skipping")
                    continue
                
                # Group by document_id
                if doc_id not in edits_by_doc:
                    edits_by_doc[doc_id] = []
                edits_by_doc[doc_id].append(item)
            
            # **PHASE 3: BATCH EDIT EXISTING FILES**
            # Process each document with batch editor
            saved_files = [f"{filename} (new)" for filename in new_files_created]
            batch_results = []
            
            for doc_id, edits in edits_by_doc.items():
                from orchestrator.utils.document_batch_editor import DocumentEditBatch
                
                batch = DocumentEditBatch(doc_id, user_id, "general_project_agent")
                
                # Initialize batch (reads content once)
                if not await batch.initialize():
                    logger.warning(f"Failed to initialize batch editor for document {doc_id} - skipping")
                    continue
                
                # Add all operations to batch
                for edit in edits:
                    section = edit.get("section", "")
                    content = edit.get("content", "")
                    action = edit.get("action", "append")
                    target_file = edit.get("target_file", "")
                    
                    if section.lower() == "frontmatter":
                        # Parse frontmatter YAML
                        import yaml
                        try:
                            frontmatter_updates = yaml.safe_load(content)
                            if not isinstance(frontmatter_updates, dict):
                                logger.warning(f"Frontmatter content is not a dict - skipping")
                                continue
                            
                            # Separate scalar and list fields
                            scalar_fields = {}
                            list_fields = {}
                            for key, value in frontmatter_updates.items():
                                if isinstance(value, list):
                                    list_fields[key] = value
                                else:
                                    scalar_fields[key] = value
                            
                            batch.add_frontmatter_update(scalar_fields, list_fields)
                        except Exception as e:
                            logger.warning(f"Failed to parse frontmatter YAML: {e}")
                    
                    elif action == "remove":
                        batch.add_section_delete(section)
                    
                    elif action == "replace":
                        batch.add_section_replace(section, content)
                    
                    else:  # append
                        batch.add_section_append(section, content)
                
                # Apply all edits atomically
                result = await batch.apply()
                batch_results.append(result)
                
                # Build saved_files list from results
                if result.get("success"):
                    operations_succeeded = result.get("operations_succeeded", 0)
                    if operations_succeeded > 0:
                        # Get target_file name for display
                        target_file_name = edits[0].get("target_file", "document")
                        saved_files.append(f"{target_file_name} ({operations_succeeded} operations)")
                else:
                    logger.warning(f"Batch edit failed for document {doc_id}: {result.get('error')}")
            
            logger.info(f"Batch editor processed {len(batch_results)} document(s) with {sum(r.get('operations_succeeded', 0) for r in batch_results)} total operations")
            
            logger.info(f"Saved content to {len(saved_files)} files: {', '.join(saved_files)}")
            
            # **PHASE 3B: HANDLE PLAN EDITS FOR INLINE EDITING**
            # If we have plan edits AND active editor is open, use inline editing (editor_operations)
            # This works in both editing_mode and generation_mode - if file is open, show diffs!
            has_active_editor = bool(active_editor and active_editor.get("document_id"))
            
            if plan_edits and has_active_editor:
                logger.info(f"📝 Storing {len(plan_edits)} plan edits for inline editing (file is open - will show diffs)")
                # Store in state for operation resolution node
                return {
                    "plan_edits": plan_edits,
                    "saved_files": saved_files,
                    "editing_mode": True,  # Force editing mode for inline diffs
                    # ✅ CRITICAL: Preserve critical state keys
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "referenced_context": state.get("referenced_context", {}),
                }
            elif plan_edits:
                # No active editor or file not open: apply plan edits directly
                logger.info(f"📝 Applying {len(plan_edits)} plan edits directly (file not open)")
                # 🔒 Use locked target_document_id to prevent race conditions during tab switches
                project_plan_doc_id = (
                    metadata.get("target_document_id") or 
                    (active_editor.get("document_id") if active_editor else None)
                )
                
                # Defensive logging: warn if document_id differs from active_editor
                if active_editor:
                    active_editor_doc_id = active_editor.get("document_id")
                    if project_plan_doc_id and active_editor_doc_id and project_plan_doc_id != active_editor_doc_id:
                        logger.warning(f"⚠️ RACE CONDITION DETECTED (general_project_save): target={project_plan_doc_id}, active_editor={active_editor_doc_id}")
                
                if project_plan_doc_id:
                    from orchestrator.utils.document_batch_editor import DocumentEditBatch
                    plan_batch = DocumentEditBatch(project_plan_doc_id, user_id, "general_project_agent")
                    if await plan_batch.initialize():
                        for edit in plan_edits:
                            section = edit.get("section", "")
                            content = edit.get("content", "")
                            action = edit.get("action", "append")
                            
                            if section.lower() == "frontmatter":
                                # Handle frontmatter updates
                                import yaml
                                try:
                                    frontmatter_updates = yaml.safe_load(content)
                                    if isinstance(frontmatter_updates, dict):
                                        scalar_fields = {}
                                        list_fields = {}
                                        for key, value in frontmatter_updates.items():
                                            if isinstance(value, list):
                                                list_fields[key] = value
                                            else:
                                                scalar_fields[key] = value
                                        plan_batch.add_frontmatter_update(scalar_fields, list_fields)
                                except Exception as e:
                                    logger.warning(f"Failed to parse frontmatter: {e}")
                            elif action == "remove":
                                plan_batch.add_section_delete(section)
                            elif action == "replace":
                                plan_batch.add_section_replace(section, content)
                            else:  # append
                                plan_batch.add_section_append(section, content)
                        
                        result = await plan_batch.apply()
                        if result.get("success"):
                            saved_files.append(f"project_plan ({result.get('operations_succeeded', 0)} operations)")
            
            # **PHASE 4: RELOAD REFERENCED CONTEXT** (Critical for subsequent operations)
            # After creating files and updating frontmatter, reload referenced_context so:
            # 1. Newly created files are available for any post-save operations
            # 2. Next agent run has fresh context with all referenced files
            # 3. Verification/gap analysis in future runs can see newly created files
            reloaded_context = referenced_context  # Default to existing context
            if new_files_created or project_plan_doc_id:
                try:
                    logger.info(f"Reloading referenced context after file creation/updates...")
                    from orchestrator.tools.reference_file_loader import load_referenced_files
                    
                    # General project reference configuration
                    reference_config = {
                        "specifications": ["specifications", "spec", "specs", "specification"],
                        "design": ["design", "design_docs", "architecture"],
                        "tasks": ["tasks", "task", "todo", "checklist"],
                        "notes": ["notes", "note", "documentation", "docs"],
                        "other": ["references", "reference", "files", "related", "documents"]
                    }
                    
                    # Reload referenced files from updated frontmatter
                    reload_result = await load_referenced_files(
                        active_editor=active_editor,
                        user_id=user_id,
                        reference_config=reference_config,
                        doc_type_filter="project"
                    )
                    
                    reloaded_context = reload_result.get("loaded_files", {})
                    reloaded_count = sum(len(docs) for docs in reloaded_context.values() if isinstance(docs, list))
                    logger.info(f"✅ Reloaded referenced context: {reloaded_count} file(s) (including {len(new_files_created)} newly created)")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to reload referenced context after file creation: {e}")
                    # Continue with existing context - not critical for this run
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Update response to include saved files information
            response = state.get("response", {})
            if isinstance(response, dict):
                response_text = response.get("response", "")
                if response_text and saved_files:
                    if len(saved_files) == 1:
                        files_info = f"\n\n**Updated file**: {saved_files[0]}"
                    else:
                        files_list = "\n".join([f"- {file}" for file in saved_files])
                        files_info = f"\n\n**Updated files**:\n{files_list}"
                    
                    response["response"] = response_text + files_info
                    logger.info(f"Added saved files info to response")
            
            return {
                "task_status": "complete",
                "saved_files": saved_files,
                "response": response,
                "referenced_context": reloaded_context,  # Update state with fresh context
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "editing_mode": state.get("editing_mode", False),
                "plan_edits": state.get("plan_edits"),
                "editor_operations": state.get("editor_operations", [])
            }
            
        except Exception as e:
            logger.error(f"Content saving failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "task_status": "error",
                "error": str(e),
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "referenced_context": state.get("referenced_context", {}),
                "editing_mode": state.get("editing_mode", False),
                "plan_edits": state.get("plan_edits"),
                "editor_operations": state.get("editor_operations", [])
            }


