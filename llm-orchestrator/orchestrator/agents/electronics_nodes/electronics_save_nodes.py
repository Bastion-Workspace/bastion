"""
Electronics Agent - Content Saving Nodes Module
Handles saving content to project files and verification

REFACTORED TO USE BATCH EDITING - All edits are atomic and efficient
"""

import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ElectronicsSaveNodes:
    """Content saving nodes for electronics agent"""
    
    def __init__(self, agent):
        """
        Initialize with reference to main agent for access to helper methods
        
        Args:
            agent: ElectronicsAgent instance (for _get_llm, _get_fast_model, etc.)
        """
        self.agent = agent
    
    async def _resolve_document_id(
        self,
        target_file: str,
        active_editor: Dict[str, Any],
        referenced_context: Dict[str, Any],
        user_id: str
    ) -> Optional[str]:
        """
        Resolve document_id from target_file name.
        
        Returns:
            document_id if found, None otherwise
        """
        doc_id = None
        
        if target_file == "project_plan":
            doc_id = active_editor.get("document_id")
            
            if not doc_id and active_editor.get("canonical_path"):
                try:
                    from orchestrator.backend_tool_client import get_backend_tool_client
                    client = await get_backend_tool_client()
                    find_result = await client.find_document_by_path(
                        file_path=active_editor.get("canonical_path"),
                        user_id=user_id
                    )
                    if find_result and find_result.get("document_id"):
                        doc_id = find_result.get("document_id")
                        logger.info(f"🔌 Found project_plan document_id via canonical_path lookup: {doc_id}")
                except Exception as e:
                    logger.warning(f"Failed to lookup document_id from canonical_path: {e}")
            
            if not doc_id and active_editor.get("filename"):
                try:
                    from orchestrator.backend_tool_client import get_backend_tool_client
                    client = await get_backend_tool_client()
                    search_results = await client.search_documents_structured(
                        query=active_editor.get("filename"),
                        user_id=user_id,
                        limit=5
                    )
                    for result in search_results:
                        doc_metadata = result.get("document", {})
                        if doc_metadata.get("filename") == active_editor.get("filename"):
                            doc_id = doc_metadata.get("document_id")
                            logger.info(f"🔌 Found project_plan document_id via filename search: {doc_id}")
                            break
                except Exception as e:
                    logger.warning(f"Failed to lookup document_id from filename: {e}")
            
            logger.info(f"🔌 Routing to project_plan (document_id: {doc_id})")
        else:
            # Find in referenced context
            # For save operations, we can work with files that are referenced OR create new ones
            # But for routing to existing files, we only use referenced_context
            for category, file_list in referenced_context.items():
                if isinstance(file_list, list):
                    for file_doc in file_list:
                        if isinstance(file_doc, dict):
                            file_filename = file_doc.get("filename", "")
                            file_doc_id = file_doc.get("document_id")
                            if file_filename == target_file or file_filename.endswith(target_file) or target_file in file_filename:
                                doc_id = file_doc_id
                                logger.info(f"🔌 Found {target_file} in {category} category (document_id: {doc_id})")
                                break
                    if doc_id:
                        break
        
        return doc_id
    
    def _remove_old_component_references_from_content(self, content: str, query: str) -> str:
        """Clean up old component references from new content"""
        old_components = self._extract_old_components_from_query(query)
        if not old_components:
            return content
        
        cleaned_content = content
        for old_comp in old_components:
            # Remove references to old component
            cleaned_content = re.sub(
                rf'\b{re.escape(old_comp)}\b',
                '',
                cleaned_content,
                flags=re.IGNORECASE
            )
        
        return cleaned_content
    
    def _extract_old_components_from_query(self, query: str) -> List[str]:
        """Extract old component names from replacement queries"""
        old_components = []
        
        # Pattern: "instead of X", "replace X", "switch from X", "change X to Y"
        patterns = [
            r'instead\s+of\s+([A-Z0-9]+)',
            r'replace\s+([A-Z0-9]+)',
            r'switch\s+from\s+([A-Z0-9]+)',
            r'change\s+([A-Z0-9]+)\s+to',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            old_components.extend(matches)
        
        return list(set(old_components))  # Remove duplicates
    
    async def save_content_node(self, state) -> Dict[str, Any]:
        """
        Save node using batch editor - groups edits by document and applies atomically.
        Preserves electronics-specific features like component verification.
        """
        try:
            # Check if pending_save_plan should be restored
            pending_save_plan = state.get("pending_save_plan")
            if pending_save_plan:
                logger.info("🔌 Restoring pending save plan from approval")
                state["save_plan"] = pending_save_plan
            
            save_plan = state.get("save_plan", {})
            if not save_plan or not save_plan.get("routing"):
                logger.info("🔌 No content to save")
                return {}
            
            user_id = state.get("user_id", "")
            metadata = state.get("metadata", {})
            active_editor = metadata.get("active_editor") or metadata.get("shared_memory", {}).get("active_editor", {})
            referenced_context = state.get("referenced_context", {})
            
            routing_items = save_plan.get("routing", [])
            
            # **PHASE 1: CREATE NEW REFERENCE FILES**
            project_plan_doc_id = active_editor.get("document_id") if active_editor else None
            new_files_created = []
            
            for item in routing_items[:]:  # Iterate over copy
                create_new = item.get("create_new_file", False)
                if not create_new:
                    continue
                
                suggested_filename = item.get("suggested_filename", "")
                file_summary = item.get("file_summary", "")
                
                if not suggested_filename:
                    logger.warning("create_new_file=true but no suggested_filename provided - skipping")
                    continue
                
                logger.info(f"🔌 Creating new reference file: {suggested_filename} ({file_summary})")
                
                try:
                    # Determine folder from active_editor
                    # PRIORITY 1: Use folder_id if available (most reliable)
                    folder_id = None
                    folder_path = None
                    
                    if active_editor:
                        folder_id = active_editor.get("folder_id")
                        if folder_id:
                            logger.info(f"🔌 Using folder_id from active_editor: {folder_id}")
                    
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
                                        logger.info(f"🔌 Extracted folder_path from canonical_path: {folder_path}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to extract folder_path from canonical_path: {e}")
                    
                    if not folder_id and not folder_path:
                        logger.warning("⚠️ Could not determine folder_id or folder_path - file will be created in My Documents root")
                    
                    # Create the new file with initial frontmatter
                    from orchestrator.tools.file_creation_tools import create_user_file_tool
                    initial_content = f"""---
type: electronics
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
                    
                    # Add to project plan frontmatter
                    if project_plan_doc_id:
                        from orchestrator.utils.document_batch_editor import DocumentEditBatch
                        plan_batch = DocumentEditBatch(project_plan_doc_id, user_id, "electronics_agent")
                        await plan_batch.initialize()
                        
                        # Determine the appropriate frontmatter list based on file type
                        list_key = "files"  # Default
                        if "component" in suggested_filename.lower():
                            list_key = "components"
                        elif "protocol" in suggested_filename.lower() or "communication" in suggested_filename.lower():
                            list_key = "protocols"
                        elif "schematic" in suggested_filename.lower() or "circuit" in suggested_filename.lower():
                            list_key = "schematics"
                        elif "spec" in suggested_filename.lower():
                            list_key = "specifications"
                        
                        plan_batch.add_frontmatter_update({}, {list_key: [f"./{suggested_filename}"]})
                        await plan_batch.apply()
                        logger.info(f"✅ Added {suggested_filename} reference to project plan frontmatter ({list_key})")
                    
                    new_files_created.append(suggested_filename)
                    routing_items.remove(item)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to create new reference file {suggested_filename}: {e}")
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
                    target_file == active_editor.get("filename", "") or
                    target_file == active_editor.get("canonical_path", "")
                )
                
                # Clean up old component references from new content
                if item.get("action") == "replace" and item.get("content"):
                    item["content"] = self._remove_old_component_references_from_content(
                        item["content"], state.get("query", "")
                    )
                
                if is_plan_edit and editing_mode:
                    # Store for inline editing (skip frontmatter updates - apply directly)
                    if item.get("section", "").lower() != "frontmatter":
                        plan_edits.append(item)
                else:
                    # Apply directly (referenced files or generation mode)
                    referenced_edits.append(item)
            
            # **PHASE 2B: GROUP REFERENCED EDITS BY DOCUMENT**
            edits_by_doc: Dict[str, List[Dict[str, Any]]] = {}
            
            for item in referenced_edits:
                target_file = item.get("target_file", "")
                if not target_file:
                    continue
                
                # Resolve document_id
                doc_id = await self._resolve_document_id(
                    target_file, active_editor, referenced_context, user_id
                )
                
                if not doc_id:
                    logger.warning(f"⚠️ Could not resolve document_id for {target_file} - skipping")
                    continue
                
                # Group by document_id
                if doc_id not in edits_by_doc:
                    edits_by_doc[doc_id] = []
                edits_by_doc[doc_id].append(item)
            
            # **PHASE 3: BATCH EDIT EXISTING FILES**
            saved_files = [f"{filename} (new)" for filename in new_files_created]
            batch_results = []
            
            # Track cleanup operations (edits/replaces, not appends) for summary reporting
            cleanup_operations = []  # List of (filename, section_name, action) tuples
            
            for doc_id, edits in edits_by_doc.items():
                from orchestrator.utils.document_batch_editor import DocumentEditBatch
                
                batch = DocumentEditBatch(doc_id, user_id, "electronics_agent")
                
                # Initialize batch (reads content once)
                if not await batch.initialize():
                    logger.warning(f"⚠️ Failed to initialize batch editor for document {doc_id} - skipping")
                    continue
                
                # Add all operations to batch
                for edit in edits:
                    section = edit.get("section", "")
                    content = edit.get("content", "")
                    action = edit.get("action", "append")
                    target_file = edit.get("target_file", "")
                    
                    if action == "remove":
                        batch.add_section_delete(section)
                        cleanup_operations.append((target_file, section, "removed"))
                    
                    elif action == "replace":
                        batch.add_section_replace(section, content)
                        cleanup_operations.append((target_file, section, "updated"))
                    
                    else:  # append
                        batch.add_section_append(section, content)
                
                # Apply all edits atomically
                result = await batch.apply()
                batch_results.append(result)
                
                # Build saved_files list from results
                if result.get("success"):
                    operations_succeeded = result.get("operations_succeeded", 0)
                    if operations_succeeded > 0:
                        target_file_name = edits[0].get("target_file", "document")
                        saved_files.append(f"{target_file_name} ({operations_succeeded} operations)")
                else:
                    logger.warning(f"⚠️ Batch edit failed for document {doc_id}: {result.get('error')}")
            
            logger.info(f"✅ Batch editor processed {len(batch_results)} document(s) with {sum(r.get('operations_succeeded', 0) for r in batch_results)} total operations")
            logger.info(f"✅ Saved content to {len(saved_files)} files: {', '.join(saved_files)}")
            
            # **PHASE 3B: HANDLE PLAN EDITS FOR INLINE EDITING**
            # If we have plan edits in editing mode, store them for operation resolution
            # Otherwise, apply plan edits directly (generation mode or frontmatter-only)
            if plan_edits and editing_mode:
                logger.info(f"📝 Storing {len(plan_edits)} plan edits for inline editing")
                # Store in state for operation resolution node
                return {
                    "plan_edits": plan_edits,
                    "saved_files": saved_files,
                    "cleanup_operations": cleanup_operations
                }
            elif plan_edits:
                # Generation mode: apply plan edits directly
                logger.info(f"📝 Applying {len(plan_edits)} plan edits directly (generation mode)")
                if project_plan_doc_id:
                    from orchestrator.utils.document_batch_editor import DocumentEditBatch
                    plan_batch = DocumentEditBatch(project_plan_doc_id, user_id, "electronics_agent")
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
            # 3. Maintenance/verification in future runs can see newly created files
            reloaded_context = referenced_context  # Default to existing context
            if new_files_created or project_plan_doc_id:
                try:
                    logger.info(f"🔌 Reloading referenced context after file creation/updates...")
                    from orchestrator.tools.reference_file_loader import load_referenced_files
                    
                    # Electronics reference configuration
                    reference_config = {
                        "components": ["components", "component", "component_docs"],
                        "protocols": ["protocols", "protocol", "protocol_docs"],
                        "schematics": ["schematics", "schematic", "schematic_docs"],
                        "specifications": ["specifications", "spec", "specs", "specification"],
                        "other": ["references", "reference", "docs", "documents", "related", "files"]
                    }
                    
                    # Reload referenced files from updated frontmatter
                    reload_result = await load_referenced_files(
                        active_editor=active_editor,
                        user_id=user_id,
                        reference_config=reference_config,
                        doc_type_filter="electronics"
                    )
                    
                    reloaded_context = reload_result.get("loaded_files", {})
                    reloaded_count = sum(len(docs) for docs in reloaded_context.values() if isinstance(docs, list))
                    logger.info(f"✅ Reloaded referenced context: {reloaded_count} file(s) (including {len(new_files_created)} newly created)")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to reload referenced context after file creation: {e}")
                    # Continue with existing context - not critical for this run
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Generate cleanup summary if cleanup operations occurred
            cleanup_summary = None
            if cleanup_operations:
                # Count operations by type
                updates_count = sum(1 for _, _, action in cleanup_operations if action == "updated")
                removals_count = sum(1 for _, _, action in cleanup_operations if action == "removed")
                
                # Build summary message
                summary_parts = []
                if updates_count > 0:
                    summary_parts.append(f"{updates_count} section{'s' if updates_count != 1 else ''} updated")
                if removals_count > 0:
                    summary_parts.append(f"{removals_count} section{'s' if removals_count != 1 else ''} removed")
                
                if summary_parts:
                    cleanup_summary = f"Documentation cleanup: {', '.join(summary_parts)} with corrections."
                    logger.info(f"🔌 {cleanup_summary}")
            
            # **PHASE 5: VERIFICATION LOOP** (Electronics-specific)
            # Check for any remaining outdated/incorrect references
            user_query = state.get("query", "")
            old_components = self._extract_old_components_from_query(user_query)
            
            change_indicators = [
                r'instead\s+of', r'replace', r'switch\s+from', r'change\s+\w+\s+to',
                r'actually', r'wrong', r'should\s+be', r'not\s+\w+\s+anymore',
                r'remove', r'delete', r'eliminate', r'update', r'correct',
            ]
            has_changes = any(re.search(pattern, user_query, re.IGNORECASE) for pattern in change_indicators)
            
            # Build return value with cleanup summary and reloaded context
            return_data = {
                "task_status": "complete",
                "saved_files": saved_files,
                "verification_needed": False,
                "referenced_context": reloaded_context  # Update state with fresh context
            }
            
            # Add cleanup summary if available
            if cleanup_summary:
                return_data["cleanup_summary"] = cleanup_summary
            
            if old_components or has_changes:
                if old_components:
                    logger.info(f"🔍 Verifying that old component references were removed: {old_components}")
                else:
                    logger.info(f"🔍 Verifying that documentation updates were applied correctly")
                
                # Store verification info for possible maintenance node processing
                # The maintenance node can do a comprehensive check if needed
                return_data["verification_needed"] = True
                return_data["old_components"] = old_components if old_components else []
            
            return return_data
            
        except Exception as e:
            logger.error(f"❌ Content saving failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "task_status": "error",
                "error": str(e)
            }


