"""
Document Editing Tools for Agents
Allows agents to update document titles and frontmatter in user's documents
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# In-memory storage for document edit proposals
# TODO: Migrate to database for persistence across restarts
_document_edit_proposals: Dict[str, Dict[str, Any]] = {}


async def update_document_metadata_tool(
    document_id: str,
    title: Optional[str] = None,
    frontmatter_type: Optional[str] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Update document title and/or frontmatter type
    
    **SECURITY**: Only updates user's own documents (collection_type='user')
    Agents cannot modify global documents
    
    Args:
        document_id: Document ID to update
        title: Optional new title (updates both database metadata and frontmatter if file has frontmatter)
        frontmatter_type: Optional frontmatter type (e.g., "electronics", "fiction", "rules") - updates file content
        user_id: User ID (required - must match document owner)
    
    Returns:
        Dict with success, message, and updated fields
    """
    try:
        logger.info(f"📝 Updating document metadata: {document_id} (title={title}, type={frontmatter_type})")
        
        # Import services
        from services.service_container import get_service_container
        from utils.frontmatter_utils import parse_frontmatter, build_frontmatter
        
        # Get service container
        container = await get_service_container()
        document_service = container.document_service
        folder_service = container.folder_service
        
        # Get document info
        doc_info = await document_service.get_document(document_id)
        if not doc_info:
            return {
                "success": False,
                "error": "Document not found",
                "message": f"Document {document_id} not found"
            }
        
        # Security check: ensure document belongs to user
        doc_user_id = getattr(doc_info, 'user_id', None)
        doc_collection_type = getattr(doc_info, 'collection_type', 'user')
        
        if doc_collection_type != "user":
            return {
                "success": False,
                "error": "Cannot modify global documents",
                "message": "Agents can only modify user documents, not global documents"
            }
        
        if doc_user_id and doc_user_id != user_id:
            return {
                "success": False,
                "error": "Access denied",
                "message": f"Document belongs to different user (document user: {doc_user_id}, requesting user: {user_id})"
            }
        
        updated_fields = []
        
        # Update database metadata (title)
        if title:
            from models.api_models import DocumentUpdateRequest
            update_request = DocumentUpdateRequest(title=title)
            success = await document_service.update_document_metadata(document_id, update_request)
            if success:
                updated_fields.append("title")
                logger.info(f"✅ Updated document title: {title}")
            else:
                logger.warning(f"⚠️ Failed to update document title")
        
        # Update frontmatter in file content (type and/or title)
        if frontmatter_type or (title and doc_info.filename and doc_info.filename.lower().endswith(('.md', '.txt', '.org'))):
            try:
                # Get file path
                file_path = await folder_service.get_document_file_path(
                    filename=doc_info.filename,
                    folder_id=getattr(doc_info, 'folder_id', None),
                    user_id=doc_user_id,
                    collection_type=doc_collection_type
                )
                
                if file_path and file_path.exists():
                    # Read current content
                    current_content = file_path.read_text(encoding='utf-8')
                    
                    # Parse frontmatter - but preserve the original frontmatter block for complex fields
                    # The simple parser only handles key-value pairs, so we need to preserve the original
                    # frontmatter block and only update specific fields
                    import re
                    frontmatter_match = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n", current_content)
                    
                    if frontmatter_match:
                        # Extract original frontmatter block and body
                        original_frontmatter_block = frontmatter_match.group(0)
                        frontmatter_text = frontmatter_match.group(1)
                        body = current_content[frontmatter_match.end():]
                        
                        # Parse simple fields
                        frontmatter, _ = parse_frontmatter(current_content)
                        
                        # Update only the fields we're changing
                        if frontmatter_type:
                            # Replace or add type field
                            if re.search(r'^type:\s*', frontmatter_text, re.MULTILINE):
                                frontmatter_text = re.sub(r'^type:\s*.*$', f'type: {frontmatter_type}', frontmatter_text, flags=re.MULTILINE)
                            else:
                                # Add after first line
                                lines = frontmatter_text.split('\n')
                                if len(lines) > 0:
                                    lines.insert(1, f'type: {frontmatter_type}')
                                else:
                                    lines.append(f'type: {frontmatter_type}')
                                frontmatter_text = '\n'.join(lines)
                            updated_fields.append("frontmatter_type")
                            logger.info(f"✅ Updated frontmatter type: {frontmatter_type}")
                        
                        if title:
                            # Replace or add title field
                            if re.search(r'^title:\s*', frontmatter_text, re.MULTILINE):
                                frontmatter_text = re.sub(r'^title:\s*.*$', f'title: {title}', frontmatter_text, flags=re.MULTILINE)
                            else:
                                # Add after first line
                                lines = frontmatter_text.split('\n')
                                if len(lines) > 0:
                                    lines.insert(1, f'title: {title}')
                                else:
                                    lines.append(f'title: {title}')
                                frontmatter_text = '\n'.join(lines)
                            if "title" not in updated_fields:
                                updated_fields.append("title (frontmatter)")
                        
                        # Rebuild frontmatter block preserving all original fields including lists
                        new_frontmatter_block = f"---\n{frontmatter_text}\n---\n"
                        new_content = new_frontmatter_block + body
                    else:
                        # No frontmatter - create new one
                        frontmatter = {}
                        if frontmatter_type:
                            frontmatter['type'] = frontmatter_type
                        if title:
                            frontmatter['title'] = title
                        new_frontmatter_block = build_frontmatter(frontmatter)
                        new_content = new_frontmatter_block + "\n" + current_content
                    
                    # Write updated content
                    file_path.write_text(new_content, encoding='utf-8')
                    logger.info(f"✅ Updated file content: {file_path}")
                    
                    # Update file size in database
                    await document_service.document_repository.update_file_size(
                        document_id, 
                        len(new_content.encode('utf-8'))
                    )
                else:
                    logger.warning(f"⚠️ File not found on disk: {file_path}")
            except Exception as e:
                logger.error(f"❌ Failed to update file frontmatter: {e}")
                # Continue - database update may have succeeded
        
        if updated_fields:
            return {
                "success": True,
                "message": f"Updated document: {', '.join(updated_fields)}",
                "updated_fields": updated_fields,
                "document_id": document_id
            }
        else:
            return {
                "success": False,
                "error": "No fields to update",
                "message": "No valid updates provided"
            }
        
    except Exception as e:
        logger.error(f"❌ Failed to update document metadata: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to update document: {str(e)}"
        }


async def update_document_content_tool(
    document_id: str,
    content: str,
    user_id: str = "system",
    append: bool = False
) -> Dict[str, Any]:
    """
    Update document content (append or replace)
    
    **SECURITY**: Only updates user's own documents (collection_type='user')
    Agents cannot modify global documents
    
    Args:
        document_id: Document ID to update
        content: New content to add (if append=True) or replace entire content (if append=False)
        user_id: User ID (required - must match document owner)
        append: If True, append content to existing; if False, replace entire content
    
    Returns:
        Dict with success, document_id, content_length, and message
    """
    try:
        logger.info(f"📝 Updating document content: {document_id} (append={append}, content_length={len(content)})")
        
        # Import services
        from services.service_container import get_service_container
        from models.api_models import ProcessingStatus
        
        # **CRITICAL**: Use proper YAML parser that handles lists and complex structures
        # The simple parser in utils.frontmatter_utils only handles key:value pairs
        # and will corrupt list fields like files: ["./file1.md"]
        try:
            import yaml
            def parse_frontmatter_yaml(text: str):
                """Parse frontmatter using proper YAML parser that handles lists"""
                import re
                m = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n", text)
                if not m:
                    return {}, text
                yaml_block = m.group(1)
                body = text[m.end():]
                try:
                    frontmatter = yaml.safe_load(yaml_block) or {}
                    if not isinstance(frontmatter, dict):
                        frontmatter = {}
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse frontmatter as YAML: {e}")
                    frontmatter = {}
                return frontmatter, body
            
            def build_frontmatter_yaml(data: dict) -> str:
                """Build frontmatter using proper YAML dumper that preserves lists"""
                if not data:
                    return ""
                try:
                    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
                    return f"---\n{yaml_str}\n---\n"
                except Exception as e:
                    logger.error(f"❌ Failed to build frontmatter: {e}")
                    return ""
            
            parse_frontmatter = parse_frontmatter_yaml
            build_frontmatter = build_frontmatter_yaml
        except ImportError:
            # Fallback to simple parser if yaml not available (shouldn't happen in production)
            from utils.frontmatter_utils import parse_frontmatter, build_frontmatter
            logger.warning("⚠️ YAML library not available - using simple frontmatter parser (WILL CORRUPT LIST FIELDS)")
        
        # Get service container
        container = await get_service_container()
        document_service = container.document_service
        folder_service = container.folder_service
        
        # Get document info
        doc_info = await document_service.get_document(document_id)
        if not doc_info:
            return {
                "success": False,
                "error": "Document not found",
                "message": f"Document {document_id} not found"
            }
        
        # Security check: ensure document belongs to user
        doc_user_id = getattr(doc_info, 'user_id', None)
        doc_collection_type = getattr(doc_info, 'collection_type', 'user')
        
        if doc_collection_type != "user":
            return {
                "success": False,
                "error": "Cannot modify global documents",
                "message": "Agents can only modify user documents, not global documents"
            }
        
        if doc_user_id and doc_user_id != user_id:
            return {
                "success": False,
                "error": "Access denied",
                "message": f"Document belongs to different user (document user: {doc_user_id}, requesting user: {user_id})"
            }
        
        # Get file path
        file_path = await folder_service.get_document_file_path(
            filename=doc_info.filename,
            folder_id=getattr(doc_info, 'folder_id', None),
            user_id=doc_user_id,
            collection_type=doc_collection_type
        )
        
        if not file_path or not file_path.exists():
            return {
                "success": False,
                "error": "File not found",
                "message": f"Document file not found on disk: {file_path}"
            }
        
        # Read current content
        current_content = file_path.read_text(encoding='utf-8')
        
        # Parse frontmatter if it exists
        frontmatter, body = parse_frontmatter(current_content)
        has_frontmatter = bool(frontmatter)
        
        # **CRITICAL**: Remove any duplicate frontmatter blocks from body
        # This prevents frontmatter duplication when appending
        if has_frontmatter and body:
            import re
            # Remove any frontmatter blocks that might exist in the body
            body = re.sub(r'^---\s*\r?\n[\s\S]*?\r?\n---\s*\r?\n', '', body, flags=re.MULTILINE)
            logger.debug(f"Cleaned body of any duplicate frontmatter blocks")
        
        if append:
            # Strip any frontmatter from content being appended (shouldn't have frontmatter)
            # This prevents frontmatter duplication
            content_to_append = content
            if content.strip().startswith('---'):
                # Content has frontmatter - extract body only
                import re
                frontmatter_match = re.match(r'^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n', content)
                if frontmatter_match:
                    # Extract body after frontmatter
                    content_to_append = content[frontmatter_match.end():].strip()
                    logger.warning(f"⚠️ Content being appended had frontmatter - stripped it to prevent duplication")
            
            # Append new content to body (preserving frontmatter)
            if has_frontmatter:
                new_body = body + "\n\n" + content_to_append
                new_content = build_frontmatter(frontmatter) + new_body
            else:
                new_content = current_content + "\n\n" + content_to_append
            logger.info(f"Appending {len(content_to_append)} chars to existing {len(current_content)} chars")
        else:
            # Replace entire content
            # **CRITICAL**: Strip frontmatter from incoming content to prevent duplication
            content_to_replace = content
            if content.strip().startswith('---'):
                # Content has frontmatter - extract body only
                import re
                frontmatter_match = re.match(r'^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n', content)
                if frontmatter_match:
                    # Extract body after frontmatter
                    content_to_replace = content[frontmatter_match.end():].strip()
                    logger.warning(f"⚠️ Content being replaced had frontmatter - stripped it to prevent duplication")
            
            if has_frontmatter:
                # Preserve existing frontmatter, replace body with cleaned content
                new_content = build_frontmatter(frontmatter) + "\n\n" + content_to_replace
            else:
                # No existing frontmatter, use cleaned content (which may or may not have had frontmatter)
                new_content = content_to_replace
            logger.info(f"Replacing entire content ({len(current_content)} chars) with new content ({len(content_to_replace)} chars)")
        
        # Write updated content to file
        file_path.write_text(new_content, encoding='utf-8')
        logger.info(f"✅ Updated file content: {file_path} ({len(new_content)} chars)")
        
        # Update file size in database
        await document_service.document_repository.update_file_size(
            document_id, 
            len(new_content.encode('utf-8'))
        )
        
        # Check if document is exempt from vectorization BEFORE processing
        is_exempt = await document_service.document_repository.is_document_exempt(document_id)
        if is_exempt:
            logger.info(f"🚫 Document {document_id} is exempt from vectorization - skipping embedding and KG extraction")
            await document_service.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            await document_service._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            return {
                "success": True,
                "document_id": document_id,
                "content_length": len(new_content),
                "message": f"Document content updated successfully ({'appended' if append else 'replaced'}) - exempt from vectorization"
            }
        
        # Re-embed the document (trigger reprocessing)
        # Update status to embedding to trigger reprocessing
        await document_service.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
        
        # Delete old vectors and knowledge graph entities
        await document_service.embedding_manager.delete_document_chunks(document_id)
        
        if document_service.kg_service:
            try:
                await document_service.kg_service.delete_document_entities(document_id)
                logger.info(f"🗑️ Deleted old knowledge graph entities for {document_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete old KG entities for {document_id}: {e}")
        
        # Re-process content into chunks
        metadata = {
            "title": getattr(doc_info, 'title', ''),
            "tags": getattr(doc_info, 'tags', []),
            "category": getattr(doc_info, 'category', '')
        }
        
        chunks = await document_service.document_processor.process_text_content(
            new_content, document_id, metadata
        )
        
        # Store chunks in vector database
        if chunks:
            await document_service.embedding_manager.embed_and_store_chunks(chunks, document_id)
            logger.info(f"✅ Re-embedded {len(chunks)} chunks for document {document_id}")
        
        # Update status to completed
        await document_service.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
        
        # Emit WebSocket notification for UI refresh (so open editor tabs update automatically)
        await document_service._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
        
        return {
            "success": True,
            "document_id": document_id,
            "content_length": len(new_content),
            "message": f"Document content updated successfully ({'appended' if append else 'replaced'})"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to update document content: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to update document content: {str(e)}"
        }


async def propose_document_edit_tool(
    document_id: str,
    edit_type: str,
    operations: Optional[List[Dict[str, Any]]] = None,
    content_edit: Optional[Dict[str, Any]] = None,
    agent_name: str = "unknown",
    summary: str = "",
    requires_preview: bool = True,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Propose a document edit for user review (universal edit proposal system)
    
    **SECURITY**: Only proposes edits for user's own documents (collection_type='user')
    
    Args:
        document_id: Document ID to edit
        edit_type: "operations" or "content"
        operations: List of EditorOperation dicts (for operation-based edits)
        content_edit: ContentEdit dict (for content-based edits)
        agent_name: Name of proposing agent
        summary: Human-readable summary of proposed changes
        requires_preview: If False and edit is small, frontend may auto-apply
        user_id: User ID (required - must match document owner)
    
    Returns:
        Dict with success, proposal_id, document_id, and message
    """
    try:
        logger.info(f"📝 Proposing document edit: {document_id} (type={edit_type}, agent={agent_name})")
        
        # Import services
        from services.service_container import get_service_container
        
        # Get service container
        container = await get_service_container()
        document_service = container.document_service
        
        # Get document info
        doc_info = await document_service.get_document(document_id)
        if not doc_info:
            return {
                "success": False,
                "error": "Document not found",
                "message": f"Document {document_id} not found"
            }
        
        # Security check: ensure document belongs to user
        doc_user_id = getattr(doc_info, 'user_id', None)
        doc_collection_type = getattr(doc_info, 'collection_type', 'user')
        
        if doc_collection_type != "user":
            return {
                "success": False,
                "error": "Cannot propose edits for global documents",
                "message": "Agents can only propose edits for user documents, not global documents"
            }
        
        if doc_user_id and doc_user_id != user_id:
            return {
                "success": False,
                "error": "Access denied",
                "message": f"Document belongs to different user (document user: {doc_user_id}, requesting user: {user_id})"
            }
        
        # Validate edit type
        if edit_type == "operations" and (not operations or len(operations) == 0):
            return {
                "success": False,
                "error": "Invalid proposal",
                "message": "operations field is required when edit_type='operations'"
            }
        
        if edit_type == "content" and not content_edit:
            return {
                "success": False,
                "error": "Invalid proposal",
                "message": "content_edit field is required when edit_type='content'"
            }
        
        # Generate proposal ID
        proposal_id = str(uuid.uuid4())
        
        # Store proposal
        proposal = {
            "proposal_id": proposal_id,
            "document_id": document_id,
            "edit_type": edit_type,
            "operations": operations or [],
            "content_edit": content_edit,
            "agent_name": agent_name,
            "summary": summary,
            "requires_preview": requires_preview,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "applied": False
        }
        
        _document_edit_proposals[proposal_id] = proposal
        
        logger.info(f"✅ Document edit proposal created: {proposal_id} for document {document_id}")
        
        return {
            "success": True,
            "proposal_id": proposal_id,
            "document_id": document_id,
            "message": f"Document edit proposal created successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to propose document edit: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to propose document edit: {str(e)}"
        }


async def apply_operations_directly(
    document_id: str,
    operations: List[Dict[str, Any]],
    user_id: str = "system",
    agent_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Apply operations directly to a document file without creating a proposal.
    
    **SECURITY**: Only allowed for specific agents (electronics_agent) editing referenced files.
    This is a restricted operation - use with caution!
    
    Args:
        document_id: Document ID to edit
        operations: List of EditorOperation dicts to apply
        user_id: User ID (required - must match document owner)
        agent_name: Name of agent requesting this operation (for security check)
    
    Returns:
        Dict with success, document_id, applied_count, and message
    """
    # Security check: Only allow specific agents
    ALLOWED_AGENTS = ["electronics_agent", "project_content_manager"]
    if agent_name not in ALLOWED_AGENTS:
        return {
            "success": False,
            "error": "Agent not authorized",
            "message": f"Agent '{agent_name}' is not authorized to apply operations directly. Allowed agents: {ALLOWED_AGENTS}"
        }
    
    try:
        logger.info(f"📝 Applying operations directly to document: {document_id} (agent: {agent_name}, {len(operations)} operations)")
        
        # Import services
        from services.service_container import get_service_container
        from models.api_models import ProcessingStatus
        
        # **CRITICAL**: Use proper YAML parser that handles lists and complex structures
        try:
            import yaml
            def parse_frontmatter_yaml(text: str):
                """Parse frontmatter using proper YAML parser that handles lists"""
                import re
                m = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n", text)
                if not m:
                    return {}, text
                yaml_block = m.group(1)
                body = text[m.end():]
                try:
                    frontmatter = yaml.safe_load(yaml_block) or {}
                    if not isinstance(frontmatter, dict):
                        frontmatter = {}
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse frontmatter as YAML: {e}")
                    frontmatter = {}
                return frontmatter, body
            
            def build_frontmatter_yaml(data: dict) -> str:
                """Build frontmatter using proper YAML dumper that preserves lists"""
                if not data:
                    return ""
                try:
                    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
                    return f"---\n{yaml_str}\n---\n"
                except Exception as e:
                    logger.error(f"❌ Failed to build frontmatter: {e}")
                    return ""
            
            parse_frontmatter = parse_frontmatter_yaml
            build_frontmatter = build_frontmatter_yaml
        except ImportError:
            # Fallback to simple parser if yaml not available (shouldn't happen in production)
            from utils.frontmatter_utils import parse_frontmatter, build_frontmatter
            logger.warning("⚠️ YAML library not available - using simple frontmatter parser (WILL CORRUPT LIST FIELDS)")
        
        container = await get_service_container()
        document_service = container.document_service
        folder_service = container.folder_service
        
        # Get document info
        doc_info = await document_service.get_document(document_id)
        if not doc_info:
            return {
                "success": False,
                "error": "Document not found",
                "message": f"Document {document_id} not found"
            }
        
        # Security check: Only user's own documents
        if getattr(doc_info, 'collection_type', 'user') != 'user' or getattr(doc_info, 'user_id', '') != user_id:
            return {
                "success": False,
                "error": "Access denied",
                "message": f"Document belongs to different user or collection"
            }
        
        # Get file path
        file_path = await folder_service.get_document_file_path(
            filename=doc_info.filename,
            folder_id=getattr(doc_info, 'folder_id', None),
            user_id=user_id,
            collection_type=getattr(doc_info, 'collection_type', 'user')
        )
        
        if not file_path or not file_path.exists():
            return {
                "success": False,
                "error": "File not found",
                "message": f"Document file not found on disk: {file_path}"
            }
        
        # Read current content
        current_content = file_path.read_text(encoding='utf-8')
        frontmatter, body = parse_frontmatter(current_content)
        has_frontmatter = bool(frontmatter)
        
        # Apply operations (same logic as apply_document_edit_proposal)
        # Sort operations by start position (highest first to keep offsets stable)
        sorted_ops = sorted(operations, key=lambda op: op.get("start", 0), reverse=True)
        
        new_content = current_content
        for op in sorted_ops:
            op_type = op.get("op_type", "replace_range")
            start = op.get("start", 0)
            end = op.get("end", start)
            text = op.get("text", "")
            
            if op_type == "delete_range":
                new_content = new_content[:start] + new_content[end:]
            elif op_type == "replace_range":
                new_content = new_content[:start] + text + new_content[end:]
            elif op_type == "insert_after_heading":
                # For insert_after_heading, we need to find the anchor and insert after it
                anchor_text = op.get("anchor_text", "")
                if anchor_text:
                    anchor_pos = new_content.find(anchor_text)
                    if anchor_pos != -1:
                        # Find end of line after anchor
                        line_end = new_content.find("\n", anchor_pos + len(anchor_text))
                        if line_end == -1:
                            line_end = len(new_content)
                        insert_pos = line_end + 1
                        new_content = new_content[:insert_pos] + text + "\n" + new_content[insert_pos:]
                    else:
                        logger.warning(f"⚠️ Anchor text not found for insert_after_heading: {anchor_text}")
                else:
                    # Fallback to end
                    new_content = new_content + "\n" + text
        
        # Write updated content to file
        file_path.write_text(new_content, encoding='utf-8')
        logger.info(f"✅ Applied operations directly: {file_path} ({len(new_content)} chars, {len(sorted_ops)} operations)")
        
        # Update file size in database
        await document_service.document_repository.update_file_size(
            document_id,
            len(new_content.encode('utf-8'))
        )
        
        # Check if document is exempt from vectorization BEFORE processing
        is_exempt = await document_service.document_repository.is_document_exempt(document_id)
        if is_exempt:
            logger.info(f"🚫 Document {document_id} is exempt from vectorization - skipping embedding and KG extraction")
            await document_service.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            await document_service._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            return {
                "success": True,
                "document_id": document_id,
                "applied_count": len(sorted_ops),
                "message": f"Applied {len(sorted_ops)} operation(s) directly to document - exempt from vectorization"
            }
        
        # Re-embed the document
        await document_service.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
        await document_service.embedding_manager.delete_document_chunks(document_id)
        
        if document_service.kg_service:
            try:
                await document_service.kg_service.delete_document_entities(document_id)
                logger.info(f"🗑️ Deleted old knowledge graph entities for {document_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete old KG entities for {document_id}: {e}")
        
        # Re-process content into chunks
        metadata = {
            "title": getattr(doc_info, 'title', ''),
            "tags": getattr(doc_info, 'tags', []),
            "category": getattr(doc_info, 'category', '')
        }
        
        chunks = await document_service.document_processor.process_text_content(
            new_content, document_id, metadata
        )
        
        # Store chunks in vector database
        if chunks:
            await document_service.embedding_manager.embed_and_store_chunks(chunks, document_id)
            logger.info(f"✅ Re-embedded {len(chunks)} chunks for document {document_id}")
        
        # Update status to completed
        await document_service.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
        
        # Emit WebSocket notification for UI refresh
        await document_service._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
        
        return {
            "success": True,
            "document_id": document_id,
            "applied_count": len(sorted_ops),
            "message": f"Applied {len(sorted_ops)} operation(s) directly to document"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to apply operations directly: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to apply operations directly: {str(e)}"
        }


async def apply_document_edit_proposal(
    proposal_id: str,
    selected_operation_indices: Optional[List[int]] = None,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Apply an approved document edit proposal
    
    **SECURITY**: Only applies proposals for user's own documents
    
    Args:
        proposal_id: ID of proposal to apply
        selected_operation_indices: Which operations to apply (None = all, only for operation-based edits)
        user_id: User ID (required - must match proposal owner)
    
    Returns:
        Dict with success, document_id, applied_count, and message
    """
    try:
        logger.info(f"📝 Applying document edit proposal: {proposal_id}")
        
        # Get proposal
        proposal = _document_edit_proposals.get(proposal_id)
        if not proposal:
            return {
                "success": False,
                "error": "Proposal not found",
                "message": f"Proposal {proposal_id} not found"
            }
        
        # Security check
        if proposal["user_id"] != user_id:
            return {
                "success": False,
                "error": "Access denied",
                "message": f"Proposal belongs to different user"
            }
        
        if proposal["applied"]:
            return {
                "success": False,
                "error": "Proposal already applied",
                "message": f"Proposal {proposal_id} has already been applied"
            }
        
        document_id = proposal["document_id"]
        edit_type = proposal["edit_type"]
        applied_count = 0
        
        # Import services
        from services.service_container import get_service_container
        from models.api_models import ProcessingStatus
        
        # **CRITICAL**: Use proper YAML parser that handles lists and complex structures
        try:
            import yaml
            def parse_frontmatter_yaml(text: str):
                """Parse frontmatter using proper YAML parser that handles lists"""
                import re
                m = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n", text)
                if not m:
                    return {}, text
                yaml_block = m.group(1)
                body = text[m.end():]
                try:
                    frontmatter = yaml.safe_load(yaml_block) or {}
                    if not isinstance(frontmatter, dict):
                        frontmatter = {}
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse frontmatter as YAML: {e}")
                    frontmatter = {}
                return frontmatter, body
            
            def build_frontmatter_yaml(data: dict) -> str:
                """Build frontmatter using proper YAML dumper that preserves lists"""
                if not data:
                    return ""
                try:
                    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
                    return f"---\n{yaml_str}\n---\n"
                except Exception as e:
                    logger.error(f"❌ Failed to build frontmatter: {e}")
                    return ""
            
            parse_frontmatter = parse_frontmatter_yaml
            build_frontmatter = build_frontmatter_yaml
        except ImportError:
            # Fallback to simple parser if yaml not available (shouldn't happen in production)
            from utils.frontmatter_utils import parse_frontmatter, build_frontmatter
            logger.warning("⚠️ YAML library not available - using simple frontmatter parser (WILL CORRUPT LIST FIELDS)")
        
        container = await get_service_container()
        document_service = container.document_service
        folder_service = container.folder_service
        
        # Get document info
        doc_info = await document_service.get_document(document_id)
        if not doc_info:
            return {
                "success": False,
                "error": "Document not found",
                "message": f"Document {document_id} not found"
            }
        
        # Get file path
        file_path = await folder_service.get_document_file_path(
            filename=doc_info.filename,
            folder_id=getattr(doc_info, 'folder_id', None),
            user_id=proposal["user_id"],
            collection_type=getattr(doc_info, 'collection_type', 'user')
        )
        
        if not file_path or not file_path.exists():
            return {
                "success": False,
                "error": "File not found",
                "message": f"Document file not found on disk: {file_path}"
            }
        
        # Read current content
        current_content = file_path.read_text(encoding='utf-8')
        frontmatter, body = parse_frontmatter(current_content)
        has_frontmatter = bool(frontmatter)
        
        if edit_type == "operations":
            # Apply selected operations
            operations = proposal["operations"]
            if selected_operation_indices is None:
                # Apply all operations
                selected_ops = operations
            else:
                # Apply only selected operations
                selected_ops = [ops[i] for i in selected_operation_indices if 0 <= i < len(operations)]
            
            # Sort operations by start position (highest first to keep offsets stable)
            selected_ops = sorted(selected_ops, key=lambda op: op.get("start", 0), reverse=True)
            
            new_content = current_content
            for op in selected_ops:
                op_type = op.get("op_type", "replace_range")
                start = op.get("start", 0)
                end = op.get("end", start)
                text = op.get("text", "")
                
                if op_type == "delete_range":
                    new_content = new_content[:start] + new_content[end:]
                elif op_type == "replace_range":
                    new_content = new_content[:start] + text + new_content[end:]
                elif op_type == "insert_after_heading":
                    # For insert_after_heading, we need to find the anchor and insert after it
                    anchor_text = op.get("anchor_text", "")
                    if anchor_text:
                        anchor_pos = new_content.find(anchor_text)
                        if anchor_pos != -1:
                            # Find end of line after anchor
                            line_end = new_content.find("\n", anchor_pos + len(anchor_text))
                            if line_end == -1:
                                line_end = len(new_content)
                            insert_pos = line_end + 1
                            new_content = new_content[:insert_pos] + text + "\n" + new_content[insert_pos:]
                        else:
                            logger.warning(f"⚠️ Anchor text not found for insert_after_heading: {anchor_text}")
                    else:
                        # Fallback to end
                        new_content = new_content + "\n" + text
            
            applied_count = len(selected_ops)
            
        elif edit_type == "content":
            # Apply content edit
            content_edit = proposal["content_edit"]
            edit_mode = content_edit.get("edit_mode", "append")
            content = content_edit.get("content", "")
            
            if edit_mode == "append":
                if has_frontmatter:
                    new_content = build_frontmatter(frontmatter) + body + "\n\n" + content
                else:
                    new_content = current_content + "\n\n" + content
            elif edit_mode == "replace":
                if has_frontmatter:
                    new_content = build_frontmatter(frontmatter) + "\n\n" + content
                else:
                    new_content = content
            elif edit_mode == "insert_at":
                insert_pos = content_edit.get("insert_position")
                if insert_pos is None:
                    # Append to end
                    new_content = current_content + "\n\n" + content
                else:
                    new_content = current_content[:insert_pos] + content + current_content[insert_pos:]
            
            applied_count = 1
        
        # Write updated content to file
        file_path.write_text(new_content, encoding='utf-8')
        logger.info(f"✅ Applied edit proposal: {file_path} ({len(new_content)} chars)")
        
        # Update file size in database
        await document_service.document_repository.update_file_size(
            document_id,
            len(new_content.encode('utf-8'))
        )
        
        # Re-embed the document
        await document_service.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
        await document_service.embedding_manager.delete_document_chunks(document_id)
        
        if document_service.kg_service:
            try:
                await document_service.kg_service.delete_document_entities(document_id)
                logger.info(f"🗑️ Deleted old knowledge graph entities for {document_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete old KG entities for {document_id}: {e}")
        
        # Re-process content into chunks
        metadata = {
            "title": getattr(doc_info, 'title', ''),
            "tags": getattr(doc_info, 'tags', []),
            "category": getattr(doc_info, 'category', '')
        }
        
        chunks = await document_service.document_processor.process_text_content(
            new_content, document_id, metadata
        )
        
        # Store chunks in vector database
        if chunks:
            await document_service.embedding_manager.embed_and_store_chunks(chunks, document_id)
            logger.info(f"✅ Re-embedded {len(chunks)} chunks for document {document_id}")
        
        # Update status to completed
        await document_service.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
        
        # Emit WebSocket notification for UI refresh (so open editor tabs update automatically)
        await document_service._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
        
        # Mark proposal as applied
        proposal["applied"] = True
        proposal["applied_at"] = datetime.now().isoformat()
        
        return {
            "success": True,
            "document_id": document_id,
            "applied_count": applied_count,
            "message": f"Document edit proposal applied successfully ({applied_count} edit(s))"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to apply document edit proposal: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to apply document edit proposal: {str(e)}"
        }


def get_document_edit_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a document edit proposal by ID (for frontend/API access)
    
    Args:
        proposal_id: Proposal ID
    
    Returns:
        Proposal dict or None if not found
    """
    return _document_edit_proposals.get(proposal_id)

