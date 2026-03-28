"""
Document Editing Tools for Agents
Allows agents to update document titles and frontmatter in user's documents
"""

import hashlib
import json
import logging
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _parse_jsonb(value: Any) -> Any:
    """
    Parse a JSONB column value from asyncpg.
    asyncpg returns JSONB as a JSON string by default; convert to Python object.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSONB value: %s", e)
            return None
    return value


def _normalize_operations(operations: Any) -> List[Dict[str, Any]]:
    """
    Ensure operations is a list of dicts. JSONB or upstream may return a list
    containing strings (e.g. JSON-stringified ops); parse and filter to dicts only.
    Also handles asyncpg returning the whole column as a JSON string.
    """
    if not operations:
        return []
    # asyncpg returns JSONB columns as strings; parse to list first
    if isinstance(operations, str):
        try:
            operations = json.loads(operations)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse operations JSONB string: %s", e)
            return []
    if not isinstance(operations, list):
        return []
    result = []
    for i, op in enumerate(operations):
        if isinstance(op, dict):
            result.append(op)
        elif isinstance(op, str):
            try:
                parsed = json.loads(op)
                if isinstance(parsed, dict):
                    result.append(parsed)
                else:
                    logger.warning("Proposal operation[%s] parsed to non-dict: %s", i, type(parsed))
            except json.JSONDecodeError as e:
                logger.warning("Proposal operation[%s] invalid JSON: %s", i, e)
        else:
            logger.warning("Proposal operation[%s] unexpected type: %s", i, type(op))
    return result


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
                    
                    # Snapshot current content before overwrite (version history)
                    try:
                        from services.document_version_service import snapshot_before_write
                        await snapshot_before_write(document_id, user_id, "metadata_update", None, None)
                    except Exception as verr:
                        logger.warning("Version snapshot before write failed (non-fatal): %s", verr)
                    
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
        
        # Snapshot current content before overwrite (version history)
        try:
            from services.document_version_service import snapshot_before_write
            await snapshot_before_write(document_id, user_id, "agent_edit", None, None)
        except Exception as verr:
            logger.warning("Version snapshot before write failed (non-fatal): %s", verr)
        
        # Write updated content to file
        file_path.write_text(new_content, encoding='utf-8')
        logger.info(f"✅ Updated file content: {file_path} ({len(new_content)} chars)")
        
        # Update file size in database
        await document_service.document_repository.update_file_size(
            document_id, 
            len(new_content.encode('utf-8'))
        )
        
        # Check if document is exempt from vectorization BEFORE processing
        is_exempt = await document_service.document_repository.is_document_exempt(document_id, user_id)
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
        
        # Get current document content for content_hash and staleness detection
        folder_service = container.folder_service
        file_path = await folder_service.get_document_file_path(
            filename=doc_info.filename,
            folder_id=getattr(doc_info, 'folder_id', None),
            user_id=doc_user_id,
            collection_type=doc_collection_type
        )
        current_content = ""
        if file_path and file_path.exists():
            current_content = file_path.read_text(encoding='utf-8')
        content_hash = hashlib.sha256(current_content.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        proposal_id = str(uuid.uuid4())
        operations_json = json.dumps(operations) if operations else '[]'
        content_edit_json = json.dumps(content_edit) if content_edit else None
        from services.database_manager.database_helpers import execute, fetch_value
        rls = {"user_id": user_id, "user_role": "user"}
        await execute(
            """
            INSERT INTO document_edit_proposals
            (proposal_id, document_id, user_id, edit_type, operations, content_edit, agent_name, summary, requires_preview, content_hash, expires_at)
            VALUES ($1::uuid, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9, $10, $11)
            """,
            proposal_id, document_id, user_id, edit_type, operations_json, content_edit_json,
            agent_name, summary or "", requires_preview, content_hash, expires_at,
            rls_context=rls
        )
        logger.info(f"Document edit proposal created: {proposal_id} for document {document_id}")
        try:
            if container.websocket_manager:
                await container.websocket_manager.send_document_status_update(
                    document_id=document_id,
                    status="edit_proposal",
                    user_id=user_id,
                    filename=None,
                    proposal_data={
                        "proposal_id": proposal_id,
                        "edit_type": edit_type,
                        "operations": operations or [],
                        "content_edit": content_edit,
                        "agent_name": agent_name,
                        "summary": summary,
                        "has_pending_proposals": True,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to send edit proposal notification: {e}")
        return {
            "success": True,
            "proposal_id": proposal_id,
            "document_id": document_id,
            "message": "Document edit proposal created successfully"
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
    
    **SECURITY**: Only allowed for specific trusted agent names (see ALLOWED_AGENTS).
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
    ALLOWED_AGENTS = ["project_content_manager"]
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
                        new_content = new_content[:insert_pos] + text + new_content[insert_pos:]
                    else:
                        logger.warning(f"⚠️ Anchor text not found for insert_after_heading: {anchor_text}")
                else:
                    # Fallback to end
                    new_content = new_content + text
        
        # Snapshot current content before overwrite (version history)
        try:
            from services.document_version_service import snapshot_before_write
            await snapshot_before_write(document_id, user_id, "direct_ops", None, operations)
        except Exception as verr:
            logger.warning("Version snapshot before write failed (non-fatal): %s", verr)
        
        # Write updated content to file
        file_path.write_text(new_content, encoding='utf-8')
        logger.info(f"✅ Applied operations directly: {file_path} ({len(new_content)} chars, {len(sorted_ops)} operations)")
        
        # Update file size in database
        await document_service.document_repository.update_file_size(
            document_id,
            len(new_content.encode('utf-8'))
        )
        
        # Check if document is exempt from vectorization BEFORE processing
        is_exempt = await document_service.document_repository.is_document_exempt(document_id, user_id)
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
        logger.info(f"Applying document edit proposal: {proposal_id}")
        from services.database_manager.database_helpers import fetch_one, execute
        rls = {"user_id": user_id, "user_role": "user"}
        row = await fetch_one(
            "SELECT document_id, user_id, edit_type, operations, content_edit, content_hash FROM document_edit_proposals WHERE proposal_id = $1::uuid",
            proposal_id,
            rls_context=rls
        )
        if not row:
            return {
                "success": False,
                "error": "Proposal not found",
                "message": f"Proposal {proposal_id} not found"
            }
        proposal = {
            "document_id": row["document_id"],
            "user_id": row["user_id"],
            "edit_type": row["edit_type"],
            "operations": _normalize_operations(row["operations"] if row["operations"] is not None else []),
            "content_edit": _parse_jsonb(row["content_edit"]),
            "content_hash": row.get("content_hash"),
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
        
        is_partial_proposal_update = False
        remaining_raw: List[Dict[str, Any]] = []

        if edit_type == "operations":
            from utils.editor_operations_resolver import resolve_operations
            resolved_ops = resolve_operations(current_content, proposal["operations"])
            raw_ops = _normalize_operations(proposal["operations"])
            n = min(len(raw_ops), len(resolved_ops))
            if len(raw_ops) != len(resolved_ops):
                logger.warning(
                    "Raw/resolved operations length mismatch (%s vs %s); applying using min length %s",
                    len(raw_ops),
                    len(resolved_ops),
                    n,
                )

            indexed_candidates = []
            if selected_operation_indices is None:
                for i in range(n):
                    op = resolved_ops[i]
                    if op.get("confidence", 0) > 0 and op.get("start", -1) >= 0:
                        indexed_candidates.append((i, op))
            else:
                seen = set()
                for i in selected_operation_indices:
                    if not isinstance(i, int) or i in seen:
                        continue
                    seen.add(i)
                    if (
                        0 <= i < n
                        and resolved_ops[i].get("confidence", 0) > 0
                        and resolved_ops[i].get("start", -1) >= 0
                    ):
                        indexed_candidates.append((i, resolved_ops[i]))
            indexed_candidates.sort(key=lambda x: x[1].get("start", 0), reverse=True)

            new_content = current_content
            applied_raw_indices = set()
            skipped_ops: List[Dict[str, Any]] = []
            for raw_idx, op in indexed_candidates:
                op_type = op.get("op_type", "replace_range")
                start = op.get("start", 0)
                end = op.get("end", start)
                text = op.get("text", "")
                try:
                    if op_type == "delete_range":
                        if 0 <= start < end <= len(new_content):
                            new_content = new_content[:start] + new_content[end:]
                            applied_raw_indices.add(raw_idx)
                        else:
                            skipped_ops.append(op)
                    elif op_type == "replace_range":
                        if 0 <= start < end <= len(new_content):
                            new_content = new_content[:start] + text + new_content[end:]
                            applied_raw_indices.add(raw_idx)
                        else:
                            skipped_ops.append(op)
                    elif op_type in ("insert_after_heading", "insert_after"):
                        if 0 <= start <= len(new_content):
                            new_content = new_content[:start] + text + new_content[start:]
                            applied_raw_indices.add(raw_idx)
                        else:
                            skipped_ops.append(op)
                    else:
                        skipped_ops.append(op)
                except Exception as e:
                    logger.error("Error applying operation: %s", e)
                    skipped_ops.append(op)
            applied_count = len(applied_raw_indices)
            if skipped_ops:
                logger.warning("Skipped %s operation(s) due to validation failures", len(skipped_ops))

            remaining_raw = [raw_ops[i] for i in range(len(raw_ops)) if i not in applied_raw_indices]
            is_partial_proposal_update = bool(
                selected_operation_indices is not None and len(remaining_raw) > 0
            )

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
        
        # Snapshot current content before overwrite (version history)
        try:
            from services.document_version_service import snapshot_before_write
            await snapshot_before_write(document_id, proposal["user_id"], "proposal_apply", proposal.get("summary"), proposal.get("operations"))
        except Exception as verr:
            logger.warning("Version snapshot before write failed (non-fatal): %s", verr)
        
        # Write updated content to file
        file_path.write_text(new_content, encoding='utf-8')
        logger.info(f"✅ Applied edit proposal: {file_path} ({len(new_content)} chars)")
        
        # Update file size in database
        await document_service.document_repository.update_file_size(
            document_id,
            len(new_content.encode('utf-8'))
        )

        has_pending_proposals = bool(is_partial_proposal_update)
        if is_partial_proposal_update:
            # asyncpg jsonb bind expects a JSON string, not a raw Python list
            await execute(
                "UPDATE document_edit_proposals SET operations = $1::jsonb WHERE proposal_id = $2::uuid",
                json.dumps(remaining_raw),
                proposal_id,
                rls_context=rls,
            )
        else:
            await execute(
                "DELETE FROM document_edit_proposals WHERE proposal_id = $1::uuid",
                proposal_id,
                rls_context=rls,
            )

        try:
            if container.websocket_manager:
                await container.websocket_manager.send_document_status_update(
                    document_id=document_id,
                    status="completed",
                    user_id=user_id,
                    filename=None,
                    proposal_data={"has_pending_proposals": has_pending_proposals},
                )
        except Exception as e:
            logger.warning("Failed to send proposal applied notification: %s", e)

        # Check if document is exempt from vectorization BEFORE processing
        is_exempt = await document_service.document_repository.is_document_exempt(document_id, user_id)
        if is_exempt:
            logger.info(f"🚫 Document {document_id} is exempt from vectorization - skipping embedding and KG extraction")
            await document_service.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
            await document_service._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            return {
                "success": True,
                "document_id": document_id,
                "applied_count": applied_count,
                "message": f"Document edit proposal applied successfully ({applied_count} edit(s)) - exempt from vectorization"
            }
        
        # Run vectorization in background so API returns quickly; frontend can refresh content immediately
        async def _reembed_after_apply() -> None:
            try:
                await document_service.document_repository.update_status(document_id, ProcessingStatus.EMBEDDING)
                await document_service.embedding_manager.delete_document_chunks(document_id)
                if document_service.kg_service:
                    try:
                        await document_service.kg_service.delete_document_entities(document_id)
                        logger.info("🗑️ Deleted old knowledge graph entities for %s", document_id)
                    except Exception as e:
                        logger.warning("⚠️ Failed to delete old KG entities for %s: %s", document_id, e)
                metadata = {
                    "title": getattr(doc_info, "title", ""),
                    "tags": getattr(doc_info, "tags", []),
                    "category": getattr(doc_info, "category", ""),
                }
                chunks = await document_service.document_processor.process_text_content(
                    new_content, document_id, metadata
                )
                if chunks:
                    await document_service.embedding_manager.embed_and_store_chunks(chunks, document_id)
                    logger.info("✅ Re-embedded %s chunks for document %s", len(chunks), document_id)
                await document_service.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
                await document_service._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
            except Exception as e:
                logger.error("❌ Background re-embed after apply failed: %s", e)
                import traceback
                logger.error("%s", traceback.format_exc())
                try:
                    await document_service.document_repository.update_status(document_id, ProcessingStatus.COMPLETED)
                    await document_service._emit_document_status_update(document_id, ProcessingStatus.COMPLETED.value, user_id)
                except Exception as e2:
                    logger.warning("Failed to set status after re-embed error: %s", e2)
        
        import asyncio
        asyncio.create_task(_reembed_after_apply())
        
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


def _row_to_proposal_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a DB row to proposal dict for API/WebSocket. Parses JSONB columns (asyncpg returns strings)."""
    return {
        "proposal_id": str(row["proposal_id"]),
        "document_id": row["document_id"],
        "user_id": row["user_id"],
        "edit_type": row["edit_type"],
        "operations": _normalize_operations(row["operations"] if row["operations"] is not None else []),
        "content_edit": _parse_jsonb(row["content_edit"]),
        "agent_name": row["agent_name"],
        "summary": row["summary"],
        "requires_preview": row["requires_preview"],
        "content_hash": row.get("content_hash"),
        "expires_at": row["expires_at"].isoformat() if row.get("expires_at") else None,
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


async def _get_document_content_for_resolution(document_id: str, user_id: str) -> Optional[str]:
    """Load current document content for JIT resolution. Returns None if document not found."""
    from services.service_container import get_service_container
    container = await get_service_container()
    document_service = container.document_service
    folder_service = container.folder_service
    doc_info = await document_service.get_document(document_id)
    if not doc_info:
        return None
    file_path = await folder_service.get_document_file_path(
        filename=doc_info.filename,
        folder_id=getattr(doc_info, 'folder_id', None),
        user_id=user_id,
        collection_type=getattr(doc_info, 'collection_type', 'user')
    )
    if not file_path or not file_path.exists():
        return None
    return file_path.read_text(encoding='utf-8')


def _dedup_cross_proposal_ops(proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove ops from older proposals whose resolved range overlaps a newer proposal's op.

    Proposals should be sorted by created_at ascending. When two ops from different
    proposals resolve to overlapping character ranges, the newer proposal's op wins
    and the older one is dropped from the API response. No DB mutation.
    """
    if len(proposals) < 2:
        return proposals

    # All resolved ranges with proposal timestamp (ISO string from API dict).
    claimed: List[tuple] = []
    for p in reversed(proposals):
        p_ts = p.get("created_at") or ""
        for op in p.get("operations") or []:
            s, e = op.get("start", -1), op.get("end", -1)
            if s >= 0 and e > s:
                claimed.append((s, e, p_ts))

    for p in proposals:
        p_ts = p.get("created_at") or ""
        kept: List[Dict[str, Any]] = []
        for op in p.get("operations") or []:
            s, e = op.get("start", -1), op.get("end", -1)
            if s < 0:
                kept.append(op)
                continue
            superseded = False
            for cs, ce, claim_ts in claimed:
                if claim_ts != p_ts and claim_ts > p_ts:
                    if not (e <= cs or s >= ce):
                        superseded = True
                        break
            if not superseded:
                kept.append(op)
        p["operations"] = kept

    return [
        p
        for p in proposals
        if p.get("edit_type") != "operations" or len(p.get("operations") or []) > 0
    ]


async def list_pending_proposals_for_document(document_id: str, user_id: str = "system") -> List[Dict[str, Any]]:
    """List pending proposals for a document. Operations are resolved JIT against current content."""
    from services.database_manager.database_helpers import fetch_all
    from utils.editor_operations_resolver import resolve_operations
    rows = await fetch_all(
        "SELECT proposal_id, document_id, user_id, edit_type, operations, content_edit, agent_name, summary, requires_preview, content_hash, expires_at, created_at FROM document_edit_proposals WHERE document_id = $1 AND (expires_at IS NULL OR expires_at > NOW()) ORDER BY created_at ASC",
        document_id,
        rls_context={"user_id": user_id, "user_role": "user"}
    )
    proposals = [_row_to_proposal_dict(row) for row in (rows or [])]
    if not proposals:
        return proposals
    current_content = await _get_document_content_for_resolution(document_id, user_id)
    if current_content is None:
        return proposals
    for p in proposals:
        if p.get("edit_type") == "operations" and p.get("operations"):
            resolved = resolve_operations(current_content, p["operations"])
            filtered = []
            for i, op in enumerate(resolved):
                if op.get("confidence", 0) > 0 and op.get("start", -1) >= 0:
                    op = dict(op)
                    op["proposal_operation_index"] = i
                    filtered.append(op)
            p["operations"] = filtered

    proposals = _dedup_cross_proposal_ops(proposals)
    return proposals


async def get_document_edit_proposal(proposal_id: str, user_id: str = "system") -> Optional[Dict[str, Any]]:
    """
    Get a document edit proposal by ID (for frontend/API access).
    Requires user_id for RLS.
    """
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        "SELECT proposal_id, document_id, user_id, edit_type, operations, content_edit, agent_name, summary, requires_preview, content_hash, expires_at, created_at FROM document_edit_proposals WHERE proposal_id = $1::uuid",
        proposal_id,
        rls_context={"user_id": user_id, "user_role": "user"}
    )
    if not row:
        return None
    return _row_to_proposal_dict(row)

