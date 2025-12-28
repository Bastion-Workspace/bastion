from fastapi import APIRouter, Depends, HTTPException
from typing import List
from datetime import datetime

from models.conversation_models import (
    CreateConversationRequest,
    CreateMessageRequest,
    ConversationResponse,
    ConversationListResponse,
    MessageListResponse,
    MessageResponse,
    UpdateConversationRequest,
)
from utils.auth_middleware import get_current_user, validate_conversation_access
from models.api_models import AuthenticatedUserResponse
import logging

logger = logging.getLogger(__name__)

from services.service_container import get_service_container

# Helper function to get services from container
async def _get_conversation_service():
    """Get conversation service from service container"""
    container = await get_service_container()
    return container.conversation_service


router = APIRouter()


@router.get("/api/conversations/{conversation_id}/checkpoints")
async def list_conversation_checkpoints(conversation_id: str, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    try:
        logger.info(f"🧭 Listing checkpoints for conversation: {conversation_id}")
        from services.langgraph_postgres_checkpointer import get_postgres_checkpointer
        checkpointer = await get_postgres_checkpointer()
        if not checkpointer.is_initialized:
            raise HTTPException(status_code=500, detail="LangGraph checkpointer not initialized")
        import asyncpg
        from config import settings
        from services.orchestrator_utils import normalize_thread_id
        normalized_thread_id = normalize_thread_id(current_user.user_id, conversation_id)
        connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        conn = await asyncpg.connect(connection_string)
        try:
            rows = await conn.fetch(
                """
                SELECT checkpoint_id, parent_checkpoint_id, created_at, type
                FROM checkpoints
                WHERE thread_id = $1
                ORDER BY created_at DESC, checkpoint_id DESC
                """,
                normalized_thread_id
            )
        finally:
            await conn.close()
        checkpoints = [
            {
                "checkpoint_id": r["checkpoint_id"],
                "parent_checkpoint_id": r["parent_checkpoint_id"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "type": r["type"]
            }
            for r in rows
        ]
        return {"success": True, "conversation_id": conversation_id, "checkpoints": checkpoints}
    except Exception as e:
        logger.error(f"❌ Failed to list checkpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/conversations", response_model=ConversationListResponse)
async def list_conversations(skip: int = 0, limit: int = 50, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    try:
        logger.info(f"💬 Listing conversations from database (skip={skip}, limit={limit})")
        conversations = []
        try:
            import asyncpg
            from config import settings
            connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            conn = await asyncpg.connect(connection_string)
            try:
                # Set RLS context for Row-Level Security policies
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", current_user.user_id)
                await conn.execute("SELECT set_config('app.current_user_role', $1, false)", current_user.role)
                
                rows = await conn.fetch(
                    """
                        SELECT 
                            conv.conversation_id,
                            conv.title,
                            conv.description,
                            conv.is_pinned,
                            conv.is_archived,
                            conv.tags,
                            conv.created_at,
                            conv.updated_at,
                            conv.last_message_at,
                            conv.manual_order,
                            conv.order_locked,
                            -- Use actual message count from conversation_messages table
                            COALESCE(
                                (SELECT COUNT(*) FROM conversation_messages cm 
                                 WHERE cm.conversation_id = conv.conversation_id 
                                 AND (cm.is_deleted IS NULL OR cm.is_deleted = FALSE)),
                                conv.message_count,
                                0
                            ) as message_count
                        FROM conversations conv
                        WHERE conv.user_id = $1
                        ORDER BY conv.updated_at DESC NULLS LAST, conv.created_at DESC
                        LIMIT $2 OFFSET $3
                    """, current_user.user_id, limit, skip)
                from models.conversation_models import ConversationSummary
                for row in rows:
                    if row['conversation_id']:
                        # Tags are stored as TEXT[] in PostgreSQL, asyncpg returns as list
                        tags_list = list(row['tags']) if row['tags'] else []
                        
                        conversation = ConversationSummary(
                            conversation_id=row['conversation_id'],
                            user_id=current_user.user_id,
                            title=row['title'] or "Untitled Conversation",
                            description=row['description'],
                            is_pinned=row['is_pinned'] or False,
                            is_archived=row['is_archived'] or False,
                            tags=tags_list,
                            metadata_json={},
                            message_count=row['message_count'] or 0,
                            last_message_at=row['last_message_at'].isoformat() if row['last_message_at'] else row['updated_at'].isoformat() if row['updated_at'] else None,
                            manual_order=row['manual_order'],
                            order_locked=row['order_locked'] or False,
                            created_at=row['created_at'].isoformat() if row['created_at'] else datetime.now().isoformat(),
                            updated_at=row['updated_at'].isoformat() if row['updated_at'] else datetime.now().isoformat()
                        )
                        conversations.append(conversation)
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"❌ Failed to query conversations from database: {e}")
            conversations = []
        
        # Get total count for pagination
        total_count = len(conversations)
        try:
            import asyncpg
            from config import settings
            connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            conn = await asyncpg.connect(connection_string)
            try:
                total_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM conversations WHERE user_id = $1",
                    current_user.user_id
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"⚠️ Failed to get total conversation count: {e}")
        
        from models.conversation_models import ConversationListResponse as ConvList
        result = ConvList(
            conversations=conversations,
            total_count=total_count,
            has_more=(skip + len(conversations)) < total_count,
            folders=[]
        )
        logger.info(f"✅ Retrieved {len(result.conversations)} conversations from database (total: {total_count})")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to list conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    conversation_service = await _get_conversation_service()
    try:
        # Check access permission
        has_access = await validate_conversation_access(
            user_id=current_user.user_id,
            conversation_id=conversation_id,
            required_permission="read"
        )
        if not has_access:
            raise HTTPException(status_code=403, detail="You do not have access to this conversation")
        
        logger.info(f"💬 Getting conversation: {conversation_id}")
        from services.langgraph_postgres_checkpointer import get_postgres_checkpointer
        checkpointer = await get_postgres_checkpointer()
        if not checkpointer.is_initialized:
            logger.error("❌ LangGraph checkpointer not initialized")
            raise HTTPException(status_code=500, detail="LangGraph checkpointer not initialized")
        conversation_dict = None
        try:
            import asyncpg
            from config import settings
            connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            conn = await asyncpg.connect(connection_string)
            try:
                from services.orchestrator_utils import normalize_thread_id
                normalized_thread_id = normalize_thread_id(current_user.user_id, conversation_id)
                row = await conn.fetchrow(
                    """
                    SELECT DISTINCT ON (c.thread_id) 
                        c.thread_id,
                        c.checkpoint,
                        c.checkpoint_id
                    FROM checkpoints c
                    WHERE c.thread_id = $1 
                    AND c.checkpoint -> 'channel_values' IS NOT NULL
                    AND c.checkpoint -> 'channel_values' ->> 'user_id' = $2
                    ORDER BY c.thread_id, c.checkpoint_id DESC
                    LIMIT 1
                """, normalized_thread_id, current_user.user_id)
                if row:
                    checkpoint_data = row['checkpoint']
                    if isinstance(checkpoint_data, str):
                        import json
                        try:
                            checkpoint_data = json.loads(checkpoint_data)
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ Failed to parse checkpoint JSON: {e}")
                            checkpoint_data = {}
                    elif checkpoint_data is None:
                        checkpoint_data = {}
                    channel_values = checkpoint_data.get('channel_values', {})
                    conversation_dict = {
                        "conversation_id": conversation_id,
                        "user_id": current_user.user_id,
                        "title": channel_values.get('conversation_title', 'New Conversation'),
                        "description": channel_values.get('conversation_description'),
                        "is_pinned": channel_values.get('is_pinned', False),
                        "is_archived": channel_values.get('is_archived', False),
                        "tags": channel_values.get('conversation_tags', []),
                        "metadata_json": {},
                        "message_count": len(channel_values.get('messages', [])),
                        "last_message_at": channel_values.get('conversation_updated_at'),
                        "manual_order": None,
                        "order_locked": False,
                        "created_at": channel_values.get('conversation_created_at', datetime.now().isoformat()),
                        "updated_at": channel_values.get('conversation_updated_at', datetime.now().isoformat()),
                        "messages": []
                    }
                    logger.info(f"✅ Found conversation in LangGraph checkpoint: {conversation_id}")
                    
                    # CRITICAL FIX: Always check database for title, even when checkpoint exists
                    # Database is source of truth for conversation metadata
                    try:
                        conversation_service = await _get_conversation_service()
                        conversation_service.set_current_user(current_user.user_id)
                        db_conversation = await conversation_service.lifecycle_manager.get_conversation_lifecycle(conversation_id, current_user.user_id)
                        if db_conversation:
                            db_title = db_conversation.get("title")
                            # Prefer database title if it exists and is not default
                            if db_title and db_title != "New Conversation" and db_title != "Untitled Conversation":
                                conversation_dict["title"] = db_title
                                logger.info(f"✅ Using database title for conversation {conversation_id}: {db_title}")
                            # Also update other metadata from database if checkpoint values are defaults
                            if db_conversation.get("is_pinned") is not None:
                                conversation_dict["is_pinned"] = db_conversation.get("is_pinned", False)
                            if db_conversation.get("is_archived") is not None:
                                conversation_dict["is_archived"] = db_conversation.get("is_archived", False)
                            if db_conversation.get("tags"):
                                conversation_dict["tags"] = db_conversation.get("tags", [])
                            if db_conversation.get("description"):
                                conversation_dict["description"] = db_conversation.get("description")
                            if db_conversation.get("manual_order") is not None:
                                conversation_dict["manual_order"] = db_conversation.get("manual_order")
                            if db_conversation.get("order_locked") is not None:
                                conversation_dict["order_locked"] = db_conversation.get("order_locked", False)
                    except Exception as db_title_error:
                        logger.warning(f"⚠️ Failed to fetch database title for conversation {conversation_id}: {db_title_error}")
                else:
                    logger.info(f"💬 Conversation {conversation_id} not found in LangGraph checkpoints")
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"⚠️ Failed to query LangGraph checkpoints for conversation {conversation_id}: {e}")
        
        # ROOSEVELT'S DUAL SOURCE FIX: Fall back to conversation database if not found in checkpoints
        if not conversation_dict:
            logger.info(f"📚 Conversation not found in checkpoint, falling back to conversation database for {conversation_id}")
            try:
                conversation_service = await _get_conversation_service()
                conversation_service.set_current_user(current_user.user_id)
                
                # Get conversation from database
                db_conversation = await conversation_service.lifecycle_manager.get_conversation_lifecycle(conversation_id, current_user.user_id)
                if db_conversation:
                    user_context = db_conversation.get("user_context", {})
                    if user_context.get("user_id") == current_user.user_id:
                        # Get message count from total_count in result
                        messages_result = await conversation_service.get_conversation_messages(
                            conversation_id=conversation_id,
                            user_id=current_user.user_id,
                            skip=0,
                            limit=1  # Just to get count
                        )
                        # ConversationService returns dict with "total_count" key
                        message_count = messages_result.get("total_count", 0) if messages_result and "total_count" in messages_result else 0
                        
                        conversation_dict = {
                            "conversation_id": conversation_id,
                            "user_id": current_user.user_id,
                            "title": db_conversation.get("title", "New Conversation"),
                            "description": db_conversation.get("description"),
                            "is_pinned": db_conversation.get("is_pinned", False),
                            "is_archived": db_conversation.get("is_archived", False),
                            "tags": db_conversation.get("tags", []),
                            "metadata_json": db_conversation.get("metadata_json", {}),
                            "message_count": message_count,
                            "last_message_at": db_conversation.get("last_message_at"),
                            "manual_order": db_conversation.get("manual_order"),
                            "order_locked": db_conversation.get("order_locked", False),
                            "created_at": db_conversation.get("created_at", datetime.now().isoformat()),
                            "updated_at": db_conversation.get("updated_at", datetime.now().isoformat()),
                            "messages": []
                        }
                        logger.info(f"✅ Found conversation in database: {conversation_id}")
                    else:
                        logger.warning(f"⚠️ User {current_user.user_id} does not own conversation {conversation_id}")
                else:
                    logger.info(f"💬 Conversation {conversation_id} not found in database either")
            except Exception as db_error:
                logger.warning(f"⚠️ Fallback to conversation database failed: {db_error}")
        
        if not conversation_dict:
            raise HTTPException(status_code=404, detail="Conversation not found")
        from models.conversation_models import ConversationDetail
        conversation_detail = ConversationDetail(
            conversation_id=conversation_dict["conversation_id"],
            user_id=conversation_dict["user_id"],
            title=conversation_dict["title"],
            description=conversation_dict.get("description"),
            is_pinned=conversation_dict.get("is_pinned", False),
            is_archived=conversation_dict.get("is_archived", False),
            tags=conversation_dict.get("tags", []),
            metadata_json=conversation_dict.get("metadata_json", {}),
            message_count=conversation_dict.get("message_count", 0),
            last_message_at=conversation_dict.get("last_message_at"),
            manual_order=conversation_dict.get("manual_order"),
            order_locked=conversation_dict.get("order_locked", False),
            created_at=conversation_dict["created_at"],
            updated_at=conversation_dict["updated_at"],
            messages=conversation_dict.get("messages", [])
        )
        logger.info(f"✅ Retrieved conversation from LangGraph checkpoint: {conversation_id}")
        return ConversationResponse(conversation=conversation_detail)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, request: UpdateConversationRequest, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    try:
        # Check edit permission
        has_access = await validate_conversation_access(
            user_id=current_user.user_id,
            conversation_id=conversation_id,
            required_permission="edit"
        )
        if not has_access:
            raise HTTPException(status_code=403, detail="You do not have permission to edit this conversation")
        
        logger.info(f"💬 Updating conversation: {conversation_id} for user: {current_user.user_id}")
        import asyncpg
        from config import settings
        from services.orchestrator_utils import normalize_thread_id
        normalized_thread_id = normalize_thread_id(current_user.user_id, conversation_id)
        connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        conn = await asyncpg.connect(connection_string)
        try:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT DISTINCT ON (c.thread_id)
                        c.thread_id,
                        c.checkpoint,
                        c.checkpoint_id
                    FROM checkpoints c
                    WHERE c.thread_id = $1 
                      AND c.checkpoint -> 'channel_values' IS NOT NULL
                      AND c.checkpoint -> 'channel_values' ->> 'user_id' = $2
                    ORDER BY c.thread_id, c.checkpoint_id DESC
                    LIMIT 1
                    """,
                    normalized_thread_id,
                    current_user.user_id,
                )
                thread_id_used = normalized_thread_id
                if not row:
                    row = await conn.fetchrow(
                        """
                        SELECT DISTINCT ON (c.thread_id)
                            c.thread_id,
                            c.checkpoint,
                            c.checkpoint_id
                        FROM checkpoints c
                        WHERE c.thread_id = $1 
                          AND c.checkpoint -> 'channel_values' IS NOT NULL
                          AND c.checkpoint -> 'channel_values' ->> 'user_id' = $2
                        ORDER BY c.thread_id, c.checkpoint_id DESC
                        LIMIT 1
                        """,
                        conversation_id,
                        current_user.user_id,
                    )
                    if row:
                        thread_id_used = conversation_id
                if row:
                    checkpoint_data = row["checkpoint"]
                    if isinstance(checkpoint_data, str):
                        import json
                        try:
                            checkpoint_data = json.loads(checkpoint_data)
                        except Exception:
                            checkpoint_data = {}
                    elif checkpoint_data is None:
                        checkpoint_data = {}
                    channel_values = checkpoint_data.get("channel_values", {})
                    if request.title is not None:
                        channel_values["conversation_title"] = request.title
                    if request.description is not None:
                        channel_values["conversation_description"] = request.description
                    if request.is_pinned is not None:
                        channel_values["is_pinned"] = request.is_pinned
                    if request.is_archived is not None:
                        channel_values["is_archived"] = request.is_archived
                    if request.tags is not None:
                        channel_values["conversation_tags"] = request.tags
                    channel_values["conversation_updated_at"] = datetime.now().isoformat()
                    checkpoint_data["channel_values"] = channel_values
                    await conn.execute(
                        """
                        UPDATE checkpoints
                        SET checkpoint = $1
                        WHERE thread_id = $2
                          AND checkpoint -> 'channel_values' IS NOT NULL
                          AND checkpoint -> 'channel_values' ->> 'user_id' = $3
                        """,
                        checkpoint_data,
                        thread_id_used,
                        current_user.user_id,
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to update LangGraph checkpoint JSON: {e}")
            # Legacy compatibility update (same connection)
            # Set RLS context for Row-Level Security policies
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", current_user.user_id)
            await conn.execute("SELECT set_config('app.current_user_role', $1, false)", current_user.role)
            
            updates = []
            values = []
            if request.title is not None:
                updates.append("title = $%d" % (len(values) + 1))
                values.append(request.title)
            if request.description is not None:
                updates.append("description = $%d" % (len(values) + 1))
                values.append(request.description)
            if request.is_pinned is not None:
                updates.append("is_pinned = $%d" % (len(values) + 1))
                values.append(request.is_pinned)
            if request.is_archived is not None:
                updates.append("is_archived = $%d" % (len(values) + 1))
                values.append(request.is_archived)
            if request.tags is not None:
                updates.append("tags = $%d" % (len(values) + 1))
                values.append(request.tags)
            if request.manual_order is not None:
                updates.append("manual_order = $%d" % (len(values) + 1))
                values.append(request.manual_order)
            if request.order_locked is not None:
                updates.append("order_locked = $%d" % (len(values) + 1))
                values.append(request.order_locked)
            if updates:
                set_clause = ", ".join(updates)
                values.extend([conversation_id, current_user.user_id])
                await conn.execute(
                    f"""
                    UPDATE conversations
                    SET {set_clause}, updated_at = NOW()
                    WHERE conversation_id = ${len(values) - 1} AND user_id = ${len(values)}
                    """,
                    *values,
                )
        finally:
            await conn.close()
        # WebSocket notification best-effort
        try:
            from utils.websocket_manager import get_websocket_manager
            websocket_manager = get_websocket_manager()
            if websocket_manager:
                await websocket_manager.send_to_session(
                    session_id=current_user.user_id,
                    message={
                        "type": "conversation_updated",
                        "data": {"conversation_id": conversation_id},
                    },
                )
        except Exception as e:
            logger.debug(f"WebSocket notify failed (non-fatal): {e}")
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




# Helper function for image cleanup
async def _cleanup_conversation_images(conversation_id: str, user_id: str):
    """Extract and clean up generated images from a conversation before deletion"""
    import re
    from pathlib import Path
    from services.database_manager.database_helpers import fetch_all
    
    try:
        logger.info(f"🖼️ Extracting image URLs from conversation: {conversation_id}")
        
        # Get all messages from the conversation before deletion
        messages = await fetch_all(
            """
            SELECT content, role 
            FROM conversation_messages 
            WHERE conversation_id = $1
            ORDER BY created_at
            """,
            conversation_id
        )
        
        if not messages:
            logger.info(f"📭 No messages found for conversation {conversation_id}")
            return
        
        # Extract image URLs from message content
        image_urls = set()
        image_pattern = re.compile(r'/static/images/(gen_[a-f0-9]+\.(?:png|jpg|jpeg|webp))')
        
        for message in messages:
            content = message.get('content', '') or ''
            if isinstance(content, str):
                matches = image_pattern.findall(content)
                for filename in matches:
                    image_urls.add(f"/static/images/{filename}")
        
        if not image_urls:
            logger.info(f"📭 No generated images found in conversation {conversation_id}")
            return
        
        logger.info(f"🖼️ Found {len(image_urls)} unique image(s) in conversation")
        
        # Check which images have been imported into document library
        # An imported image would have a document with doc_type='image' and file_path matching
        images_path = Path(f"{settings.UPLOAD_DIR}/web_sources/images")
        imported_images = set()
        
        for image_url in image_urls:
            # Extract filename from URL
            filename = image_url.replace('/static/images/', '')
            
            # Check if this image has been imported as a document
            # We check by looking for documents with this filename
            imported_docs = await fetch_all(
                """
                SELECT document_id, filename 
                FROM document_metadata 
                WHERE user_id = $1 
                AND doc_type = 'image'
                AND filename = $2
                """,
                user_id, filename
            )
            
            if imported_docs:
                logger.info(f"✅ Image {filename} has been imported - skipping deletion")
                imported_images.add(image_url)
            else:
                logger.info(f"🗑️ Image {filename} not imported - will be deleted")
        
        # Delete only non-imported images
        deleted_count = 0
        for image_url in image_urls:
            if image_url in imported_images:
                continue  # Skip imported images
            
            filename = image_url.replace('/static/images/', '')
            image_file_path = images_path / filename
            
            # Also check subdirectories (some images may be in document_id subdirectories)
            if not image_file_path.exists():
                found = False
                for subdir in images_path.iterdir():
                    if subdir.is_dir():
                        potential_path = subdir / filename
                        if potential_path.exists():
                            image_file_path = potential_path
                            found = True
                            break
                
                if not found:
                    logger.warning(f"⚠️ Image file not found: {filename}")
                    continue
            
            try:
                if image_file_path.exists():
                    image_file_path.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted image file: {image_file_path}")
            except Exception as e:
                logger.error(f"❌ Failed to delete image file {image_file_path}: {e}")
        
        logger.info(f"✅ Cleaned up {deleted_count} image file(s) for conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup conversation images: {e}")
        # Don't raise - image cleanup failure shouldn't block conversation deletion


@router.post("/api/conversations", response_model=ConversationResponse)
async def create_conversation(request: CreateConversationRequest, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Create a new conversation"""
    conversation_service = await _get_conversation_service()
    try:
        logger.info(f"💬 Creating new conversation: {request.title}")
        
        # Set the current user for this operation
        conversation_service.set_current_user(current_user.user_id)
        
        # Pass the user_id and initial message for title generation
        conversation_summary = await conversation_service.create_conversation(
            user_id=current_user.user_id,
            initial_message=request.initial_message,
            initial_mode="chat",
            metadata={"title": request.title} if request.title else None
        )
        
        # Convert dictionary response to ConversationDetail for the response
        from models.conversation_models import ConversationDetail
        conversation_detail = ConversationDetail(
            conversation_id=conversation_summary["conversation_id"],
            user_id=conversation_summary["user_id"],
            title=conversation_summary.get("title"),
            description=conversation_summary.get("description"),
            is_pinned=conversation_summary.get("is_pinned", False),
            is_archived=conversation_summary.get("is_archived", False),
            tags=conversation_summary.get("tags", []),
            metadata_json=conversation_summary.get("metadata_json", {}),
            message_count=conversation_summary.get("message_count", 0),
            last_message_at=conversation_summary.get("last_message_at"),
            manual_order=conversation_summary.get("manual_order"),
            order_locked=conversation_summary.get("order_locked", False),
            created_at=conversation_summary["created_at"],
            updated_at=conversation_summary["updated_at"],
            messages=[]  # New conversation has no messages yet
        )
        
        logger.info(f"✅ Conversation created: {conversation_summary['conversation_id']}")
        return ConversationResponse(conversation=conversation_detail)
        
    except Exception as e:
        logger.error(f"❌ Failed to create conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/conversations/{conversation_id}/messages", response_model=MessageListResponse)
async def get_conversation_messages(conversation_id: str, skip: int = 0, limit: int = 100, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """
    Get messages for a conversation - MIGRATION-COMPATIBLE APPROACH
    
    Priority order (consistent with orchestrator migration):
    1. Conversation database (primary source - populated by backend proxy)
    2. LangGraph checkpoints (fallback for legacy conversations)
    
    This ensures new orchestrator conversations work correctly while maintaining
    backward compatibility with old conversations that only have checkpoints.
    """
    conversation_service = await _get_conversation_service()
    try:
        # Check read permission
        from utils.auth_middleware import validate_conversation_access
        has_access = await validate_conversation_access(
            user_id=current_user.user_id,
            conversation_id=conversation_id,
            required_permission="read"
        )
        if not has_access:
            raise HTTPException(status_code=403, detail="You do not have access to this conversation")
        
        logger.info(f"💬 Getting messages for conversation: {conversation_id}")
        
        messages = []
        messages_from_checkpoint = False  # Track if we got messages from checkpoint
        messages_from_database = False  # Track if we got messages from database
        
        # PRIORITY 1: Read from conversation database (primary source for orchestrator conversations)
        try:
            conversation_service = await _get_conversation_service()
            conversation_service.set_current_user(current_user.user_id)
            
            db_messages_result = await conversation_service.get_conversation_messages(
                conversation_id=conversation_id,
                user_id=current_user.user_id,
                skip=skip,
                limit=limit
            )
            
            # ConversationService returns dict with "messages" key
            logger.info(f"🔍 get_conversation_messages result keys: {list(db_messages_result.keys()) if db_messages_result else 'None'}")
            if db_messages_result and "messages" in db_messages_result:
                db_messages = db_messages_result.get("messages", [])
                logger.info(f"🔍 get_conversation_messages returned {len(db_messages)} messages")
                if db_messages:
                    logger.info(f"✅ Retrieved {len(db_messages)} messages from conversation database (primary source)")
                    
                    # Convert database messages to API format
                    for msg in db_messages:
                        metadata_json = msg.get("metadata_json", {})
                        
                        # Extract editor_operations and manuscript_edit from metadata for top-level access
                        editor_operations = metadata_json.get("editor_operations", []) if isinstance(metadata_json, dict) else []
                        manuscript_edit = metadata_json.get("manuscript_edit") if isinstance(metadata_json, dict) else None
                        
                        message_obj = {
                            "message_id": msg.get("message_id"),
                            "conversation_id": conversation_id,
                            "message_type": msg.get("message_type", "user"),
                            "role": msg.get("message_type", "user"),
                            "content": msg.get("content", ""),
                            "sequence_number": msg.get("sequence_number", 0),
                            "created_at": msg.get("created_at").isoformat() if hasattr(msg.get("created_at"), "isoformat") else str(msg.get("created_at")),
                            "updated_at": msg.get("updated_at").isoformat() if hasattr(msg.get("updated_at"), "isoformat") else str(msg.get("updated_at")),
                            "metadata_json": metadata_json,
                            "citations": metadata_json.get("citations", []) if isinstance(metadata_json, dict) else [],
                            "edit_history": []
                        }
                        
                        # Add editor_operations and manuscript_edit as top-level fields if present
                        if editor_operations:
                            message_obj["editor_operations"] = editor_operations
                        if manuscript_edit:
                            message_obj["manuscript_edit"] = manuscript_edit
                        
                        messages.append(message_obj)
                    messages_from_database = True
                else:
                    logger.info(f"📚 No messages in conversation database, will try checkpoints as fallback")
            else:
                logger.debug(f"📚 Conversation database query returned no messages")
        except Exception as db_error:
            logger.warning(f"⚠️ Failed to load messages from conversation database: {db_error}")
        
        # PRIORITY 2: Fallback to LangGraph checkpoints (for legacy conversations)
        if not messages_from_database:
            logger.info(f"📚 Falling back to LangGraph checkpoints for conversation {conversation_id}")
            try:
                from services.langgraph_postgres_checkpointer import get_postgres_checkpointer
                checkpointer = await get_postgres_checkpointer()
                
                if not checkpointer.is_initialized:
                    logger.warning("⚠️ LangGraph checkpointer not initialized, skipping checkpoint fallback")
                else:
                    # ROOSEVELT'S THREAD_ID FIX: Use normalized thread_id for proper conversation lookup
                    from services.orchestrator_utils import normalize_thread_id
                    normalized_thread_id = normalize_thread_id(current_user.user_id, conversation_id)
                    
                    # Get conversation state from LangGraph checkpoint
                    config = {"configurable": {"thread_id": normalized_thread_id}}
                    
                    # Try to get current state with all messages using the actual checkpointer instance
                    if checkpointer.checkpointer and not checkpointer.using_fallback:
                        actual_checkpointer = checkpointer.checkpointer
                        if hasattr(actual_checkpointer, 'aget_tuple'):
                            checkpoint_tuple = await actual_checkpointer.aget_tuple(config)
                        else:
                            logger.warning("⚠️ aget_tuple method not available on checkpointer")
                            checkpoint_tuple = None
                            
                        if checkpoint_tuple and checkpoint_tuple.checkpoint:
                            # Extract messages from checkpoint state
                            checkpoint_state = checkpoint_tuple.checkpoint
                            # ROOSEVELT'S POSTGRESQL JSON FIX: Handle both dict and JSON string formats
                            if isinstance(checkpoint_state, str):
                                import json
                                try:
                                    checkpoint_state = json.loads(checkpoint_state)
                                except json.JSONDecodeError as e:
                                    logger.error(f"❌ Failed to parse checkpoint state JSON: {e}")
                                    checkpoint_state = {}
                            elif checkpoint_state is None:
                                checkpoint_state = {}
                            
                            state_data = checkpoint_state.get("channel_values", {})
                            logger.info(f"🔍 Checkpoint state keys: {list(state_data.keys())}")
                            logger.info(f"🔍 Checkpoint state data: {state_data}")
                            
                            if "messages" in state_data:
                                langgraph_messages = state_data["messages"]
                                logger.info(f"✅ Found {len(langgraph_messages)} messages in LangGraph checkpoint")
                                
                                # ROOSEVELT'S DEBUG LOGGING: Log message types for debugging
                                for i, msg in enumerate(langgraph_messages):
                                    msg_type = "unknown"
                                    if hasattr(msg, '__class__'):
                                        msg_type = str(msg.__class__)
                                    elif hasattr(msg, 'type'):
                                        msg_type = msg.type
                                    logger.debug(f"🔍 Message {i}: type={msg_type}, content_length={len(msg.content) if hasattr(msg, 'content') else 0}")
                                
                                # Convert LangGraph messages to API format
                                for i, msg in enumerate(langgraph_messages):
                                    if hasattr(msg, 'content'):
                                        # ROOSEVELT'S MESSAGE TYPE FIX: Proper LangGraph message type detection
                                        message_type = "user"
                                        role = "user"
                                        
                                        # Check for HumanMessage (user messages)
                                        if hasattr(msg, '__class__') and 'HumanMessage' in str(msg.__class__):
                                            message_type = "user"
                                            role = "user"
                                        # Check for AIMessage (assistant messages)  
                                        elif hasattr(msg, '__class__') and 'AIMessage' in str(msg.__class__):
                                            message_type = "assistant"
                                            role = "assistant"
                                        # Fallback to type attribute if available
                                        elif hasattr(msg, 'type'):
                                            if msg.type == "human":
                                                message_type = "user"
                                                role = "user"
                                            elif msg.type == "ai":
                                                message_type = "assistant"
                                                role = "assistant"
                                        
                                        # ROOSEVELT'S CITATION FIX: Extract citations from THIS MESSAGE's additional_kwargs
                                        citations = []
                                        if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                                            citations = msg.additional_kwargs.get("citations", [])
                                            if citations:
                                                logger.info(f"🔗 EXTRACTED {len(citations)} CITATIONS from message {i} additional_kwargs")
                                        
                                        # Build metadata from additional_kwargs
                                        metadata_json = {}
                                        if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                                            metadata_json = {
                                                "citations": citations,
                                                "research_mode": msg.additional_kwargs.get("research_mode"),
                                                "timestamp": msg.additional_kwargs.get("timestamp")
                                            }
                                        
                                        messages.append({
                                            "message_id": f"lg_{conversation_id}_{i}",
                                            "conversation_id": conversation_id,
                                            "message_type": message_type,
                                            "role": role,
                                            "content": msg.content,
                                            "sequence_number": i,
                                            "created_at": datetime.now().isoformat(),
                                            "updated_at": datetime.now().isoformat(),
                                            "metadata_json": metadata_json if metadata_json.get("citations") else {},
                                            "citations": citations,
                                            "edit_history": []
                                        })
                                messages_from_checkpoint = True  # Mark that we got messages from checkpoint
                            else:
                                logger.info(f"⚠️ No messages found in checkpoint state for {conversation_id}")
                        else:
                            logger.info(f"⚠️ No checkpoint found for conversation {conversation_id}")
                    else:
                        logger.warning("⚠️ aget_tuple method not available on checkpointer")
            except Exception as checkpoint_error:
                # This is normal for new conversations that don't have checkpoints yet
                if "aget_tuple" in str(checkpoint_error) or "get_next_version" in str(checkpoint_error):
                    logger.debug(f"🔧 LangGraph checkpointer API issue (expected for new conversations): {checkpoint_error}")
                else:
                    logger.warning(f"⚠️ Failed to read from LangGraph checkpoint: {checkpoint_error}")
        
        total_count = len(messages)
        has_more = False  # Since we're getting all messages from checkpoint or database
        
        # Determine source for logging
        if messages_from_checkpoint and total_count > 0:
            source = "checkpoint"
        elif total_count > 0:
            source = "database"
        else:
            source = "none"
        
        logger.info(f"✅ Retrieved {total_count} messages for conversation {conversation_id} (source: {source})")
        return MessageListResponse(
            messages=messages, 
            total_count=total_count,
            has_more=has_more
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get conversation messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/conversations/{conversation_id}/messages", response_model=MessageResponse)
async def add_message_to_conversation(conversation_id: str, request: CreateMessageRequest, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Add a message to a conversation"""
    conversation_service = await _get_conversation_service()
    try:
        # Check comment permission
        from utils.auth_middleware import validate_conversation_access
        has_access = await validate_conversation_access(
            user_id=current_user.user_id,
            conversation_id=conversation_id,
            required_permission="comment"
        )
        if not has_access:
            raise HTTPException(status_code=403, detail="You do not have permission to add messages to this conversation")
        
        logger.info(f"💬 Adding message to conversation: {conversation_id}")
        
        # Set the current user for this operation
        conversation_service.set_current_user(current_user.user_id)
        
        message = await conversation_service.add_message(
            conversation_id=conversation_id,
            user_id=current_user.user_id,
            role=request.message_type.value,  # Convert enum to string
            content=request.content,
            metadata=request.metadata
        )
        
        logger.info(f"✅ Message added to conversation {conversation_id}")
        
        # Broadcast message to all conversation participants
        try:
            from services.conversation_sharing_service import get_conversation_sharing_service
            from utils.websocket_manager import get_websocket_manager
            
            sharing_service = await get_conversation_sharing_service()
            participants = await sharing_service.get_conversation_participants(
                conversation_id=conversation_id,
                user_id=current_user.user_id
            )
            
            websocket_manager = get_websocket_manager()
            if websocket_manager and participants:
                for participant in participants:
                    if participant["user_id"] != current_user.user_id:  # Don't notify sender
                        try:
                            await websocket_manager.send_to_session(
                                message={
                                    "type": "participant_message",
                                    "data": {
                                        "conversation_id": conversation_id,
                                        "sender_id": current_user.user_id,
                                        "message_id": message.get("message_id"),
                                        "content": message.get("content", "")[:100]  # Preview
                                    }
                                },
                                session_id=participant["user_id"]
                            )
                        except Exception as ws_error:
                            logger.debug(f"Failed to notify participant {participant['user_id']}: {ws_error}")
        except Exception as collab_error:
            logger.debug(f"Collaboration notification failed (non-fatal): {collab_error}")
        
        return MessageResponse(message=message)
        
    except Exception as e:
        logger.error(f"❌ Failed to add message to conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Delete a conversation and all its messages (LangGraph + legacy)"""
    conversation_service = await _get_conversation_service()
    try:
        logger.info(f"💬 Deleting conversation: {conversation_id} for user: {current_user.user_id}")
        
        # Step 0: Clean up generated images before deletion
        await _cleanup_conversation_images(conversation_id, current_user.user_id)
        
        # ROOSEVELT'S DUAL DELETION: Delete from both LangGraph checkpoints AND legacy tables
        
        # Step 1: Delete from LangGraph checkpoints
        import asyncpg
        from config import settings
        
        connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        conn = await asyncpg.connect(connection_string)
        
        try:
            # Delete all checkpoints for this conversation/thread
            deleted_checkpoints = await conn.execute("""
                DELETE FROM checkpoints 
                WHERE thread_id = $1 
                AND checkpoint -> 'channel_values' IS NOT NULL
                AND checkpoint -> 'channel_values' ->> 'user_id' = $2
            """, conversation_id, current_user.user_id)
            
            # Also delete from checkpoint_blobs and checkpoint_writes
            await conn.execute("""
                DELETE FROM checkpoint_blobs 
                WHERE thread_id = $1
            """, conversation_id)
            
            await conn.execute("""
                DELETE FROM checkpoint_writes 
                WHERE thread_id = $1
            """, conversation_id)
            
            logger.info(f"🗑️ Deleted LangGraph checkpoints: {deleted_checkpoints}")
            
        finally:
            await conn.close()
        
        # Step 2: Delete from legacy conversation tables (for completeness)
        conversation_service.set_current_user(current_user.user_id)
        legacy_success = await conversation_service.delete_conversation(conversation_id)
        
        logger.info(f"🔍 Legacy delete result: {legacy_success}")
        logger.info(f"✅ Conversation deleted from both LangGraph and legacy systems: {conversation_id}")
        return {"status": "success", "message": f"Conversation {conversation_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/conversations")
async def delete_all_conversations(current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Delete ALL conversations for the current user (LangGraph + legacy)"""
    conversation_service = await _get_conversation_service()
    try:
        logger.info(f"💬 Deleting ALL conversations for user: {current_user.user_id}")
        
        # ROOSEVELT'S MASS DELETION: Delete from both LangGraph checkpoints AND legacy tables
        
        # Step 0: Clean up generated images from all conversations before deletion
        from services.database_manager.database_helpers import fetch_all
        
        # Get all conversation IDs for this user before deletion
        conversations = await fetch_all(
            """
            SELECT conversation_id 
            FROM conversations 
            WHERE user_id = $1
            """,
            current_user.user_id
        )
        
        for conv in conversations:
            conv_id = conv.get('conversation_id')
            if conv_id:
                await _cleanup_conversation_images(conv_id, current_user.user_id)
        
        # Step 1: Delete from LangGraph checkpoints
        import asyncpg
        from config import settings
        
        connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        conn = await asyncpg.connect(connection_string)
        
        try:
            # Delete all checkpoints for this user
            deleted_checkpoints = await conn.execute("""
                DELETE FROM checkpoints 
                WHERE checkpoint -> 'channel_values' IS NOT NULL
                AND checkpoint -> 'channel_values' ->> 'user_id' = $1
            """, current_user.user_id)
            
            # Also delete from checkpoint_blobs and checkpoint_writes for this user
            # First get all thread_ids for this user
            user_threads = await conn.fetch("""
                SELECT DISTINCT thread_id FROM checkpoints 
                WHERE checkpoint -> 'channel_values' IS NOT NULL
                AND checkpoint -> 'channel_values' ->> 'user_id' = $1
            """, current_user.user_id)
            
            thread_ids = [row['thread_id'] for row in user_threads]
            
            if thread_ids:
                # Delete from checkpoint_blobs and checkpoint_writes for user's threads
                await conn.execute("""
                    DELETE FROM checkpoint_blobs 
                    WHERE thread_id = ANY($1)
                """, thread_ids)
                
                await conn.execute("""
                    DELETE FROM checkpoint_writes 
                    WHERE thread_id = ANY($1)
                """, thread_ids)
            
            logger.info(f"🗑️ Deleted LangGraph checkpoints for user: {deleted_checkpoints}")
            
        finally:
            await conn.close()
        
        # Step 2: Delete from legacy conversation tables (for completeness)
        from services.service_container import get_service_container
        container = await get_service_container()
        conversation_service = container.conversation_service
        conversation_service.set_current_user(current_user.user_id)
        
        # Get all conversations for the user to delete them one by one (to ensure cleanup)
        conversations_result = await conversation_service.list_conversations(skip=0, limit=1000)
        user_conversations = conversations_result.get("conversations", [])
        
        for conv in user_conversations:
            await conversation_service.delete_conversation(conv["conversation_id"])
            
        logger.info(f"✅ ALL conversations deleted for user: {current_user.user_id}")
        return {"status": "success", "message": "All conversations deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete all conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
