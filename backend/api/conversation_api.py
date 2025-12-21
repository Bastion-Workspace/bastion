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
                        from services.conversation_service import ConversationService
                        conversation_service = ConversationService()
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
                from services.conversation_service import ConversationService
                conversation_service = ConversationService()
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


