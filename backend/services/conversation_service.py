"""
Conversation History Service for Bastion Workspace
Handles persistent conversation storage with multi-user support
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple

import asyncpg
from config import settings
from models.conversation_models import *
from repositories.document_repository import DocumentRepository
# Title generation moved to LLM orchestrator - no longer needed in backend
from utils.citation_utils import citations_to_json, citations_from_json

logger = logging.getLogger(__name__)


def _conversation_message_row_to_dict(row: Any) -> Dict[str, Any]:
    """Map a conversation_messages row to API-shaped dict."""
    parsed_metadata = json.loads(row["metadata_json"] or "{}")
    return {
        "message_id": row["message_id"],
        "conversation_id": row["conversation_id"],
        "message_type": row["message_type"],
        "role": row["message_type"],
        "content": row["content"],
        "content_hash": row.get("content_hash"),
        "model_used": row.get("model_used"),
        "query_time": row.get("query_time"),
        "token_count": row.get("token_count"),
        "sequence_number": row["sequence_number"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
        "metadata_json": parsed_metadata,
        "metadata": parsed_metadata,
        "citations": json.loads(row["citations"] or "[]") if "citations" in row else [],
        "parent_message_id": row.get("parent_message_id"),
        "branch_id": row.get("branch_id"),
        "is_edited": row.get("is_edited", False),
        "edit_history": [],
    }


class ConversationLifecycleManager:
    """Manages the complete lifecycle of conversations with single source of truth"""
    
    def __init__(self):
        self.db_pool = None
        logger.debug("🔄 Initializing ConversationLifecycleManager...")
    
    async def _get_db_pool(self):
        """Get database connection pool"""
        if not self.db_pool:
            self.db_pool = await asyncpg.create_pool(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB,
                min_size=1,
                max_size=10
            )
        return self.db_pool
    
    async def create_conversation(self, user_id: str, initial_message: str = None, 
                                initial_mode: str = "chat", metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new conversation with complete lifecycle tracking"""
        conversation_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        # Initialize conversation metadata
        conversation_metadata = {
            "lifecycle": {
                "created_at": created_at.isoformat(),
                "initial_mode": initial_mode,
                "current_mode": initial_mode,
                "mode_transitions": [],
                "total_messages": 0,
                "total_user_messages": 0,
                "total_assistant_messages": 0,
                "last_activity": created_at.isoformat(),
                "status": "active"
            },
            "execution_stats": {
                "research_plans_generated": 0,
                "research_plans_executed": 0,
                "web_searches_performed": 0,
                "documents_ingested": 0,
                "total_processing_time": 0
            },
            "user_context": {
                "user_id": user_id,
                "session_id": None,  # Will be set when first message is sent
                "preferred_model": None
            }
        }
        
        # Merge with provided metadata
        if metadata:
            conversation_metadata.update(metadata)
        
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            # Set user context for RLS policies
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            # Generate default title if none provided
            default_title = "New Conversation"
            if metadata and metadata.get("title"):
                default_title = metadata.get("title")
            elif initial_message:
                # Use first 50 characters of initial message as title
                default_title = initial_message[:50] + ("..." if len(initial_message) > 50 else "")
            
            # Create conversation
            conversation = await conn.fetchrow("""
                INSERT INTO conversations (conversation_id, user_id, title, created_at, updated_at, metadata_json, message_sequence)
                VALUES ($1, $2, $3, $4, $5, $6, 0)
                RETURNING conversation_id, user_id, title, created_at, updated_at, metadata_json
            """, conversation_id, user_id, default_title, created_at, created_at, json.dumps(conversation_metadata))
            
            logger.info(f"✅ Created conversation {conversation_id} with lifecycle tracking")
            
            # Return a complete conversation dict with all required fields
            conversation_dict = dict(conversation)
            
            # Parse metadata_json if it's a string
            metadata_json = conversation_dict.get("metadata_json")
            if isinstance(metadata_json, str):
                conversation_dict["metadata_json"] = json.loads(metadata_json)
            elif metadata_json is None:
                conversation_dict["metadata_json"] = {}
            
            conversation_dict.update({
                "description": None,
                "is_pinned": False,
                "is_archived": False,
                "tags": [],
                "message_count": 0,
                "last_message_at": None,
                "manual_order": None,
                "order_locked": False
            })
            return conversation_dict
    
    async def _ensure_conversation_exists_in_connection(self, conn, conversation_id: str, user_id: str, 
                                                        initial_message: str = None) -> bool:
        """
        Safely ensure conversation exists in database without overwriting existing data.
        Uses provided connection to avoid race conditions.
        
        This method is idempotent - it only creates if conversation doesn't exist.
        If conversation exists, it verifies ownership and returns True.
        If conversation doesn't exist, it creates it with safe defaults.
        
        Returns:
            True if conversation exists (created or already existed), False on error
        """
        # Check if conversation exists
        conversation = await conn.fetchrow(
            "SELECT * FROM conversations WHERE conversation_id = $1",
            conversation_id
        )
        
        if conversation:
            # Conversation exists - verify ownership
            if conversation['user_id'] != user_id:
                logger.warning(f"⚠️ Conversation {conversation_id} exists but owned by different user (expected {user_id}, found {conversation['user_id']})")
                return False
            logger.debug(f"✅ Conversation {conversation_id} already exists")
            return True
        
        # Conversation doesn't exist - create it safely
        logger.info(f"📝 Auto-creating conversation {conversation_id} for user {user_id}")
        created_at = datetime.now(timezone.utc)
        
        # Initialize conversation metadata with safe defaults
        conversation_metadata = {
            "lifecycle": {
                "created_at": created_at.isoformat(),
                "initial_mode": "chat",
                "current_mode": "chat",
                "mode_transitions": [],
                "total_messages": 0,
                "total_user_messages": 0,
                "total_assistant_messages": 0,
                "last_activity": created_at.isoformat(),
                "status": "active"
            },
            "execution_stats": {
                "research_plans_generated": 0,
                "research_plans_executed": 0,
                "web_searches_performed": 0,
                "documents_ingested": 0,
                "total_processing_time": 0
            },
            "user_context": {
                "user_id": user_id,
                "session_id": None,
                "preferred_model": None
            }
        }
        
        # Generate default title
        default_title = "New Conversation"
        if initial_message:
            default_title = initial_message[:50] + ("..." if len(initial_message) > 50 else "")
        
        try:
            # Use INSERT ... ON CONFLICT DO NOTHING to prevent race conditions
            await conn.execute("""
                INSERT INTO conversations (conversation_id, user_id, title, created_at, updated_at, metadata_json, message_sequence)
                VALUES ($1, $2, $3, $4, $5, $6, 0)
                ON CONFLICT (conversation_id) DO NOTHING
            """, conversation_id, user_id, default_title, created_at, created_at, json.dumps(conversation_metadata))
            
            logger.info(f"✅ Auto-created conversation {conversation_id} in database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to auto-create conversation {conversation_id}: {e}")
            return False
    
    async def add_message(self, conversation_id: str, user_id: str, role: str, 
                         content: str, message_type: str = "text", 
                         metadata: Dict[str, Any] = None, 
                         mode_transition: str = None,
                         parent_message_id: Optional[str] = None,
                         message_branch_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a message and update conversation lifecycle"""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            # Set user context for RLS policies
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            # Ensure conversation exists (safe auto-creation if needed) - using same connection
            conversation_exists = await self._ensure_conversation_exists_in_connection(
                conn, conversation_id, user_id, initial_message=content if role == "user" else None
            )
            if not conversation_exists:
                raise ValueError(f"Conversation {conversation_id} could not be created or accessed")
            
            return await self._add_message_in_connection(
                conn,
                conversation_id,
                user_id,
                role,
                content,
                metadata,
                mode_transition,
                parent_message_id,
                message_branch_id,
            )

    async def _add_message_in_connection(
        self,
        conn: asyncpg.Connection,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        mode_transition: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        message_branch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert one message row and bump lifecycle (caller must set RLS and ensure conversation exists)."""
        # Get current conversation (should exist now, in same transaction)
        conversation = await conn.fetchrow(
            "SELECT * FROM conversations WHERE conversation_id = $1",
            conversation_id
        )
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found after ensure")

        effective_parent = parent_message_id
        if effective_parent is None:
            effective_parent = conversation.get("current_node_message_id")

        # Parse existing metadata
        conv_metadata = json.loads(conversation['metadata_json'] or "{}")
        lifecycle = conv_metadata.get("lifecycle", {})
        execution_stats = conv_metadata.get("execution_stats", {})
        
        # Update lifecycle metadata
        current_time = datetime.now(timezone.utc)
        lifecycle["last_activity"] = current_time.isoformat()
        lifecycle["total_messages"] = lifecycle.get("total_messages", 0) + 1

        if role == "user":
            lifecycle["total_user_messages"] = lifecycle.get("total_user_messages", 0) + 1
        elif role == "assistant":
            lifecycle["total_assistant_messages"] = lifecycle.get("total_assistant_messages", 0) + 1

        # Handle mode transitions
        if mode_transition and mode_transition != lifecycle.get("current_mode"):
            lifecycle["mode_transitions"].append({
                "from_mode": lifecycle.get("current_mode"),
                "to_mode": mode_transition,
                "timestamp": current_time.isoformat(),
                "triggered_by_message": content[:100]  # First 100 chars
            })
            lifecycle["current_mode"] = mode_transition

        # Update execution stats based on message content/type
        if metadata:
            if metadata.get("is_research_plan"):
                execution_stats["research_plans_generated"] = execution_stats.get("research_plans_generated", 0) + 1
            if metadata.get("execution_mode") == "execute":
                execution_stats["research_plans_executed"] = execution_stats.get("research_plans_executed", 0) + 1
            if metadata.get("web_search_performed"):
                execution_stats["web_searches_performed"] = execution_stats.get("web_searches_performed", 0) + 1
            if metadata.get("documents_ingested"):
                execution_stats["documents_ingested"] = execution_stats.get("documents_ingested", 0) + metadata["documents_ingested"]
            if metadata.get("processing_time"):
                execution_stats["total_processing_time"] = execution_stats.get("total_processing_time", 0) + metadata["processing_time"]

        # Update conversation metadata
        conv_metadata["lifecycle"] = lifecycle
        conv_metadata["execution_stats"] = execution_stats

        # Update conversation; any new message marks the thread for session-level memory analysis
        await conn.execute("""
            UPDATE conversations
            SET metadata_json = $1, updated_at = $2, message_sequence = message_sequence + 1,
                needs_session_summary = TRUE
            WHERE conversation_id = $3
        """, json.dumps(conv_metadata), current_time, conversation_id)

        # Get the new sequence number
        sequence_result = await conn.fetchval(
            "SELECT message_sequence FROM conversations WHERE conversation_id = $1",
            conversation_id
        )

        # Title generation is now handled by the orchestrator for better context
        # Only generate fallback title here for non-orchestrator flows
        # Check if this is from orchestrator (metadata flag or orchestrator_system flag)
        is_orchestrator_flow = (
            metadata and (
                metadata.get("orchestrator_system") or
                metadata.get("orchestrator_handles_title") or
                metadata.get("skip_title_generation")
            )
        )

        # Generate simple fallback title only for non-orchestrator flows
        if not is_orchestrator_flow:
            if not conversation['title'] or conversation['title'] == "New Conversation":
                if role == "user":
                    # Check if this is the first user message (no previous user messages)
                    user_message_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM conversation_messages
                        WHERE conversation_id = $1 AND message_type = 'user'
                    """, conversation_id)

                    # Only generate fallback title for the very first user message
                    if user_message_count == 0:
                        # Simple fallback title (orchestrator will generate better one)
                        title = content[:100] + ("..." if len(content) > 100 else "")
                        await conn.execute(
                            "UPDATE conversations SET title = $1 WHERE conversation_id = $2",
                            title, conversation_id
                        )
                        logger.debug(f"Generated fallback title for conversation {conversation_id}: {title}")
                    else:
                        # Not the first message, just update with simple title if still "New Conversation"
                        title = content[:100] + ("..." if len(content) > 100 else "")
                        await conn.execute(
                            "UPDATE conversations SET title = $1 WHERE conversation_id = $2",
                            title, conversation_id
                        )

        # Add the message
        message_id = str(uuid.uuid4())
        message = await conn.fetchrow("""
            INSERT INTO conversation_messages (
                message_id, conversation_id, message_type, content, sequence_number,
                created_at, metadata_json, parent_message_id, branch_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING message_id, conversation_id, message_type, content, sequence_number,
                created_at, metadata_json, parent_message_id, branch_id
        """, message_id, conversation_id, role, content, sequence_result, current_time,
            json.dumps(metadata or {}), effective_parent, message_branch_id)

        await conn.execute(
            "UPDATE conversations SET current_node_message_id = $1 WHERE conversation_id = $2",
            message_id, conversation_id,
        )

        logger.info(
            "Added message to conversation %s (sequence: %s, mode: %s)",
            conversation_id, sequence_result, lifecycle["current_mode"],
        )
        return dict(message)
    
    async def update_conversation_metadata(self, conversation_id: str, 
                                         updates: Dict[str, Any]) -> bool:
        """Update conversation metadata while preserving lifecycle tracking"""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            conversation = await conn.fetchrow(
                "SELECT * FROM conversations WHERE conversation_id = $1",
                conversation_id
            )
            if not conversation:
                return False
            
            # Set user context for RLS policies
            user_id = conversation['user_id']
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            # Parse existing metadata
            conv_metadata = json.loads(conversation['metadata_json'] or "{}")
            
            # Update specified fields
            for key, value in updates.items():
                if key == "lifecycle":
                    # Merge lifecycle updates carefully
                    existing_lifecycle = conv_metadata.get("lifecycle", {})
                    existing_lifecycle.update(value)
                    conv_metadata["lifecycle"] = existing_lifecycle
                elif key == "execution_stats":
                    # Merge execution stats carefully
                    existing_stats = conv_metadata.get("execution_stats", {})
                    existing_stats.update(value)
                    conv_metadata["execution_stats"] = existing_stats
                else:
                    conv_metadata[key] = value
            
            # Update conversation
            await conn.execute(
                "UPDATE conversations SET metadata_json = $1, updated_at = $2 WHERE conversation_id = $3",
                json.dumps(conv_metadata), datetime.now(timezone.utc), conversation_id
            )
            
            logger.info(f"✅ Updated metadata for conversation {conversation_id}")
            return True

    async def ensure_conversation_exists(
        self, conversation_id: str, user_id: str, initial_message: str = None
    ) -> bool:
        """Ensure a conversation exists; create it if not. Returns True if it exists or was created."""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            return await self._ensure_conversation_exists_in_connection(
                conn, conversation_id, user_id, initial_message=initial_message
            )
    
    async def get_conversation_lifecycle(self, conversation_id: str, user_id: str = None) -> Dict[str, Any]:
        """Get complete conversation lifecycle information"""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            # Set user context for RLS policies if provided
            if user_id:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            conversation = await conn.fetchrow(
                "SELECT * FROM conversations WHERE conversation_id = $1",
                conversation_id
            )
            if not conversation:
                    return None
                
            # If user_id wasn't provided, get it from conversation and set context
            if not user_id:
                user_id = conversation['user_id']
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            conv_metadata = json.loads(conversation['metadata_json'] or "{}")
            
            # Get message count for verification
            message_count = await conn.fetchval(
                "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id = $1",
                conversation_id
            )
            
            lifecycle = conv_metadata.get("lifecycle", {})
            lifecycle["verified_message_count"] = message_count
            
            return {
                "conversation_id": conversation_id,
                "user_id": conversation['user_id'],
                "title": conversation['title'],
                "description": None,
                "is_pinned": conversation.get('is_pinned', False),
                "is_archived": conversation.get('is_archived', False),
                "tags": conversation.get('tags') or [],
                "metadata_json": conv_metadata,  # Already parsed as dict
                "message_count": message_count,
                "last_message_at": conversation.get('last_message_at'),
                "manual_order": conversation.get('manual_order'),
                "order_locked": conversation.get('order_locked', False),
                "created_at": conversation['created_at'],
                "updated_at": conversation['updated_at'],
                "lifecycle": lifecycle,
                "execution_stats": conv_metadata.get("execution_stats", {}),
                "user_context": conv_metadata.get("user_context", {}),
                "current_node_message_id": conversation.get("current_node_message_id"),
            }
    
    async def list_conversations_with_lifecycle(self, user_id: str, 
                                              skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """List conversations with complete lifecycle information"""
        pool = await self._get_db_pool()
        async with pool.acquire() as conn:
            # Set user context for RLS policies
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
            
            conversations = await conn.fetch(
                """
                SELECT c.conversation_id, c.user_id, c.title, c.description, c.is_pinned, c.is_archived, 
                       c.tags, c.metadata_json, c.manual_order, c.order_locked, c.created_at, c.updated_at,
                       COUNT(cm.message_id) as message_count,
                       MAX(cm.created_at) as last_message_time
                FROM conversations c
                LEFT JOIN conversation_messages cm ON c.conversation_id = cm.conversation_id
                WHERE c.user_id = $1
                GROUP BY c.conversation_id, c.user_id, c.title, c.description, c.is_pinned, c.is_archived, 
                         c.tags, c.metadata_json, c.manual_order, c.order_locked, c.created_at, c.updated_at
                ORDER BY c.updated_at DESC
                LIMIT $2 OFFSET $3
                """, user_id, limit, skip
            )
            
            result = []
            for row in conversations:
                conv_metadata = json.loads(row['metadata_json'] or "{}")
                lifecycle = conv_metadata.get("lifecycle", {})
                
                result.append({
                    "conversation_id": row['conversation_id'],
                    "user_id": row.get('user_id', user_id),
                    "title": row['title'],
                    "description": row.get('description'),
                    "is_pinned": row.get('is_pinned', False),
                    "is_archived": row.get('is_archived', False),
                    "tags": row.get('tags') or [],
                    "metadata_json": conv_metadata,  # Already parsed as dict
                    "message_count": row['message_count'],
                    "last_message_at": row['last_message_time'],
                    "manual_order": row.get('manual_order'),
                    "order_locked": row.get('order_locked', False),
                    "created_at": row['created_at'],
                    "updated_at": row['updated_at'],
                    # Additional lifecycle info for backward compatibility
                    "last_message_time": row['last_message_time'].isoformat() if row['last_message_time'] else None,
                    "current_mode": lifecycle.get("current_mode", "chat"),
                    "status": lifecycle.get("status", "active"),
                    "total_processing_time": conv_metadata.get("execution_stats", {}).get("total_processing_time", 0)
                })
            
            return result


class ConversationService:
    """Service for managing conversations with unified lifecycle tracking"""
    
    def __init__(self):
        self.lifecycle_manager = ConversationLifecycleManager()
        # Title generation moved to LLM orchestrator - no longer needed here
        logger.debug("🗨️ Initializing Conversation Service...")
        
        # Note: Database connection is handled by the lifecycle manager
        logger.debug("✅ Conversation Service initialized with lifecycle manager")
        
        # User context is handled internally via lifecycle manager
        self.current_user_id = None
    
    def set_current_user(self, user_id: str):
        """Set the current user for operations (for compatibility with existing code)"""
        self.current_user_id = user_id
        logger.debug(f"🔄 ConversationService: Set current user to {user_id}")
    
    async def create_conversation(self, user_id: str, initial_message: str = None, 
                                initial_mode: str = "chat", metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new conversation with lifecycle tracking"""
        try:
            conversation = await self.lifecycle_manager.create_conversation(
                user_id=user_id,
                initial_message=initial_message,
                initial_mode=initial_mode,
                metadata=metadata
            )
            
            # Title generation is handled by LLM orchestrator after first agent response
            # Use simple fallback title for new conversations (will be replaced by orchestrator)
            if initial_message and (not conversation.get("title") or conversation.get("title") == "New Conversation"):
                fallback_title = initial_message[:100] + ("..." if len(initial_message) > 100 else "")
                await self.lifecycle_manager.update_conversation_metadata(
                    conversation["conversation_id"],
                    {"title": fallback_title}
                )
                conversation["title"] = fallback_title
                logger.debug(f"Set fallback title for new conversation: {fallback_title}")
            
            return conversation
        except Exception as e:
            logger.error(f"❌ Failed to create conversation: {e}")
            raise
    
    async def add_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None,
        mode_transition: str = None,
        parent_message_id: Optional[str] = None,
        message_branch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a message with lifecycle tracking"""
        try:
            message = await self.lifecycle_manager.add_message(
                conversation_id=conversation_id,
                user_id=user_id,
                role=role,
                content=content,
                metadata=metadata,
                mode_transition=mode_transition,
                parent_message_id=parent_message_id,
                message_branch_id=message_branch_id,
            )

            meta = json.loads(message["metadata_json"] or "{}")
            return {
                "message_id": message["message_id"],
                "conversation_id": message["conversation_id"],
                "role": message["message_type"],
                "content": message["content"],
                "sequence_number": message["sequence_number"],
                "created_at": message["created_at"].isoformat(),
                "metadata": meta,
                "parent_message_id": message.get("parent_message_id"),
                "branch_id": message.get("branch_id"),
            }
        except Exception as e:
            logger.error(f"❌ Failed to add message: {e}")
            raise

    async def create_branch(
        self,
        conversation_id: str,
        user_id: str,
        original_message_id: str,
        new_content: str,
    ) -> Dict[str, Any]:
        """Fork: new user message as sibling of original; new branch row for checkpoint suffix."""
        pool = await self.lifecycle_manager._get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, false)", user_id
            )
            conv = await conn.fetchrow(
                "SELECT * FROM conversations WHERE conversation_id = $1",
                conversation_id,
            )
            if not conv or conv["user_id"] != user_id:
                raise ValueError("Conversation not found or access denied")

            async with conn.transaction():
                orig = await conn.fetchrow(
                    """
                    SELECT * FROM conversation_messages
                    WHERE message_id = $1 AND conversation_id = $2
                    """,
                    original_message_id,
                    conversation_id,
                )
                if not orig or orig["message_type"] != "user":
                    raise ValueError("Original message not found or not a user message")

                new_branch_id = str(uuid.uuid4())
                thread_suffix = str(uuid.uuid4())
                parent_branch_id = orig.get("branch_id")

                await conn.execute(
                    """
                    INSERT INTO conversation_branches (
                        branch_id, conversation_id, parent_branch_id,
                        forked_from_message_id, thread_id_suffix
                    ) VALUES ($1, $2, $3, $4, $5)
                    """,
                    new_branch_id,
                    conversation_id,
                    parent_branch_id,
                    original_message_id,
                    thread_suffix,
                )

                meta = {
                    "orchestrator_system": True,
                    "streaming": True,
                    "forked_from_message_id": original_message_id,
                    "branch_edit_resend": True,
                }
                msg = await self.lifecycle_manager._add_message_in_connection(
                    conn,
                    conversation_id,
                    user_id,
                    "user",
                    new_content,
                    meta,
                    None,
                    orig.get("parent_message_id"),
                    new_branch_id,
                )

                await conn.execute(
                    """
                    UPDATE conversation_branches
                    SET first_message_id = $1
                    WHERE branch_id = $2
                    """,
                    msg["message_id"],
                    new_branch_id,
                )

            active = await self.get_active_path_messages(conversation_id, user_id)
            return {
                "message": {
                    "message_id": msg["message_id"],
                    "conversation_id": msg["conversation_id"],
                    "role": msg["message_type"],
                    "content": msg["content"],
                    "sequence_number": msg["sequence_number"],
                    "created_at": msg["created_at"].isoformat(),
                    "metadata": json.loads(msg["metadata_json"] or "{}"),
                    "parent_message_id": msg.get("parent_message_id"),
                    "branch_id": msg.get("branch_id"),
                },
                "branch_id": new_branch_id,
                "thread_id_suffix": thread_suffix,
                "active_path_messages": active,
            }

    async def get_active_path_messages(
        self, conversation_id: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Messages from root to current leaf (for orchestrator conversation_history)."""
        data = await self.get_conversation_messages(
            conversation_id=conversation_id,
            user_id=user_id,
            limit=10000,
            most_recent=False,
            include_tree=False,
        )
        return data.get("messages") or []

    async def get_message_siblings(
        self, conversation_id: str, user_id: str, message_id: str
    ) -> Dict[str, Any]:
        """All messages sharing the same parent_message_id, ordered by created_at."""
        lifecycle_info = await self.lifecycle_manager.get_conversation_lifecycle(
            conversation_id, user_id
        )
        if not lifecycle_info or lifecycle_info.get("user_context", {}).get("user_id") != user_id:
            raise ValueError("Conversation not found or access denied")

        pool = await self.lifecycle_manager._get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, false)", user_id
            )
            row = await conn.fetchrow(
                """
                SELECT parent_message_id FROM conversation_messages
                WHERE message_id = $1 AND conversation_id = $2
                """,
                message_id,
                conversation_id,
            )
            if not row:
                raise ValueError("Message not found")

            parent_id = row["parent_message_id"]
            siblings = await conn.fetch(
                """
                SELECT * FROM conversation_messages
                WHERE conversation_id = $1
                  AND parent_message_id IS NOT DISTINCT FROM $2::varchar
                ORDER BY created_at ASC, sequence_number ASC
                """,
                conversation_id,
                parent_id,
            )
            sibs = [_conversation_message_row_to_dict(r) for r in siblings]
            idx = next(
                (i for i, m in enumerate(sibs) if m["message_id"] == message_id),
                0,
            )
            return {
                "siblings": sibs,
                "current_index": idx,
                "total": len(sibs),
            }

    async def switch_active_branch(
        self, conversation_id: str, user_id: str, target_message_id: str
    ) -> Dict[str, Any]:
        """Set current leaf to deepest descendant of target (max sequence_number per level)."""
        lifecycle_info = await self.lifecycle_manager.get_conversation_lifecycle(
            conversation_id, user_id
        )
        if not lifecycle_info or lifecycle_info.get("user_context", {}).get("user_id") != user_id:
            raise ValueError("Conversation not found or access denied")

        pool = await self.lifecycle_manager._get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, false)", user_id
            )
            target = await conn.fetchrow(
                """
                SELECT message_id FROM conversation_messages
                WHERE message_id = $1 AND conversation_id = $2
                """,
                target_message_id,
                conversation_id,
            )
            if not target:
                raise ValueError("Target message not found")

            all_rows = await conn.fetch(
                """
                SELECT message_id, parent_message_id, sequence_number
                FROM conversation_messages
                WHERE conversation_id = $1
                """,
                conversation_id,
            )
            by_parent: Dict[Optional[str], List[Any]] = {}
            for r in all_rows:
                p = r["parent_message_id"]
                by_parent.setdefault(p, []).append(r)

            for key in list(by_parent.keys()):
                by_parent[key].sort(key=lambda x: x["sequence_number"], reverse=True)

            cur = target_message_id
            while True:
                children = by_parent.get(cur) or []
                if not children:
                    break
                cur = children[0]["message_id"]

            await conn.execute(
                """
                UPDATE conversations
                SET current_node_message_id = $1, updated_at = NOW()
                WHERE conversation_id = $2
                """,
                cur,
                conversation_id,
            )

        path = await self.get_active_path_messages(conversation_id, user_id)
        return {
            "current_node_message_id": cur,
            "active_path": path,
        }

    async def get_branch_thread_id(
        self, conversation_id: str, user_id: str
    ) -> Dict[str, Optional[str]]:
        """Resolve LangGraph thread_id for the active path (branch suffix from deepest branch on path)."""
        lifecycle_info = await self.lifecycle_manager.get_conversation_lifecycle(
            conversation_id, user_id
        )
        if not lifecycle_info or lifecycle_info.get("user_context", {}).get("user_id") != user_id:
            raise ValueError("Conversation not found or access denied")

        pool = await self.lifecycle_manager._get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, false)", user_id
            )
            conv = await conn.fetchrow(
                """
                SELECT current_node_message_id FROM conversations
                WHERE conversation_id = $1
                """,
                conversation_id,
            )
            cur_id = conv["current_node_message_id"] if conv else None
            if not cur_id:
                return {
                    "full_thread_id": f"{user_id}:{conversation_id}",
                    "thread_id_suffix": None,
                    "branch_uuid": None,
                }

            rows = await conn.fetch(
                """
                SELECT message_id, parent_message_id, branch_id
                FROM conversation_messages
                WHERE conversation_id = $1
                """,
                conversation_id,
            )
            by_id = {r["message_id"]: r for r in rows}

            path_ids: List[str] = []
            c = cur_id
            while c:
                path_ids.append(c)
                row = by_id.get(c)
                c = row["parent_message_id"] if row else None
            path_ids.reverse()

            branch_uuid: Optional[str] = None
            for mid in path_ids:
                row = by_id.get(mid)
                bid = row["branch_id"] if row else None
                if bid:
                    branch_uuid = bid

            if not branch_uuid:
                return {
                    "full_thread_id": f"{user_id}:{conversation_id}",
                    "thread_id_suffix": None,
                    "branch_uuid": None,
                }

            br = await conn.fetchrow(
                """
                SELECT thread_id_suffix FROM conversation_branches
                WHERE branch_id = $1 AND conversation_id = $2
                """,
                branch_uuid,
                conversation_id,
            )
        suffix = br["thread_id_suffix"] if br else None
        if not suffix:
            return {
                "full_thread_id": f"{user_id}:{conversation_id}",
                "thread_id_suffix": None,
                "branch_uuid": branch_uuid,
            }
        full_tid = f"{user_id}:{conversation_id}:branch_{suffix}"
        return {
            "full_thread_id": full_tid,
            "thread_id_suffix": suffix,
            "branch_uuid": branch_uuid,
        }

    async def get_message_by_id(
        self, conversation_id: str, user_id: str, message_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load a single message row if it belongs to the conversation and user."""
        lifecycle_info = await self.lifecycle_manager.get_conversation_lifecycle(
            conversation_id, user_id
        )
        if not lifecycle_info or lifecycle_info.get("user_context", {}).get("user_id") != user_id:
            return None
        pool = await self.lifecycle_manager._get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, false)", user_id
            )
            row = await conn.fetchrow(
                """
                SELECT * FROM conversation_messages
                WHERE message_id = $1 AND conversation_id = $2
                """,
                message_id,
                conversation_id,
            )
            return dict(row) if row else None

    async def delete_conversation_message(
        self, conversation_id: str, user_id: str, message_id: str
    ) -> bool:
        """Delete one message row after verifying conversation ownership."""
        if not message_id or not conversation_id:
            return False
        lifecycle_info = await self.lifecycle_manager.get_conversation_lifecycle(
            conversation_id, user_id
        )
        if not lifecycle_info or lifecycle_info.get("user_context", {}).get("user_id") != user_id:
            return False
        pool = await self.lifecycle_manager._get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, false)", user_id
            )
            row = await conn.fetchrow(
                """
                DELETE FROM conversation_messages
                WHERE message_id = $1 AND conversation_id = $2
                RETURNING message_id
                """,
                message_id,
                conversation_id,
            )
            return row is not None

    async def get_conversation(self, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """Get conversation with complete lifecycle information"""
        try:
            lifecycle_info = await self.lifecycle_manager.get_conversation_lifecycle(conversation_id, user_id)
            if not lifecycle_info:
                return None
            
            # Verify user ownership (check both user_id field and user_context)
            conversation_user_id = lifecycle_info.get("user_id") or lifecycle_info.get("user_context", {}).get("user_id")
            if conversation_user_id != user_id:
                logger.warning(f"⚠️ User {user_id} attempted to access conversation {conversation_id} owned by {conversation_user_id}")
                return None
            
            return lifecycle_info
        except Exception as e:
            logger.error(f"❌ Failed to get conversation: {e}")
            raise
    
    async def get_conversation_messages(
        self,
        conversation_id: str,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
        most_recent: bool = False,
        include_tree: bool = False,
    ) -> Dict[str, Any]:
        """
        Get conversation messages with lifecycle verification.

        When most_recent is True, returns the newest ``limit`` messages in chronological order
        (skip is ignored). When False, uses ascending order with skip/limit for pagination
        (oldest-first window).

        When include_tree is True, returns all messages in the conversation (all branches) ordered
        by sequence_number, plus current_node_message_id.

        When include_tree is False and current_node_message_id is set, returns only the active
        path from root to the current leaf (branch-aware transcript).
        """
        try:
            # First verify conversation ownership and get lifecycle info
            lifecycle_info = await self.lifecycle_manager.get_conversation_lifecycle(conversation_id, user_id)
            if not lifecycle_info:
                return {"messages": [], "has_more": False}
            
            if lifecycle_info.get("user_context", {}).get("user_id") != user_id:
                logger.warning(f"⚠️ User {user_id} attempted to access messages for conversation {conversation_id}")
                return {"messages": [], "has_more": False}
            
            # Get messages from database
            pool = await self.lifecycle_manager._get_db_pool()
            async with pool.acquire() as conn:
                # Set user context for RLS policies
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                logger.info(f"🔍 Set user context for conversation messages: {user_id}")

                # Verify RLS context was set
                rls_context = await conn.fetchval("SELECT current_setting('app.current_user_id', true)")
                logger.info(f"🔍 Verified RLS context: {rls_context}")

                conv_row = await conn.fetchrow(
                    "SELECT current_node_message_id FROM conversations WHERE conversation_id = $1",
                    conversation_id,
                )
                current_node_message_id = (
                    conv_row["current_node_message_id"] if conv_row else None
                )

                if include_tree:
                    messages = await conn.fetch(
                        """
                        SELECT * FROM conversation_messages
                        WHERE conversation_id = $1
                        ORDER BY sequence_number ASC
                        """,
                        conversation_id,
                    )
                elif current_node_message_id:
                    messages = await conn.fetch(
                        """
                        WITH RECURSIVE ancestors AS (
                            SELECT cm.* FROM conversation_messages cm
                            WHERE cm.message_id = $1 AND cm.conversation_id = $2
                            UNION ALL
                            SELECT cm.* FROM conversation_messages cm
                            INNER JOIN ancestors a ON cm.message_id = a.parent_message_id
                        )
                        SELECT * FROM ancestors ORDER BY sequence_number ASC
                        """,
                        current_node_message_id,
                        conversation_id,
                    )
                    logger.info(
                        "Loaded %s messages on active path for conversation %s",
                        len(messages),
                        conversation_id,
                    )
                elif most_recent:
                    messages = await conn.fetch(
                        """
                        SELECT * FROM (
                            SELECT cm.*, c.metadata_json as conversation_metadata
                            FROM conversation_messages cm
                            JOIN conversations c ON cm.conversation_id = c.conversation_id
                            WHERE cm.conversation_id = $1
                            ORDER BY cm.sequence_number DESC
                            LIMIT $2
                        ) sub
                        ORDER BY sub.sequence_number ASC
                        """,
                        conversation_id,
                        limit,
                    )
                    if not messages:
                        messages = await conn.fetch(
                            """
                            SELECT * FROM (
                                SELECT * FROM conversation_messages
                                WHERE conversation_id = $1
                                ORDER BY sequence_number DESC
                                LIMIT $2
                            ) sub
                            ORDER BY sequence_number ASC
                            """,
                            conversation_id,
                            limit,
                        )
                    logger.info(
                        "Loaded %s messages for conversation %s (most_recent=True)",
                        len(messages),
                        conversation_id,
                    )
                else:
                    conversation_exists = await conn.fetchval(
                        "SELECT COUNT(*) FROM conversations WHERE conversation_id = $1",
                        conversation_id,
                    )
                    logger.info(
                        "Conversation %s exists check: %s rows",
                        conversation_id,
                        conversation_exists,
                    )

                    message_count_before = await conn.fetchval(
                        "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id = $1",
                        conversation_id,
                    )
                    logger.info(
                        "Message count for conversation %s: %s messages",
                        conversation_id,
                        message_count_before,
                    )

                    messages_direct = await conn.fetch(
                        """
                        SELECT * FROM conversation_messages
                        WHERE conversation_id = $1
                        ORDER BY sequence_number ASC
                        LIMIT $2 OFFSET $3
                        """,
                        conversation_id,
                        limit,
                        skip,
                    )
                    messages = await conn.fetch(
                        """
                        SELECT cm.*, c.metadata_json as conversation_metadata
                        FROM conversation_messages cm
                        JOIN conversations c ON cm.conversation_id = c.conversation_id
                        WHERE cm.conversation_id = $1
                        ORDER BY cm.sequence_number ASC
                        LIMIT $2 OFFSET $3
                        """,
                        conversation_id,
                        limit,
                        skip,
                    )

                    logger.info(
                        "Direct query (no JOIN) returned %s messages",
                        len(messages_direct),
                    )

                    logger.info(
                        "JOIN query returned %s messages from database for conversation %s (most_recent=%s)",
                        len(messages),
                        conversation_id,
                        most_recent,
                    )
                    if len(messages) == 0 and message_count_before > 0:
                        logger.warning(
                            "Query returned 0 messages but message_count shows %s - possible RLS issue",
                            message_count_before,
                        )
                    if len(messages_direct) > 0 and len(messages) == 0:
                        logger.warning(
                            "Direct query found %s messages but JOIN query found 0 - RLS on conversations",
                            len(messages_direct),
                        )
                        messages = messages_direct

                message_list = [_conversation_message_row_to_dict(row) for row in messages]

                total_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id = $1",
                    conversation_id,
                )

                if include_tree or current_node_message_id:
                    return {
                        "messages": message_list,
                        "has_more": False,
                        "total_count": total_count,
                        "lifecycle": lifecycle_info,
                        "current_node_message_id": current_node_message_id,
                    }

                if most_recent:
                    has_more = total_count > limit
                else:
                    has_more = (skip + limit) < total_count

                return {
                    "messages": message_list,
                    "has_more": has_more,
                    "total_count": total_count,
                    "lifecycle": lifecycle_info,
                    "current_node_message_id": current_node_message_id,
                }
        except Exception as e:
            logger.error(f"❌ Failed to get conversation messages: {e}")
            raise
    
    async def list_conversations(self, user_id: str, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """List conversations with lifecycle information"""
        try:
            conversations = await self.lifecycle_manager.list_conversations_with_lifecycle(
                user_id=user_id,
                skip=skip,
                limit=limit
            )
            return conversations
        except Exception as e:
            logger.error(f"❌ Failed to list conversations: {e}")
            raise
    
    async def update_conversation_metadata(self, conversation_id: str, user_id: str, 
                                         updates: Dict[str, Any]) -> bool:
        """Update conversation metadata with lifecycle preservation"""
        try:
            # Verify ownership first
            lifecycle_info = await self.lifecycle_manager.get_conversation_lifecycle(conversation_id, user_id)
            if not lifecycle_info or lifecycle_info.get("user_context", {}).get("user_id") != user_id:
                return False
            
            return await self.lifecycle_manager.update_conversation_metadata(conversation_id, updates)
        except Exception as e:
            logger.error(f"❌ Failed to update conversation metadata: {e}")
            return False

    async def ensure_conversation_exists(self, conversation_id: str, user_id: str) -> bool:
        """Ensure a conversation exists for the user; create it if not. Returns True if it exists or was created."""
        try:
            return await self.lifecycle_manager.ensure_conversation_exists(conversation_id, user_id)
        except Exception as e:
            logger.error(f"❌ Failed to ensure conversation exists: {e}")
            return False
    
    async def get_conversation_analytics(self, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a conversation"""
        try:
            lifecycle_info = await self.lifecycle_manager.get_conversation_lifecycle(conversation_id, user_id)
            if not lifecycle_info or lifecycle_info.get("user_context", {}).get("user_id") != user_id:
                return None
            
            # Calculate additional analytics
            lifecycle = lifecycle_info.get("lifecycle", {})
            execution_stats = lifecycle_info.get("execution_stats", {})
            
            analytics = {
                "conversation_id": conversation_id,
                "title": lifecycle_info.get("title"),
                "created_at": lifecycle_info.get("created_at"),
                "last_activity": lifecycle_info.get("updated_at"),
                "current_mode": lifecycle.get("current_mode", "chat"),
                "status": lifecycle.get("status", "active"),
                "message_stats": {
                    "total_messages": lifecycle.get("total_messages", 0),
                    "user_messages": lifecycle.get("total_user_messages", 0),
                    "assistant_messages": lifecycle.get("total_assistant_messages", 0),
                    "verified_count": lifecycle.get("verified_message_count", 0)
                },
                "execution_stats": execution_stats,
                "mode_transitions": lifecycle.get("mode_transitions", []),
                "performance": {
                    "total_processing_time": execution_stats.get("total_processing_time", 0),
                    "avg_processing_time": execution_stats.get("total_processing_time", 0) / max(lifecycle.get("total_messages", 1), 1)
                }
            }
            
            return analytics
        except Exception as e:
            logger.error(f"❌ Failed to get conversation analytics: {e}")
            return None

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages, including attachments"""
        try:
            logger.info(f"🗑️ Deleting conversation: {conversation_id}")
            
            # Get the current user from the lifecycle manager
            current_user_id = self.current_user_id
            if not current_user_id:
                logger.error("❌ No current user set for conversation deletion")
                return False
            
            # Verify ownership first
            logger.info(f"🔍 Looking up conversation lifecycle for: {conversation_id} with user: {current_user_id}")
            lifecycle_info = await self.lifecycle_manager.get_conversation_lifecycle(conversation_id, current_user_id)
            logger.info(f"🔍 Lifecycle info result: {lifecycle_info}")
            
            if not lifecycle_info:
                logger.warning(f"⚠️ Conversation {conversation_id} not found or not owned by user {current_user_id}")
                return False
            
            # Double-check ownership
            conversation_user_id = lifecycle_info.get("user_id")
            if conversation_user_id != current_user_id:
                logger.warning(f"⚠️ User {current_user_id} attempted to delete conversation {conversation_id} owned by {conversation_user_id}")
                return False
            
            # Clean up attachments before deleting messages
            try:
                from services.chat_attachment_service import chat_attachment_service
                await chat_attachment_service.initialize()
                await chat_attachment_service.cleanup_conversation_attachments(conversation_id)
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to cleanup attachments for conversation {conversation_id}: {cleanup_error}")
                # Continue with deletion even if cleanup fails
            
            # Delete the conversation and all its messages
            pool = await self.lifecycle_manager._get_db_pool()
            async with pool.acquire() as conn:
                # Set user context for RLS policies
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", current_user_id)
                
                async with conn.transaction():
                    # Delete all messages first
                    messages_deleted = await conn.execute(
                        "DELETE FROM conversation_messages WHERE conversation_id = $1",
                        conversation_id
                    )
                    logger.info(f"🗑️ Deleted {messages_deleted} messages for conversation: {conversation_id}")
                    
                    # Delete the conversation
                    result = await conn.execute(
                        "DELETE FROM conversations WHERE conversation_id = $1 AND user_id = $2",
                        conversation_id, current_user_id
                    )
                    logger.info(f"🗑️ Delete conversation result: {result}")
                    
                    if result == "DELETE 1":
                        logger.info(f"✅ Successfully deleted conversation: {conversation_id}")
                        return True
                    else:
                        logger.warning(f"⚠️ No conversation deleted for ID: {conversation_id}")
                        logger.warning(f"⚠️ Result was: {result}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Failed to delete conversation {conversation_id}: {e}")
            return False
    
    async def update_agent_metadata(
        self,
        conversation_id: str,
        user_id: str,
        primary_agent_selected: str,
        last_agent: Optional[str] = None,
        agent_profile_id: Optional[str] = None,
        clear_agent_profile_id: bool = False,
        active_line_id: Optional[str] = None,
        active_line_name: Optional[str] = None,
        clear_active_line: bool = False,
    ) -> bool:
        """
        Update agent routing metadata in conversation

        This stores which agent is currently handling the conversation,
        enabling proper agent continuity across requests.

        Args:
            conversation_id: The conversation to update
            user_id: The user ID (for RLS)
            primary_agent_selected: The agent handling this conversation
            last_agent: The most recent agent that processed a request
            agent_profile_id: Optional Agent Factory profile UUID for sticky routing and lookback
            clear_agent_profile_id: When True, remove agent_profile_id from conversation metadata
            active_line_id: When set, persist agent line UUID for sticky line chat dispatch
            active_line_name: Display name for the active line (UI)
            clear_active_line: When True, remove active_line_id and active_line_name

        Returns:
            bool: True if update succeeded, False otherwise
        """
        try:
            pool = await self.lifecycle_manager._get_db_pool()
            async with pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Get current metadata
                row = await conn.fetchrow(
                    "SELECT metadata_json FROM conversations WHERE conversation_id = $1",
                    conversation_id
                )
                
                if not row:
                    logger.warning(f"⚠️ Conversation {conversation_id} not found for agent metadata update")
                    return False
                
                # Parse existing metadata
                metadata = json.loads(row['metadata_json'] or "{}")
                
                # Update agent routing metadata
                metadata["primary_agent_selected"] = primary_agent_selected
                if last_agent:
                    metadata["last_agent"] = last_agent
                if clear_agent_profile_id:
                    metadata.pop("agent_profile_id", None)
                elif agent_profile_id:
                    metadata["agent_profile_id"] = agent_profile_id
                if clear_active_line:
                    metadata.pop("active_line_id", None)
                    metadata.pop("active_line_name", None)
                else:
                    if active_line_id:
                        metadata["active_line_id"] = active_line_id
                    if active_line_name is not None:
                        metadata["active_line_name"] = active_line_name
                metadata["agent_updated_at"] = datetime.now(timezone.utc).isoformat()
                
                # Save back to database
                await conn.execute(
                    "UPDATE conversations SET metadata_json = $1, updated_at = NOW() WHERE conversation_id = $2",
                    json.dumps(metadata),
                    conversation_id
                )
                
                logger.info(
                    "✅ Updated agent metadata for conversation %s: primary_agent=%s, last_agent=%s, agent_profile_id=%s, active_line=%s",
                    conversation_id,
                    primary_agent_selected,
                    last_agent,
                    "(cleared)" if clear_agent_profile_id else (agent_profile_id or "(unchanged)"),
                    "(cleared)" if clear_active_line else (active_line_id or "(unchanged)"),
                )
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to update agent metadata for conversation {conversation_id}: {e}")
            return False
    
    async def get_agent_metadata(self, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get agent routing metadata from conversation
        
        Returns agent continuity information needed for proper routing.
        
        Args:
            conversation_id: The conversation ID
            user_id: The user ID (for RLS)
            
        Returns:
            Dict with primary_agent_selected, last_agent, and optional agent_profile_id, or empty dict
        """
        try:
            pool = await self.lifecycle_manager._get_db_pool()
            async with pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Get metadata
                row = await conn.fetchrow(
                    "SELECT metadata_json FROM conversations WHERE conversation_id = $1",
                    conversation_id
                )
                
                if not row:
                    return {}
                
                metadata = json.loads(row['metadata_json'] or "{}")
                
                # Extract agent routing info
                agent_metadata = {}
                if "primary_agent_selected" in metadata:
                    agent_metadata["primary_agent_selected"] = metadata["primary_agent_selected"]
                if "last_agent" in metadata:
                    agent_metadata["last_agent"] = metadata["last_agent"]
                if "agent_profile_id" in metadata:
                    agent_metadata["agent_profile_id"] = metadata["agent_profile_id"]
                if "active_line_id" in metadata:
                    agent_metadata["active_line_id"] = metadata["active_line_id"]
                if "active_line_name" in metadata:
                    agent_metadata["active_line_name"] = metadata["active_line_name"]
                
                if agent_metadata:
                    logger.debug(f"📚 Loaded agent metadata for conversation {conversation_id}: {agent_metadata}")
                
                return agent_metadata
                
        except Exception as e:
            logger.debug(f"⚠️ Failed to get agent metadata for conversation {conversation_id}: {e}")
            return {}
    
    async def add_reaction(self, conversation_id: str, message_id: str, user_id: str, emoji: str) -> Optional[Dict[str, Any]]:
        """
        Add or remove a reaction to a message
        
        If the user has already reacted with this emoji, remove it (toggle behavior).
        Otherwise, add the reaction.
        
        Args:
            conversation_id: The conversation ID
            message_id: The message ID
            user_id: The user ID reacting
            emoji: The emoji reaction (👍, 👎, 😂, ❤️, 😢)
            
        Returns:
            Dict with reactions map, or None if message not found
        """
        try:
            pool = await self.lifecycle_manager._get_db_pool()
            async with pool.acquire() as conn:
                # Set user context for RLS policies
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Get the message
                message = await conn.fetchrow(
                    "SELECT * FROM conversation_messages WHERE message_id = $1 AND conversation_id = $2",
                    message_id, conversation_id
                )
                
                if not message:
                    logger.warning(f"⚠️ Message {message_id} not found in conversation {conversation_id}")
                    return None
                
                # Parse existing metadata
                metadata = json.loads(message['metadata_json'] or "{}")
                
                # Get or initialize reactions
                reactions = metadata.get("reactions", {})
                if not isinstance(reactions, dict):
                    reactions = {}
                
                # Get current users for this emoji
                emoji_users = reactions.get(emoji, [])
                if not isinstance(emoji_users, list):
                    emoji_users = []
                
                # Toggle: if user already reacted, remove; otherwise add
                if user_id in emoji_users:
                    emoji_users.remove(user_id)
                    logger.info(f"🔄 Removed reaction {emoji} from message {message_id} by user {user_id}")
                else:
                    emoji_users.append(user_id)
                    logger.info(f"✅ Added reaction {emoji} to message {message_id} by user {user_id}")
                
                # Update reactions (remove emoji if no users left)
                if emoji_users:
                    reactions[emoji] = emoji_users
                else:
                    reactions.pop(emoji, None)
                
                # Update metadata
                metadata["reactions"] = reactions
                
                # Save back to database
                await conn.execute(
                    "UPDATE conversation_messages SET metadata_json = $1, updated_at = NOW() WHERE message_id = $2",
                    json.dumps(metadata),
                    message_id
                )
                
                logger.info(f"✅ Updated reactions for message {message_id}: {reactions}")
                return {"reactions": reactions}
                
        except Exception as e:
            logger.error(f"❌ Failed to add reaction to message {message_id}: {e}")
            raise