"""
Conversation Sharing Service - Multi-User Thread Support
Handles sharing conversations with full checkpoint replication
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
import asyncpg
from config import settings
from services.orchestrator_utils import normalize_thread_id

logger = logging.getLogger(__name__)


class ConversationSharingService:
    """Service for managing conversation sharing with checkpoint replication"""
    
    def __init__(self):
        self.db_pool = None
        logger.info("Initializing ConversationSharingService")
    
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
    
    async def share_conversation(
        self,
        conversation_id: str,
        shared_by_user_id: str,
        shared_with_user_id: str,
        share_type: str = "read",
        expires_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Share a conversation with another user by replicating checkpoints
        
        Args:
            conversation_id: The conversation to share
            shared_by_user_id: User sharing the conversation
            shared_with_user_id: User receiving the share
            share_type: Permission level (read, comment, edit)
            expires_at: Optional expiration date
            
        Returns:
            Dictionary with share_id and success status
        """
        try:
            # Validate conversation ownership
            if not await self._validate_conversation_ownership(conversation_id, shared_by_user_id):
                raise ValueError(f"User {shared_by_user_id} does not own conversation {conversation_id}")
            
            # Check if share already exists
            existing_share = await self._get_existing_share(
                conversation_id, shared_by_user_id, shared_with_user_id
            )
            if existing_share:
                logger.info(f"Share already exists: {existing_share['share_id']}")
                return {
                    "success": True,
                    "share_id": existing_share["share_id"],
                    "message": "Conversation already shared with this user"
                }
            
            # Get source thread_id
            source_thread_id = normalize_thread_id(shared_by_user_id, conversation_id)
            
            # Get target thread_id (for recipient)
            target_thread_id = normalize_thread_id(shared_with_user_id, conversation_id)
            
            # Replicate checkpoints
            replication_success = await self.replicate_checkpoints(
                source_thread_id=source_thread_id,
                target_thread_id=target_thread_id,
                target_user_id=shared_with_user_id,
                conversation_id=conversation_id
            )
            
            if not replication_success:
                raise Exception("Failed to replicate checkpoints")
            
            # Create share record
            share_id = str(uuid.uuid4())
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", shared_by_user_id)
                
                await conn.execute("""
                    INSERT INTO conversation_shares 
                    (share_id, conversation_id, shared_by_user_id, shared_with_user_id, share_type, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, share_id, conversation_id, shared_by_user_id, shared_with_user_id, share_type, expires_at)
            
            logger.info(f"Successfully shared conversation {conversation_id} with user {shared_with_user_id}")
            
            return {
                "success": True,
                "share_id": share_id,
                "message": "Conversation shared successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to share conversation: {e}")
            raise
    
    async def replicate_checkpoints(
        self,
        source_thread_id: str,
        target_thread_id: str,
        target_user_id: str,
        conversation_id: str
    ) -> bool:
        """
        Replicate all checkpoint data from source to target thread
        
        This copies:
        - checkpoints table (conversation state)
        - checkpoint_blobs table (large objects)
        - checkpoint_writes table (pending writes)
        
        Preserves checkpoint_id hierarchy for state continuity.
        """
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                # Set user context
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", target_user_id)
                
                # Step 1: Replicate checkpoints
                checkpoint_rows = await conn.fetch("""
                    SELECT checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                    FROM checkpoints
                    WHERE thread_id = $1
                    ORDER BY created_at ASC
                """, source_thread_id)
                
                logger.info(f"Replicating {len(checkpoint_rows)} checkpoints from {source_thread_id} to {target_thread_id}")
                
                for row in checkpoint_rows:
                    await conn.execute("""
                        INSERT INTO checkpoints 
                        (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO NOTHING
                    """, target_thread_id, row['checkpoint_ns'], row['checkpoint_id'], 
                        row['parent_checkpoint_id'], row['type'], row['checkpoint'], row['metadata'])
                
                # Step 2: Replicate checkpoint blobs
                blob_rows = await conn.fetch("""
                    SELECT checkpoint_id, channel, version, type, blob
                    FROM checkpoint_blobs
                    WHERE thread_id = $1
                """, source_thread_id)
                
                logger.info(f"Replicating {len(blob_rows)} checkpoint blobs")
                
                for row in blob_rows:
                    await conn.execute("""
                        INSERT INTO checkpoint_blobs 
                        (thread_id, checkpoint_ns, checkpoint_id, channel, version, type, blob)
                        VALUES ($1, '', $2, $3, $4, $5, $6)
                        ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
                    """, target_thread_id, row['checkpoint_id'], row['channel'], 
                        row['version'], row['type'], row['blob'])
                
                # Step 3: Replicate checkpoint writes (if any)
                write_rows = await conn.fetch("""
                    SELECT channel, channel_version, checkpoint_id, task_id, checkpoint_ns
                    FROM checkpoint_writes
                    WHERE thread_id = $1
                """, source_thread_id)
                
                if write_rows:
                    logger.info(f"Replicating {len(write_rows)} checkpoint writes")
                    for row in write_rows:
                        await conn.execute("""
                            INSERT INTO checkpoint_writes 
                            (thread_id, checkpoint_ns, channel, channel_version, checkpoint_id, task_id)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (thread_id, checkpoint_ns, channel, channel_version) DO NOTHING
                        """, target_thread_id, row['checkpoint_ns'] or '', row['channel'], 
                            row['channel_version'], row['checkpoint_id'], row['task_id'])
                
                logger.info(f"Successfully replicated all checkpoint data to {target_thread_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to replicate checkpoints: {e}")
            return False
    
    async def unshare_conversation(
        self,
        conversation_id: str,
        share_id: str,
        user_id: str
    ) -> bool:
        """
        Remove a share (revoke access)
        
        Note: This does NOT delete the recipient's checkpoint history.
        They keep their copy but lose access to future updates.
        """
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Verify user has permission to remove this share
                share = await conn.fetchrow("""
                    SELECT shared_by_user_id, shared_with_user_id
                    FROM conversation_shares
                    WHERE share_id = $1 AND conversation_id = $2
                """, share_id, conversation_id)
                
                if not share:
                    raise ValueError("Share not found")
                
                # Only owner or the shared user can remove
                if share['shared_by_user_id'] != user_id and share['shared_with_user_id'] != user_id:
                    # Check if user is owner
                    owner = await conn.fetchrow("""
                        SELECT user_id FROM conversations
                        WHERE conversation_id = $1
                    """, conversation_id)
                    
                    if not owner or owner['user_id'] != user_id:
                        raise ValueError("Insufficient permissions to remove share")
                
                # Delete the share record
                await conn.execute("""
                    DELETE FROM conversation_shares
                    WHERE share_id = $1 AND conversation_id = $2
                """, share_id, conversation_id)
                
                logger.info(f"Removed share {share_id} for conversation {conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unshare conversation: {e}")
            raise
    
    async def get_conversation_shares(
        self,
        conversation_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get all shares for a conversation"""
        try:
            # Verify user has access
            if not await self._has_conversation_access(conversation_id, user_id, "read"):
                raise ValueError("User does not have access to this conversation")
            
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                rows = await conn.fetch("""
                    SELECT 
                        cs.share_id,
                        cs.conversation_id,
                        cs.shared_by_user_id,
                        cs.shared_with_user_id,
                        cs.share_type,
                        cs.is_public,
                        cs.expires_at,
                        cs.created_at,
                        u.username,
                        u.email
                    FROM conversation_shares cs
                    LEFT JOIN users u ON cs.shared_with_user_id = u.user_id
                    WHERE cs.conversation_id = $1
                    AND (cs.expires_at IS NULL OR cs.expires_at > NOW())
                    ORDER BY cs.created_at DESC
                """, conversation_id)
                
                shares = []
                for row in rows:
                    shares.append({
                        "share_id": row['share_id'],
                        "conversation_id": row['conversation_id'],
                        "shared_by_user_id": row['shared_by_user_id'],
                        "shared_with_user_id": row['shared_with_user_id'],
                        "share_type": row['share_type'],
                        "is_public": row['is_public'],
                        "expires_at": row['expires_at'].isoformat() if row['expires_at'] else None,
                        "created_at": row['created_at'].isoformat(),
                        "username": row['username'],
                        "email": row['email']
                    })
                
                return shares
                
        except Exception as e:
            logger.error(f"Failed to get conversation shares: {e}")
            raise
    
    async def get_shared_conversations(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all conversations shared with the user"""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                rows = await conn.fetch("""
                    SELECT 
                        c.conversation_id,
                        c.user_id as owner_id,
                        c.title,
                        c.description,
                        c.is_pinned,
                        c.is_archived,
                        c.tags,
                        c.metadata_json,
                        c.message_count,
                        c.last_message_at,
                        c.created_at,
                        c.updated_at,
                        cs.share_id,
                        cs.share_type,
                        cs.shared_by_user_id,
                        cs.created_at as shared_at,
                        u.username as owner_username,
                        u.email as owner_email
                    FROM conversation_shares cs
                    JOIN conversations c ON cs.conversation_id = c.conversation_id
                    LEFT JOIN users u ON c.user_id = u.user_id
                    WHERE cs.shared_with_user_id = $1
                    AND (cs.expires_at IS NULL OR cs.expires_at > NOW())
                    ORDER BY cs.created_at DESC
                    LIMIT $2 OFFSET $3
                """, user_id, limit, skip)
                
                conversations = []
                for row in rows:
                    conversations.append({
                        "conversation_id": row['conversation_id'],
                        "user_id": row['owner_id'],
                        "title": row['title'],
                        "description": row['description'],
                        "is_pinned": row['is_pinned'],
                        "is_archived": row['is_archived'],
                        "tags": row['tags'] or [],
                        "metadata_json": row['metadata_json'] or {},
                        "message_count": row['message_count'] or 0,
                        "last_message_at": row['last_message_at'].isoformat() if row['last_message_at'] else None,
                        "created_at": row['created_at'].isoformat(),
                        "updated_at": row['updated_at'].isoformat(),
                        "share_id": row['share_id'],
                        "share_type": row['share_type'],
                        "shared_by_user_id": row['shared_by_user_id'],
                        "shared_at": row['shared_at'].isoformat(),
                        "owner_username": row['owner_username'],
                        "owner_email": row['owner_email'],
                        "is_shared": True
                    })
                
                return conversations
                
        except Exception as e:
            logger.error(f"Failed to get shared conversations: {e}")
            raise
    
    async def get_conversation_participants(
        self,
        conversation_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get all participants (owner + shared users) for a conversation"""
        try:
            # Verify user has access
            if not await self._has_conversation_access(conversation_id, user_id, "read"):
                raise ValueError("User does not have access to this conversation")
            
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Get owner
                owner = await conn.fetchrow("""
                    SELECT 
                        c.user_id,
                        u.username,
                        u.email,
                        u.display_name
                    FROM conversations c
                    LEFT JOIN users u ON c.user_id = u.user_id
                    WHERE c.conversation_id = $1
                """, conversation_id)
                
                participants = []
                if owner:
                    participants.append({
                        "user_id": owner['user_id'],
                        "username": owner['username'],
                        "email": owner['email'],
                        "display_name": owner['display_name'],
                        "share_type": "edit",  # Owner has full permissions
                        "is_owner": True
                    })
                
                # Get shared users
                shares = await conn.fetch("""
                    SELECT 
                        cs.shared_with_user_id,
                        cs.share_type,
                        u.username,
                        u.email,
                        u.display_name
                    FROM conversation_shares cs
                    LEFT JOIN users u ON cs.shared_with_user_id = u.user_id
                    WHERE cs.conversation_id = $1
                    AND (cs.expires_at IS NULL OR cs.expires_at > NOW())
                """, conversation_id)
                
                for share in shares:
                    participants.append({
                        "user_id": share['shared_with_user_id'],
                        "username": share['username'],
                        "email": share['email'],
                        "display_name": share['display_name'],
                        "share_type": share['share_type'],
                        "is_owner": False
                    })
                
                return participants
                
        except Exception as e:
            logger.error(f"Failed to get conversation participants: {e}")
            raise
    
    async def update_share_permissions(
        self,
        conversation_id: str,
        share_id: str,
        user_id: str,
        new_share_type: str
    ) -> bool:
        """Update share permissions (read/comment/edit)"""
        try:
            # Verify user is owner
            if not await self._validate_conversation_ownership(conversation_id, user_id):
                raise ValueError("Only conversation owner can update share permissions")
            
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                await conn.execute("""
                    UPDATE conversation_shares
                    SET share_type = $1
                    WHERE share_id = $2 AND conversation_id = $3
                """, new_share_type, share_id, conversation_id)
                
                logger.info(f"Updated share {share_id} permissions to {new_share_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update share permissions: {e}")
            raise
    
    async def _validate_conversation_ownership(
        self,
        conversation_id: str,
        user_id: str
    ) -> bool:
        """Check if user owns the conversation"""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                row = await conn.fetchrow("""
                    SELECT user_id FROM conversations
                    WHERE conversation_id = $1
                """, conversation_id)
                
                return row and row['user_id'] == user_id
                
        except Exception as e:
            logger.error(f"Failed to validate ownership: {e}")
            return False
    
    async def _has_conversation_access(
        self,
        conversation_id: str,
        user_id: str,
        required_permission: str = "read"
    ) -> bool:
        """
        Check if user has access to conversation
        
        Permission hierarchy: read < comment < edit
        """
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Check if owner
                owner = await conn.fetchrow("""
                    SELECT user_id FROM conversations
                    WHERE conversation_id = $1
                """, conversation_id)
                
                if owner and owner['user_id'] == user_id:
                    return True
                
                # Check if shared
                share = await conn.fetchrow("""
                    SELECT share_type, expires_at
                    FROM conversation_shares
                    WHERE conversation_id = $1
                    AND shared_with_user_id = $2
                    AND (expires_at IS NULL OR expires_at > NOW())
                """, conversation_id, user_id)
                
                if not share:
                    return False
                
                # Check permission level
                permission_levels = {"read": 1, "comment": 2, "edit": 3}
                required_level = permission_levels.get(required_permission, 1)
                user_level = permission_levels.get(share['share_type'], 0)
                
                return user_level >= required_level
                
        except Exception as e:
            logger.error(f"Failed to check access: {e}")
            return False
    
    async def _get_existing_share(
        self,
        conversation_id: str,
        shared_by_user_id: str,
        shared_with_user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Check if share already exists"""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", shared_by_user_id)
                
                row = await conn.fetchrow("""
                    SELECT share_id, share_type, expires_at
                    FROM conversation_shares
                    WHERE conversation_id = $1
                    AND shared_by_user_id = $2
                    AND shared_with_user_id = $3
                    AND (expires_at IS NULL OR expires_at > NOW())
                """, conversation_id, shared_by_user_id, shared_with_user_id)
                
                if row:
                    return {
                        "share_id": row['share_id'],
                        "share_type": row['share_type'],
                        "expires_at": row['expires_at']
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to check existing share: {e}")
            return None


# Global instance
_conversation_sharing_service = None


async def get_conversation_sharing_service() -> ConversationSharingService:
    """Get or create global conversation sharing service instance"""
    global _conversation_sharing_service
    if _conversation_sharing_service is None:
        _conversation_sharing_service = ConversationSharingService()
    return _conversation_sharing_service

