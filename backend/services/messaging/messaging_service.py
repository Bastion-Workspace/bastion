"""
Roosevelt's Messaging Service
Core service for user-to-user messaging operations

BULLY! A well-organized messaging system is like a well-organized cavalry charge!
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncpg

from config import settings
from .encryption_service import encryption_service
from utils.shared_db_pool import get_shared_db_pool

logger = logging.getLogger(__name__)


class MessagingService:
    """
    Service for managing chat rooms, messages, and user presence
    
    Handles:
    - Room creation and management
    - Message sending and retrieval
    - Emoji reactions
    - User presence tracking
    - Unread message counts
    """
    
    def __init__(self):
        self.db_pool = None
    
    async def initialize(self, shared_db_pool=None):
        """Initialize with database pool"""
        if shared_db_pool:
            self.db_pool = shared_db_pool
        else:
            self.db_pool = await get_shared_db_pool()
        logger.info("🏇 BULLY! Messaging service initialized!")
    
    async def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self.db_pool:
            await self.initialize()
    
    # =====================
    # ROOM OPERATIONS
    # =====================
    
    async def create_room(
        self, 
        creator_id: str, 
        participant_ids: List[str], 
        room_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new chat room
        
        Args:
            creator_id: User ID of room creator
            participant_ids: List of other participant user IDs
            room_name: Optional custom room name
        
        Returns:
            Dict with room details
        """
        await self._ensure_initialized()
        
        # Determine room type
        all_participants = [creator_id] + participant_ids
        room_type = 'direct' if len(all_participants) == 2 else 'group'
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get creator's role for RLS context
                creator_role_row = await conn.fetchrow("""
                    SELECT role FROM users WHERE user_id = $1
                """, creator_id)
                creator_role = creator_role_row["role"] if creator_role_row else "user"
                
                # Set user context for RLS (both user_id and role)
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", creator_id)
                await conn.execute("SELECT set_config('app.current_user_role', $1, false)", creator_role)
                
                # Create room
                room_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO chat_rooms (room_id, room_name, room_type, created_by)
                    VALUES ($1, $2, $3, $4)
                """, room_id, room_name, room_type, creator_id)
                
                # Add all participants (deduplicate to avoid duplicates)
                unique_participants = list(dict.fromkeys(all_participants))  # Preserves order
                for participant_id in unique_participants:
                    # Check if participant already exists (idempotent)
                    existing = await conn.fetchval("""
                        SELECT 1 FROM room_participants 
                        WHERE room_id = $1 AND user_id = $2
                    """, room_id, participant_id)
                    
                    if not existing:
                        await conn.execute("""
                            INSERT INTO room_participants (room_id, user_id)
                            VALUES ($1, $2)
                        """, room_id, participant_id)
                
                # Create encryption key for future E2EE (only if encryption is enabled)
                if encryption_service.is_encryption_enabled():
                    room_key = encryption_service.derive_room_key(room_id)
                    encrypted_key = encryption_service.encrypt_room_key(room_key)
                    if encrypted_key:  # Only insert if we successfully got an encrypted key
                        await conn.execute("""
                            INSERT INTO room_encryption_keys (room_id, encrypted_key)
                            VALUES ($1, $2)
                        """, room_id, encrypted_key)
                        logger.info(f"🔐 Created encryption key for room {room_id}")
                
                logger.info(f"✅ Created room {room_id} with {len(all_participants)} participants")
                
                return {
                    "room_id": room_id,
                    "room_name": room_name,
                    "room_type": room_type,
                    "created_by": creator_id,
                    "participant_ids": all_participants,
                    "created_at": datetime.utcnow().isoformat()
                }
        
        except Exception as e:
            logger.error(f"❌ Failed to create room: {e}")
            raise
    
    async def get_user_rooms(
        self, 
        user_id: str, 
        limit: int = 20,
        include_participants: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all rooms for a user, sorted by last message time
        
        Args:
            user_id: User ID
            limit: Maximum number of rooms to return
            include_participants: Whether to include participant details
        
        Returns:
            List of room dicts
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Get rooms
                rows = await conn.fetch("""
                    SELECT 
                        r.room_id, r.room_name, r.room_type, r.created_by,
                        r.created_at, r.last_message_at,
                        (SELECT COUNT(*) FROM chat_messages cm 
                         WHERE cm.room_id = r.room_id 
                         AND cm.deleted_at IS NULL) as message_count
                    FROM chat_rooms r
                    JOIN room_participants rp ON r.room_id = rp.room_id
                    WHERE rp.user_id = $1
                    ORDER BY r.last_message_at DESC
                    LIMIT $2
                """, user_id, limit)
                
                rooms = []
                for row in rows:
                    room_dict = dict(row)
                    
                    # Get participants if requested
                    if include_participants:
                        participants = await conn.fetch("""
                            SELECT 
                                u.user_id, u.username, u.display_name, u.avatar_url
                            FROM room_participants rp
                            JOIN users u ON rp.user_id = u.user_id
                            WHERE rp.room_id = $1
                        """, room_dict['room_id'])
                        room_dict['participants'] = [dict(p) for p in participants]
                        
                        # For direct rooms without custom name, use other person's name
                        if room_dict['room_type'] == 'direct' and not room_dict['room_name']:
                            other_participant = [p for p in room_dict['participants'] if p['user_id'] != user_id]
                            if other_participant:
                                room_dict['display_name'] = other_participant[0]['display_name'] or other_participant[0]['username']
                        else:
                            room_dict['display_name'] = room_dict['room_name'] or 'Unnamed Room'
                    
                    # Get unread count
                    unread_count = await conn.fetchval("""
                        SELECT COUNT(*)
                        FROM chat_messages cm
                        WHERE cm.room_id = $1
                        AND cm.created_at > (
                            SELECT COALESCE(last_read_at, '1970-01-01')
                            FROM room_participants
                            WHERE room_id = $1 AND user_id = $2
                        )
                        AND cm.sender_id != $2
                        AND cm.deleted_at IS NULL
                    """, room_dict['room_id'], user_id)
                    room_dict['unread_count'] = unread_count
                    
                    # Get notification settings for this user
                    notification_settings = await conn.fetchval("""
                        SELECT notification_settings
                        FROM room_participants
                        WHERE room_id = $1 AND user_id = $2
                    """, room_dict['room_id'], user_id)
                    room_dict['notification_settings'] = notification_settings or {}
                    
                    rooms.append(room_dict)
                
                return rooms
        
        except Exception as e:
            logger.error(f"❌ Failed to get user rooms: {e}")
            return []
    
    async def update_room_name(
        self, 
        room_id: str, 
        user_id: str, 
        new_name: str
    ) -> bool:
        """
        Update room name (must be participant)
        
        Args:
            room_id: Room UUID
            user_id: User making the update
            new_name: New room name
        
        Returns:
            True if successful
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Update room name
                result = await conn.execute("""
                    UPDATE chat_rooms
                    SET room_name = $1, updated_at = NOW()
                    WHERE room_id = $2
                    AND room_id IN (
                        SELECT room_id FROM room_participants WHERE user_id = $3
                    )
                """, new_name, room_id, user_id)
                
                if result == "UPDATE 1":
                    logger.info(f"✅ Updated room {room_id} name to '{new_name}'")
                    return True
                else:
                    logger.warning(f"⚠️ Failed to update room {room_id} - not a participant")
                    return False
        
        except Exception as e:
            logger.error(f"❌ Failed to update room name: {e}")
            return False
    
    async def update_notification_settings(
        self,
        room_id: str,
        user_id: str,
        settings: Dict[str, Any]
    ) -> bool:
        """
        Update notification settings for a user in a room
        
        Args:
            room_id: Room UUID
            user_id: User ID
            settings: Dictionary of notification settings (e.g., {"muted": True})
        
        Returns:
            True if successful
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Update notification settings
                import json
                result = await conn.execute("""
                    UPDATE room_participants
                    SET notification_settings = $1
                    WHERE room_id = $2 AND user_id = $3
                """, json.dumps(settings), room_id, user_id)
                
                if result == "UPDATE 1":
                    logger.info(f"✅ Updated notification settings for room {room_id}, user {user_id}")
                    return True
                else:
                    logger.warning(f"⚠️ Failed to update notification settings - not a participant")
                    return False
        
        except Exception as e:
            logger.error(f"❌ Failed to update notification settings: {e}")
            return False
    
    async def delete_room(
        self, 
        room_id: str, 
        user_id: str
    ) -> bool:
        """
        Delete a room (must be a participant)
        
        Args:
            room_id: Room UUID
            user_id: User requesting deletion
        
        Returns:
            True if successful
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Verify user is a participant
                is_participant = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM room_participants
                        WHERE room_id = $1 AND user_id = $2
                    )
                """, room_id, user_id)
                
                if not is_participant:
                    logger.warning(f"⚠️ User {user_id} not authorized to delete room {room_id}")
                    return False
                
                # Delete room attachments before deleting room
                from services.messaging.messaging_attachment_service import messaging_attachment_service
                await messaging_attachment_service.initialize(shared_db_pool=self.db_pool)
                await messaging_attachment_service.delete_room_attachments(room_id)
                
                # Delete room (cascades will handle participants, messages, etc.)
                result = await conn.execute("""
                    DELETE FROM chat_rooms
                    WHERE room_id = $1
                """, room_id)
                
                if result == "DELETE 1":
                    logger.info(f"🗑️ Deleted room {room_id}")
                    return True
                
                return False
        
        except Exception as e:
            logger.error(f"❌ Failed to delete room: {e}")
            return False
    
    async def add_participant(
        self,
        room_id: str,
        user_id: str,
        added_by: str,
        share_history: bool = False
    ) -> bool:
        """
        Add a participant to an existing room
        
        Args:
            room_id: Room UUID
            user_id: User ID to add
            added_by: User ID adding the participant
            share_history: Whether new participant can see message history
        
        Returns:
            True if successful
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", added_by)
                
                # Verify adding user is a participant
                is_participant = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM room_participants
                        WHERE room_id = $1 AND user_id = $2
                    )
                """, room_id, added_by)
                
                if not is_participant:
                    logger.warning(f"⚠️ User {added_by} not authorized to add participants to room {room_id}")
                    return False
                
                # Check if user is already a participant
                already_participant = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM room_participants
                        WHERE room_id = $1 AND user_id = $2
                    )
                """, room_id, user_id)
                
                if already_participant:
                    logger.warning(f"⚠️ User {user_id} is already a participant in room {room_id}")
                    return False
                
                # Add participant
                await conn.execute("""
                    INSERT INTO room_participants (room_id, user_id)
                    VALUES ($1, $2)
                """, room_id, user_id)
                
                # If not sharing history, mark all existing messages as read for this user
                if not share_history:
                    await conn.execute("""
                        UPDATE room_participants
                        SET last_read_at = NOW()
                        WHERE room_id = $1 AND user_id = $2
                    """, room_id, user_id)
                    logger.info(f"📭 Added {user_id} to room {room_id} (no history)")
                else:
                    logger.info(f"📬 Added {user_id} to room {room_id} (with history)")
                
                # Update room's updated_at timestamp
                await conn.execute("""
                    UPDATE chat_rooms
                    SET updated_at = NOW()
                    WHERE room_id = $1
                """, room_id)
                
                return True
        
        except Exception as e:
            logger.error(f"❌ Failed to add participant: {e}")
            return False
    
    # =====================
    # MESSAGE OPERATIONS
    # =====================
    
    async def send_message(
        self,
        room_id: str,
        sender_id: str,
        content: str,
        message_type: str = 'text',
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Send a message to a room
        
        Args:
            room_id: Room UUID
            sender_id: User ID of sender
            content: Message content
            message_type: 'text', 'ai_share', or 'system'
            metadata: Optional metadata (for AI shares, etc.)
        
        Returns:
            Message dict or None if failed
        """
        await self._ensure_initialized()
        
        # Validate message length
        if len(content) > settings.MESSAGE_MAX_LENGTH:
            logger.warning(f"⚠️ Message exceeds max length ({len(content)} > {settings.MESSAGE_MAX_LENGTH})")
            content = content[:settings.MESSAGE_MAX_LENGTH]
        
        # Encrypt content if enabled
        encrypted_content = encryption_service.encrypt_message(content)
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", sender_id)
                
                # Insert message
                message_id = str(uuid.uuid4())
                # Convert metadata to JSON string if present, or None
                metadata_json = json.dumps(metadata) if metadata else None
                
                row = await conn.fetchrow("""
                    INSERT INTO chat_messages 
                    (message_id, room_id, sender_id, message_content, message_type, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    RETURNING message_id, created_at
                """, message_id, room_id, sender_id, encrypted_content, message_type, metadata_json)
                
                # Update room's last_message_at
                await conn.execute("""
                    UPDATE chat_rooms
                    SET last_message_at = NOW()
                    WHERE room_id = $1
                """, room_id)
                
                logger.info(f"✅ Message sent to room {room_id} by {sender_id}")
                
                return {
                    "message_id": message_id,
                    "room_id": room_id,
                    "sender_id": sender_id,
                    "content": content,  # Return decrypted for immediate display
                    "message_type": message_type,
                    "metadata": metadata,  # Return original metadata dict
                    "created_at": row['created_at'].isoformat()
                }
        
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return None
    
    async def get_room_messages(
        self,
        room_id: str,
        user_id: str,
        limit: int = 50,
        before_message_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a room (paginated)
        
        Args:
            room_id: Room UUID
            user_id: User requesting messages
            limit: Maximum messages to return
            before_message_id: For pagination, get messages before this ID
        
        Returns:
            List of message dicts
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Build query based on pagination
                if before_message_id:
                    rows = await conn.fetch("""
                        SELECT 
                            m.message_id, m.sender_id, m.message_content, 
                            m.message_type, m.metadata, m.created_at,
                            u.username, u.display_name, u.avatar_url
                        FROM chat_messages m
                        JOIN users u ON m.sender_id = u.user_id
                        WHERE m.room_id = $1
                        AND m.deleted_at IS NULL
                        AND m.created_at < (
                            SELECT created_at FROM chat_messages WHERE message_id = $2
                        )
                        ORDER BY m.created_at DESC
                        LIMIT $3
                    """, room_id, before_message_id, limit)
                else:
                    rows = await conn.fetch("""
                        SELECT 
                            m.message_id, m.sender_id, m.message_content, 
                            m.message_type, m.metadata, m.created_at,
                            u.username, u.display_name, u.avatar_url
                        FROM chat_messages m
                        JOIN users u ON m.sender_id = u.user_id
                        WHERE m.room_id = $1
                        AND m.deleted_at IS NULL
                        ORDER BY m.created_at DESC
                        LIMIT $2
                    """, room_id, limit)
                
                messages = []
                for row in rows:
                    msg_dict = dict(row)
                    # Decrypt content
                    msg_dict['content'] = encryption_service.decrypt_message(msg_dict['message_content'])
                    del msg_dict['message_content']  # Remove encrypted version
                    
                    # Get reactions
                    reactions = await conn.fetch("""
                        SELECT emoji, user_id, reaction_id
                        FROM message_reactions
                        WHERE message_id = $1
                    """, msg_dict['message_id'])
                    msg_dict['reactions'] = [dict(r) for r in reactions]
                    
                    messages.append(msg_dict)
                
                # Reverse to get chronological order
                messages.reverse()
                
                # Mark as read
                await conn.execute("""
                    UPDATE room_participants
                    SET last_read_at = NOW()
                    WHERE room_id = $1 AND user_id = $2
                """, room_id, user_id)
                
                return messages
        
        except Exception as e:
            logger.error(f"❌ Failed to get room messages: {e}")
            return []
    
    async def delete_message(
        self,
        message_id: str,
        user_id: str,
        delete_for: str = 'me'
    ) -> bool:
        """
        Soft delete a message
        
        Args:
            message_id: Message UUID
            user_id: User requesting deletion
            delete_for: 'me' or 'everyone'
        
        Returns:
            True if successful
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                if delete_for == 'everyone':
                    # Only sender can delete for everyone
                    result = await conn.execute("""
                        UPDATE chat_messages
                        SET deleted_at = NOW()
                        WHERE message_id = $1 AND sender_id = $2
                    """, message_id, user_id)
                else:
                    # For now, we don't support per-user deletion
                    # Just mark as deleted if user is sender
                    result = await conn.execute("""
                        UPDATE chat_messages
                        SET deleted_at = NOW()
                        WHERE message_id = $1 AND sender_id = $2
                    """, message_id, user_id)
                
                return result == "UPDATE 1"
        
        except Exception as e:
            logger.error(f"❌ Failed to delete message: {e}")
            return False
    
    # =====================
    # REACTION OPERATIONS
    # =====================
    
    async def add_reaction(
        self,
        message_id: str,
        user_id: str,
        emoji: str
    ) -> Optional[str]:
        """
        Add emoji reaction to a message
        
        Args:
            message_id: Message UUID
            user_id: User adding reaction
            emoji: Emoji character
        
        Returns:
            Reaction ID or None if failed
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                reaction_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO message_reactions (reaction_id, message_id, user_id, emoji)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (message_id, user_id, emoji) DO NOTHING
                """, reaction_id, message_id, user_id, emoji)
                
                return reaction_id
        
        except Exception as e:
            logger.error(f"❌ Failed to add reaction: {e}")
            return None
    
    async def remove_reaction(
        self,
        reaction_id: str,
        user_id: str
    ) -> bool:
        """
        Remove an emoji reaction
        
        Args:
            reaction_id: Reaction UUID
            user_id: User removing reaction
        
        Returns:
            True if successful
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                result = await conn.execute("""
                    DELETE FROM message_reactions
                    WHERE reaction_id = $1 AND user_id = $2
                """, reaction_id, user_id)
                
                return result == "DELETE 1"
        
        except Exception as e:
            logger.error(f"❌ Failed to remove reaction: {e}")
            return False
    
    # =====================
    # PRESENCE OPERATIONS
    # =====================
    
    async def update_user_presence(
        self,
        user_id: str,
        status: str = 'online',
        status_message: Optional[str] = None
    ) -> bool:
        """
        Update user presence status
        
        Args:
            user_id: User ID
            status: 'online', 'offline', or 'away'
            status_message: Optional status message
        
        Returns:
            True if successful
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                await conn.execute("""
                    INSERT INTO user_presence (user_id, status, last_seen_at, status_message)
                    VALUES ($1, $2, NOW(), $3)
                    ON CONFLICT (user_id) DO UPDATE
                    SET status = $2, last_seen_at = NOW(), status_message = $3
                """, user_id, status, status_message)
                
                return True
        
        except Exception as e:
            logger.error(f"❌ Failed to update user presence: {e}")
            return False
    
    async def get_user_presence(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single user's presence
        
        Args:
            user_id: User ID
        
        Returns:
            Presence dict or None
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT user_id, status, last_seen_at, status_message
                    FROM user_presence
                    WHERE user_id = $1
                """, user_id)
                
                if row:
                    return dict(row)
                return None
        
        except Exception as e:
            logger.error(f"❌ Failed to get user presence: {e}")
            return None
    
    async def get_room_participant_presence(
        self, 
        room_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get presence for all participants in a room
        
        Args:
            room_id: Room UUID
        
        Returns:
            Dict mapping user_id to presence info
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        u.user_id, u.username, u.display_name,
                        COALESCE(p.status, 'offline') as status,
                        p.last_seen_at, p.status_message
                    FROM room_participants rp
                    JOIN users u ON rp.user_id = u.user_id
                    LEFT JOIN user_presence p ON u.user_id = p.user_id
                    WHERE rp.room_id = $1
                """, room_id)
                
                presence_map = {}
                for row in rows:
                    presence_map[row['user_id']] = dict(row)
                
                return presence_map
        
        except Exception as e:
            logger.error(f"❌ Failed to get room participant presence: {e}")
            return {}
    
    async def cleanup_stale_presence(self) -> int:
        """
        Mark users as offline if they haven't updated presence recently
        
        Returns:
            Number of users marked offline
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                threshold = datetime.utcnow() - timedelta(seconds=settings.PRESENCE_OFFLINE_THRESHOLD_SECONDS)
                
                result = await conn.execute("""
                    UPDATE user_presence
                    SET status = 'offline'
                    WHERE status != 'offline'
                    AND last_seen_at < $1
                """, threshold)
                
                # Extract count from "UPDATE N" result
                count = int(result.split()[-1]) if result.startswith("UPDATE") else 0
                
                if count > 0:
                    logger.info(f"🧹 Marked {count} users as offline due to inactivity")
                
                return count
        
        except Exception as e:
            logger.error(f"❌ Failed to cleanup stale presence: {e}")
            return 0
    
    async def get_unread_counts(self, user_id: str) -> Dict[str, int]:
        """
        Get unread message counts for all user's rooms
        
        Args:
            user_id: User ID
        
        Returns:
            Dict mapping room_id to unread count
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                rows = await conn.fetch("""
                    SELECT 
                        rp.room_id,
                        COUNT(cm.message_id) as unread_count
                    FROM room_participants rp
                    LEFT JOIN chat_messages cm ON cm.room_id = rp.room_id
                        AND cm.created_at > COALESCE(rp.last_read_at, '1970-01-01')
                        AND cm.sender_id != $1
                        AND cm.deleted_at IS NULL
                    WHERE rp.user_id = $1
                    GROUP BY rp.room_id
                """, user_id)
                
                return {row['room_id']: row['unread_count'] for row in rows}
        
        except Exception as e:
            logger.error(f"❌ Failed to get unread counts: {e}")
            return {}


# Global messaging service instance
messaging_service = MessagingService()

