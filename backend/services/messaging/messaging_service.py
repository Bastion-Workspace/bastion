"""
Messaging service
Core service for user-to-user messaging operations

Room-based messaging, presence, and delivery helpers.
"""

import logging
import json
import uuid
from datetime import datetime, timedelta, timezone
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
        logger.info("Messaging service initialized")
    
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
                
                # Get full participant details for response
                participants = await conn.fetch("""
                    SELECT 
                        u.user_id, u.username, u.display_name, u.avatar_url
                    FROM room_participants rp
                    JOIN users u ON rp.user_id = u.user_id
                    WHERE rp.room_id = $1
                """, room_id)
                
                participant_list = [dict(p) for p in participants]
                
                # Set display_name for direct rooms
                display_name = room_name
                if room_type == 'direct' and not room_name:
                    # Find the participant that isn't the creator
                    other_participant = next((p for p in participant_list if p['user_id'] != creator_id), None)
                    if other_participant:
                        display_name = other_participant.get('display_name') or other_participant.get('username')
                        logger.info(f"🏷️ Set direct room display_name to '{display_name}' (other participant)")
                    else:
                        logger.warning(f"⚠️ No other participant found for direct room {room_id}")
                
                if not display_name:
                    display_name = 'Unnamed Room'
                    logger.info(f"🏷️ Fallback display_name for room {room_id}: '{display_name}'")
                
                return {
                    "room_id": room_id,
                    "room_name": room_name,
                    "room_type": room_type,
                    "created_by": creator_id,
                    "participant_ids": all_participants,
                    "participants": participant_list,
                    "display_name": display_name,
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
                        r.federation_metadata,
                        p.status AS federation_peer_status,
                        p.peer_url AS federation_peer_url,
                        p.display_name AS federation_peer_display_name,
                        (SELECT COUNT(*) FROM chat_messages cm 
                         WHERE cm.room_id = r.room_id 
                         AND cm.deleted_at IS NULL) as message_count
                    FROM chat_rooms r
                    JOIN room_participants rp ON r.room_id = rp.room_id
                    LEFT JOIN federation_peers p
                      ON r.room_type = 'federated'
                     AND p.peer_id = (r.federation_metadata->>'peer_id')::uuid
                    WHERE rp.user_id = $1
                    ORDER BY r.last_message_at DESC
                    LIMIT $2
                """, user_id, limit)
                
                rooms = []
                for row in rows:
                    room_dict = dict(row)
                    fm = room_dict.get("federation_metadata")
                    if isinstance(fm, str):
                        try:
                            room_dict["federation_metadata"] = json.loads(fm)
                        except Exception:
                            room_dict["federation_metadata"] = {}
                    
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
                        logger.info(f"👥 Room {room_dict['room_id']} has {len(room_dict['participants'])} visible participants")
                        
                        # For direct rooms without custom name, use other person's name
                        if room_dict['room_type'] == 'direct' and not room_dict['room_name']:
                            other_participant = [p for p in room_dict['participants'] if p['user_id'] != user_id]
                            if other_participant:
                                room_dict['display_name'] = other_participant[0].get('display_name') or other_participant[0].get('username') or 'Unknown User'
                                logger.info(f"🏷️ Set direct room {room_dict['room_id']} display_name to '{room_dict['display_name']}'")
                            else:
                                room_dict['display_name'] = 'Unnamed Room'
                                logger.warning(f"⚠️ No other participant found for direct room {room_dict['room_id']}")
                        else:
                            room_dict['display_name'] = room_dict['room_name'] or 'Unnamed Room'
                            logger.info(f"🏷️ Room {room_dict['room_id']} display_name set to '{room_dict['display_name']}'")
                    
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
                        AND cm.sender_id IS DISTINCT FROM $2
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
        metadata: Optional[Dict[str, Any]] = None,
        reply_to_message_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send a message to a room
        
        Args:
            room_id: Room UUID
            sender_id: User ID of sender
            content: Message content
            message_type: 'text', 'ai_share', or 'system'
            metadata: Optional metadata (for AI shares, mentions, etc.)
            reply_to_message_id: Optional message ID being replied to
        
        Returns:
            Message dict or None if failed
        """
        await self._ensure_initialized()
        
        if len(content) > settings.MESSAGE_MAX_LENGTH:
            content = content[:settings.MESSAGE_MAX_LENGTH]
        
        encrypted_content = encryption_service.encrypt_message(content)
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", sender_id)
                
                message_id = str(uuid.uuid4())
                metadata_json = json.dumps(metadata) if metadata else None
                
                row = await conn.fetchrow("""
                    INSERT INTO chat_messages 
                    (message_id, room_id, sender_id, message_content, message_type, metadata, reply_to_message_id)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                    RETURNING message_id, created_at
                """, message_id, room_id, sender_id, encrypted_content, message_type, metadata_json,
                   reply_to_message_id)
                
                await conn.execute("""
                    UPDATE chat_rooms SET last_message_at = NOW() WHERE room_id = $1
                """, room_id)

                # Fetch sender info for the response
                sender_row = await conn.fetchrow(
                    "SELECT username, display_name, avatar_url FROM users WHERE user_id = $1",
                    sender_id,
                )
                
                result = {
                    "message_id": message_id,
                    "room_id": room_id,
                    "sender_id": sender_id,
                    "content": content,
                    "message_type": message_type,
                    "metadata": metadata,
                    "reply_to_message_id": reply_to_message_id,
                    "created_at": row['created_at'].isoformat(),
                }
                if sender_row:
                    result["username"] = sender_row["username"]
                    result["display_name"] = sender_row["display_name"]
                    result["avatar_url"] = sender_row["avatar_url"]

                fed_res = None
                try:
                    fed_res = await self._maybe_deliver_federated(room_id, result)
                    if fed_res:
                        result["federation_delivery"] = fed_res
                except Exception as fed_e:
                    logger.warning("Federated outbound delivery failed (non-fatal): %s", fed_e)

                rt = await conn.fetchrow(
                    "SELECT room_type FROM chat_rooms WHERE room_id = $1::uuid", room_id
                )
                if rt and (rt.get("room_type") or "") == "federated":
                    if fed_res and fed_res.get("ok"):
                        st = "delivered"
                    elif fed_res and (fed_res.get("reason") or "") == "peer_suspended":
                        st = "peer_suspended"
                    else:
                        st = "failed"
                    await conn.execute(
                        """
                        UPDATE chat_messages
                        SET federation_delivery_status = $2
                        WHERE message_id = $1::uuid
                        """,
                        message_id,
                        st,
                    )
                    result["federation_delivery_status"] = st

                return result
        
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None

    async def _maybe_deliver_federated(
        self, room_id: str, message_dict: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """If room is federated, sign and deliver the message to the peer (HTTP or outbox)."""
        if not getattr(settings, "FEDERATION_ENABLED", False):
            return None
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT room_type FROM chat_rooms WHERE room_id = $1::uuid
                    """,
                    room_id,
                )
            if not row or (row.get("room_type") or "") != "federated":
                return None
            from services.federation_message_service import federation_message_service

            return await federation_message_service.deliver_outbound_message(room_id, message_dict)
        except Exception as e:
            logger.warning("_maybe_deliver_federated: %s", e)
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
                            m.message_id, m.sender_id, m.federated_sender_id, m.message_content, 
                            m.message_type, m.metadata, m.created_at,
                            m.federation_delivery_status,
                            m.reply_to_message_id, m.is_edited, m.edited_at,
                            u.username, u.display_name, u.avatar_url,
                            fu.federated_address AS federated_address,
                            fu.display_name AS federated_display_name,
                            fu.avatar_url AS federated_avatar_url,
                            COALESCE(ru.display_name, rfu.display_name) AS reply_sender_name,
                            rm.message_content AS reply_content
                        FROM chat_messages m
                        LEFT JOIN users u ON m.sender_id = u.user_id
                        LEFT JOIN federated_users fu ON m.federated_sender_id = fu.federated_user_id
                        LEFT JOIN chat_messages rm ON m.reply_to_message_id = rm.message_id
                        LEFT JOIN users ru ON rm.sender_id = ru.user_id
                        LEFT JOIN federated_users rfu ON rm.federated_sender_id = rfu.federated_user_id
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
                            m.message_id, m.sender_id, m.federated_sender_id, m.message_content, 
                            m.message_type, m.metadata, m.created_at,
                            m.federation_delivery_status,
                            m.reply_to_message_id, m.is_edited, m.edited_at,
                            u.username, u.display_name, u.avatar_url,
                            fu.federated_address AS federated_address,
                            fu.display_name AS federated_display_name,
                            fu.avatar_url AS federated_avatar_url,
                            COALESCE(ru.display_name, rfu.display_name) AS reply_sender_name,
                            rm.message_content AS reply_content
                        FROM chat_messages m
                        LEFT JOIN users u ON m.sender_id = u.user_id
                        LEFT JOIN federated_users fu ON m.federated_sender_id = fu.federated_user_id
                        LEFT JOIN chat_messages rm ON m.reply_to_message_id = rm.message_id
                        LEFT JOIN users ru ON rm.sender_id = ru.user_id
                        LEFT JOIN federated_users rfu ON rm.federated_sender_id = rfu.federated_user_id
                        WHERE m.room_id = $1
                        AND m.deleted_at IS NULL
                        ORDER BY m.created_at DESC
                        LIMIT $2
                    """, room_id, limit)
                
                messages = []
                for row in rows:
                    msg_dict = dict(row)
                    if msg_dict.get("federated_sender_id"):
                        msg_dict["display_name"] = (
                            msg_dict.get("federated_display_name")
                            or msg_dict.get("federated_address")
                            or msg_dict.get("display_name")
                        )
                        msg_dict["username"] = msg_dict.get("federated_address") or msg_dict.get("username")
                        msg_dict["avatar_url"] = msg_dict.get("federated_avatar_url") or msg_dict.get("avatar_url")
                        msg_dict["is_federated"] = True
                    for k in (
                        "federated_display_name",
                        "federated_avatar_url",
                    ):
                        msg_dict.pop(k, None)
                    msg_dict['content'] = encryption_service.decrypt_message(msg_dict['message_content'])
                    del msg_dict['message_content']
                    
                    if msg_dict.get('reply_content'):
                        msg_dict['reply_preview'] = {
                            'sender_name': msg_dict.pop('reply_sender_name', None),
                            'content': encryption_service.decrypt_message(msg_dict.pop('reply_content'))[:200],
                        }
                    else:
                        msg_dict.pop('reply_sender_name', None)
                        msg_dict.pop('reply_content', None)
                    
                    reactions = await conn.fetch("""
                        SELECT emoji, user_id, reaction_id, federated_user_id
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
    
    async def edit_message(
        self,
        message_id: str,
        user_id: str,
        new_content: str,
    ) -> Optional[Dict[str, Any]]:
        """Edit a message (sender only). Returns updated message dict."""
        await self._ensure_initialized()
        if len(new_content) > settings.MESSAGE_MAX_LENGTH:
            new_content = new_content[:settings.MESSAGE_MAX_LENGTH]
        encrypted = encryption_service.encrypt_message(new_content)
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                row = await conn.fetchrow("""
                    UPDATE chat_messages
                    SET message_content = $1, is_edited = true, edited_at = NOW()
                    WHERE message_id = $2 AND sender_id = $3 AND deleted_at IS NULL
                    RETURNING message_id, room_id, sender_id, message_type, metadata, created_at, edited_at
                """, encrypted, message_id, user_id)
                if not row:
                    return None
                return {
                    "message_id": row["message_id"],
                    "room_id": row["room_id"],
                    "sender_id": row["sender_id"],
                    "content": new_content,
                    "message_type": row["message_type"],
                    "metadata": row["metadata"],
                    "is_edited": True,
                    "edited_at": row["edited_at"].isoformat() if row["edited_at"] else None,
                    "created_at": row["created_at"].isoformat(),
                }
        except Exception as e:
            logger.error(f"Failed to edit message: {e}")
            return None

    async def search_messages(
        self,
        room_id: str,
        user_id: str,
        query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search messages in a room using ILIKE."""
        await self._ensure_initialized()
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                pattern = f"%{query}%"
                rows = await conn.fetch("""
                    SELECT m.message_id, m.sender_id, m.message_content,
                           m.message_type, m.metadata, m.created_at,
                           u.username, u.display_name,
                           fu.federated_address AS federated_address,
                           fu.display_name AS federated_display_name
                    FROM chat_messages m
                    LEFT JOIN users u ON m.sender_id = u.user_id
                    LEFT JOIN federated_users fu ON m.federated_sender_id = fu.federated_user_id
                    WHERE m.room_id = $1 AND m.deleted_at IS NULL
                    AND m.message_content ILIKE $2
                    ORDER BY m.created_at DESC
                    LIMIT $3
                """, room_id, pattern, limit)
                results = []
                for row in rows:
                    d = dict(row)
                    if d.get("federated_address"):
                        d["display_name"] = d.get("federated_display_name") or d.get("federated_address")
                        d["username"] = d.get("federated_address")
                    d.pop("federated_display_name", None)
                    d["content"] = encryption_service.decrypt_message(d.pop("message_content"))
                    results.append(d)
                return results
        except Exception as e:
            logger.error(f"Failed to search messages: {e}")
            return []

    # =====================
    # REACTION OPERATIONS
    # =====================
    
    async def add_reaction(
        self,
        message_id: str,
        user_id: str,
        emoji: str
    ) -> Optional[Dict[str, Any]]:
        """
        Add emoji reaction to a message.
        Returns dict with reaction_id, message_id, room_id, emoji on success.
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                reaction_id = str(uuid.uuid4())
                ins = await conn.fetchrow(
                    """
                    INSERT INTO message_reactions (reaction_id, message_id, user_id, emoji)
                    SELECT $1::uuid, $2::uuid, $3, $4
                    WHERE NOT EXISTS (
                        SELECT 1 FROM message_reactions
                        WHERE message_id = $2::uuid AND user_id = $3 AND emoji = $4
                    )
                    RETURNING reaction_id
                    """,
                    reaction_id,
                    message_id,
                    user_id,
                    emoji,
                )
                row = await conn.fetchrow(
                    "SELECT room_id FROM chat_messages WHERE message_id = $1::uuid",
                    message_id,
                )
                if ins and ins.get("reaction_id"):
                    rid = ins["reaction_id"]
                else:
                    rid = await conn.fetchval(
                        """
                        SELECT reaction_id FROM message_reactions
                        WHERE message_id = $1::uuid AND user_id = $2 AND emoji = $3
                        """,
                        message_id,
                        user_id,
                        emoji,
                    )
                if not rid or not row:
                    return None
                return {
                    "reaction_id": str(rid),
                    "message_id": message_id,
                    "room_id": str(row["room_id"]),
                    "emoji": emoji,
                }
        
        except Exception as e:
            logger.error(f"❌ Failed to add reaction: {e}")
            return None
    
    async def remove_reaction(
        self,
        reaction_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Remove an emoji reaction. Returns deleted row info dict or None.
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                prev = await conn.fetchrow(
                    """
                    SELECT reaction_id, message_id, room_id, emoji, user_id, federated_user_id
                    FROM message_reactions
                    WHERE reaction_id = $1::uuid AND user_id = $2
                    """,
                    reaction_id,
                    user_id,
                )
                result = await conn.execute("""
                    DELETE FROM message_reactions
                    WHERE reaction_id = $1 AND user_id = $2
                """, reaction_id, user_id)
                
                if result == "DELETE 1" and prev:
                    return {
                        "reaction_id": str(prev["reaction_id"]),
                        "message_id": str(prev["message_id"]),
                        "room_id": str(prev["room_id"]),
                        "emoji": prev["emoji"],
                    }
                return None
        
        except Exception as e:
            logger.error(f"❌ Failed to remove reaction: {e}")
            return None
    
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

                exists = await conn.fetchval(
                    "SELECT 1 FROM users WHERE user_id = $1",
                    user_id,
                )
                if not exists:
                    logger.warning(
                        "Skipping user_presence update: no users row for user_id=%s "
                        "(stale JWT after DB reset, or token not yet tied to a row)",
                        (user_id[:48] + "...") if user_id and len(user_id) > 48 else user_id,
                    )
                    return False

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
        room_id: str,
        user_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get presence for all participants in a room
        
        Args:
            room_id: Room UUID
            user_id: User ID requesting the presence (for RLS)
        
        Returns:
            Dict mapping user_id to presence info
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get user role for RLS
                user_role_row = await conn.fetchrow("""
                    SELECT role FROM users WHERE user_id = $1
                """, user_id)
                user_role = user_role_row["role"] if user_role_row else "user"
                
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_role)
                
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
    
    async def cleanup_stale_presence(self) -> List[str]:
        """
        Mark users as offline if they haven't updated presence recently.

        Returns:
            user_id values that were updated to offline (for WebSocket broadcast).
        """
        await self._ensure_initialized()

        try:
            async with self.db_pool.acquire() as conn:
                threshold = datetime.utcnow() - timedelta(seconds=settings.PRESENCE_OFFLINE_THRESHOLD_SECONDS)

                rows = await conn.fetch(
                    """
                    UPDATE user_presence
                    SET status = 'offline'
                    WHERE status != 'offline'
                    AND last_seen_at < $1
                    RETURNING user_id
                    """,
                    threshold,
                )

                user_ids = [str(r["user_id"]) for r in rows]
                if user_ids:
                    logger.info(
                        "Marked %d users offline due to stale presence (threshold %ss)",
                        len(user_ids),
                        settings.PRESENCE_OFFLINE_THRESHOLD_SECONDS,
                    )
                return user_ids

        except Exception as e:
            logger.error(f"Failed to cleanup stale presence: {e}")
            return []
    
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

    async def get_message_federation_wire_id(self, message_id: str) -> Optional[str]:
        """BFP message id for cross-instance reactions (remote id if mirrored, else local)."""
        await self._ensure_initialized()
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT COALESCE(
                        NULLIF(metadata->>'federation_remote_message_id', ''),
                        message_id::text
                    ) AS wire
                    FROM chat_messages
                    WHERE message_id = $1::uuid AND deleted_at IS NULL
                    """,
                    message_id,
                )
                return str(row["wire"]) if row and row.get("wire") else None
        except Exception as e:
            logger.warning("get_message_federation_wire_id failed: %s", e)
            return None

    async def get_room_participants(self, room_id: str) -> List[Dict[str, Any]]:
        """
        Get all participants in a specific room
        
        Args:
            room_id: Room UUID
        
        Returns:
            List of participant dicts
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # We use admin context here to bypass RLS since this is a system utility
                await conn.execute("SELECT set_config('app.current_user_role', 'admin', false)")
                
                rows = await conn.fetch("""
                    SELECT 
                        u.user_id, u.username, u.display_name, u.avatar_url
                    FROM room_participants rp
                    JOIN users u ON rp.user_id = u.user_id
                    WHERE rp.room_id = $1
                """, room_id)
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"❌ Failed to get room participants: {e}")
            return []

    async def mark_room_as_read(self, room_id: str, user_id: str) -> bool:
        """
        Update the last_read_at timestamp for a user in a room
        
        Args:
            room_id: Room UUID
            user_id: User ID
        
        Returns:
            True if successful
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                result = await conn.execute("""
                    UPDATE room_participants
                    SET last_read_at = NOW()
                    WHERE room_id = $1 AND user_id = $2
                """, room_id, user_id)
                ok = result == "UPDATE 1"
                if ok and getattr(settings, "FEDERATION_ENABLED", False):
                    rtype = await conn.fetchrow(
                        "SELECT room_type FROM chat_rooms WHERE room_id = $1::uuid",
                        room_id,
                    )
                    if rtype and (rtype.get("room_type") or "") == "federated":
                        pref = await conn.fetchrow(
                            """
                            SELECT federation_share_read_receipts
                            FROM users WHERE user_id = $1
                            """,
                            user_id,
                        )
                        share = True
                        if pref and pref.get("federation_share_read_receipts") is False:
                            share = False
                        if share:
                            from services.federation_message_service import (
                                federation_message_service,
                            )

                            ts = datetime.now(timezone.utc).isoformat()
                            try:
                                await federation_message_service.deliver_outbound_read_receipt(
                                    room_id, user_id, ts
                                )
                            except Exception as fre:
                                logger.warning(
                                    "Federated read receipt forward failed: %s", fre
                                )
                return ok
        except Exception as e:
            logger.error(f"❌ Failed to mark room {room_id} as read for user {user_id}: {e}")
            return False


# Global messaging service instance
messaging_service = MessagingService()

