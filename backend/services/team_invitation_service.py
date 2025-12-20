"""
Team Invitation Service - Handles team invitations and messaging integration
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import asyncpg

from utils.shared_db_pool import get_shared_db_pool
from services.team_service import TeamService

logger = logging.getLogger(__name__)


class TeamInvitationService:
    """
    Service for managing team invitations
    
    Handles:
    - Creating invitations
    - Linking invitations to chat messages
    - Accepting/rejecting invitations
    - Expiration handling
    """
    
    def __init__(self):
        self.db_pool = None
        self.messaging_service = None
        self.team_service = None
    
    async def initialize(self, shared_db_pool=None, messaging_service=None, team_service=None):
        """Initialize with database pool and services"""
        if shared_db_pool:
            self.db_pool = shared_db_pool
        else:
            self.db_pool = await get_shared_db_pool()
        
        self.messaging_service = messaging_service
        self.team_service = team_service
        logger.info("Team invitation service initialized")
    
    async def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self.db_pool:
            await self.initialize()
    
    async def create_invitation(
        self,
        team_id: str,
        invited_user_id: str,
        invited_by: str
    ) -> Dict[str, Any]:
        """
        Create a team invitation and send message
        
        Args:
            team_id: Team ID
            invited_user_id: User ID to invite
            invited_by: User ID creating invitation
        
        Returns:
            Invitation dict with message_id
        """
        await self._ensure_initialized()
        
        # Check admin permission
        if self.team_service:
            role = await self.team_service.check_team_access(team_id, invited_by)
            if role != "admin":
                raise PermissionError("Only team admins can create invitations")
        
        # Check if already member
        if self.team_service:
            existing_role = await self.team_service.check_team_access(team_id, invited_user_id)
            if existing_role:
                raise ValueError("User is already a team member")
        
        # Check for existing pending invitation
        invitation_id = str(uuid.uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", invited_by)
                
                # Check for existing pending invitation
                existing = await conn.fetchrow("""
                    SELECT invitation_id FROM team_invitations
                    WHERE team_id = $1 
                    AND invited_user_id = $2 
                    AND status = 'pending'
                    AND expires_at > CURRENT_TIMESTAMP
                """, team_id, invited_user_id)
                
                if existing:
                    raise ValueError("Pending invitation already exists")
                
                # Get team and inviter info
                team_row = await conn.fetchrow("""
                    SELECT team_name FROM teams WHERE team_id = $1
                """, team_id)
                
                inviter_row = await conn.fetchrow("""
                    SELECT username, display_name FROM users WHERE user_id = $1
                """, invited_by)
                
                if not team_row or not inviter_row:
                    raise ValueError("Team or inviter not found")
                
                # Create invitation
                await conn.execute("""
                    INSERT INTO team_invitations (
                        invitation_id, team_id, invited_user_id, invited_by,
                        status, expires_at
                    )
                    VALUES ($1, $2, $3, $4, 'pending', $5)
                """, invitation_id, team_id, invited_user_id, invited_by, expires_at)
                
                # Create invitation message
                message_id = None
                if self.messaging_service:
                    try:
                        # Get or create direct room between inviter and invitee
                        rooms = await self.messaging_service.get_user_rooms(invited_by, limit=100)
                        direct_room = None
                        
                        for room in rooms:
                            if room["room_type"] == "direct":
                                participants = room.get("participants", [])
                                participant_ids = [p.get("user_id") for p in participants if p]
                                if invited_user_id in participant_ids and len(participant_ids) == 2:
                                    direct_room = room
                                    break
                        
                        if not direct_room:
                            # Create direct room
                            direct_room = await self.messaging_service.create_room(
                                creator_id=invited_by,
                                participant_ids=[invited_user_id],
                                room_name=None
                            )
                        
                        # Send invitation message
                        inviter_name = inviter_row["display_name"] or inviter_row["username"]
                        message_content = f"Invitation to join team: {team_row['team_name']}"
                        
                        message = await self.messaging_service.send_message(
                            room_id=direct_room["room_id"],
                            sender_id=invited_by,
                            content=message_content,
                            message_type="team_invitation",
                            metadata={
                                "invitation_id": invitation_id,
                                "team_id": team_id,
                                "team_name": team_row["team_name"],
                                "invited_by": invited_by,
                                "inviter_name": inviter_name
                            }
                        )
                        
                        message_id = message.get("message_id")
                        
                        # Update invitation with message_id
                        await conn.execute("""
                            UPDATE team_invitations
                            SET message_id = $1
                            WHERE invitation_id = $2
                        """, message_id, invitation_id)
                        
                        logger.info(f"Created invitation message {message_id} for invitation {invitation_id}")
                    except Exception as e:
                        logger.warning(f"Failed to create invitation message: {e}")
                
                logger.info(f"Created invitation {invitation_id} for team {team_id}")
                
                return {
                    "invitation_id": invitation_id,
                    "team_id": team_id,
                    "team_name": team_row["team_name"],
                    "invited_user_id": invited_user_id,
                    "invited_by": invited_by,
                    "inviter_name": inviter_row["display_name"] or inviter_row["username"],
                    "status": "pending",
                    "message_id": message_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "expires_at": expires_at.isoformat()
                }
        
        except (PermissionError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Failed to create invitation: {e}")
            raise
    
    async def get_pending_invitations(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get user's pending invitations
        
        Args:
            user_id: User ID
        
        Returns:
            List of invitation dicts
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Get pending invitations
                rows = await conn.fetch("""
                    SELECT 
                        ti.*,
                        t.team_name,
                        u_inviter.username as inviter_username,
                        u_inviter.display_name as inviter_display_name
                    FROM team_invitations ti
                    INNER JOIN teams t ON t.team_id = ti.team_id
                    INNER JOIN users u_inviter ON u_inviter.user_id = ti.invited_by
                    WHERE ti.invited_user_id = $1
                    AND ti.status = 'pending'
                    AND ti.expires_at > CURRENT_TIMESTAMP
                    ORDER BY ti.created_at DESC
                """, user_id)
                
                return [
                    {
                        "invitation_id": str(row["invitation_id"]),
                        "team_id": str(row["team_id"]),
                        "team_name": row["team_name"],
                        "invited_user_id": row["invited_user_id"],
                        "invited_by": row["invited_by"],
                        "inviter_name": row["inviter_display_name"] or row["inviter_username"],
                        "status": row["status"],
                        "message_id": str(row["message_id"]) if row["message_id"] else None,
                        "created_at": row["created_at"].isoformat(),
                        "expires_at": row["expires_at"].isoformat(),
                        "responded_at": row["responded_at"].isoformat() if row["responded_at"] else None
                    }
                    for row in rows
                ]
        
        except Exception as e:
            logger.error(f"Failed to get pending invitations: {e}")
            raise
    
    async def accept_invitation(self, invitation_id: str, user_id: str) -> Dict[str, Any]:
        """
        Accept team invitation
        
        Args:
            invitation_id: Invitation ID
            user_id: User ID accepting
        
        Returns:
            Team dict
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Get invitation
                inv_row = await conn.fetchrow("""
                    SELECT * FROM team_invitations
                    WHERE invitation_id = $1 AND invited_user_id = $2
                """, invitation_id, user_id)
                
                if not inv_row:
                    raise ValueError("Invitation not found")
                
                if inv_row["status"] != "pending":
                    raise ValueError("Invitation already responded to")
                
                if inv_row["expires_at"] < datetime.now(timezone.utc):
                    # Mark as expired
                    await conn.execute("""
                        UPDATE team_invitations
                        SET status = 'expired'
                        WHERE invitation_id = $1
                    """, invitation_id)
                    raise ValueError("Invitation has expired")
                
                team_id = str(inv_row["team_id"])
                
                # Add user to team as member
                if self.team_service:
                    await self.team_service.add_member(
                        team_id=team_id,
                        user_id=user_id,
                        role="member",  # Default role for invited members
                        added_by=inv_row["invited_by"]
                    )
                
                # Update invitation
                await conn.execute("""
                    UPDATE team_invitations
                    SET status = 'accepted', responded_at = CURRENT_TIMESTAMP
                    WHERE invitation_id = $1
                """, invitation_id)
                
                # Update invitation message if exists
                if inv_row["message_id"] and self.messaging_service:
                    try:
                        await conn.execute("""
                            UPDATE chat_messages
                            SET metadata = jsonb_set(
                                COALESCE(metadata, '{}'::jsonb),
                                '{invitation_status}',
                                '"accepted"'
                            )
                            WHERE message_id = $1
                        """, inv_row["message_id"])
                    except Exception as e:
                        logger.warning(f"Failed to update invitation message: {e}")
                
                logger.info(f"User {user_id} accepted invitation {invitation_id} to team {team_id}")
                
                # Get team details
                if self.team_service:
                    return await self.team_service.get_team(team_id, user_id)
                else:
                    return {"team_id": team_id}
        
        except (ValueError, PermissionError):
            raise
        except Exception as e:
            logger.error(f"Failed to accept invitation: {e}")
            raise
    
    async def reject_invitation(self, invitation_id: str, user_id: str) -> bool:
        """
        Reject team invitation
        
        Args:
            invitation_id: Invitation ID
            user_id: User ID rejecting
        
        Returns:
            True if rejected
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Get invitation
                inv_row = await conn.fetchrow("""
                    SELECT * FROM team_invitations
                    WHERE invitation_id = $1 AND invited_user_id = $2
                """, invitation_id, user_id)
                
                if not inv_row:
                    raise ValueError("Invitation not found")
                
                if inv_row["status"] != "pending":
                    raise ValueError("Invitation already responded to")
                
                # Update invitation
                result = await conn.execute("""
                    UPDATE team_invitations
                    SET status = 'rejected', responded_at = CURRENT_TIMESTAMP
                    WHERE invitation_id = $1
                """, invitation_id)
                
                # Update invitation message if exists
                if inv_row["message_id"] and self.messaging_service:
                    try:
                        await conn.execute("""
                            UPDATE chat_messages
                            SET metadata = jsonb_set(
                                COALESCE(metadata, '{}'::jsonb),
                                '{invitation_status}',
                                '"rejected"'
                            )
                            WHERE message_id = $1
                        """, inv_row["message_id"])
                    except Exception as e:
                        logger.warning(f"Failed to update invitation message: {e}")
                
                if result == "UPDATE 1":
                    logger.info(f"User {user_id} rejected invitation {invitation_id}")
                    return True
                else:
                    return False
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to reject invitation: {e}")
            raise
    
    async def cancel_invitation(self, invitation_id: str, cancelled_by: str) -> bool:
        """
        Cancel invitation (admin only)
        
        Args:
            invitation_id: Invitation ID
            cancelled_by: User ID cancelling
        
        Returns:
            True if cancelled
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", cancelled_by)
                
                # Get invitation and check admin permission
                inv_row = await conn.fetchrow("""
                    SELECT ti.*, tm.role
                    FROM team_invitations ti
                    INNER JOIN team_members tm ON tm.team_id = ti.team_id AND tm.user_id = $2
                    WHERE ti.invitation_id = $1
                """, invitation_id, cancelled_by)
                
                if not inv_row:
                    raise ValueError("Invitation not found")
                
                if inv_row["role"] != "admin":
                    raise PermissionError("Only team admins can cancel invitations")
                
                if inv_row["status"] != "pending":
                    raise ValueError("Can only cancel pending invitations")
                
                # Update invitation
                result = await conn.execute("""
                    UPDATE team_invitations
                    SET status = 'expired', responded_at = CURRENT_TIMESTAMP
                    WHERE invitation_id = $1
                """, invitation_id)
                
                if result == "UPDATE 1":
                    logger.info(f"Admin {cancelled_by} cancelled invitation {invitation_id}")
                    return True
                else:
                    return False
        
        except (ValueError, PermissionError):
            raise
        except Exception as e:
            logger.error(f"Failed to cancel invitation: {e}")
            raise

