"""
Team Service - Handles team management, members, and permissions
"""

import logging
import uuid
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import asyncpg

from utils.shared_db_pool import get_shared_db_pool
from models.team_models import TeamRole

logger = logging.getLogger(__name__)


class TeamService:
    """
    Service for managing teams, members, and permissions
    
    Handles:
    - Team CRUD operations
    - Member management with roles
    - Permission checks
    - Team room creation
    """
    
    def __init__(self):
        self.db_pool = None
        self.messaging_service = None
    
    async def initialize(self, shared_db_pool=None, messaging_service=None):
        """Initialize with database pool and messaging service"""
        if shared_db_pool:
            self.db_pool = shared_db_pool
        else:
            try:
                self.db_pool = await get_shared_db_pool()
            except Exception as e:
                logger.error(f"Failed to get shared DB pool: {e}")
                raise
        
        self.messaging_service = messaging_service
        logger.info("Team service initialized")
    
    async def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self.db_pool:
            # Try to get shared pool from service container if available
            try:
                from services.service_container import service_container
                if hasattr(service_container, 'db_pool') and service_container.db_pool:
                    self.db_pool = service_container.db_pool
                    if self.messaging_service is None:
                        from services.messaging.messaging_service import messaging_service
                        self.messaging_service = messaging_service
                    return
            except Exception:
                pass
            
            # Fallback to creating own pool
            await self.initialize()
    
    async def create_team(
        self,
        name: str,
        description: Optional[str],
        creator_id: str,
        avatar_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new team
        
        Args:
            name: Team name
            description: Team description
            creator_id: User ID of creator
            avatar_url: Optional avatar URL
        
        Returns:
            Dict with team details
        """
        await self._ensure_initialized()
        
        team_id = str(uuid.uuid4())
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get creator's role for RLS context and folder creation
                creator_role_row = await conn.fetchrow("""
                    SELECT role FROM users WHERE user_id = $1
                """, creator_id)
                creator_role = creator_role_row["role"] if creator_role_row else "user"
                
                # Set user context for RLS (both user_id and role)
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", creator_id)
                await conn.execute("SELECT set_config('app.current_user_role', $1, false)", creator_role)
                
                # Create team
                await conn.execute("""
                    INSERT INTO teams (team_id, team_name, description, created_by, avatar_url)
                    VALUES ($1, $2, $3, $4, $5)
                """, team_id, name, description, creator_id, avatar_url)
                
                # Add creator as admin
                await conn.execute("""
                    INSERT INTO team_members (team_id, user_id, role, invited_by)
                    VALUES ($1, $2, $3, $2)
                """, team_id, creator_id, "admin")
                
                logger.info(f"Created team {team_id} by user {creator_id}")
                
                # Create team folder
                try:
                    from services.folder_service import FolderService
                    folder_service = FolderService()
                    await folder_service.initialize()
                    
                    # Create root folder for team documents
                    team_folder = await folder_service.create_folder(
                        name=f"{name} Documents",
                        parent_folder_id=None,
                        user_id=None,
                        collection_type="team",
                        current_user_role=creator_role,
                        admin_user_id=creator_id,
                        team_id=team_id
                    )
                    logger.info(f"Created team folder {team_folder.folder_id} for team {team_id}")
                except Exception as e:
                    logger.warning(f"Failed to create team folder: {e}")
                
                # Create team chat room
                if self.messaging_service:
                    try:
                        # Create room with creator only (participant_ids should not include creator)
                        # create_room automatically adds creator_id to participants
                        room = await self.messaging_service.create_room(
                            creator_id=creator_id,
                            participant_ids=[],  # Empty - creator will be added automatically
                            room_name=name
                        )
                        
                        # Link room to team
                        await conn.execute("""
                            UPDATE chat_rooms
                            SET team_id = $1
                            WHERE room_id = $2
                        """, team_id, room["room_id"])
                        
                        logger.info(f"Created team room {room['room_id']} for team {team_id}")
                    except Exception as e:
                        logger.warning(f"Failed to create team room: {e}")
                
                # Get created team
                team_row = await conn.fetchrow("""
                    SELECT t.*, COUNT(tm.user_id) as member_count
                    FROM teams t
                    LEFT JOIN team_members tm ON tm.team_id = t.team_id
                    WHERE t.team_id = $1
                    GROUP BY t.team_id
                """, team_id)
                
                # Parse settings JSONB (asyncpg returns it as dict, but ensure it's a dict)
                settings = team_row["settings"]
                if isinstance(settings, str):
                    try:
                        settings = json.loads(settings) if settings else {}
                    except (json.JSONDecodeError, TypeError):
                        settings = {}
                elif settings is None:
                    settings = {}
                
                return {
                    "team_id": str(team_row["team_id"]),
                    "team_name": team_row["team_name"],
                    "description": team_row["description"],
                    "created_by": team_row["created_by"],
                    "created_at": team_row["created_at"].isoformat(),
                    "updated_at": team_row["updated_at"].isoformat(),
                    "avatar_url": team_row["avatar_url"],
                    "settings": settings,
                    "member_count": team_row["member_count"] or 0,
                    "user_role": "admin"
                }
        
        except Exception as e:
            logger.error(f"Failed to create team: {e}")
            raise
    
    async def get_team(self, team_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get team details with user's role
        
        Args:
            team_id: Team ID
            user_id: User ID requesting team
        
        Returns:
            Team dict or None if not found/not member
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Get team with member count and user's role
                row = await conn.fetchrow("""
                    SELECT 
                        t.*,
                        COUNT(DISTINCT tm.user_id) as member_count,
                        tm_user.role as user_role
                    FROM teams t
                    LEFT JOIN team_members tm ON tm.team_id = t.team_id
                    LEFT JOIN team_members tm_user ON tm_user.team_id = t.team_id AND tm_user.user_id = $2
                    WHERE t.team_id = $1
                    GROUP BY t.team_id, tm_user.role
                """, team_id, user_id)
                
                if not row:
                    return None
                
                # Parse settings JSONB
                settings = row["settings"]
                if isinstance(settings, str):
                    try:
                        settings = json.loads(settings) if settings else {}
                    except (json.JSONDecodeError, TypeError):
                        settings = {}
                elif settings is None:
                    settings = {}
                
                return {
                    "team_id": str(row["team_id"]),
                    "team_name": row["team_name"],
                    "description": row["description"],
                    "created_by": row["created_by"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "avatar_url": row["avatar_url"],
                    "settings": settings,
                    "member_count": row["member_count"] or 0,
                    "user_role": row["user_role"]
                }
        
        except Exception as e:
            logger.error(f"Failed to get team: {e}")
            raise
    
    async def list_user_teams(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all teams user is member of
        
        Args:
            user_id: User ID
        
        Returns:
            List of team dicts
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get user's role for RLS context
                user_role_row = await conn.fetchrow("""
                    SELECT role FROM users WHERE user_id = $1
                """, user_id)
                user_role = user_role_row["role"] if user_role_row else "user"
                
                # Set user context for RLS (both user_id and role)
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                await conn.execute("SELECT set_config('app.current_user_role', $1, false)", user_role)
                
                rows = await conn.fetch("""
                    SELECT 
                        t.*,
                        COUNT(DISTINCT tm.user_id) as member_count,
                        tm_user.role as user_role,
                        tm_user.muted,
                        t.updated_at as last_activity
                    FROM teams t
                    INNER JOIN team_members tm_user ON tm_user.team_id = t.team_id AND tm_user.user_id = $1
                    LEFT JOIN team_members tm ON tm.team_id = t.team_id
                    GROUP BY t.team_id, tm_user.role, tm_user.muted
                    ORDER BY last_activity DESC
                """, user_id)
                
                teams = []
                for row in rows:
                    # Parse settings JSONB
                    settings = row["settings"]
                    if isinstance(settings, str):
                        try:
                            settings = json.loads(settings) if settings else {}
                        except (json.JSONDecodeError, TypeError):
                            settings = {}
                    elif settings is None:
                        settings = {}
                    
                    teams.append({
                        "team_id": str(row["team_id"]),
                        "team_name": row["team_name"],
                        "description": row["description"],
                        "created_by": row["created_by"],
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat(),
                        "avatar_url": row["avatar_url"],
                        "settings": settings,
                        "member_count": row["member_count"] or 0,
                        "user_role": row["user_role"],
                        "muted": row.get("muted", False)
                    })
                
                return teams
        
        except Exception as e:
            logger.error(f"Failed to list user teams: {e}")
            raise
    
    async def update_team(
        self,
        team_id: str,
        updates: Dict[str, Any],
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Update team details (admin only)
        
        Args:
            team_id: Team ID
            updates: Dict with fields to update
            user_id: User ID making update
        
        Returns:
            Updated team dict or None if not found/not admin
        """
        await self._ensure_initialized()
        
        # Check admin permission
        role = await self.check_team_access(team_id, user_id)
        if role != TeamRole.ADMIN.value:
            raise PermissionError("Only team admins can update team details")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Build update query
                update_fields = []
                update_values = []
                param_num = 1
                
                if "team_name" in updates:
                    update_fields.append(f"team_name = ${param_num}")
                    update_values.append(updates["team_name"])
                    param_num += 1
                
                if "description" in updates:
                    update_fields.append(f"description = ${param_num}")
                    update_values.append(updates["description"])
                    param_num += 1
                
                if "avatar_url" in updates:
                    update_fields.append(f"avatar_url = ${param_num}")
                    update_values.append(updates["avatar_url"])
                    param_num += 1
                
                if "settings" in updates:
                    update_fields.append(f"settings = ${param_num}")
                    update_values.append(updates["settings"])
                    param_num += 1
                
                if not update_fields:
                    return await self.get_team(team_id, user_id)
                
                update_fields.append(f"updated_at = CURRENT_TIMESTAMP")
                update_values.extend([team_id])
                
                query = f"""
                    UPDATE teams
                    SET {', '.join(update_fields)}
                    WHERE team_id = ${param_num}
                """
                
                await conn.execute(query, *update_values)
                
                logger.info(f"Updated team {team_id} by user {user_id}")
                
                return await self.get_team(team_id, user_id)
        
        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Failed to update team: {e}")
            raise
    
    async def delete_team(self, team_id: str, user_id: str) -> bool:
        """
        Delete team (admin only, cascades to members, posts, etc.)
        Also deletes team folder from database and disk.
        
        Args:
            team_id: Team ID
            user_id: User ID deleting team
        
        Returns:
            True if deleted
        """
        await self._ensure_initialized()
        
        # Check admin permission
        role = await self.check_team_access(team_id, user_id)
        if role != TeamRole.ADMIN.value:
            raise PermissionError("Only team admins can delete teams")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Find and delete team folder(s) before deleting team
                try:
                    from services.folder_service import FolderService
                    from pathlib import Path
                    import shutil
                    from config import settings
                    
                    folder_service = FolderService()
                    await folder_service.initialize()
                    
                    # Get all folders for this team
                    team_folders = await folder_service.document_repository.get_folders_by_teams([team_id])
                    
                    # Delete all team folders from database (cascade will handle documents)
                    # We delete directly from repository to bypass the team root folder restriction
                    for folder_data in team_folders:
                        folder_id = folder_data.get('folder_id')
                        if folder_id:
                            try:
                                # Delete from database (cascade will handle subfolders and documents)
                                await folder_service.document_repository.delete_folder(folder_id)
                                logger.info(f"Deleted team folder {folder_id} from database")
                            except Exception as e:
                                logger.warning(f"Failed to delete team folder {folder_id} from database: {e}")
                    
                    # Delete entire team directory from disk (Teams/{team_id})
                    # This will delete all team files including documents, posts, etc.
                    team_base_path = folder_service.get_team_base_path(team_id)
                    team_dir = team_base_path.parent  # Go up one level to get Teams/{team_id}
                    
                    if team_dir.exists():
                        try:
                            shutil.rmtree(team_dir)
                            logger.info(f"Deleted team directory from disk: {team_dir}")
                        except Exception as e:
                            logger.warning(f"Failed to delete team directory {team_dir} from disk: {e}")
                    else:
                        logger.info(f"Team directory not found on disk: {team_dir}")
                        
                except Exception as e:
                    logger.warning(f"Failed to delete team folder(s) for team {team_id}: {e}")
                    # Continue with team deletion even if folder deletion fails
                
                # Delete team (cascades to members, posts, etc.)
                result = await conn.execute("""
                    DELETE FROM teams WHERE team_id = $1
                """, team_id)
                
                if result == "DELETE 1":
                    logger.info(f"Deleted team {team_id} by user {user_id}")
                    return True
                else:
                    return False
        
        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete team: {e}")
            raise
    
    async def add_member(
        self,
        team_id: str,
        user_id: str,
        role: str,  # "admin", "member", or "viewer"
        added_by: str
    ) -> bool:
        """
        Add member to team (admin only)
        
        Args:
            team_id: Team ID
            user_id: User ID to add
            role: Member role
            added_by: User ID adding member
        
        Returns:
            True if added
        """
        await self._ensure_initialized()
        
        # Check admin permission
        admin_role = await self.check_team_access(team_id, added_by)
        if admin_role != TeamRole.ADMIN.value:
            raise PermissionError("Only team admins can add members")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", added_by)
                
                # Check if already member (may not work due to RLS, but try anyway)
                try:
                    existing = await conn.fetchval("""
                        SELECT 1 FROM team_members
                        WHERE team_id = $1 AND user_id = $2
                    """, team_id, user_id)
                    
                    if existing:
                        raise ValueError("User is already a team member")
                except Exception:
                    # RLS might prevent seeing the row, continue to try insert
                    pass
                
                # Add member (with ON CONFLICT to handle race conditions and RLS edge cases)
                result = await conn.execute("""
                    INSERT INTO team_members (team_id, user_id, role, invited_by)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (team_id, user_id) DO NOTHING
                """, team_id, user_id, role, added_by)
                
                # Check if insert actually happened
                # asyncpg returns "INSERT 0 1" for success, "INSERT 0 0" for conflict
                if result == "INSERT 0 0":
                    raise ValueError("User is already a team member")
                
                # Add to team room if exists
                if self.messaging_service:
                    try:
                        room_row = await conn.fetchrow("""
                            SELECT room_id FROM chat_rooms WHERE team_id = $1
                        """, team_id)
                        
                        if room_row:
                            await conn.execute("""
                                INSERT INTO room_participants (room_id, user_id)
                                VALUES ($1, $2)
                                ON CONFLICT (room_id, user_id) DO NOTHING
                            """, room_row["room_id"], user_id)
                    except Exception as e:
                        logger.warning(f"Failed to add member to team room: {e}")
                
                logger.info(f"Added member {user_id} to team {team_id} with role {role}")
                return True
        
        except (PermissionError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Failed to add member: {e}")
            raise
    
    async def remove_member(self, team_id: str, user_id: str, removed_by: str) -> bool:
        """
        Remove member from team (admin only, cannot remove self)
        
        Args:
            team_id: Team ID
            user_id: User ID to remove
            removed_by: User ID removing member
        
        Returns:
            True if removed
        """
        await self._ensure_initialized()
        
        # Check admin permission
        admin_role = await self.check_team_access(team_id, removed_by)
        if admin_role != TeamRole.ADMIN.value:
            raise PermissionError("Only team admins can remove members")
        
        # Cannot remove self
        if user_id == removed_by:
            raise ValueError("Cannot remove yourself from team")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", removed_by)
                
                # Remove member
                result = await conn.execute("""
                    DELETE FROM team_members
                    WHERE team_id = $1 AND user_id = $2
                """, team_id, user_id)
                
                # Remove from team room
                if self.messaging_service:
                    try:
                        room_row = await conn.fetchrow("""
                            SELECT room_id FROM chat_rooms WHERE team_id = $1
                        """, team_id)
                        
                        if room_row:
                            await conn.execute("""
                                DELETE FROM room_participants
                                WHERE room_id = $1 AND user_id = $2
                            """, room_row["room_id"], user_id)
                    except Exception as e:
                        logger.warning(f"Failed to remove member from team room: {e}")
                
                if result == "DELETE 1":
                    logger.info(f"Removed member {user_id} from team {team_id}")
                    return True
                else:
                    return False
        
        except (PermissionError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Failed to remove member: {e}")
            raise
    
    async def update_member_role(
        self,
        team_id: str,
        user_id: str,
        new_role: str,  # "admin", "member", or "viewer"
        updated_by: str
    ) -> bool:
        """
        Update member role (admin only)
        
        Args:
            team_id: Team ID
            user_id: User ID to update
            new_role: New role
            updated_by: User ID making update
        
        Returns:
            True if updated
        """
        await self._ensure_initialized()
        
        # Check admin permission
        admin_role = await self.check_team_access(team_id, updated_by)
        if admin_role != TeamRole.ADMIN.value:
            raise PermissionError("Only team admins can update member roles")
        
        # Cannot change own role
        if user_id == updated_by:
            raise ValueError("Cannot change your own role")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", updated_by)
                
                # Update role
                result = await conn.execute("""
                    UPDATE team_members
                    SET role = $1
                    WHERE team_id = $2 AND user_id = $3
                """, new_role, team_id, user_id)
                
                if result == "UPDATE 1":
                    logger.info(f"Updated member {user_id} role to {new_role} in team {team_id}")
                    return True
                else:
                    return False
        
        except (PermissionError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Failed to update member role: {e}")
            raise
    
    async def get_team_members(self, team_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        Get team members with online presence
        
        Args:
            team_id: Team ID
            user_id: User ID requesting (must be member)
        
        Returns:
            List of member dicts
        """
        await self._ensure_initialized()
        
        # Check access
        role = await self.check_team_access(team_id, user_id)
        if not role:
            raise PermissionError("Not a team member")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                # Get members with user info and presence
                rows = await conn.fetch("""
                    SELECT 
                        tm.user_id,
                        tm.role,
                        tm.joined_at,
                        tm.invited_by,
                        u.username,
                        u.display_name,
                        u.avatar_url,
                        up.status as presence_status,
                        up.last_seen_at
                    FROM team_members tm
                    INNER JOIN users u ON u.user_id = tm.user_id
                    LEFT JOIN user_presence up ON up.user_id = tm.user_id
                    WHERE tm.team_id = $1
                    ORDER BY tm.role DESC, tm.joined_at ASC
                """, team_id)
                
                return [
                    {
                        "user_id": row["user_id"],
                        "username": row["username"],
                        "display_name": row["display_name"],
                        "avatar_url": row["avatar_url"],
                        "role": row["role"],
                        "joined_at": row["joined_at"].isoformat(),
                        "invited_by": row["invited_by"],
                        "is_online": row["presence_status"] == "online",
                        "last_seen": row["last_seen_at"].isoformat() if row["last_seen_at"] else None
                    }
                    for row in rows
                ]
        
        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Failed to get team members: {e}")
            raise
    
    async def check_team_access(self, team_id: str, user_id: str) -> Optional[str]:
        """
        Check if user has access to team and return their role
        
        Args:
            team_id: Team ID
            user_id: User ID
        
        Returns:
            Role string or None if no access
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                role = await conn.fetchval("""
                    SELECT role FROM team_members
                    WHERE team_id = $1 AND user_id = $2
                """, team_id, user_id)
                
                return role
        
        except Exception as e:
            logger.error(f"Failed to check team access: {e}")
            return None
    
    async def get_team_permissions(self, team_id: str, user_id: str) -> Dict[str, bool]:
        """
        Get user's permissions for team
        
        Args:
            team_id: Team ID
            user_id: User ID
        
        Returns:
            Dict with permission flags
        """
        role = await self.check_team_access(team_id, user_id)
        
        if not role:
            return {
                "can_view": False,
                "can_post": False,
                "can_comment": False,
                "can_react": False,
                "can_manage_members": False,
                "can_manage_team": False
            }
        
        is_admin = role == "admin"
        is_member = role in ["admin", "member"]
        is_viewer = role == "viewer"
        
        return {
            "can_view": True,
            "can_post": is_member,
            "can_comment": is_member,
            "can_react": is_member,
            "can_manage_members": is_admin,
            "can_manage_team": is_admin
        }
    
    async def get_unread_post_counts(self, user_id: str) -> Dict[str, int]:
        """
        Get unread post counts for all user's teams (excluding muted teams)
        
        Args:
            user_id: User ID
        
        Returns:
            Dict mapping team_id to unread count
        """
        await self._ensure_initialized()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                logger.debug(f"📊 Fetching unread counts for user {user_id}")
                rows = await conn.fetch("""
                    SELECT 
                        tm.team_id,
                        tm.muted,
                        tm.last_read_at,
                        tm.joined_at,
                        COUNT(tp.post_id) as unread_count,
                        COUNT(CASE WHEN tp.created_at > COALESCE(tm.last_read_at, tm.joined_at) THEN 1 END) as posts_after_last_read
                    FROM team_members tm
                    LEFT JOIN team_posts tp ON tp.team_id = tm.team_id
                        AND tp.author_id != $1
                        AND tp.deleted_at IS NULL
                    WHERE tm.user_id = $1
                    GROUP BY tm.team_id, tm.muted, tm.last_read_at, tm.joined_at
                """, user_id)
                
                # Build result and log details
                result = {}
                for row in rows:
                    team_id = str(row['team_id'])
                    is_muted = row.get('muted', False)
                    unread_count = row['posts_after_last_read'] if not is_muted else 0
                    
                    logger.debug(f"  Team {team_id}: muted={is_muted}, posts_after_last_read={row['posts_after_last_read']}, will show={unread_count}")
                    
                    if unread_count > 0:
                        result[team_id] = unread_count
                
                logger.debug(f"📊 Final unread counts for user {user_id}: {result}")
                return result
        
        except Exception as e:
            logger.error(f"Failed to get unread post counts: {e}", exc_info=True)
            return {}
    
    async def mark_team_posts_as_read(self, team_id: str, user_id: str) -> bool:
        """
        Mark all posts in a team as read for a user
        
        Args:
            team_id: Team ID
            user_id: User ID
        
        Returns:
            True if updated
        """
        await self._ensure_initialized()
        
        # Check access
        role = await self.check_team_access(team_id, user_id)
        if not role:
            raise PermissionError("Not a team member")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                await conn.execute("""
                    UPDATE team_members
                    SET last_read_at = CURRENT_TIMESTAMP
                    WHERE team_id = $1 AND user_id = $2
                """, team_id, user_id)
                
                logger.info(f"Marked team {team_id} posts as read for user {user_id}")
                return True
        
        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Failed to mark team posts as read: {e}")
            raise
    
    async def mute_team(self, team_id: str, user_id: str, muted: bool = True) -> bool:
        """
        Mute or unmute a team for a user
        
        Args:
            team_id: Team ID
            user_id: User ID
            muted: True to mute, False to unmute
        
        Returns:
            True if updated
        """
        await self._ensure_initialized()
        
        # Check access
        role = await self.check_team_access(team_id, user_id)
        if not role:
            raise PermissionError("Not a team member")
        
        try:
            async with self.db_pool.acquire() as conn:
                # Set user context for RLS
                await conn.execute("SELECT set_config('app.current_user_id', $1, false)", user_id)
                
                await conn.execute("""
                    UPDATE team_members
                    SET muted = $1
                    WHERE team_id = $2 AND user_id = $3
                """, muted, team_id, user_id)
                
                action = "muted" if muted else "unmuted"
                logger.info(f"Team {team_id} {action} for user {user_id}")
                return True
        
        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Failed to mute/unmute team: {e}")
            raise

