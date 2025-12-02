import logging
from typing import Optional, Dict, Any
from datetime import datetime

from db.connection_manager import DatabaseConnectionManager

logger = logging.getLogger(__name__)


async def check_workspace_permission(
    db_manager: DatabaseConnectionManager,
    workspace_id: str,
    user_id: str,
    required_permission: str  # 'read', 'write', 'admin'
) -> bool:
    """
    Check if user has required permission on workspace
    
    Checks in order:
    1. Ownership (always has admin permission)
    2. Direct shares (shared_with_user_id)
    3. Team membership shares (shared_with_team_id)
    4. Public shares (is_public = true)
    
    Permission hierarchy: read < write < admin
    """
    try:
        # Check ownership first
        ownership_query = """
            SELECT user_id FROM data_workspaces WHERE workspace_id = $1
        """
        owner = await db_manager.fetchval(ownership_query, workspace_id)
        
        if owner == user_id:
            return True  # Owner has all permissions
        
        # Permission hierarchy mapping
        permission_levels = {'read': 1, 'write': 2, 'admin': 3}
        required_level = permission_levels.get(required_permission, 0)
        
        if required_level == 0:
            return False
        
        # Check direct user shares
        user_share_query = """
            SELECT permission_level, expires_at
            FROM data_workspace_shares
            WHERE workspace_id = $1 
            AND shared_with_user_id = $2
            AND (expires_at IS NULL OR expires_at > NOW())
        """
        user_share = await db_manager.fetchrow(user_share_query, workspace_id, user_id)
        
        if user_share:
            share_level = permission_levels.get(user_share['permission_level'], 0)
            if share_level >= required_level:
                return True
        
        # Note: Team membership checks need to be done in backend API layer
        # since teams table is in main database, not data workspace database
        # This function will be called from backend with team info passed in
        
        # Check public shares
        public_share_query = """
            SELECT permission_level, expires_at
            FROM data_workspace_shares
            WHERE workspace_id = $1 
            AND is_public = TRUE
            AND (expires_at IS NULL OR expires_at > NOW())
        """
        public_share = await db_manager.fetchrow(public_share_query, workspace_id)
        
        if public_share:
            share_level = permission_levels.get(public_share['permission_level'], 0)
            if share_level >= required_level:
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to check workspace permission: {e}")
        return False


async def get_user_workspace_permission(
    db_manager: DatabaseConnectionManager,
    workspace_id: str,
    user_id: str,
    user_team_ids: Optional[list] = None
) -> Optional[str]:
    """
    Get user's permission level for a workspace
    
    Returns: 'admin', 'write', 'read', or None (no access)
    """
    try:
        # Check ownership
        ownership_query = """
            SELECT user_id FROM data_workspaces WHERE workspace_id = $1
        """
        owner = await db_manager.fetchval(ownership_query, workspace_id)
        
        if owner == user_id:
            return 'admin'
        
        # Check direct user shares
        user_share_query = """
            SELECT permission_level, expires_at
            FROM data_workspace_shares
            WHERE workspace_id = $1 
            AND shared_with_user_id = $2
            AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY 
                CASE permission_level
                    WHEN 'admin' THEN 3
                    WHEN 'write' THEN 2
                    WHEN 'read' THEN 1
                END DESC
            LIMIT 1
        """
        user_share = await db_manager.fetchrow(user_share_query, workspace_id, user_id)
        
        if user_share:
            return user_share['permission_level']
        
        # Check team shares (if team IDs provided)
        if user_team_ids:
            team_share_query = """
                SELECT permission_level, expires_at
                FROM data_workspace_shares
                WHERE workspace_id = $1 
                AND shared_with_team_id = ANY($2::VARCHAR[])
                AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY 
                    CASE permission_level
                        WHEN 'admin' THEN 3
                        WHEN 'write' THEN 2
                        WHEN 'read' THEN 1
                    END DESC
                LIMIT 1
            """
            team_share = await db_manager.fetchrow(team_share_query, workspace_id, user_team_ids)
            
            if team_share:
                return team_share['permission_level']
        
        # Check public shares
        public_share_query = """
            SELECT permission_level, expires_at
            FROM data_workspace_shares
            WHERE workspace_id = $1 
            AND is_public = TRUE
            AND (expires_at IS NULL OR expires_at > NOW())
            LIMIT 1
        """
        public_share = await db_manager.fetchrow(public_share_query, workspace_id)
        
        if public_share:
            return public_share['permission_level']
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get user workspace permission: {e}")
        return None

