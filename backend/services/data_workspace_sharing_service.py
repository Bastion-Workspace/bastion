"""
Data Workspace Sharing Service
Direct database access for sharing operations (bypasses gRPC for now)
"""

import logging
import uuid
import asyncpg
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class DataWorkspaceSharingService:
    """Service for managing workspace shares via direct database access"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection pool"""
        if self._initialized:
            return
        
        try:
            # Get data-service database connection details
            host = os.getenv("DATA_SERVICE_DB_HOST", "postgres-data")
            port = int(os.getenv("DATA_SERVICE_DB_PORT", "5432"))
            database = os.getenv("DATA_SERVICE_DB_NAME", "data_workspace")
            user = os.getenv("DATA_SERVICE_DB_USER", "data_user")
            password = os.getenv("DATA_SERVICE_DB_PASSWORD", "data_workspace_secure_password")
            
            self.pool = await asyncpg.create_pool(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                min_size=2,
                max_size=5
            )
            
            self._initialized = True
            logger.info("Data workspace sharing service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize sharing service: {e}")
            raise
    
    async def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self._initialized:
            await self.initialize()
    
    async def share_workspace(
        self,
        workspace_id: str,
        shared_by_user_id: str,
        shared_with_user_id: Optional[str] = None,
        shared_with_team_id: Optional[str] = None,
        permission_level: str = 'read',
        is_public: bool = False,
        expires_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Share a workspace"""
        await self._ensure_initialized()
        
        async with self.pool.acquire() as conn:
            # Verify workspace exists and user is owner
            workspace = await conn.fetchrow(
                "SELECT user_id FROM data_workspaces WHERE workspace_id = $1",
                workspace_id
            )
            
            if not workspace:
                raise ValueError("Workspace not found")
            
            if workspace['user_id'] != shared_by_user_id:
                raise ValueError("Only workspace owner can share")
            
            # Validate share parameters
            if not is_public and not shared_with_user_id and not shared_with_team_id:
                raise ValueError("Must specify user, team, or set is_public=True")
            
            share_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            row = await conn.fetchrow(
                """
                INSERT INTO data_workspace_shares
                (share_id, workspace_id, shared_by_user_id, shared_with_user_id, 
                 shared_with_team_id, permission_level, is_public, expires_at, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING share_id, workspace_id, shared_by_user_id, shared_with_user_id,
                          shared_with_team_id, permission_level, is_public, expires_at,
                          created_at, access_count
                """,
                share_id, workspace_id, shared_by_user_id, shared_with_user_id,
                shared_with_team_id, permission_level, is_public, expires_at, now
            )
            
            return self._share_row_to_dict(row)
    
    async def list_workspace_shares(
        self,
        workspace_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """List all shares for a workspace"""
        await self._ensure_initialized()
        
        async with self.pool.acquire() as conn:
            # Verify user is owner
            workspace = await conn.fetchrow(
                "SELECT user_id FROM data_workspaces WHERE workspace_id = $1",
                workspace_id
            )
            
            if not workspace:
                raise ValueError("Workspace not found")
            
            if workspace['user_id'] != user_id:
                raise ValueError("Only workspace owner can view shares")
            
            rows = await conn.fetch(
                """
                SELECT share_id, workspace_id, shared_by_user_id, shared_with_user_id,
                       shared_with_team_id, permission_level, is_public, expires_at,
                       created_at, access_count
                FROM data_workspace_shares
                WHERE workspace_id = $1
                ORDER BY created_at DESC
                """,
                workspace_id
            )
            
            return [self._share_row_to_dict(row) for row in rows]
    
    async def revoke_share(
        self,
        workspace_id: str,
        share_id: str,
        user_id: str
    ) -> bool:
        """Revoke a workspace share"""
        await self._ensure_initialized()
        
        async with self.pool.acquire() as conn:
            # Verify user is owner
            workspace = await conn.fetchrow(
                "SELECT user_id FROM data_workspaces WHERE workspace_id = $1",
                workspace_id
            )
            
            if not workspace:
                raise ValueError("Workspace not found")
            
            if workspace['user_id'] != user_id:
                raise ValueError("Only workspace owner can revoke shares")
            
            result = await conn.execute(
                "DELETE FROM data_workspace_shares WHERE workspace_id = $1 AND share_id = $2",
                workspace_id, share_id
            )
            
            return result.split()[-1] != '0'
    
    async def list_shared_workspaces(
        self,
        user_id: str,
        user_team_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List workspaces shared with a user"""
        await self._ensure_initialized()
        
        async with self.pool.acquire() as conn:
            # Build query conditions
            conditions = []
            params = [user_id]
            param_idx = 1
            
            # Direct user shares
            conditions.append(f"dws.shared_with_user_id = ${param_idx}")
            param_idx += 1
            
            # Team shares
            if user_team_ids:
                conditions.append(f"dws.shared_with_team_id = ANY(${param_idx}::VARCHAR[])")
                params.append(user_team_ids)
                param_idx += 1
            
            # Public shares
            conditions.append("dws.is_public = TRUE")
            
            where_clause = " OR ".join([f"({c})" for c in conditions])
            
            query = f"""
                SELECT DISTINCT dw.workspace_id, dw.user_id, dw.name, dw.description, 
                       dw.icon, dw.color, dw.is_pinned, dw.metadata_json,
                       dw.created_at, dw.updated_at,
                       dws.permission_level,
                       CASE 
                           WHEN dws.shared_with_user_id = $1 THEN 'user'
                           WHEN dws.shared_with_team_id IS NOT NULL THEN 'team'
                           WHEN dws.is_public = TRUE THEN 'public'
                       END as share_type
                FROM data_workspaces dw
                JOIN data_workspace_shares dws ON dw.workspace_id = dws.workspace_id
                WHERE ({where_clause})
                AND (dws.expires_at IS NULL OR dws.expires_at > NOW())
                AND dw.user_id != $1
                ORDER BY dw.created_at DESC
            """
            
            rows = await conn.fetch(query, *params)
            return [self._workspace_row_to_dict(row) for row in rows]
    
    async def check_workspace_permission(
        self,
        workspace_id: str,
        user_id: str,
        required_permission: str,
        user_team_ids: Optional[List[str]] = None
    ) -> bool:
        """Check if user has required permission on workspace"""
        await self._ensure_initialized()
        
        async with self.pool.acquire() as conn:
            # Check ownership
            owner = await conn.fetchval(
                "SELECT user_id FROM data_workspaces WHERE workspace_id = $1",
                workspace_id
            )
            
            if owner == user_id:
                return True  # Owner has all permissions
            
            permission_levels = {'read': 1, 'write': 2, 'admin': 3}
            required_level = permission_levels.get(required_permission, 0)
            
            if required_level == 0:
                return False
            
            # Check direct user shares
            user_share = await conn.fetchrow(
                """
                SELECT permission_level, expires_at
                FROM data_workspace_shares
                WHERE workspace_id = $1 
                AND shared_with_user_id = $2
                AND (expires_at IS NULL OR expires_at > NOW())
                """,
                workspace_id, user_id
            )
            
            if user_share:
                share_level = permission_levels.get(user_share['permission_level'], 0)
                if share_level >= required_level:
                    return True
            
            # Check team shares
            if user_team_ids:
                team_share = await conn.fetchrow(
                    """
                    SELECT permission_level, expires_at
                    FROM data_workspace_shares
                    WHERE workspace_id = $1 
                    AND shared_with_team_id = ANY($2::VARCHAR[])
                    AND (expires_at IS NULL OR expires_at > NOW())
                    """,
                    workspace_id, user_team_ids
                )
                
                if team_share:
                    share_level = permission_levels.get(team_share['permission_level'], 0)
                    if share_level >= required_level:
                        return True
            
            # Check public shares
            public_share = await conn.fetchrow(
                """
                SELECT permission_level, expires_at
                FROM data_workspace_shares
                WHERE workspace_id = $1 
                AND is_public = TRUE
                AND (expires_at IS NULL OR expires_at > NOW())
                """,
                workspace_id
            )
            
            if public_share:
                share_level = permission_levels.get(public_share['permission_level'], 0)
                if share_level >= required_level:
                    return True
            
            return False
    
    def _workspace_row_to_dict(self, row) -> Dict[str, Any]:
        """Convert workspace row with share info to dictionary"""
        return {
            'workspace_id': row['workspace_id'],
            'user_id': row['user_id'],
            'name': row['name'],
            'description': row['description'],
            'icon': row['icon'],
            'color': row['color'],
            'is_pinned': row['is_pinned'],
            'metadata_json': row['metadata_json'] if isinstance(row['metadata_json'], str) else str(row['metadata_json']),
            'created_at': row['created_at'].isoformat() if row['created_at'] else None,
            'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None,
            'permission_level': row.get('permission_level'),
            'is_shared': True,
            'share_type': row.get('share_type')
        }
    
    def _share_row_to_dict(self, row) -> Dict[str, Any]:
        """Convert share row to dictionary"""
        if not row:
            return {}
        
        return {
            'share_id': row['share_id'],
            'workspace_id': row['workspace_id'],
            'shared_by_user_id': row['shared_by_user_id'],
            'shared_with_user_id': row['shared_with_user_id'],
            'shared_with_team_id': row['shared_with_team_id'],
            'permission_level': row['permission_level'],
            'is_public': row['is_public'],
            'expires_at': row['expires_at'].isoformat() if row['expires_at'] else None,
            'created_at': row['created_at'].isoformat() if row['created_at'] else None,
            'access_count': row['access_count']
        }
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self._initialized = False


# Global instance
_sharing_service: Optional[DataWorkspaceSharingService] = None


async def get_sharing_service() -> DataWorkspaceSharingService:
    """Get the global sharing service instance"""
    global _sharing_service
    if _sharing_service is None:
        _sharing_service = DataWorkspaceSharingService()
        await _sharing_service.initialize()
    return _sharing_service

