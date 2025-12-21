import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import json

from db.connection_manager import DatabaseConnectionManager

logger = logging.getLogger(__name__)


class WorkspaceService:
    """Service for managing data workspaces"""
    
    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db = db_manager
    
    async def create_workspace(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        icon: Optional[str] = None,
        color: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new workspace"""
        try:
            workspace_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            query = """
                INSERT INTO data_workspaces 
                (workspace_id, user_id, name, description, icon, color, is_pinned, metadata_json, created_at, updated_at, updated_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING workspace_id, user_id, name, description, icon, color, is_pinned, 
                          metadata_json, created_at, updated_at, updated_by
            """
            
            row = await self.db.fetchrow(
                query,
                workspace_id, user_id, name, description, icon, color,
                False, json.dumps({}), now, now, user_id
            )
            
            logger.info(f"Created workspace: {workspace_id} for user: {user_id}")
            return self._row_to_dict(row)
            
        except Exception as e:
            logger.error(f"Failed to create workspace: {e}")
            raise
    
    async def list_workspaces(
        self, 
        user_id: str,
        include_shared: bool = False,
        user_team_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List all workspaces for a user (owned + optionally shared)"""
        try:
            # Get owned workspaces
            owned_query = """
                SELECT workspace_id, user_id, name, description, icon, color, is_pinned,
                       metadata_json, created_at, updated_at, updated_by
                FROM data_workspaces
                WHERE user_id = $1
                ORDER BY is_pinned DESC, created_at DESC
            """
            
            owned_rows = await self.db.fetch(
                owned_query, 
                user_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            workspaces = [self._row_to_dict(row) for row in owned_rows]
            
            # Add shared workspaces if requested
            if include_shared:
                shared_workspaces = await self.list_shared_workspaces(user_id, user_team_ids)
                workspaces.extend(shared_workspaces)
            
            return workspaces
            
        except Exception as e:
            logger.error(f"Failed to list workspaces: {e}")
            raise
    
    async def get_workspace(
        self, 
        workspace_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a single workspace by ID"""
        try:
            query = """
                SELECT workspace_id, user_id, name, description, icon, color, is_pinned,
                       metadata_json, created_at, updated_at, updated_by
                FROM data_workspaces
                WHERE workspace_id = $1
            """
            
            row = await self.db.fetchrow(
                query, 
                workspace_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            return self._row_to_dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Failed to get workspace {workspace_id}: {e}")
            raise
    
    async def update_workspace(
        self,
        workspace_id: str,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        icon: Optional[str] = None,
        color: Optional[str] = None,
        is_pinned: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a workspace"""
        try:
            # Get current workspace
            current = await self.get_workspace(workspace_id)
            if not current:
                return None
            
            # Update only provided fields
            updates = {
                'name': name if name is not None else current['name'],
                'description': description if description is not None else current['description'],
                'icon': icon if icon is not None else current['icon'],
                'color': color if color is not None else current['color'],
                'is_pinned': is_pinned if is_pinned is not None else current['is_pinned'],
                'updated_at': datetime.utcnow()
            }
            
            query = """
                UPDATE data_workspaces
                SET name = $2, description = $3, icon = $4, color = $5, 
                    is_pinned = $6, updated_at = $7, updated_by = $8
                WHERE workspace_id = $1
                RETURNING workspace_id, user_id, name, description, icon, color, is_pinned,
                          metadata_json, created_at, updated_at, updated_by
            """
            
            row = await self.db.fetchrow(
                query,
                workspace_id,
                updates['name'],
                updates['description'],
                updates['icon'],
                updates['color'],
                updates['is_pinned'],
                updates['updated_at'],
                user_id,
                user_id=user_id,
                user_team_ids=None
            )
            
            logger.info(f"Updated workspace: {workspace_id}")
            return self._row_to_dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Failed to update workspace {workspace_id}: {e}")
            raise
    
    async def delete_workspace(
        self, 
        workspace_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> bool:
        """Delete a workspace and all associated data"""
        try:
            query = "DELETE FROM data_workspaces WHERE workspace_id = $1"
            result = await self.db.execute(
                query, 
                workspace_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            # Check if any rows were deleted
            deleted = result.split()[-1] != '0'
            
            if deleted:
                logger.info(f"Deleted workspace: {workspace_id}")
            else:
                logger.warning(f"Workspace not found for deletion: {workspace_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete workspace {workspace_id}: {e}")
            raise
    
    async def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get statistics for a workspace"""
        try:
            # Count databases
            db_count_query = """
                SELECT COUNT(*) FROM custom_databases WHERE workspace_id = $1
            """
            db_count = await self.db.fetchval(db_count_query, workspace_id)
            
            # Count tables
            table_count_query = """
                SELECT COUNT(ct.table_id)
                FROM custom_tables ct
                JOIN custom_databases cd ON ct.database_id = cd.database_id
                WHERE cd.workspace_id = $1
            """
            table_count = await self.db.fetchval(table_count_query, workspace_id)
            
            # Count total rows
            row_count_query = """
                SELECT COALESCE(SUM(ct.row_count), 0)
                FROM custom_tables ct
                JOIN custom_databases cd ON ct.database_id = cd.database_id
                WHERE cd.workspace_id = $1
            """
            row_count = await self.db.fetchval(row_count_query, workspace_id)
            
            # Count visualizations
            viz_count_query = """
                SELECT COUNT(*) FROM data_visualizations WHERE workspace_id = $1
            """
            viz_count = await self.db.fetchval(viz_count_query, workspace_id)
            
            return {
                'workspace_id': workspace_id,
                'database_count': db_count,
                'table_count': table_count,
                'total_rows': row_count,
                'visualization_count': viz_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get workspace stats {workspace_id}: {e}")
            raise
    
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
        """Share a workspace with a user, team, or make it public"""
        try:
            # Verify workspace exists and user is owner
            workspace = await self.get_workspace(workspace_id)
            if not workspace:
                raise ValueError("Workspace not found")
            
            if workspace['user_id'] != shared_by_user_id:
                raise ValueError("Only workspace owner can share")
            
            # Validate share parameters
            if not is_public and not shared_with_user_id and not shared_with_team_id:
                raise ValueError("Must specify user, team, or set is_public=True")
            
            if is_public and (shared_with_user_id or shared_with_team_id):
                raise ValueError("Public shares cannot specify user or team")
            
            share_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            query = """
                INSERT INTO data_workspace_shares
                (share_id, workspace_id, shared_by_user_id, shared_with_user_id, 
                 shared_with_team_id, permission_level, is_public, expires_at, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING share_id, workspace_id, shared_by_user_id, shared_with_user_id,
                          shared_with_team_id, permission_level, is_public, expires_at,
                          created_at, access_count
            """
            
            row = await self.db.fetchrow(
                query,
                share_id, workspace_id, shared_by_user_id, shared_with_user_id,
                shared_with_team_id, permission_level, is_public, expires_at, now
            )
            
            logger.info(f"Shared workspace {workspace_id} via share {share_id}")
            return self._share_row_to_dict(row)
            
        except Exception as e:
            logger.error(f"Failed to share workspace: {e}")
            raise
    
    async def list_workspace_shares(
        self,
        workspace_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """List all shares for a workspace (only owner can view)"""
        try:
            # Verify user is owner
            workspace = await self.get_workspace(workspace_id)
            if not workspace:
                raise ValueError("Workspace not found")
            
            if workspace['user_id'] != user_id:
                raise ValueError("Only workspace owner can view shares")
            
            query = """
                SELECT share_id, workspace_id, shared_by_user_id, shared_with_user_id,
                       shared_with_team_id, permission_level, is_public, expires_at,
                       created_at, access_count
                FROM data_workspace_shares
                WHERE workspace_id = $1
                ORDER BY created_at DESC
            """
            
            rows = await self.db.fetch(query, workspace_id)
            return [self._share_row_to_dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list workspace shares: {e}")
            raise
    
    async def revoke_share(
        self,
        workspace_id: str,
        share_id: str,
        user_id: str
    ) -> bool:
        """Revoke a workspace share (only owner can revoke)"""
        try:
            # Verify user is owner
            workspace = await self.get_workspace(workspace_id)
            if not workspace:
                raise ValueError("Workspace not found")
            
            if workspace['user_id'] != user_id:
                raise ValueError("Only workspace owner can revoke shares")
            
            query = """
                DELETE FROM data_workspace_shares
                WHERE workspace_id = $1 AND share_id = $2
            """
            
            result = await self.db.execute(query, workspace_id, share_id)
            deleted = result.split()[-1] != '0'
            
            if deleted:
                logger.info(f"Revoked share {share_id} for workspace {workspace_id}")
            else:
                logger.warning(f"Share not found for revocation: {share_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to revoke share: {e}")
            raise
    
    async def list_shared_workspaces(
        self,
        user_id: str,
        user_team_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List workspaces shared with a user (via direct share or team membership)"""
        try:
            # Build query for shared workspaces
            # Include: direct user shares, team shares, and public shares
            conditions = []
            params = [user_id]
            param_idx = 1
            
            # Direct user shares
            conditions.append(f"dws.shared_with_user_id = ${param_idx}")
            param_idx += 1
            
            # Team shares (if user is in teams)
            if user_team_ids:
                conditions.append(f"dws.shared_with_team_id = ANY(${param_idx}::VARCHAR[])")
                params.append(user_team_ids)
                param_idx += 1
            
            # Public shares
            conditions.append(f"dws.is_public = TRUE")
            
            # Combine conditions with OR
            where_clause = " OR ".join([f"({c})" for c in conditions])
            
            query = f"""
                SELECT DISTINCT dw.workspace_id, dw.user_id, dw.name, dw.description, 
                       dw.icon, dw.color, dw.is_pinned, dw.metadata_json,
                       dw.created_at, dw.updated_at, dw.updated_by,
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
            
            rows = await self.db.fetch(query, *params)
            return [self._row_to_dict_with_share(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list shared workspaces: {e}")
            raise
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        if not row:
            return {}
        
        return {
            'workspace_id': row['workspace_id'],
            'user_id': row['user_id'],
            'name': row['name'],
            'description': row['description'],
            'icon': row['icon'],
            'color': row['color'],
            'is_pinned': row['is_pinned'],
            'metadata_json': row['metadata_json'] if isinstance(row['metadata_json'], str) else json.dumps(row['metadata_json']),
            'created_at': row['created_at'].isoformat() if row['created_at'] else None,
            'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None,
            'updated_by': row.get('updated_by')
        }
    
    def _row_to_dict_with_share(self, row) -> Dict[str, Any]:
        """Convert workspace row with share info to dictionary"""
        result = self._row_to_dict(row)
        if row and 'permission_level' in row:
            result['permission_level'] = row['permission_level']
        if row and 'share_type' in row:
            result['share_type'] = row['share_type']
        # Ensure updated_by is included
        if row and 'updated_by' in row:
            result['updated_by'] = row['updated_by']
        return result
    
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









