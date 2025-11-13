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
                (workspace_id, user_id, name, description, icon, color, is_pinned, metadata_json, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING workspace_id, user_id, name, description, icon, color, is_pinned, 
                          metadata_json, created_at, updated_at
            """
            
            row = await self.db.fetchrow(
                query,
                workspace_id, user_id, name, description, icon, color,
                False, json.dumps({}), now, now
            )
            
            logger.info(f"Created workspace: {workspace_id} for user: {user_id}")
            return self._row_to_dict(row)
            
        except Exception as e:
            logger.error(f"Failed to create workspace: {e}")
            raise
    
    async def list_workspaces(self, user_id: str) -> List[Dict[str, Any]]:
        """List all workspaces for a user"""
        try:
            query = """
                SELECT workspace_id, user_id, name, description, icon, color, is_pinned,
                       metadata_json, created_at, updated_at
                FROM data_workspaces
                WHERE user_id = $1
                ORDER BY is_pinned DESC, created_at DESC
            """
            
            rows = await self.db.fetch(query, user_id)
            return [self._row_to_dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list workspaces: {e}")
            raise
    
    async def get_workspace(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """Get a single workspace by ID"""
        try:
            query = """
                SELECT workspace_id, user_id, name, description, icon, color, is_pinned,
                       metadata_json, created_at, updated_at
                FROM data_workspaces
                WHERE workspace_id = $1
            """
            
            row = await self.db.fetchrow(query, workspace_id)
            return self._row_to_dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Failed to get workspace {workspace_id}: {e}")
            raise
    
    async def update_workspace(
        self,
        workspace_id: str,
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
                    is_pinned = $6, updated_at = $7
                WHERE workspace_id = $1
                RETURNING workspace_id, user_id, name, description, icon, color, is_pinned,
                          metadata_json, created_at, updated_at
            """
            
            row = await self.db.fetchrow(
                query,
                workspace_id,
                updates['name'],
                updates['description'],
                updates['icon'],
                updates['color'],
                updates['is_pinned'],
                updates['updated_at']
            )
            
            logger.info(f"Updated workspace: {workspace_id}")
            return self._row_to_dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Failed to update workspace {workspace_id}: {e}")
            raise
    
    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace and all associated data"""
        try:
            query = "DELETE FROM data_workspaces WHERE workspace_id = $1"
            result = await self.db.execute(query, workspace_id)
            
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
            'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None
        }





