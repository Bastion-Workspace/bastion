"""
Data Workspace Permission Middleware
Helper functions for enforcing workspace permissions in API endpoints
"""

import logging
from typing import Optional, List
from fastapi import HTTPException

from models.api_models import AuthenticatedUserResponse
from services.data_workspace_sharing_service import get_sharing_service
from services.team_service import TeamService

logger = logging.getLogger(__name__)


async def _get_user_team_ids(user_id: str) -> List[str]:
    """Helper to get user's team IDs"""
    try:
        team_service = TeamService()
        await team_service.initialize()
        user_teams = await team_service.list_user_teams(user_id)
        return [team['team_id'] for team in user_teams]
    except Exception as e:
        logger.warning(f"Failed to get user teams: {e}")
        return []


async def require_workspace_permission(
    workspace_id: str,
    required_permission: str,  # 'read', 'write', 'admin'
    current_user: AuthenticatedUserResponse
) -> None:
    """
    Raise HTTPException(403) if user lacks required permission on workspace
    
    Args:
        workspace_id: Workspace to check access for
        required_permission: Required permission level ('read', 'write', 'admin')
        current_user: Authenticated user making the request
        
    Raises:
        HTTPException: 403 if user lacks permission, 404 if workspace not found
    """
    try:
        sharing_service = await get_sharing_service()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        
        has_access = await sharing_service.check_workspace_permission(
            workspace_id, 
            current_user.user_id, 
            required_permission, 
            user_team_ids
        )
        
        if not has_access:
            raise HTTPException(
                status_code=403, 
                detail=f"{required_permission.capitalize()} access required for this workspace"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Permission check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to verify workspace access"
        )


async def get_workspace_for_database(
    database_id: str,
    current_user: AuthenticatedUserResponse,
    grpc_client=None
) -> str:
    """
    Lookup workspace_id for a database and verify user has read access
    
    Args:
        database_id: Database to lookup
        current_user: Authenticated user
        grpc_client: Optional gRPC client (to avoid circular imports)
        
    Returns:
        workspace_id: Workspace ID that contains the database
        
    Raises:
        HTTPException: 404 if database not found, 403 if no access
    """
    if grpc_client is None:
        from services.data_workspace_grpc_client import DataWorkspaceGRPCClient
        grpc_client = DataWorkspaceGRPCClient()
    
    try:
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        
        database = await grpc_client.get_database(
            database_id,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        
        if not database:
            raise HTTPException(status_code=404, detail="Database not found")
        
        workspace_id = database.get('workspace_id')
        if not workspace_id:
            raise HTTPException(status_code=404, detail="Database workspace not found")
        
        # Verify user has at least read access
        await require_workspace_permission(workspace_id, 'read', current_user)
        
        return workspace_id
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workspace for database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_workspace_for_table(
    table_id: str,
    current_user: AuthenticatedUserResponse,
    grpc_client=None
) -> str:
    """
    Lookup workspace_id for a table and verify user has read access
    
    Args:
        table_id: Table to lookup
        current_user: Authenticated user
        grpc_client: Optional gRPC client (to avoid circular imports)
        
    Returns:
        workspace_id: Workspace ID that contains the table
        
    Raises:
        HTTPException: 404 if table not found, 403 if no access
    """
    if grpc_client is None:
        from services.data_workspace_grpc_client import DataWorkspaceGRPCClient
        grpc_client = DataWorkspaceGRPCClient()
    
    try:
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        
        table = await grpc_client.get_table(
            table_id,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        
        if not table:
            raise HTTPException(status_code=404, detail="Table not found")
        
        database_id = table.get('database_id')
        if not database_id:
            raise HTTPException(status_code=404, detail="Table database not found")
        
        # Get workspace via database
        return await get_workspace_for_database(database_id, current_user, grpc_client)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workspace for table: {e}")
        raise HTTPException(status_code=500, detail=str(e))

