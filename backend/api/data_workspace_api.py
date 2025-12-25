from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
import logging
import os
import uuid

from models.data_workspace_models import (
    CreateWorkspaceRequest, UpdateWorkspaceRequest, WorkspaceResponse,
    CreateDatabaseRequest, DatabaseResponse,
    CreateTableRequest, TableResponse,
    PreviewImportRequest, PreviewImportResponse,
    ExecuteImportRequest, ImportJobResponse,
    TableDataResponse, InsertRowRequest, UpdateRowRequest, UpdateCellRequest,
    ShareWorkspaceRequest, WorkspaceShareResponse,
    SQLQueryRequest, NaturalLanguageQueryRequest, QueryResultResponse,
    RecalculateTableResponse
)
from models.api_models import AuthenticatedUserResponse
from utils.auth_middleware import get_current_user
from utils.data_workspace_middleware import (
    require_workspace_permission,
    get_workspace_for_database,
    get_workspace_for_table
)
from services.data_workspace_grpc_client import DataWorkspaceGRPCClient
from services.data_workspace_sharing_service import get_sharing_service
from services.team_service import TeamService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/data", tags=["data_workspace"])

# Global gRPC client instance
_grpc_client = None


def get_grpc_client() -> DataWorkspaceGRPCClient:
    """Get or create gRPC client"""
    global _grpc_client
    if _grpc_client is None:
        _grpc_client = DataWorkspaceGRPCClient()
    return _grpc_client


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


# Workspace Endpoints
@router.post("/workspaces", response_model=WorkspaceResponse)
async def create_workspace(
    request: CreateWorkspaceRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a new workspace"""
    try:
        client = get_grpc_client()
        workspace = await client.create_workspace(
            user_id=current_user.user_id,
            name=request.name,
            description=request.description,
            icon=request.icon,
            color=request.color
        )
        return workspace
    except Exception as e:
        logger.error(f"Failed to create workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces", response_model=List[WorkspaceResponse])
async def list_workspaces(
    include_shared: bool = True,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List all workspaces for the current user (owned + shared)"""
    try:
        client = get_grpc_client()
        
        # Get user's team IDs for shared workspace lookup
        user_team_ids = None
        if include_shared:
            try:
                team_service = TeamService()
                await team_service.initialize()
                user_teams = await team_service.list_user_teams(current_user.user_id)
                user_team_ids = [team['team_id'] for team in user_teams]
            except Exception as e:
                logger.warning(f"Failed to get user teams for shared workspaces: {e}")
                user_team_ids = []
        
        # Get owned workspaces
        workspaces = await client.list_workspaces(current_user.user_id)
        
        # Add shared workspaces if requested
        if include_shared:
            try:
                sharing_service = await get_sharing_service()
                shared_workspaces = await sharing_service.list_shared_workspaces(
                    current_user.user_id,
                    user_team_ids
                )
                workspaces.extend(shared_workspaces)
            except Exception as e:
                logger.warning(f"Failed to get shared workspaces: {e}")
        
        return workspaces
    except Exception as e:
        logger.error(f"Failed to list workspaces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get a single workspace (requires read permission)"""
    try:
        # Check permission
        sharing_service = await get_sharing_service()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        has_access = await sharing_service.check_workspace_permission(
            workspace_id, current_user.user_id, 'read', user_team_ids
        )
        
        if not has_access:
            raise HTTPException(status_code=403, detail="Access denied")
        
        client = get_grpc_client()
        workspace = await client.get_workspace(
            workspace_id,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return workspace
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/workspaces/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: str,
    request: UpdateWorkspaceRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update a workspace (requires write permission)"""
    try:
        # Check permission
        sharing_service = await get_sharing_service()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        has_access = await sharing_service.check_workspace_permission(
            workspace_id, current_user.user_id, 'write', user_team_ids
        )
        
        if not has_access:
            raise HTTPException(status_code=403, detail="Write access required")
        
        client = get_grpc_client()
        workspace = await client.update_workspace(
            workspace_id=workspace_id,
            user_id=current_user.user_id,
            name=request.name,
            description=request.description,
            icon=request.icon,
            color=request.color,
            is_pinned=request.is_pinned
        )
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return workspace
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workspaces/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a workspace (requires admin permission)"""
    try:
        # Check permission
        sharing_service = await get_sharing_service()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        has_access = await sharing_service.check_workspace_permission(
            workspace_id, current_user.user_id, 'admin', user_team_ids
        )
        
        if not has_access:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        client = get_grpc_client()
        success = await client.delete_workspace(
            workspace_id,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        if not success:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return {"success": True, "message": "Workspace deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Database Endpoints
@router.post("/databases", response_model=DatabaseResponse)
async def create_database(
    request: CreateDatabaseRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a new database in a workspace (requires write permission)"""
    try:
        # Check workspace write permission
        await require_workspace_permission(request.workspace_id, 'write', current_user)
        
        client = get_grpc_client()
        database = await client.create_database(
            workspace_id=request.workspace_id,
            name=request.name,
            user_id=current_user.user_id,
            description=request.description,
            source_type=request.source_type
        )
        return database
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces/{workspace_id}/databases", response_model=List[DatabaseResponse])
async def list_databases(
    workspace_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List all databases in a workspace (requires read permission)"""
    try:
        # Check workspace read permission
        await require_workspace_permission(workspace_id, 'read', current_user)
        
        client = get_grpc_client()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        databases = await client.list_databases(
            workspace_id,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        return databases
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list databases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/databases/{database_id}", response_model=DatabaseResponse)
async def get_database(
    database_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get a single database (requires read permission)"""
    try:
        # Verify workspace access via database lookup
        workspace_id = await get_workspace_for_database(database_id, current_user)
        
        client = get_grpc_client()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        database = await client.get_database(
            database_id,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        if not database:
            raise HTTPException(status_code=404, detail="Database not found")
        return database
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/databases/{database_id}")
async def delete_database(
    database_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a database (requires admin permission)"""
    try:
        # Verify workspace admin access via database lookup
        workspace_id = await get_workspace_for_database(database_id, current_user)
        await require_workspace_permission(workspace_id, 'admin', current_user)
        
        client = get_grpc_client()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        success = await client.delete_database(
            database_id,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        if not success:
            raise HTTPException(status_code=404, detail="Database not found")
        return {"success": True, "message": "Database deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Import Endpoints
@router.post("/import/upload")
async def upload_import_file(
    workspace_id: str,
    file: UploadFile = File(...),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Upload a file for import"""
    try:
        # Save file to uploads directory
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        file_path = f"/app/uploads/data_imports/{file_id}{file_extension}"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Uploaded file for import: {file_path}")
        
        return {
            "file_id": file_id,
            "file_path": file_path,
            "filename": file.filename,
            "file_size": len(content)
        }
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/preview", response_model=PreviewImportResponse)
async def preview_import(
    request: PreviewImportRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Preview imported file and infer schema"""
    try:
        client = get_grpc_client()
        preview = await client.preview_import(
            file_path=request.file_path,
            file_type=request.file_type,
            preview_rows=request.preview_rows
        )
        return preview
    except Exception as e:
        logger.error(f"Failed to preview import: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import/execute", response_model=ImportJobResponse)
async def execute_import(
    request: ExecuteImportRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Execute import job (requires write permission)"""
    try:
        # Check workspace write permission
        await require_workspace_permission(request.workspace_id, 'write', current_user)
        
        client = get_grpc_client()
        job = await client.execute_import(
            workspace_id=request.workspace_id,
            database_id=request.database_id,
            table_name=request.table_name,
            file_path=request.file_path,
            user_id=current_user.user_id,
            field_mapping=request.field_mapping
        )
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute import: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/import/jobs/{job_id}", response_model=ImportJobResponse)
async def get_import_status(
    job_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get import job status"""
    try:
        client = get_grpc_client()
        job = await client.get_import_status(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Import job not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get import status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Table Endpoints
@router.post("/tables", response_model=TableResponse)
async def create_table(
    request: CreateTableRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a new table in a database (requires write permission)"""
    try:
        # Verify workspace write access via database lookup
        workspace_id = await get_workspace_for_database(request.database_id, current_user)
        await require_workspace_permission(workspace_id, 'write', current_user)
        
        client = get_grpc_client()
        table = await client.create_table(
            database_id=request.database_id,
            name=request.name,
            user_id=current_user.user_id,
            description=request.description,
            table_schema=request.table_schema
        )
        return table
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/databases/{database_id}/tables", response_model=List[TableResponse])
async def list_tables(
    database_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List all tables in a database (requires read permission)"""
    try:
        # Verify workspace read access via database lookup
        workspace_id = await get_workspace_for_database(database_id, current_user)
        
        client = get_grpc_client()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        tables = await client.list_tables(
            database_id,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        return tables
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables/{table_id}", response_model=TableResponse)
async def get_table(
    table_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get a single table (requires read permission)"""
    try:
        # Verify workspace read access via table lookup
        workspace_id = await get_workspace_for_table(table_id, current_user)
        
        client = get_grpc_client()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        table = await client.get_table(
            table_id,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        if not table:
            raise HTTPException(status_code=404, detail="Table not found")
        return table
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tables/{table_id}")
async def delete_table(
    table_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a table (requires admin permission)"""
    try:
        # Verify workspace admin access via table lookup
        workspace_id = await get_workspace_for_table(table_id, current_user)
        await require_workspace_permission(workspace_id, 'admin', current_user)
        
        client = get_grpc_client()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        success = await client.delete_table(
            table_id,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        if not success:
            raise HTTPException(status_code=404, detail="Table not found")
        return {"success": True, "message": "Table deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Table Data Endpoints
@router.get("/tables/{table_id}/data", response_model=TableDataResponse)
async def get_table_data(
    table_id: str,
    offset: int = 0,
    limit: int = 100,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get table data with pagination (requires read permission)"""
    try:
        # Verify workspace read access via table lookup
        workspace_id = await get_workspace_for_table(table_id, current_user)
        
        client = get_grpc_client()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        data = await client.get_table_data(
            table_id, 
            offset, 
            limit,
            user_id=current_user.user_id,
            user_team_ids=user_team_ids
        )
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get table data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tables/{table_id}/rows")
async def insert_table_row(
    table_id: str,
    request: InsertRowRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Insert a new row into a table (requires write permission)"""
    try:
        # Verify workspace write access via table lookup
        workspace_id = await get_workspace_for_table(table_id, current_user)
        await require_workspace_permission(workspace_id, 'write', current_user)
        
        client = get_grpc_client()
        row = await client.insert_table_row(table_id, request.row_data, current_user.user_id)
        return row
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to insert row: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/tables/{table_id}/rows/{row_id}")
async def update_table_row(
    table_id: str,
    row_id: str,
    request: UpdateRowRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update an existing row in a table (requires write permission)"""
    try:
        # Verify workspace write access via table lookup
        workspace_id = await get_workspace_for_table(table_id, current_user)
        await require_workspace_permission(workspace_id, 'write', current_user)
        
        client = get_grpc_client()
        row = await client.update_table_row(table_id, row_id, request.row_data, current_user.user_id)
        return row
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update row: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/tables/{table_id}/rows/{row_id}/cells")
async def update_table_cell(
    table_id: str,
    row_id: str,
    request: UpdateCellRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update a single cell in a row with optional formula"""
    try:
        # Verify workspace write access via table lookup
        workspace_id = await get_workspace_for_table(table_id, current_user)
        await require_workspace_permission(workspace_id, 'write', current_user)
        
        client = get_grpc_client()
        result = await client.update_table_cell(
            table_id, row_id, request.column_name, request.value, current_user.user_id, request.formula
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update cell: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tables/{table_id}/rows/{row_id}")
async def delete_table_row(
    table_id: str,
    row_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a row from a table (requires write permission)"""
    try:
        # Verify workspace write access via table lookup
        workspace_id = await get_workspace_for_table(table_id, current_user)
        await require_workspace_permission(workspace_id, 'write', current_user)
        
        client = get_grpc_client()
        success = await client.delete_table_row(table_id, row_id)
        if not success:
            raise HTTPException(status_code=404, detail="Row not found")
        return {"success": True, "message": "Row deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete row: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tables/{table_id}/recalculate", response_model=RecalculateTableResponse)
async def recalculate_table(
    table_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Manually trigger recalculation of all formulas in table (requires write permission)"""
    try:
        # Verify workspace write access via table lookup
        workspace_id = await get_workspace_for_table(table_id, current_user)
        await require_workspace_permission(workspace_id, 'write', current_user)
        
        client = get_grpc_client()
        result = await client.recalculate_table(table_id, current_user.user_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to recalculate table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Sharing Endpoints
@router.post("/workspaces/{workspace_id}/share", response_model=WorkspaceShareResponse)
async def share_workspace(
    workspace_id: str,
    request: ShareWorkspaceRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Share a workspace with a user, team, or make it public"""
    try:
        sharing_service = await get_sharing_service()
        share = await sharing_service.share_workspace(
            workspace_id=workspace_id,
            shared_by_user_id=current_user.user_id,
            shared_with_user_id=request.shared_with_user_id,
            shared_with_team_id=request.shared_with_team_id,
            permission_level=request.permission_level.value,
            is_public=request.is_public,
            expires_at=request.expires_at
        )
        return share
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to share workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces/{workspace_id}/shares", response_model=List[WorkspaceShareResponse])
async def list_workspace_shares(
    workspace_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List all shares for a workspace"""
    try:
        sharing_service = await get_sharing_service()
        shares = await sharing_service.list_workspace_shares(
            workspace_id=workspace_id,
            user_id=current_user.user_id
        )
        return shares
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to list workspace shares: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workspaces/{workspace_id}/shares/{share_id}")
async def revoke_share(
    workspace_id: str,
    share_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Revoke a workspace share"""
    try:
        sharing_service = await get_sharing_service()
        success = await sharing_service.revoke_share(
            workspace_id=workspace_id,
            share_id=share_id,
            user_id=current_user.user_id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Share not found")
        return {"success": True, "message": "Share revoked successfully"}
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke share: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces/shared", response_model=List[WorkspaceResponse])
async def list_shared_workspaces(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List workspaces shared with the current user"""
    try:
        # Get user's team IDs
        user_team_ids = None
        try:
            team_service = TeamService()
            await team_service.initialize()
            user_teams = await team_service.list_user_teams(current_user.user_id)
            user_team_ids = [team['team_id'] for team in user_teams]
        except Exception as e:
            logger.warning(f"Failed to get user teams: {e}")
            user_team_ids = []
        
        sharing_service = await get_sharing_service()
        workspaces = await sharing_service.list_shared_workspaces(
            current_user.user_id,
            user_team_ids
        )
        return workspaces
    except Exception as e:
        logger.error(f"Failed to list shared workspaces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Query Endpoints
@router.post("/workspaces/{workspace_id}/query/sql", response_model=QueryResultResponse)
async def execute_sql_query(
    workspace_id: str,
    request: SQLQueryRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Execute SQL query against workspace databases (requires read permission)"""
    try:
        # Check workspace read permission
        await require_workspace_permission(workspace_id, 'read', current_user)
        
        client = get_grpc_client()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        
        result = await client.execute_sql_query(
            workspace_id=workspace_id,
            query=request.query,
            user_id=current_user.user_id,
            limit=request.limit,
            user_team_ids=user_team_ids
        )
        
        # Parse results_json to List[Dict]
        import json
        results = json.loads(result['results_json']) if result.get('results_json') else []
        
        return QueryResultResponse(
            query_id=result['query_id'],
            column_names=result['column_names'],
            results=results,
            result_count=result['result_count'],
            execution_time_ms=result['execution_time_ms'],
            generated_sql=result.get('generated_sql'),
            error_message=result.get('error_message')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute SQL query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workspaces/{workspace_id}/query/natural-language", response_model=QueryResultResponse)
async def execute_nl_query(
    workspace_id: str,
    request: NaturalLanguageQueryRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Execute natural language query (requires read permission)"""
    try:
        # Check workspace read permission
        await require_workspace_permission(workspace_id, 'read', current_user)
        
        client = get_grpc_client()
        user_team_ids = await _get_user_team_ids(current_user.user_id)
        
        result = await client.execute_nl_query(
            workspace_id=workspace_id,
            natural_query=request.query,
            user_id=current_user.user_id,
            include_documents=request.include_documents,
            user_team_ids=user_team_ids
        )
        
        # Parse results_json to List[Dict]
        import json
        results = json.loads(result['results_json']) if result.get('results_json') else []
        
        return QueryResultResponse(
            query_id=result['query_id'],
            column_names=result['column_names'],
            results=results,
            result_count=result['result_count'],
            execution_time_ms=result['execution_time_ms'],
            generated_sql=result.get('generated_sql'),
            error_message=result.get('error_message')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute NL query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

