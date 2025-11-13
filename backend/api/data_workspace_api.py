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
    TableDataResponse, InsertRowRequest, UpdateRowRequest, UpdateCellRequest
)
from models.api_models import AuthenticatedUserResponse
from utils.auth_middleware import get_current_user
from services.data_workspace_grpc_client import DataWorkspaceGRPCClient

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
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List all workspaces for the current user"""
    try:
        client = get_grpc_client()
        workspaces = await client.list_workspaces(current_user.user_id)
        return workspaces
    except Exception as e:
        logger.error(f"Failed to list workspaces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get a single workspace"""
    try:
        client = get_grpc_client()
        workspace = await client.get_workspace(workspace_id)
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
    """Update a workspace"""
    try:
        client = get_grpc_client()
        workspace = await client.update_workspace(
            workspace_id=workspace_id,
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
    """Delete a workspace"""
    try:
        client = get_grpc_client()
        success = await client.delete_workspace(workspace_id)
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
    """Create a new database in a workspace"""
    try:
        client = get_grpc_client()
        database = await client.create_database(
            workspace_id=request.workspace_id,
            name=request.name,
            description=request.description,
            source_type=request.source_type
        )
        return database
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces/{workspace_id}/databases", response_model=List[DatabaseResponse])
async def list_databases(
    workspace_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List all databases in a workspace"""
    try:
        client = get_grpc_client()
        databases = await client.list_databases(workspace_id)
        return databases
    except Exception as e:
        logger.error(f"Failed to list databases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/databases/{database_id}", response_model=DatabaseResponse)
async def get_database(
    database_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get a single database"""
    try:
        client = get_grpc_client()
        database = await client.get_database(database_id)
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
    """Delete a database"""
    try:
        client = get_grpc_client()
        success = await client.delete_database(database_id)
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
    """Execute import job"""
    try:
        client = get_grpc_client()
        job = await client.execute_import(
            workspace_id=request.workspace_id,
            database_id=request.database_id,
            table_name=request.table_name,
            file_path=request.file_path,
            field_mapping=request.field_mapping
        )
        return job
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
    """Create a new table in a database"""
    try:
        client = get_grpc_client()
        table = await client.create_table(
            database_id=request.database_id,
            name=request.name,
            description=request.description,
            table_schema=request.table_schema
        )
        return table
    except Exception as e:
        logger.error(f"Failed to create table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/databases/{database_id}/tables", response_model=List[TableResponse])
async def list_tables(
    database_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List all tables in a database"""
    try:
        client = get_grpc_client()
        tables = await client.list_tables(database_id)
        return tables
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables/{table_id}", response_model=TableResponse)
async def get_table(
    table_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get a single table"""
    try:
        client = get_grpc_client()
        table = await client.get_table(table_id)
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
    """Delete a table"""
    try:
        client = get_grpc_client()
        success = await client.delete_table(table_id)
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
    """Get table data with pagination"""
    try:
        client = get_grpc_client()
        data = await client.get_table_data(table_id, offset, limit)
        return data
    except Exception as e:
        logger.error(f"Failed to get table data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tables/{table_id}/rows")
async def insert_table_row(
    table_id: str,
    request: InsertRowRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Insert a new row into a table"""
    try:
        client = get_grpc_client()
        row = await client.insert_table_row(table_id, request.row_data)
        return row
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
    """Update an existing row in a table"""
    try:
        client = get_grpc_client()
        row = await client.update_table_row(table_id, row_id, request.row_data)
        return row
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
    """Update a single cell in a row"""
    try:
        client = get_grpc_client()
        result = await client.update_table_cell(
            table_id, row_id, request.column_name, request.value
        )
        return result
    except Exception as e:
        logger.error(f"Failed to update cell: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tables/{table_id}/rows/{row_id}")
async def delete_table_row(
    table_id: str,
    row_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a row from a table"""
    try:
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

