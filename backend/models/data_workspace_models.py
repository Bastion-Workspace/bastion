from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Workspace Models
class CreateWorkspaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None


class UpdateWorkspaceRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    is_pinned: Optional[bool] = None


class WorkspacePermission(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class WorkspaceResponse(BaseModel):
    workspace_id: str
    user_id: str
    name: str
    description: Optional[str]
    icon: Optional[str]
    color: Optional[str]
    is_pinned: bool
    metadata_json: str
    created_at: str
    updated_at: str
    updated_by: Optional[str] = None
    permission_level: Optional[str] = None  # User's permission level (if shared)
    is_shared: Optional[bool] = False  # Whether workspace is shared with user
    share_type: Optional[str] = None  # 'user', 'team', or 'public' if shared


# Database Models
class CreateDatabaseRequest(BaseModel):
    workspace_id: str
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    source_type: str = Field(default="imported")


class DatabaseResponse(BaseModel):
    database_id: str
    workspace_id: str
    name: str
    description: Optional[str]
    source_type: str
    table_count: int
    total_rows: int
    created_at: str
    created_by: Optional[str] = None
    updated_by: Optional[str] = None


# Table Models
class ColumnDefinition(BaseModel):
    name: str
    type: str
    nullable: bool = True


class CreateTableRequest(BaseModel):
    database_id: str
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    table_schema: Dict[str, Any]


class TableResponse(BaseModel):
    table_id: str
    database_id: str
    name: str
    description: Optional[str]
    row_count: int
    table_schema_json: str
    styling_rules_json: str
    created_by: Optional[str] = None
    updated_by: Optional[str] = None


class RowData(BaseModel):
    row_id: str
    row_data: Dict[str, Any]
    row_index: int
    row_color: Optional[str] = None


class TableDataResponse(BaseModel):
    table_id: str
    rows: List[RowData]
    total_rows: int
    offset: int
    limit: int
    table_schema: Dict[str, Any]


class InsertRowRequest(BaseModel):
    row_data: Dict[str, Any]


class UpdateRowRequest(BaseModel):
    row_data: Dict[str, Any]


class UpdateCellRequest(BaseModel):
    column_name: str
    value: Any
    formula: Optional[str] = None  # Formula string if cell contains formula


# Import Models
class PreviewImportRequest(BaseModel):
    workspace_id: str
    file_path: str
    file_type: str
    preview_rows: int = 10


class PreviewImportResponse(BaseModel):
    column_names: List[str]
    inferred_types: List[Dict[str, Any]]
    preview_data: List[Dict[str, Any]]
    estimated_rows: int
    total_columns: int


class ExecuteImportRequest(BaseModel):
    workspace_id: str
    database_id: str
    table_name: str
    file_path: str
    field_mapping: Optional[Dict[str, str]] = None


class ImportJobResponse(BaseModel):
    job_id: str
    workspace_id: str
    database_id: Optional[str]
    table_id: Optional[str]
    status: str
    source_file: str
    rows_processed: int
    rows_total: int
    error_log: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    created_by: Optional[str] = None
    progress_percent: int


# Query Models
class SQLQueryRequest(BaseModel):
    query: str
    limit: int = 1000


class NaturalLanguageQueryRequest(BaseModel):
    workspace_id: str
    query: str
    include_documents: bool = False


class QueryResultResponse(BaseModel):
    query_id: str
    column_names: List[str]
    results: List[Dict[str, Any]]
    result_count: int
    execution_time_ms: int
    generated_sql: Optional[str] = None
    error_message: Optional[str] = None


# Transformation Models
class FilterCondition(BaseModel):
    column: str
    operator: str  # eq, ne, gt, lt, gte, lte, contains, startswith, endswith
    value: Any


class FilterRequest(BaseModel):
    table_id: str
    conditions: List[FilterCondition]
    logic: str = "AND"  # AND or OR


class AggregationRequest(BaseModel):
    table_id: str
    group_by: List[str]
    aggregations: Dict[str, str]  # column: function (sum, avg, count, min, max)


class JoinRequest(BaseModel):
    left_table_id: str
    right_table_id: str
    join_type: str  # inner, left, right, outer
    left_column: str
    right_column: str


class TransformResultResponse(BaseModel):
    transformation_id: str
    result_preview: List[Dict[str, Any]]
    result_rows: int


# Visualization Models
class CreateVisualizationRequest(BaseModel):
    workspace_id: str
    table_id: str
    name: str
    viz_type: str  # bar, line, scatter, pie, map, table, heatmap
    config: Dict[str, Any]
    color_scheme: Optional[str] = None


class VisualizationResponse(BaseModel):
    visualization_id: str
    workspace_id: str
    table_id: str
    name: str
    viz_type: str
    config: Dict[str, Any]
    color_scheme: Optional[str]
    is_pinned: bool
    created_at: str


# Styling Models
class CreateStylingRuleRequest(BaseModel):
    table_id: str
    rule_name: str
    rule_type: str  # row, column, cell, conditional
    target_column: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    color: str
    background_color: Optional[str] = None
    priority: int = 0


class StylingRuleResponse(BaseModel):
    rule_id: str
    table_id: str
    rule_name: str
    rule_type: str
    target_column: Optional[str]
    condition: Optional[Dict[str, Any]]
    color: str
    background_color: Optional[str]
    priority: int
    is_active: bool


class ApplyStylingRequest(BaseModel):
    table_id: str
    data: List[Dict[str, Any]]


class StyledDataResponse(BaseModel):
    styled_data: List[Dict[str, Any]]
    applied_rules: List[StylingRuleResponse]


# Sharing Models
class ShareWorkspaceRequest(BaseModel):
    shared_with_user_id: Optional[str] = None  # None for team/public share
    shared_with_team_id: Optional[str] = None  # None for user/public share
    permission_level: WorkspacePermission = WorkspacePermission.READ
    is_public: bool = False
    expires_at: Optional[datetime] = None


class WorkspaceShareResponse(BaseModel):
    share_id: str
    workspace_id: str
    shared_by_user_id: str
    shared_with_user_id: Optional[str]
    shared_with_team_id: Optional[str]
    permission_level: str
    is_public: bool
    expires_at: Optional[str]
    created_at: str
    access_count: int


class RecalculateTableResponse(BaseModel):
    success: bool
    cells_recalculated: int
    error_message: Optional[str] = None

