import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import json
import pandas as pd
from pathlib import Path

from db.connection_manager import DatabaseConnectionManager
from services.table_service import TableService
from services.database_service import DatabaseService
from config.settings import settings

logger = logging.getLogger(__name__)


class DataImportService:
    """Service for importing data from various file formats"""
    
    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db = db_manager
        self.table_service = TableService(db_manager)
        self.database_service = DatabaseService(db_manager)
    
    async def preview_import(
        self,
        file_path: str,
        file_type: str,
        preview_rows: int = 10
    ) -> Dict[str, Any]:
        """Preview imported file and infer schema"""
        try:
            # Read file based on type
            df = self._read_file(file_path, file_type)
            
            if df.empty:
                return {'error': 'File is empty or could not be parsed'}
            
            # Get preview data
            preview_df = df.head(preview_rows)
            
            # Infer types
            column_info = []
            for col_name in df.columns:
                dtype = df[col_name].dtype
                has_nulls = df[col_name].isnull().any()
                
                # Map pandas dtype to SQL type
                if pd.api.types.is_integer_dtype(dtype):
                    inferred_type = 'INTEGER'
                elif pd.api.types.is_float_dtype(dtype):
                    inferred_type = 'FLOAT'
                elif pd.api.types.is_bool_dtype(dtype):
                    inferred_type = 'BOOLEAN'
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    inferred_type = 'TIMESTAMP'
                else:
                    inferred_type = 'TEXT'
                
                column_info.append({
                    'name': col_name,
                    'type': inferred_type,
                    'nullable': bool(has_nulls)
                })
            
            # Convert preview to dict
            preview_data = preview_df.to_dict(orient='records')
            
            return {
                'column_names': [col['name'] for col in column_info],
                'inferred_types': column_info,
                'preview_data': preview_data,
                'estimated_rows': len(df),
                'total_columns': len(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Failed to preview import {file_path}: {e}")
            raise
    
    async def execute_import(
        self,
        workspace_id: str,
        database_id: str,
        table_name: str,
        file_path: str,
        file_type: str,
        user_id: str,
        field_mapping: Optional[Dict[str, str]] = None
    ) -> str:
        """Execute data import job"""
        try:
            job_id = str(uuid.uuid4())
            
            # Create import job record
            await self._create_import_job(
                job_id, workspace_id, database_id, file_path, 'pending', user_id
            )
            
            # Read file
            df = self._read_file(file_path, file_type)
            
            if df.empty:
                await self._update_import_job_status(
                    job_id, 'failed', error_log='File is empty'
                )
                return job_id
            
            # Apply field mapping if provided
            if field_mapping:
                df = df.rename(columns=field_mapping)
            
            # Update job status to processing
            await self._update_import_job_status(
                job_id, 'processing', rows_total=len(df)
            )
            
            # Infer schema
            schema = await self.table_service.infer_schema_from_data(
                df.head(1000).to_dict(orient='records')
            )
            
            # Create table
            table = await self.table_service.create_table(
                database_id=database_id,
                name=table_name,
                schema=schema,
                user_id=user_id,
                description=f"Imported from {Path(file_path).name}"
            )
            
            table_id = table['table_id']
            
            # Update job with table_id
            await self._update_import_job_table(job_id, table_id)
            
            # Convert DataFrame to list of dicts
            rows_data = df.to_dict(orient='records')
            
            # Bulk insert with progress tracking
            batch_size = settings.IMPORT_BATCH_SIZE
            total_rows = len(rows_data)
            
            for i in range(0, total_rows, batch_size):
                batch = rows_data[i:i + batch_size]
                await self.table_service.bulk_insert_rows(
                    table_id, batch, user_id, batch_size
                )
                
                # Update progress
                rows_processed = min(i + batch_size, total_rows)
                await self._update_import_job_progress(job_id, rows_processed)
            
            # Update database stats
            await self.database_service.update_database_stats(database_id, user_id)
            
            # Mark job as completed
            await self._update_import_job_status(
                job_id, 'completed', rows_processed=total_rows
            )
            
            logger.info(f"Import job {job_id} completed: {total_rows} rows")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to execute import: {e}")
            await self._update_import_job_status(
                job_id, 'failed', error_log=str(e)
            )
            raise
    
    async def get_import_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get import job status"""
        try:
            query = """
                SELECT job_id, workspace_id, database_id, table_id, status,
                       source_file, file_size, rows_processed, rows_total,
                       field_mapping_json, error_log, started_at, completed_at, created_at, created_by
                FROM data_import_jobs
                WHERE job_id = $1
            """
            
            row = await self.db.fetchrow(query, job_id)
            
            if not row:
                return None
            
            return {
                'job_id': row['job_id'],
                'workspace_id': row['workspace_id'],
                'database_id': row['database_id'],
                'table_id': row['table_id'],
                'status': row['status'],
                'source_file': row['source_file'],
                'rows_processed': row['rows_processed'],
                'rows_total': row['rows_total'],
                'error_log': row['error_log'],
                'started_at': row['started_at'].isoformat() if row['started_at'] else None,
                'completed_at': row['completed_at'].isoformat() if row['completed_at'] else None,
                'created_by': row.get('created_by'),
                'progress_percent': int((row['rows_processed'] / row['rows_total'] * 100)) if row['rows_total'] > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get import status {job_id}: {e}")
            raise
    
    def _read_file(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Read file based on type"""
        try:
            if file_type.lower() in ['csv', 'text/csv']:
                return pd.read_csv(file_path)
            elif file_type.lower() in ['json', 'application/json']:
                return pd.read_json(file_path)
            elif file_type.lower() in ['jsonl', 'ndjson', 'application/x-ndjson']:
                # Read JSONL (JSON Lines) format - one JSON object per line
                return pd.read_json(file_path, lines=True)
            elif file_type.lower() in ['xlsx', 'xls', 'excel', 
                                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                       'application/vnd.ms-excel']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise
    
    async def _create_import_job(
        self,
        job_id: str,
        workspace_id: str,
        database_id: str,
        source_file: str,
        status: str,
        user_id: str
    ):
        """Create import job record"""
        query = """
            INSERT INTO data_import_jobs 
            (job_id, workspace_id, database_id, status, source_file, 
             rows_processed, rows_total, created_at, created_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        await self.db.execute(
            query,
            job_id, workspace_id, database_id, status, source_file,
            0, 0, datetime.utcnow(), user_id
        )
    
    async def _update_import_job_status(
        self,
        job_id: str,
        status: str,
        rows_total: Optional[int] = None,
        rows_processed: Optional[int] = None,
        error_log: Optional[str] = None
    ):
        """Update import job status"""
        updates = ['status = $2']
        params = [job_id, status]
        param_count = 2
        
        if status == 'processing':
            param_count += 1
            updates.append(f'started_at = ${param_count}')
            params.append(datetime.utcnow())
        
        if status in ['completed', 'failed']:
            param_count += 1
            updates.append(f'completed_at = ${param_count}')
            params.append(datetime.utcnow())
        
        if rows_total is not None:
            param_count += 1
            updates.append(f'rows_total = ${param_count}')
            params.append(rows_total)
        
        if rows_processed is not None:
            param_count += 1
            updates.append(f'rows_processed = ${param_count}')
            params.append(rows_processed)
        
        if error_log is not None:
            param_count += 1
            updates.append(f'error_log = ${param_count}')
            params.append(error_log)
        
        query = f"""
            UPDATE data_import_jobs
            SET {', '.join(updates)}
            WHERE job_id = $1
        """
        
        await self.db.execute(query, *params)
    
    async def _update_import_job_progress(self, job_id: str, rows_processed: int):
        """Update import job progress"""
        query = """
            UPDATE data_import_jobs
            SET rows_processed = $2
            WHERE job_id = $1
        """
        await self.db.execute(query, job_id, rows_processed)
    
    async def _update_import_job_table(self, job_id: str, table_id: str):
        """Update import job with table_id"""
        query = """
            UPDATE data_import_jobs
            SET table_id = $2
            WHERE job_id = $1
        """
        await self.db.execute(query, job_id, table_id)


