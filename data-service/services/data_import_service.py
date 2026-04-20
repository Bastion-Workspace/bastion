import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

import json
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.json as pajson
import pyarrow.types as pat
from pathlib import Path

from db.connection_manager import DatabaseConnectionManager
from services.table_service import TableService
from services.database_service import DatabaseService
from config.settings import settings

logger = logging.getLogger(__name__)


def _arrow_type_to_sql(pa_type: pa.DataType) -> str:
    """Map Arrow column type to Data Workspace SQL schema type label."""
    if pat.is_integer(pa_type):
        return "INTEGER"
    if pat.is_floating(pa_type):
        return "REAL"
    if pat.is_boolean(pa_type):
        return "BOOLEAN"
    if pat.is_timestamp(pa_type):
        return "TIMESTAMP"
    if pat.is_temporal(pa_type):
        return "TIMESTAMP"
    return "TEXT"


class DataImportService:
    """Service for importing data from various file formats"""

    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db = db_manager
        self.table_service = TableService(db_manager)
        self.database_service = DatabaseService(db_manager)

    def _read_file_as_arrow_table(self, file_path: str, file_type: str) -> pa.Table:
        """Load file as a columnar Arrow Table (Excel via pandas)."""
        ft = file_type.lower()
        try:
            if ft in ["csv", "text/csv"]:
                return pacsv.read_csv(file_path)
            if ft in ["json", "application/json"]:
                try:
                    return pajson.read_json(file_path)
                except Exception as e:
                    logger.warning("Arrow JSON read failed, using pandas: %s", e)
                    return pa.Table.from_pandas(pd.read_json(file_path), preserve_index=False)
            if ft in ["jsonl", "ndjson", "application/x-ndjson"]:
                try:
                    return pajson.read_json(file_path)
                except Exception as e:
                    logger.warning("Arrow JSONL read failed, using pandas: %s", e)
                    return pa.Table.from_pandas(
                        pd.read_json(file_path, lines=True), preserve_index=False
                    )
            if ft in [
                "xlsx",
                "xls",
                "excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
            ]:
                return pa.Table.from_pandas(pd.read_excel(file_path), preserve_index=False)
            raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error("Failed to read file %s: %s", file_path, e)
            raise

    async def preview_import(
        self,
        file_path: str,
        file_type: str,
        preview_rows: int = 10,
    ) -> Dict[str, Any]:
        """Preview imported file and infer schema"""
        try:
            table = self._read_file_as_arrow_table(file_path, file_type)

            if table.num_rows == 0:
                return {"error": "File is empty or could not be parsed"}

            preview_table = table.slice(0, min(preview_rows, table.num_rows))

            column_info = []
            for i, name in enumerate(table.column_names):
                col = table.column(i)
                pa_type = col.type
                inferred_type = _arrow_type_to_sql(pa_type)
                has_nulls = col.null_count > 0
                column_info.append(
                    {
                        "name": name,
                        "type": inferred_type,
                        "nullable": bool(has_nulls),
                    }
                )

            preview_data = preview_table.to_pylist()

            return {
                "column_names": [col["name"] for col in column_info],
                "inferred_types": column_info,
                "preview_data": preview_data,
                "estimated_rows": table.num_rows,
                "total_columns": table.num_columns,
            }

        except Exception as e:
            logger.error("Failed to preview import %s: %s", file_path, e)
            raise

    async def execute_import(
        self,
        workspace_id: str,
        database_id: str,
        table_name: str,
        file_path: str,
        file_type: str,
        user_id: str,
        field_mapping: Optional[Dict[str, str]] = None,
        type_overrides: Optional[Dict[str, str]] = None,
    ) -> str:
        """Execute data import job"""
        job_id: Optional[str] = None
        try:
            job_id = str(uuid.uuid4())

            await self._create_import_job(
                job_id, workspace_id, database_id, file_path, "pending", user_id
            )

            table = self._read_file_as_arrow_table(file_path, file_type)

            if table.num_rows == 0:
                await self._update_import_job_status(
                    job_id, "failed", error_log="File is empty"
                )
                return job_id

            if field_mapping:
                new_names = [field_mapping.get(n, n) for n in table.column_names]
                table = table.rename_columns(new_names)

            await self._update_import_job_status(
                job_id, "processing", rows_total=table.num_rows
            )

            sample_n = min(1000, table.num_rows)
            sample_rows = table.slice(0, sample_n).to_pylist()
            schema = await self.table_service.infer_schema_from_data(sample_rows)

            if type_overrides and isinstance(schema, dict) and isinstance(schema.get("columns"), list):
                for col in schema["columns"]:
                    if not isinstance(col, dict):
                        continue
                    n = col.get("name")
                    if not n:
                        continue
                    override = type_overrides.get(str(n))
                    if override:
                        col["type"] = str(override).upper()

            tbl = await self.table_service.create_table(
                database_id=database_id,
                name=table_name,
                schema=schema,
                user_id=user_id,
                description=f"Imported from {Path(file_path).name}",
            )

            table_id = tbl["table_id"]

            await self._update_import_job_table(job_id, table_id)

            batch_size = settings.IMPORT_BATCH_SIZE
            total_rows = table.num_rows

            for i in range(0, total_rows, batch_size):
                chunk = table.slice(i, min(batch_size, total_rows - i))
                rows_data = chunk.to_pylist()
                await self.table_service.bulk_insert_rows(
                    table_id, rows_data, user_id, batch_size
                )

                rows_processed = min(i + batch_size, total_rows)
                await self._update_import_job_progress(job_id, rows_processed)

            await self.database_service.update_database_stats(database_id, user_id)

            await self._update_import_job_status(
                job_id, "completed", rows_processed=total_rows
            )

            logger.info("Import job %s completed: %s rows", job_id, total_rows)
            return job_id

        except Exception as e:
            logger.error("Failed to execute import: %s", e)
            if job_id:
                await self._update_import_job_status(job_id, "failed", error_log=str(e))
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
                "job_id": row["job_id"],
                "workspace_id": row["workspace_id"],
                "database_id": row["database_id"],
                "table_id": row["table_id"],
                "status": row["status"],
                "source_file": row["source_file"],
                "rows_processed": row["rows_processed"],
                "rows_total": row["rows_total"],
                "error_log": row["error_log"],
                "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                "created_by": row.get("created_by"),
                "progress_percent": int((row["rows_processed"] / row["rows_total"] * 100))
                if row["rows_total"] > 0
                else 0,
            }

        except Exception as e:
            logger.error("Failed to get import status %s: %s", job_id, e)
            raise

    async def _create_import_job(
        self,
        job_id: str,
        workspace_id: str,
        database_id: str,
        source_file: str,
        status: str,
        user_id: str,
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
            job_id,
            workspace_id,
            database_id,
            status,
            source_file,
            0,
            0,
            datetime.utcnow(),
            user_id,
        )

    async def _update_import_job_status(
        self,
        job_id: str,
        status: str,
        rows_total: Optional[int] = None,
        rows_processed: Optional[int] = None,
        error_log: Optional[str] = None,
    ):
        """Update import job status"""
        updates = ["status = $2"]
        params: List[Any] = [job_id, status]
        param_count = 2

        if status == "processing":
            param_count += 1
            updates.append(f"started_at = ${param_count}")
            params.append(datetime.utcnow())

        if status in ["completed", "failed"]:
            param_count += 1
            updates.append(f"completed_at = ${param_count}")
            params.append(datetime.utcnow())

        if rows_total is not None:
            param_count += 1
            updates.append(f"rows_total = ${param_count}")
            params.append(rows_total)

        if rows_processed is not None:
            param_count += 1
            updates.append(f"rows_processed = ${param_count}")
            params.append(rows_processed)

        if error_log is not None:
            param_count += 1
            updates.append(f"error_log = ${param_count}")
            params.append(error_log)

        query = f"""
            UPDATE data_import_jobs
            SET {", ".join(updates)}
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
