import io
import logging
import os
import tempfile
import uuid
import re
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

import json
import pandas as pd

from db.connection_manager import DatabaseConnectionManager
from services.formula_evaluator import FormulaEvaluator

logger = logging.getLogger(__name__)

# Map Data Workspace schema types to PostgreSQL types
SCHEMA_TYPE_TO_PG = {
    'TEXT': 'TEXT',
    'INTEGER': 'BIGINT',
    'REAL': 'DOUBLE PRECISION',
    'BOOLEAN': 'BOOLEAN',
    'TIMESTAMP': 'TIMESTAMP WITH TIME ZONE',
    'DATE': 'DATE',
    'JSON': 'JSONB',
}
DEFAULT_PG_TYPE = 'TEXT'


class TableService:
    """Service for managing tables and their data"""
    
    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db = db_manager
        self.formula_evaluator = FormulaEvaluator()
    
    @staticmethod
    def _sanitize_sql_identifier(name: str) -> str:
        """Sanitize table/column name for use as SQL identifier (alphanumeric and underscore only)."""
        if not name:
            return 'unnamed'
        s = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        return s[:63] if len(s) > 63 else (s if s else 'unnamed')
    
    async def _get_workspace_id_for_database(
        self, database_id: str, user_id: Optional[str] = None, user_team_ids: Optional[List[str]] = None
    ) -> Optional[str]:
        """Resolve workspace_id from database_id."""
        row = await self.db.fetchrow(
            "SELECT workspace_id FROM custom_databases WHERE database_id = $1",
            database_id,
            user_id=user_id,
            user_team_ids=user_team_ids
        )
        return row['workspace_id'] if row else None
    
    def _schema_columns_to_pg_ddl(self, schema: Dict[str, Any]) -> List[str]:
        """Convert schema JSON columns to PostgreSQL column definitions. Includes row_id and row_index."""
        parts = [
            "row_id VARCHAR(255) PRIMARY KEY",
            "row_index INTEGER NOT NULL DEFAULT 0",
        ]
        columns = schema.get('columns') if isinstance(schema, dict) else []
        if not isinstance(columns, list):
            return parts
        for col in columns:
            name = col.get('name') if isinstance(col, dict) else None
            if not name:
                continue
            safe_name = self._sanitize_sql_identifier(str(name))
            col_type = col.get('type', 'TEXT') if isinstance(col, dict) else 'TEXT'
            pg_type = SCHEMA_TYPE_TO_PG.get(str(col_type).upper(), DEFAULT_PG_TYPE)
            parts.append(f'"{safe_name}" {pg_type}')
        return parts
    
    async def _ensure_workspace_schema_exists(
        self, workspace_id: str, user_id: Optional[str] = None, user_team_ids: Optional[List[str]] = None
    ) -> None:
        """Create schema ws_<workspace_id> if it does not exist."""
        safe_schema = 'ws_' + re.sub(r'[^a-zA-Z0-9_]', '_', workspace_id)[:50]
        await self.db.execute(
            f'CREATE SCHEMA IF NOT EXISTS "{safe_schema}"',
            user_id=user_id,
            user_team_ids=user_team_ids
        )
    
    def _workspace_schema_name(self, workspace_id: str) -> str:
        """Return the schema name for a workspace (for use in qualified names)."""
        return 'ws_' + re.sub(r'[^a-zA-Z0-9_]', '_', workspace_id)[:50]
    
    async def _create_native_table_and_view(
        self,
        workspace_id: str,
        table_id: str,
        name: str,
        schema: Dict[str, Any],
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> None:
        """Create real table and view in workspace schema. Call after INSERT into custom_tables."""
        await self._ensure_workspace_schema_exists(workspace_id, user_id=user_id, user_team_ids=user_team_ids)
        schema_name = self._workspace_schema_name(workspace_id)
        table_name = f't_{table_id.replace("-", "_")}'
        cols = self._schema_columns_to_pg_ddl(schema)
        create_table_sql = f'CREATE TABLE "{schema_name}"."{table_name}" ({", ".join(cols)})'
        await self.db.execute(create_table_sql, user_id=user_id, user_team_ids=user_team_ids)
        view_base = self._sanitize_sql_identifier(name)
        view_name = view_base
        try:
            await self.db.execute(
                f'CREATE VIEW "{schema_name}"."{view_name}" AS SELECT * FROM "{schema_name}"."{table_name}"',
                user_id=user_id,
                user_team_ids=user_team_ids
            )
        except Exception as e:
            if 'already exists' in str(e).lower():
                view_name = view_base + '_' + table_id.replace('-', '')[:8]
                await self.db.execute(
                    f'CREATE VIEW "{schema_name}"."{view_name}" AS SELECT * FROM "{schema_name}"."{table_name}"',
                    user_id=user_id,
                    user_team_ids=user_team_ids
                )
            else:
                raise
        logger.info(f"Created native table and view: {schema_name}.{table_name} / {view_name}")
    
    async def _drop_native_table_and_view(
        self,
        workspace_id: str,
        table_id: str,
        name: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> None:
        """Drop view and table in workspace schema. View may be name or name_shortid."""
        schema_name = self._workspace_schema_name(workspace_id)
        table_name = f't_{table_id.replace("-", "_")}'
        view_base = self._sanitize_sql_identifier(name)
        for candidate in [view_base, view_base + '_' + table_id.replace('-', '')[:8]]:
            try:
                await self.db.execute(
                    f'DROP VIEW IF EXISTS "{schema_name}"."{candidate}"',
                    user_id=user_id,
                    user_team_ids=user_team_ids
                )
            except Exception:
                pass
        await self.db.execute(
            f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"',
            user_id=user_id,
            user_team_ids=user_team_ids
        )
        logger.info(f"Dropped native table: {schema_name}.{table_name}")
    
    async def create_table(
        self,
        database_id: str,
        name: str,
        schema: Dict[str, Any],
        user_id: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new table. New tables use native storage (real PostgreSQL table in workspace schema)."""
        try:
            table_id = str(uuid.uuid4())
            now = datetime.utcnow()
            storage_type = 'native'
            metadata_json = json.dumps(metadata) if metadata else json.dumps({})
            query = """
                INSERT INTO custom_tables
                (table_id, database_id, name, description, row_count, schema_json,
                 styling_rules_json, metadata_json, created_at, updated_at, created_by, updated_by, storage_type)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING table_id, database_id, name, description, row_count,
                          schema_json, styling_rules_json, metadata_json, created_at, updated_at, created_by, updated_by, storage_type
            """
            row = await self.db.fetchrow(
                query,
                table_id, database_id, name, description, 0,
                json.dumps(schema), json.dumps({}), metadata_json, now, now, user_id, user_id,
                storage_type,
                user_id=user_id,
                user_team_ids=None
            )
            result = self._row_to_dict(row)
            if storage_type == 'native':
                workspace_id = await self._get_workspace_id_for_database(database_id, user_id=user_id)
                if workspace_id:
                    await self._create_native_table_and_view(
                        workspace_id, table_id, name, schema, user_id=user_id
                    )
            logger.info(f"Created table: {table_id} in database: {database_id} (storage={storage_type})")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise
    
    async def _list_workspace_schema_relations(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List user-visible relations (tables and views) in the workspace schema.
        Returns DDL-created base tables (not t_*) and views. Used to merge with custom_tables so DB is source of truth.
        """
        schema_name = self._workspace_schema_name(workspace_id)
        try:
            exists = await self.db.fetchval(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = $1",
                schema_name,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            if not exists:
                return []
            rows = await self.db.fetch(
                """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = $1
                  AND (table_type = 'VIEW' OR (table_type = 'BASE TABLE' AND table_name NOT LIKE 't\_%'))
                ORDER BY table_type, table_name
                """,
                schema_name,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            result = []
            for row in rows:
                name = row['table_name']
                table_type = row['table_type']
                row_count = 0
                if table_type == 'BASE TABLE':
                    try:
                        n = await self.db.fetchval(
                            f'SELECT COUNT(*) FROM "{schema_name}"."{name}"',
                            user_id=user_id,
                            user_team_ids=user_team_ids
                        )
                        row_count = n or 0
                    except Exception:
                        pass
                cols = await self.db.fetch(
                    """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = $1 AND table_name = $2
                    ORDER BY ordinal_position
                    """,
                    schema_name, name,
                    user_id=user_id,
                    user_team_ids=user_team_ids
                )
                columns = [
                    {
                        'name': c['column_name'],
                        'type': (c['data_type'] or 'text').upper(),
                        'nullable': c['is_nullable'] == 'YES',
                        'default_value': c['column_default'] if c.get('column_default') else None
                    }
                    for c in cols
                ]
                result.append({
                    'table_name': name,
                    'table_type': table_type,
                    'row_count': row_count,
                    'schema_json': json.dumps({'columns': columns}),
                })
            return result
        except Exception as e:
            logger.warning(f"Could not list workspace schema relations: {e}")
            return []
    
    async def list_tables(
        self, 
        database_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List all tables in a database. Includes custom_tables plus schema-only (DDL-created) tables so DB is source of truth."""
        try:
            query = """
                SELECT table_id, database_id, name, description, row_count,
                       schema_json, styling_rules_json, metadata_json, created_at, updated_at, created_by, updated_by,
                       COALESCE(storage_type, 'jsonb') AS storage_type
                FROM custom_tables
                WHERE database_id = $1
                ORDER BY created_at DESC
            """
            rows = await self.db.fetch(
                query, 
                database_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            from_custom = [self._row_to_dict(row) for row in rows]
            names_from_custom = {t['name'].lower() for t in from_custom}
            workspace_id = await self._get_workspace_id_for_database(database_id, user_id=user_id, user_team_ids=user_team_ids)
            if workspace_id:
                schema_relations = await self._list_workspace_schema_relations(workspace_id, user_id=user_id, user_team_ids=user_team_ids)
                for rel in schema_relations:
                    name = rel['table_name']
                    if name.lower() in names_from_custom:
                        continue
                    synthetic = {
                        'table_id': name,
                        'database_id': database_id,
                        'name': name,
                        'description': None,
                        'row_count': rel.get('row_count', 0),
                        'schema_json': rel.get('schema_json', '{"columns":[]}'),
                        'styling_rules_json': None,
                        'metadata_json': None,
                        'created_at': None,
                        'updated_at': None,
                        'created_by': None,
                        'updated_by': None,
                        'storage_type': 'native',
                        '_schema_only': True,
                    }
                    from_custom.append(synthetic)
                    names_from_custom.add(name.lower())
            return from_custom
            
        except Exception as e:
            logger.error(f"Failed to list tables for database {database_id}: {e}")
            raise
    
    async def get_table(
        self, 
        table_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None,
        database_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get table metadata. For schema-only (DDL-created) tables pass database_id so we can resolve from workspace schema."""
        try:
            query = """
                SELECT table_id, database_id, name, description, row_count,
                       schema_json, styling_rules_json, metadata_json, created_at, updated_at, created_by, updated_by,
                       COALESCE(storage_type, 'jsonb') AS storage_type
                FROM custom_tables
                WHERE table_id = $1
            """
            row = await self.db.fetchrow(
                query, 
                table_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            if row:
                return self._row_to_dict(row)
            if not database_id:
                return None
            return await self._get_schema_only_table(table_id, database_id, user_id=user_id, user_team_ids=user_team_ids)
            
        except Exception as e:
            logger.error(f"Failed to get table {table_id}: {e}")
            raise
    
    async def _get_schema_only_table(
        self,
        table_id: str,
        database_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Return synthetic table metadata for a relation that exists in workspace schema but not in custom_tables."""
        workspace_id = await self._get_workspace_id_for_database(database_id, user_id=user_id, user_team_ids=user_team_ids)
        if not workspace_id:
            return None
        schema_name = self._workspace_schema_name(workspace_id)
        try:
            exists = await self.db.fetchval(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = $1 AND table_name = $2",
                schema_name, table_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            if not exists:
                return None
            row_count = 0
            try:
                n = await self.db.fetchval(
                    f'SELECT COUNT(*) FROM "{schema_name}"."{table_id}"',
                    user_id=user_id,
                    user_team_ids=user_team_ids
                )
                row_count = n or 0
            except Exception:
                pass
            cols = await self.db.fetch(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2
                ORDER BY ordinal_position
                """,
                schema_name, table_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            columns = [
                {
                    'name': c['column_name'],
                    'type': (c['data_type'] or 'text').upper(),
                    'nullable': c['is_nullable'] == 'YES',
                    'default_value': c['column_default'] if c.get('column_default') else None
                }
                for c in cols
            ]
            return {
                'table_id': table_id,
                'database_id': database_id,
                'name': table_id,
                'description': None,
                'row_count': row_count,
                'schema_json': json.dumps({'columns': columns}),
                'styling_rules_json': None,
                'metadata_json': None,
                'created_at': None,
                'updated_at': None,
                'created_by': None,
                'updated_by': None,
                'storage_type': 'native',
                '_schema_only': True,
            }
        except Exception as e:
            logger.warning(f"Could not get schema-only table {table_id}: {e}")
            return None
    
    async def update_table(
        self,
        table_id: str,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Update table metadata (name, description, schema/columns, metadata_json)."""
        try:
            updates = []
            params = []
            if name is not None:
                params.append(name)
                updates.append(f"name = ${len(params)}")
            if description is not None:
                params.append(description)
                updates.append(f"description = ${len(params)}")
            if schema is not None:
                params.append(json.dumps(schema))
                updates.append(f"schema_json = ${len(params)}")
            if metadata is not None:
                params.append(json.dumps(metadata))
                updates.append(f"metadata_json = ${len(params)}")
            if not updates:
                return await self.get_table(table_id, user_id=user_id, user_team_ids=user_team_ids)
            params.append(datetime.utcnow())
            updates.append(f"updated_at = ${len(params)}")
            params.append(user_id)
            updates.append(f"updated_by = ${len(params)}")
            params.append(table_id)
            where_idx = len(params)
            query = f"""
                UPDATE custom_tables
                SET {", ".join(updates)}
                WHERE table_id = ${where_idx}
                RETURNING table_id, database_id, name, description, row_count,
                          schema_json, styling_rules_json, metadata_json, created_at, updated_at, created_by, updated_by
            """
            row = await self.db.fetchrow(
                query,
                *params,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            return self._row_to_dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to update table {table_id}: {e}")
            raise
    
    async def delete_table(
        self, 
        table_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> bool:
        """Delete a table and all its data. For native tables, drops real table and view first. For schema-only, drops the relation only."""
        try:
            table = await self.get_table(table_id, user_id=user_id, user_team_ids=user_team_ids)
            if not table:
                return False
            if table.get('_schema_only'):
                workspace_id = await self._get_workspace_id_for_database(
                    table['database_id'], user_id=user_id, user_team_ids=user_team_ids
                )
                if not workspace_id:
                    return False
                schema_name = self._workspace_schema_name(workspace_id)
                await self.db.execute(
                    f'DROP TABLE IF EXISTS "{schema_name}"."{table_id}"',
                    user_id=user_id,
                    user_team_ids=user_team_ids
                )
                logger.info(f"Dropped schema-only table {table_id}")
                return True
            if table.get('storage_type') == 'native':
                workspace_id = await self._get_workspace_id_for_database(
                    table['database_id'], user_id=user_id, user_team_ids=user_team_ids
                )
                if workspace_id:
                    await self._drop_native_table_and_view(
                        workspace_id, table_id, table['name'],
                        user_id=user_id, user_team_ids=user_team_ids
                    )
            delete_rows_query = "DELETE FROM custom_data_rows WHERE table_id = $1"
            await self.db.execute(
                delete_rows_query, 
                table_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            delete_table_query = "DELETE FROM custom_tables WHERE table_id = $1"
            result = await self.db.execute(
                delete_table_query, 
                table_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            deleted = result.split()[-1] != '0'
            if deleted:
                logger.info(f"Deleted table {table_id}")
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete table {table_id}: {e}")
            raise
    
    def _pack_table_data_response(
        self,
        table_id: str,
        data_rows: List[Dict[str, Any]],
        total_rows: int,
        offset: int,
        limit: int,
        schema: Dict[str, Any],
        prefer_arrow: bool,
    ) -> Dict[str, Any]:
        """Build get_table_data payload; optional Arrow IPC for the row page."""
        base: Dict[str, Any] = {
            'table_id': table_id,
            'total_rows': total_rows,
            'offset': offset,
            'limit': limit,
            'schema': schema,
        }
        if prefer_arrow:
            from utils.arrow_utils import table_page_rows_to_ipc_bytes

            base['rows'] = []
            base['arrow_data'] = table_page_rows_to_ipc_bytes(data_rows)
            base['has_arrow_data'] = True
            base['native_arrow'] = False
        else:
            base['rows'] = data_rows
            base['has_arrow_data'] = False
            base['arrow_data'] = b''
            base['native_arrow'] = False
        return base

    async def _get_native_table_data(
        self,
        table_id: str,
        table: Dict[str, Any],
        schema: Dict[str, Any],
        offset: int,
        limit: int,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None,
        prefer_arrow: bool = False,
    ) -> Dict[str, Any]:
        """Get paginated data from a native (real) table in workspace schema."""
        workspace_id = await self._get_workspace_id_for_database(
            table['database_id'], user_id=user_id, user_team_ids=user_team_ids
        )
        if not workspace_id:
            return {'error': 'Workspace not found'}
        schema_name = self._workspace_schema_name(workspace_id)
        is_schema_only = table.get('_schema_only') is True
        table_name = table_id if is_schema_only else f't_{table_id.replace("-", "_")}'
        data_columns = [c['name'] for c in schema.get('columns', []) if isinstance(c, dict) and c.get('name')]
        
        count_query = f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"'
        total_rows = await self.db.fetchval(count_query, user_id=user_id, user_team_ids=user_team_ids)
        
        if is_schema_only:
            select_query = f'SELECT * FROM "{schema_name}"."{table_name}" ORDER BY 1 LIMIT $1 OFFSET $2'
            rows = await self.db.fetch(
                select_query, limit, offset,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            data_rows = []
            for i, row in enumerate(rows):
                row_data = {k: row[k] for k in row.keys()}
                for k, v in list(row_data.items()):
                    if hasattr(v, 'isoformat'):
                        row_data[k] = v.isoformat() if v is not None else None
                data_rows.append({
                    'row_id': str(offset + i),
                    'row_data': row_data,
                    'row_index': offset + i,
                    'row_color': None,
                    'formula_data': {}
                })
            return self._pack_table_data_response(
                table_id, data_rows, total_rows or 0, offset, limit, schema, prefer_arrow
            )
        
        if not data_columns:
            select_query = f'SELECT row_id, row_index FROM "{schema_name}"."{table_name}" ORDER BY row_index LIMIT $1 OFFSET $2'
        else:
            select_query = f'''
                SELECT row_id, row_index, {", ".join(f'"{self._sanitize_sql_identifier(c)}"' for c in data_columns)}
                FROM "{schema_name}"."{table_name}"
                ORDER BY row_index
                LIMIT $1 OFFSET $2
            '''
        rows = await self.db.fetch(
            select_query, limit, offset,
            user_id=user_id,
            user_team_ids=user_team_ids
        )

        if prefer_arrow:
            from utils.arrow_utils import (
                asyncpg_rows_to_record_batch,
                empty_ipc_record_batch_bytes,
                record_batch_to_ipc_bytes,
            )

            if rows:
                col_names = list(rows[0].keys())
                batch = asyncpg_rows_to_record_batch(rows, col_names)
                arrow_bytes = record_batch_to_ipc_bytes(batch)
            else:
                arrow_bytes = empty_ipc_record_batch_bytes()
            return {
                'table_id': table_id,
                'total_rows': total_rows or 0,
                'offset': offset,
                'limit': limit,
                'schema': schema,
                'rows': [],
                'arrow_data': arrow_bytes,
                'has_arrow_data': True,
                'native_arrow': True,
            }

        data_rows = []
        for row in rows:
            row_data = {k: row[k] for k in row.keys() if k not in ('row_id', 'row_index')}
            data_rows.append({
                'row_id': row['row_id'],
                'row_data': row_data,
                'row_index': row['row_index'],
                'row_color': None,
                'formula_data': {}
            })

        return self._pack_table_data_response(
            table_id, data_rows, total_rows or 0, offset, limit, schema, prefer_arrow
        )
    
    async def _insert_native_row(
        self,
        table_id: str,
        table: Dict[str, Any],
        schema: Dict[str, Any],
        row_data: Dict[str, Any],
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Insert one row into a native table."""
        workspace_id = await self._get_workspace_id_for_database(
            table['database_id'], user_id=user_id, user_team_ids=user_team_ids
        )
        if not workspace_id:
            raise ValueError('Workspace not found')
        await self._ensure_workspace_schema_exists(workspace_id, user_id=user_id, user_team_ids=user_team_ids)
        schema_name = self._workspace_schema_name(workspace_id)
        table_name = f't_{table_id.replace("-", "_")}'
        columns = schema.get('columns') or []
        data_cols = [c['name'] for c in columns if isinstance(c, dict) and c.get('name')]
        
        max_index_q = f'SELECT COALESCE(MAX(row_index), -1) FROM "{schema_name}"."{table_name}"'
        max_index = await self.db.fetchval(max_index_q, user_id=user_id, user_team_ids=user_team_ids)
        new_index = (max_index or 0) + 1
        row_id = str(uuid.uuid4())
        
        col_list = ['row_id', 'row_index'] + [self._sanitize_sql_identifier(c) for c in data_cols]
        placeholders = ', '.join([f'${i+1}' for i in range(len(col_list))])
        cols_quoted = ', '.join(f'"{c}"' for c in col_list)
        insert_sql = f'INSERT INTO "{schema_name}"."{table_name}" ({cols_quoted}) VALUES ({placeholders})'
        values = [row_id, new_index] + [row_data.get(c) for c in data_cols]
        
        await self.db.execute(
            insert_sql, *values,
            user_id=user_id,
            user_team_ids=user_team_ids
        )
        await self._update_table_row_count(table_id, user_id)
        
        out_row_data = {c: row_data.get(c) for c in data_cols}
        return {
            'row_id': row_id,
            'row_data': out_row_data,
            'row_index': new_index,
            'row_color': None,
            'formula_data': {}
        }
    
    async def _update_native_row(
        self,
        table_id: str,
        table: Dict[str, Any],
        row_id: str,
        row_data: Dict[str, Any],
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Update one row in a native table."""
        workspace_id = await self._get_workspace_id_for_database(
            table['database_id'], user_id=user_id, user_team_ids=user_team_ids
        )
        if not workspace_id:
            return None
        schema = json.loads(table['schema_json']) if isinstance(table['schema_json'], str) else table['schema_json']
        schema_name = self._workspace_schema_name(workspace_id)
        table_name = f't_{table_id.replace("-", "_")}'
        data_cols = [c['name'] for c in (schema.get('columns') or []) if isinstance(c, dict) and c.get('name')]
        set_parts = []
        values = []
        for i, col in enumerate(data_cols):
            if col in row_data:
                safe = self._sanitize_sql_identifier(col)
                set_parts.append(f'"{safe}" = ${i+1}')
                values.append(row_data[col])
        if not set_parts:
            row = await self.db.fetchrow(
                f'SELECT row_id, row_index FROM "{schema_name}"."{table_name}" WHERE row_id = $1',
                row_id, user_id=user_id, user_team_ids=user_team_ids
            )
            if row:
                return {'row_id': row['row_id'], 'row_data': {}, 'row_index': row['row_index'], 'row_color': None, 'formula_data': {}}
            return None
        values.append(row_id)
        set_sql = ', '.join(set_parts)
        update_sql = f'UPDATE "{schema_name}"."{table_name}" SET {set_sql} WHERE row_id = ${len(values)} RETURNING row_id, row_index, ' + ', '.join(f'"{self._sanitize_sql_identifier(c)}"' for c in data_cols)
        row = await self.db.fetchrow(update_sql, *values, user_id=user_id, user_team_ids=user_team_ids)
        if not row:
            return None
        row_data_out = {k: row[k] for k in row.keys() if k not in ('row_id', 'row_index')}
        return {'row_id': row['row_id'], 'row_data': row_data_out, 'row_index': row['row_index'], 'row_color': None, 'formula_data': {}}
    
    async def _update_native_cell(
        self,
        table_id: str,
        table: Dict[str, Any],
        row_id: str,
        column_name: str,
        value: Any,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a single cell in a native table."""
        schema = json.loads(table['schema_json']) if isinstance(table['schema_json'], str) else table['schema_json']
        data_cols = [c['name'] for c in (schema.get('columns') or []) if isinstance(c, dict) and c.get('name')]
        if column_name not in data_cols:
            return None
        workspace_id = await self._get_workspace_id_for_database(
            table['database_id'], user_id=user_id, user_team_ids=user_team_ids
        )
        if not workspace_id:
            return None
        schema_name = self._workspace_schema_name(workspace_id)
        table_name = f't_{table_id.replace("-", "_")}'
        safe_col = self._sanitize_sql_identifier(column_name)
        return_cols = ', '.join(f'"{self._sanitize_sql_identifier(c)}"' for c in data_cols)
        update_sql = f'UPDATE "{schema_name}"."{table_name}" SET "{safe_col}" = $1 WHERE row_id = $2 RETURNING row_id, row_index, {return_cols}'
        row = await self.db.fetchrow(update_sql, value, row_id, user_id=user_id, user_team_ids=user_team_ids)
        if not row:
            return None
        row_data_out = {k: row[k] for k in row.keys() if k not in ('row_id', 'row_index')}
        return {'row_id': row['row_id'], 'row_data': row_data_out, 'row_index': row['row_index'], 'row_color': None, 'formula_data': {}}
    
    async def get_table_data(
        self,
        table_id: str,
        offset: int = 0,
        limit: int = 100,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None,
        database_id: Optional[str] = None,
        prefer_arrow: bool = False,
    ) -> Dict[str, Any]:
        """Get table data with pagination and formula evaluation. Pass database_id to resolve schema-only (DDL-created) tables."""
        try:
            table = await self.get_table(table_id, user_id=user_id, user_team_ids=user_team_ids, database_id=database_id)
            if not table:
                return {'error': 'Table not found'}
            
            storage_type = table.get('storage_type') or 'jsonb'
            schema = json.loads(table['schema_json']) if isinstance(table['schema_json'], str) else table['schema_json']
            
            if storage_type == 'native':
                return await self._get_native_table_data(
                    table_id=table_id,
                    table=table,
                    schema=schema,
                    offset=offset,
                    limit=limit,
                    user_id=user_id,
                    user_team_ids=user_team_ids,
                    prefer_arrow=prefer_arrow,
                )
            
            # JSONB path: Get all rows for formula evaluation context (needed for range functions)
            # Try with formula_data first, fallback if column doesn't exist
            has_formula_column = True
            try:
                all_rows_query = """
                    SELECT row_id, row_data, row_index, row_color, 
                           COALESCE(formula_data, '{}'::jsonb) as formula_data
                    FROM custom_data_rows
                    WHERE table_id = $1
                    ORDER BY row_index
                """
                all_rows_raw = await self.db.fetch(
                    all_rows_query,
                    table_id,
                    user_id=user_id,
                    user_team_ids=user_team_ids
                )
            except Exception as e:
                if 'formula_data' in str(e).lower() or 'column' in str(e).lower():
                    # Column doesn't exist, use fallback query
                    logger.warning(f"formula_data column not found, using fallback query: {e}")
                    has_formula_column = False
                    all_rows_query = """
                        SELECT row_id, row_data, row_index, row_color
                        FROM custom_data_rows
                        WHERE table_id = $1
                        ORDER BY row_index
                    """
                    all_rows_raw = await self.db.fetch(
                        all_rows_query,
                        table_id,
                        user_id=user_id,
                        user_team_ids=user_team_ids
                    )
                else:
                    raise
            
            # Parse all rows for formula context
            all_rows = []
            for row in all_rows_raw:
                row_data = json.loads(row['row_data']) if isinstance(row['row_data'], str) else row['row_data']
                if has_formula_column:
                    formula_data = json.loads(row['formula_data']) if row.get('formula_data') and isinstance(row['formula_data'], str) else (row.get('formula_data') or {})
                else:
                    formula_data = {}
                all_rows.append({
                    'row_id': row['row_id'],
                    'row_data': row_data,
                    'row_index': row['row_index'],
                    'row_color': row['row_color'],
                    'formula_data': formula_data
                })
            
            # Get paginated rows
            # Use same approach as all_rows query
            if has_formula_column:
                query = """
                    SELECT row_id, row_data, row_index, row_color, 
                           COALESCE(formula_data, '{}'::jsonb) as formula_data
                    FROM custom_data_rows
                    WHERE table_id = $1
                    ORDER BY row_index
                    LIMIT $2 OFFSET $3
                """
            else:
                query = """
                    SELECT row_id, row_data, row_index, row_color
                    FROM custom_data_rows
                    WHERE table_id = $1
                    ORDER BY row_index
                    LIMIT $2 OFFSET $3
                """
            
            rows = await self.db.fetch(
                query, 
                table_id, 
                limit, 
                offset,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            data_rows = []
            for row in rows:
                row_data = json.loads(row['row_data']) if isinstance(row['row_data'], str) else row['row_data']
                if has_formula_column:
                    formula_data = json.loads(row['formula_data']) if row.get('formula_data') and isinstance(row['formula_data'], str) else (row.get('formula_data') or {})
                else:
                    formula_data = {}
                
                # Evaluate formulas for this row
                evaluated_data = row_data.copy()
                for column_name, formula in formula_data.items():
                    if formula and self.formula_evaluator.is_formula(formula):
                        # Find current row index
                        current_row_index = row['row_index']
                        # Evaluate formula
                        result = self.formula_evaluator.evaluate_formula(
                            formula,
                            row_data,
                            all_rows,
                            current_row_index,
                            schema
                        )
                        evaluated_data[column_name] = result
                
                data_rows.append({
                    'row_id': row['row_id'],
                    'row_data': evaluated_data,
                    'row_index': row['row_index'],
                    'row_color': row['row_color'],
                    'formula_data': formula_data  # Include formula data for frontend
                })
            
            return self._pack_table_data_response(
                table_id,
                data_rows,
                table['row_count'],
                offset,
                limit,
                schema,
                prefer_arrow,
            )
            
        except Exception as e:
            logger.error(f"Failed to get table data {table_id}: {e}")
            raise
    
    async def insert_row(
        self,
        table_id: str,
        row_data: Dict[str, Any],
        user_id: str,
        row_color: Optional[str] = None,
        formula_data: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Insert a single row with optional formula data"""
        try:
            table = await self.get_table(table_id, user_id=user_id)
            if not table:
                raise ValueError('Table not found')
            storage_type = table.get('storage_type') or 'jsonb'
            schema = json.loads(table['schema_json']) if isinstance(table['schema_json'], str) else table['schema_json']
            
            if storage_type == 'native':
                return await self._insert_native_row(
                    table_id=table_id,
                    table=table,
                    schema=schema,
                    row_data=row_data,
                    user_id=user_id,
                    user_team_ids=None
                )
            
            row_id = str(uuid.uuid4())
            
            # Get current max row_index
            max_index_query = """
                SELECT COALESCE(MAX(row_index), -1) FROM custom_data_rows WHERE table_id = $1
            """
            max_index = await self.db.fetchval(max_index_query, table_id)
            new_index = max_index + 1
            
            # Separate formulas from values
            actual_row_data = {}
            actual_formula_data = formula_data or {}
            
            for key, value in row_data.items():
                if self.formula_evaluator.is_formula(value):
                    actual_formula_data[key] = value
                    # Don't store formula in row_data, will be evaluated
                else:
                    actual_row_data[key] = value
            
            # Insert row
            insert_query = """
                INSERT INTO custom_data_rows 
                (row_id, table_id, row_data, row_index, row_color, formula_data, created_at, updated_at, created_by, updated_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING row_id, row_data, row_index, row_color, formula_data
            """
            
            now = datetime.utcnow()
            formula_json = json.dumps(actual_formula_data) if actual_formula_data else None
            row = await self.db.fetchrow(
                insert_query,
                row_id, table_id, json.dumps(actual_row_data), new_index, row_color, formula_json, now, now, user_id, user_id
            )
            
            # Update table row count
            await self._update_table_row_count(table_id, user_id)
            
            logger.info(f"Inserted row {row_id} into table {table_id}")
            
            return {
                'row_id': row['row_id'],
                'row_data': json.loads(row['row_data']) if isinstance(row['row_data'], str) else row['row_data'],
                'row_index': row['row_index'],
                'row_color': row['row_color'],
                'formula_data': json.loads(row['formula_data']) if row.get('formula_data') and isinstance(row['formula_data'], str) else (row.get('formula_data') or {})
            }
            
        except Exception as e:
            logger.error(f"Failed to insert row into table {table_id}: {e}")
            raise
    
    def _coerce_native_cell_value(
        self, col_def: Dict[str, Any], value: Any
    ) -> Any:
        """Normalize a cell value for insertion into a typed native column."""
        if value is None:
            return None
        ctype = str(col_def.get("type", "TEXT")).upper()
        if ctype == "JSON":
            if isinstance(value, (dict, list)):
                return json.dumps(value)
            return value
        if ctype == "INTEGER":
            if hasattr(value, "item"):
                try:
                    return int(value.item())
                except (TypeError, ValueError):
                    pass
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        if ctype == "REAL":
            if hasattr(value, "item"):
                try:
                    return float(value.item())
                except (TypeError, ValueError):
                    pass
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
        if ctype == "BOOLEAN":
            if hasattr(value, "item"):
                try:
                    return bool(value.item())
                except (TypeError, ValueError):
                    pass
            return bool(value)
        return value

    async def _bulk_insert_native_rows(
        self,
        table_id: str,
        table: Dict[str, Any],
        schema: Dict[str, Any],
        rows_data: List[Dict[str, Any]],
        user_id: str,
        batch_size: int = 1000,
    ) -> int:
        """Bulk insert into a native workspace table using PostgreSQL COPY."""
        workspace_id = await self._get_workspace_id_for_database(
            table["database_id"], user_id=user_id, user_team_ids=None
        )
        if not workspace_id:
            raise ValueError("Workspace not found")
        schema_name = self._workspace_schema_name(workspace_id)
        table_name = f't_{table_id.replace("-", "_")}'
        columns_meta = [
            c
            for c in (schema.get("columns") or [])
            if isinstance(c, dict) and c.get("name")
        ]
        data_cols = [str(c["name"]) for c in columns_meta]
        safe_cols = [self._sanitize_sql_identifier(c) for c in data_cols]
        pg_columns = ["row_id", "row_index"] + safe_cols

        max_index_q = (
            f'SELECT COALESCE(MAX(row_index), -1) FROM "{schema_name}"."{table_name}"'
        )
        max_index = await self.db.fetchval(
            max_index_q, user_id=user_id, user_team_ids=None
        )
        start_idx = (max_index if max_index is not None else -1) + 1

        total_inserted = 0
        for i in range(0, len(rows_data), batch_size):
            batch = rows_data[i : i + batch_size]
            records = []
            for idx, row in enumerate(batch):
                row_id = str(uuid.uuid4())
                ri = start_idx + total_inserted + idx
                vals = [
                    self._coerce_native_cell_value(col_def, row.get(orig))
                    for orig, col_def in zip(data_cols, columns_meta)
                ]
                records.append(tuple([row_id, ri] + vals))
            async with self.db.acquire(
                user_id=user_id, user_team_ids=None
            ) as conn:
                await conn.copy_records_to_table(
                    table_name,
                    schema_name=schema_name,
                    records=records,
                    columns=pg_columns,
                )
            total_inserted += len(batch)
            logger.info(
                "Native COPY batch: %s/%s rows into %s.%s",
                total_inserted,
                len(rows_data),
                schema_name,
                table_name,
            )

        await self._update_table_row_count(table_id, user_id)
        logger.info(
            "Native bulk insert complete: %s rows into table %s",
            total_inserted,
            table_id,
        )
        return total_inserted

    async def bulk_insert_rows(
        self,
        table_id: str,
        rows_data: List[Dict[str, Any]],
        user_id: str,
        batch_size: int = 1000
    ) -> int:
        """Bulk insert rows efficiently (native tables use COPY; JSONB uses executemany)."""
        try:
            table = await self.get_table(table_id, user_id=user_id)
            if not table:
                raise ValueError("Table not found")
            storage_type = table.get("storage_type") or "jsonb"
            sch = table["schema_json"]
            schema = json.loads(sch) if isinstance(sch, str) else sch

            if storage_type == "native":
                return await self._bulk_insert_native_rows(
                    table_id,
                    table,
                    schema,
                    rows_data,
                    user_id,
                    batch_size,
                )

            max_index_query = """
                SELECT COALESCE(MAX(row_index), -1) FROM custom_data_rows WHERE table_id = $1
            """
            start_index = await self.db.fetchval(max_index_query, table_id) + 1

            now = datetime.utcnow()
            insert_query = """
                INSERT INTO custom_data_rows
                (row_id, table_id, row_data, row_index, created_at, updated_at, created_by, updated_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """

            total_inserted = 0
            for i in range(0, len(rows_data), batch_size):
                batch = rows_data[i : i + batch_size]
                args_list = []

                for idx, row_data in enumerate(batch):
                    row_id = str(uuid.uuid4())
                    row_index = start_index + total_inserted + idx
                    args_list.append(
                        (
                            row_id,
                            table_id,
                            json.dumps(row_data),
                            row_index,
                            now,
                            now,
                            user_id,
                            user_id,
                        )
                    )

                await self.db.executemany(insert_query, args_list)
                total_inserted += len(batch)

                logger.info(
                    "Inserted batch: %s/%s rows into table %s",
                    total_inserted,
                    len(rows_data),
                    table_id,
                )

            await self._update_table_row_count(table_id, user_id)

            logger.info(
                "Bulk insert complete: %s rows into table %s",
                total_inserted,
                table_id,
            )
            return total_inserted

        except Exception as e:
            logger.error(f"Failed to bulk insert rows into table {table_id}: {e}")
            raise

    async def _native_fetch_rows_export_page(
        self,
        table_id: str,
        table: Dict[str, Any],
        schema: Dict[str, Any],
        offset: int,
        limit: int,
        user_id: Optional[str],
        user_team_ids: Optional[List[str]],
    ) -> List[Any]:
        workspace_id = await self._get_workspace_id_for_database(
            table['database_id'], user_id=user_id, user_team_ids=user_team_ids
        )
        if not workspace_id:
            return []
        schema_name = self._workspace_schema_name(workspace_id)
        is_schema_only = table.get('_schema_only') is True
        table_name = table_id if is_schema_only else f't_{table_id.replace("-", "_")}'
        data_columns = [
            c['name']
            for c in schema.get('columns', [])
            if isinstance(c, dict) and c.get('name')
        ]
        if is_schema_only:
            q = (
                f'SELECT * FROM "{schema_name}"."{table_name}" '
                f'ORDER BY 1 LIMIT $1 OFFSET $2'
            )
        elif not data_columns:
            q = (
                f'SELECT row_id, row_index FROM "{schema_name}"."{table_name}" '
                f'ORDER BY row_index LIMIT $1 OFFSET $2'
            )
        else:
            cols = ', '.join(
                f'"{self._sanitize_sql_identifier(c)}"' for c in data_columns
            )
            q = (
                f'SELECT row_id, row_index, {cols} '
                f'FROM "{schema_name}"."{table_name}" '
                f'ORDER BY row_index LIMIT $1 OFFSET $2'
            )
        return await self.db.fetch(
            q, limit, offset, user_id=user_id, user_team_ids=user_team_ids
        )

    def _jsonb_rows_to_arrow_records(
        self, rows_raw: List[Any], has_formula_column: bool
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rows_raw:
            rd = r['row_data']
            rd_s = rd if isinstance(rd, str) else json.dumps(rd)
            fd_s = ''
            if has_formula_column:
                fd = r.get('formula_data')
                fd_s = (
                    fd
                    if isinstance(fd, str)
                    else json.dumps(fd if fd is not None else {})
                )
            out.append(
                {
                    'row_id': r['row_id'],
                    'row_index': r['row_index'],
                    'row_color': r.get('row_color') or '',
                    'row_data_json': rd_s,
                    'formula_data_json': fd_s,
                }
            )
        return out

    async def _export_native_stream(
        self,
        table_id: str,
        table: Dict[str, Any],
        schema: Dict[str, Any],
        user_id: str,
        user_team_ids: Optional[List[str]],
        fmt: str,
        batch_size: int,
    ) -> AsyncIterator[Dict[str, Any]]:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq

        from utils.arrow_utils import (
            asyncpg_rows_to_record_batch,
            record_batch_to_ipc_bytes,
        )

        if fmt == 'parquet':
            tmp_path = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False).name
            writer = None
            try:
                offset = 0
                while True:
                    rows = await self._native_fetch_rows_export_page(
                        table_id,
                        table,
                        schema,
                        offset,
                        batch_size,
                        user_id,
                        user_team_ids,
                    )
                    if not rows:
                        break
                    batch = asyncpg_rows_to_record_batch(
                        rows, list(rows[0].keys())
                    )
                    tab = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, tab.schema)
                    writer.write_table(tab)
                    offset += len(rows)
                    if len(rows) < batch_size:
                        break
                if writer is None:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    yield {'data': b'', 'is_last': True}
                    return
                if writer:
                    writer.close()
                    writer = None
                with open(tmp_path, 'rb') as f:
                    while True:
                        blk = f.read(1024 * 1024)
                        if not blk:
                            break
                        yield {'data': blk, 'is_last': False}
                yield {'data': b'', 'is_last': True}
            finally:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            return

        offset = 0
        first_csv = True
        while True:
            rows = await self._native_fetch_rows_export_page(
                table_id,
                table,
                schema,
                offset,
                batch_size,
                user_id,
                user_team_ids,
            )
            if not rows:
                yield {'data': b'', 'is_last': True}
                return
            batch = asyncpg_rows_to_record_batch(rows, list(rows[0].keys()))
            if fmt == 'arrow_ipc':
                yield {'data': record_batch_to_ipc_bytes(batch), 'is_last': False}
            else:
                buf = io.BytesIO()
                pacsv.write_csv(
                    pa.Table.from_batches([batch]),
                    buf,
                    write_options=pacsv.WriteOptions(include_header=first_csv),
                )
                first_csv = False
                yield {'data': buf.getvalue(), 'is_last': False}
            offset += len(rows)
            if len(rows) < batch_size:
                yield {'data': b'', 'is_last': True}
                return

    async def _export_jsonb_stream(
        self,
        table_id: str,
        user_id: str,
        user_team_ids: Optional[List[str]],
        fmt: str,
        batch_size: int,
        has_formula_column: bool,
    ) -> AsyncIterator[Dict[str, Any]]:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq

        from utils.arrow_utils import (
            asyncpg_rows_to_record_batch,
            record_batch_to_ipc_bytes,
        )

        if fmt == 'parquet':
            tmp_path = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False).name
            writer = None
            try:
                offset = 0
                while True:
                    if has_formula_column:
                        q = """
                            SELECT row_id, row_data, row_index, row_color,
                                   COALESCE(formula_data, '{}'::jsonb) as formula_data
                            FROM custom_data_rows WHERE table_id = $1
                            ORDER BY row_index LIMIT $2 OFFSET $3
                        """
                    else:
                        q = """
                            SELECT row_id, row_data, row_index, row_color
                            FROM custom_data_rows WHERE table_id = $1
                            ORDER BY row_index LIMIT $2 OFFSET $3
                        """
                    rows_raw = await self.db.fetch(
                        q,
                        table_id,
                        batch_size,
                        offset,
                        user_id=user_id,
                        user_team_ids=user_team_ids,
                    )
                    if not rows_raw:
                        break
                    recs = self._jsonb_rows_to_arrow_records(
                        rows_raw, has_formula_column
                    )
                    batch = asyncpg_rows_to_record_batch(
                        recs, list(recs[0].keys())
                    )
                    tab = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, tab.schema)
                    writer.write_table(tab)
                    offset += len(rows_raw)
                    if len(rows_raw) < batch_size:
                        break
                if writer is None:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    yield {'data': b'', 'is_last': True}
                    return
                if writer:
                    writer.close()
                    writer = None
                with open(tmp_path, 'rb') as f:
                    while True:
                        blk = f.read(1024 * 1024)
                        if not blk:
                            break
                        yield {'data': blk, 'is_last': False}
                yield {'data': b'', 'is_last': True}
            finally:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            return

        offset = 0
        first_csv = True
        while True:
            if has_formula_column:
                q = """
                    SELECT row_id, row_data, row_index, row_color,
                           COALESCE(formula_data, '{}'::jsonb) as formula_data
                    FROM custom_data_rows WHERE table_id = $1
                    ORDER BY row_index LIMIT $2 OFFSET $3
                """
            else:
                q = """
                    SELECT row_id, row_data, row_index, row_color
                    FROM custom_data_rows WHERE table_id = $1
                    ORDER BY row_index LIMIT $2 OFFSET $3
                """
            rows_raw = await self.db.fetch(
                q,
                table_id,
                batch_size,
                offset,
                user_id=user_id,
                user_team_ids=user_team_ids,
            )
            if not rows_raw:
                yield {'data': b'', 'is_last': True}
                return
            recs = self._jsonb_rows_to_arrow_records(
                rows_raw, has_formula_column
            )
            batch = asyncpg_rows_to_record_batch(recs, list(recs[0].keys()))
            if fmt == 'arrow_ipc':
                yield {'data': record_batch_to_ipc_bytes(batch), 'is_last': False}
            else:
                buf = io.BytesIO()
                pacsv.write_csv(
                    pa.Table.from_batches([batch]),
                    buf,
                    write_options=pacsv.WriteOptions(include_header=first_csv),
                )
                first_csv = False
                yield {'data': buf.getvalue(), 'is_last': False}
            offset += len(rows_raw)
            if len(rows_raw) < batch_size:
                yield {'data': b'', 'is_last': True}
                return

    async def export_table_data_stream(
        self,
        table_id: str,
        user_id: str,
        fmt: str,
        user_team_ids: Optional[List[str]] = None,
        database_id: Optional[str] = None,
        batch_size: int = 10000,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream table export chunks. Each dict has 'data' (bytes) and 'is_last' (bool).
        Supported formats: arrow_ipc, csv, parquet.
        """
        fmt = (fmt or 'arrow_ipc').lower().strip()
        if fmt not in ('arrow_ipc', 'csv', 'parquet'):
            raise ValueError(f'Unsupported export format: {fmt}')

        table = await self.get_table(
            table_id,
            user_id=user_id,
            user_team_ids=user_team_ids,
            database_id=database_id,
        )
        if not table:
            raise ValueError('Table not found')
        storage_type = table.get('storage_type') or 'jsonb'
        schema = (
            json.loads(table['schema_json'])
            if isinstance(table['schema_json'], str)
            else table['schema_json']
        )

        if storage_type == 'native':
            async for part in self._export_native_stream(
                table_id,
                table,
                schema,
                user_id,
                user_team_ids,
                fmt,
                batch_size,
            ):
                yield part
            return

        has_formula_column = True
        try:
            await self.db.fetch(
                """
                SELECT row_id, row_data, row_index, row_color,
                       COALESCE(formula_data, '{}'::jsonb) as formula_data
                FROM custom_data_rows WHERE table_id = $1 ORDER BY row_index LIMIT 1
                """,
                table_id,
                user_id=user_id,
                user_team_ids=user_team_ids,
            )
        except Exception as e:
            err = str(e).lower()
            if 'formula_data' in err or 'column' in err:
                has_formula_column = False
            else:
                raise

        async for part in self._export_jsonb_stream(
            table_id,
            user_id,
            user_team_ids,
            fmt,
            batch_size,
            has_formula_column,
        ):
            yield part

    async def update_row(
        self,
        table_id: str,
        row_id: str,
        row_data: Dict[str, Any],
        user_id: str,
        formula_data: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a row with optional formula data"""
        try:
            table = await self.get_table(table_id, user_id=user_id)
            if not table:
                return None
            storage_type = table.get('storage_type') or 'jsonb'
            if storage_type == 'native':
                return await self._update_native_row(
                    table_id=table_id,
                    table=table,
                    row_id=row_id,
                    row_data=row_data,
                    user_id=user_id,
                    user_team_ids=None
                )
            # Get current formula_data
            current_query = "SELECT formula_data FROM custom_data_rows WHERE row_id = $1 AND table_id = $2"
            current_result = await self.db.fetchrow(current_query, row_id, table_id)
            current_formula_data = {}
            if current_result and current_result.get('formula_data'):
                current_formula_data = json.loads(current_result['formula_data']) if isinstance(current_result['formula_data'], str) else current_result['formula_data']
            
            # Merge formula updates
            if formula_data:
                current_formula_data.update(formula_data)
            
            # Separate formulas from values
            actual_row_data = {}
            for key, value in row_data.items():
                if self.formula_evaluator.is_formula(value):
                    current_formula_data[key] = value
                else:
                    actual_row_data[key] = value
                    # Remove from formula_data if it was a formula before
                    if key in current_formula_data:
                        del current_formula_data[key]
            
            query = """
                UPDATE custom_data_rows
                SET row_data = $1, formula_data = $2, updated_at = $3, updated_by = $4
                WHERE row_id = $5 AND table_id = $6
                RETURNING row_id, row_data, row_index, row_color, formula_data
            """
            
            formula_json = json.dumps(current_formula_data) if current_formula_data else None
            row = await self.db.fetchrow(
                query,
                json.dumps(actual_row_data),
                formula_json,
                datetime.utcnow(),
                user_id,
                row_id,
                table_id
            )
            
            if row:
                logger.info(f"Updated row {row_id}")
                return {
                    'row_id': row['row_id'],
                    'row_data': json.loads(row['row_data']) if isinstance(row['row_data'], str) else row['row_data'],
                    'row_index': row['row_index'],
                    'row_color': row['row_color'],
                    'formula_data': json.loads(row['formula_data']) if row.get('formula_data') and isinstance(row['formula_data'], str) else (row.get('formula_data') or {})
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to update row {row_id}: {e}")
            raise
    
    async def update_cell(
        self,
        table_id: str,
        row_id: str,
        column_name: str,
        value: Any,
        user_id: str,
        formula: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Update a single cell in a row, with optional formula"""
        try:
            table = await self.get_table(table_id, user_id=user_id)
            if not table:
                return None
            storage_type = table.get('storage_type') or 'jsonb'
            if storage_type == 'native':
                return await self._update_native_cell(
                    table_id=table_id,
                    table=table,
                    row_id=row_id,
                    column_name=column_name,
                    value=value,
                    user_id=user_id,
                    user_team_ids=None
                )
            # Get current row data and formula_data
            query = "SELECT row_data, formula_data FROM custom_data_rows WHERE row_id = $1 AND table_id = $2"
            result = await self.db.fetchrow(query, row_id, table_id)
            
            if not result:
                return None
            
            # Parse current data
            row_data = json.loads(result['row_data']) if isinstance(result['row_data'], str) else result['row_data']
            formula_data = json.loads(result['formula_data']) if result.get('formula_data') and isinstance(result['formula_data'], str) else (result.get('formula_data') or {})
            
            # Determine if this is a formula
            is_formula = False
            if formula:
                is_formula = True
                formula_data[column_name] = formula
                # Remove from row_data if it was there
                if column_name in row_data:
                    del row_data[column_name]
            elif self.formula_evaluator.is_formula(value):
                is_formula = True
                formula_data[column_name] = value
                # Don't store formula in row_data
            else:
                # Regular value - remove from formula_data if present
                row_data[column_name] = value
                if column_name in formula_data:
                    del formula_data[column_name]
            
            # Save updated data
            update_query = """
                UPDATE custom_data_rows
                SET row_data = $1, formula_data = $2, updated_at = $3, updated_by = $4
                WHERE row_id = $5 AND table_id = $6
                RETURNING row_id, row_data, row_index, row_color, formula_data
            """
            
            formula_json = json.dumps(formula_data) if formula_data else None
            updated_row = await self.db.fetchrow(
                update_query,
                json.dumps(row_data),
                formula_json,
                datetime.utcnow(),
                user_id,
                row_id,
                table_id
            )
            
            if updated_row:
                logger.info(f"Updated cell {column_name} in row {row_id}")
                return {
                    'row_id': updated_row['row_id'],
                    'row_data': json.loads(updated_row['row_data']) if isinstance(updated_row['row_data'], str) else updated_row['row_data'],
                    'row_index': updated_row['row_index'],
                    'row_color': updated_row['row_color'],
                    'formula_data': json.loads(updated_row['formula_data']) if updated_row.get('formula_data') and isinstance(updated_row['formula_data'], str) else (updated_row.get('formula_data') or {})
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to update cell in row {row_id}: {e}")
            raise
    
    async def delete_row(self, table_id: str, row_id: str) -> bool:
        """Delete a row"""
        try:
            table = await self.get_table(table_id)
            if not table:
                return False
            storage_type = table.get('storage_type') or 'jsonb'
            if storage_type == 'native':
                workspace_id = await self._get_workspace_id_for_database(table['database_id'])
                if not workspace_id:
                    return False
                schema_name = self._workspace_schema_name(workspace_id)
                table_name = f't_{table_id.replace("-", "_")}'
                result = await self.db.execute(
                    f'DELETE FROM "{schema_name}"."{table_name}" WHERE row_id = $1',
                    row_id,
                    user_id=None,
                    user_team_ids=None
                )
                deleted = result.split()[-1] != '0'
                if deleted:
                    await self._update_table_row_count(table_id, None)
                return deleted
            # JSONB path
            delete_query = "DELETE FROM custom_data_rows WHERE row_id = $1 AND table_id = $2"
            result = await self.db.execute(delete_query, row_id, table_id)
            
            deleted = result.split()[-1] != '0'
            
            if deleted and table_id:
                # Update table row count (user_id not available for deletes, pass None)
                await self._update_table_row_count(table_id, None)
                logger.info(f"Deleted row {row_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete row {row_id}: {e}")
            raise
    
    async def infer_schema_from_data(
        self,
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Infer schema from data using pandas"""
        try:
            if not data:
                return {'columns': []}
            
            # Convert to DataFrame for type inference
            df = pd.DataFrame(data)
            
            columns = []
            for col_name in df.columns:
                dtype = df[col_name].dtype
                
                # Map pandas dtype to SQL type
                if pd.api.types.is_integer_dtype(dtype):
                    col_type = 'INTEGER'
                elif pd.api.types.is_float_dtype(dtype):
                    col_type = 'FLOAT'
                elif pd.api.types.is_bool_dtype(dtype):
                    col_type = 'BOOLEAN'
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    col_type = 'TIMESTAMP'
                else:
                    col_type = 'TEXT'
                
                # Check for nulls
                has_nulls = df[col_name].isnull().any()
                
                columns.append({
                    'name': col_name,
                    'type': col_type,
                    'nullable': bool(has_nulls)
                })
            
            return {'columns': columns}
            
        except Exception as e:
            logger.error(f"Failed to infer schema: {e}")
            raise
    
    async def _update_table_row_count(self, table_id: str, user_id: Optional[str] = None):
        """Update the row count for a table (JSONB or native)."""
        try:
            meta = await self.db.fetchrow(
                "SELECT storage_type, database_id FROM custom_tables WHERE table_id = $1",
                table_id
            )
            if not meta:
                return
            storage_type = (meta.get('storage_type') or 'jsonb')
            if storage_type == 'native':
                workspace_id = await self._get_workspace_id_for_database(
                    meta['database_id'], user_id=user_id
                )
                if not workspace_id:
                    return
                schema_name = self._workspace_schema_name(workspace_id)
                table_name = f't_{table_id.replace("-", "_")}'
                count = await self.db.fetchval(
                    f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"',
                    user_id=user_id
                )
            else:
                count_query = "SELECT COUNT(*) FROM custom_data_rows WHERE table_id = $1"
                count = await self.db.fetchval(count_query, table_id)
            
            update_query = """
                UPDATE custom_tables 
                SET row_count = $2, updated_at = $3, updated_by = $4
                WHERE table_id = $1
            """
            await self.db.execute(update_query, table_id, count, datetime.utcnow(), user_id)
            
        except Exception as e:
            logger.error(f"Failed to update row count for table {table_id}: {e}")
            raise
    
    async def recalculate_table(self, table_id: str, user_id: str) -> Dict[str, Any]:
        """Recalculate all formulas in a table"""
        try:
            # Get table schema
            table = await self.get_table(table_id)
            if not table:
                return {'success': False, 'error_message': 'Table not found', 'cells_recalculated': 0}
            
            schema = json.loads(table['schema_json']) if isinstance(table['schema_json'], str) else table['schema_json']
            
            # Get all rows with formulas
            query = """
                SELECT row_id, row_data, row_index, formula_data
                FROM custom_data_rows
                WHERE table_id = $1 AND formula_data IS NOT NULL
                ORDER BY row_index
            """
            rows = await self.db.fetch(query, table_id)
            
            if not rows:
                return {'success': True, 'cells_recalculated': 0, 'error_message': None}
            
            # Parse all rows for context
            all_rows = []
            for row in rows:
                row_data = json.loads(row['row_data']) if isinstance(row['row_data'], str) else row['row_data']
                formula_data = json.loads(row['formula_data']) if row.get('formula_data') and isinstance(row['formula_data'], str) else (row.get('formula_data') or {})
                all_rows.append({
                    'row_id': row['row_id'],
                    'row_data': row_data,
                    'row_index': row['row_index'],
                    'formula_data': formula_data
                })
            
            # Recalculate each row
            cells_recalculated = 0
            for row in rows:
                row_data = json.loads(row['row_data']) if isinstance(row['row_data'], str) else row['row_data']
                formula_data = json.loads(row['formula_data']) if row.get('formula_data') and isinstance(row['formula_data'], str) else (row.get('formula_data') or {})
                
                # Evaluate each formula
                updated_row_data = row_data.copy()
                for column_name, formula in formula_data.items():
                    if formula and self.formula_evaluator.is_formula(formula):
                        result = self.formula_evaluator.evaluate_formula(
                            formula,
                            row_data,
                            all_rows,
                            row['row_index'],
                            schema
                        )
                        updated_row_data[column_name] = result
                        cells_recalculated += 1
                
                # Update row with recalculated values
                update_query = """
                    UPDATE custom_data_rows
                    SET row_data = $1, updated_at = $2, updated_by = $3
                    WHERE row_id = $4 AND table_id = $5
                """
                await self.db.execute(
                    update_query,
                    json.dumps(updated_row_data),
                    datetime.utcnow(),
                    user_id,
                    row['row_id'],
                    table_id
                )
            
            logger.info(f"Recalculated {cells_recalculated} cells in table {table_id}")
            return {
                'success': True,
                'cells_recalculated': cells_recalculated,
                'error_message': None
            }
            
        except Exception as e:
            logger.error(f"Failed to recalculate table {table_id}: {e}")
            return {
                'success': False,
                'cells_recalculated': 0,
                'error_message': str(e)
            }
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        if not row:
            return {}
        
        return {
            'table_id': row['table_id'],
            'database_id': row['database_id'],
            'name': row['name'],
            'description': row['description'],
            'row_count': row['row_count'],
            'schema_json': row['schema_json'] if isinstance(row['schema_json'], str) else json.dumps(row['schema_json']),
            'styling_rules_json': row['styling_rules_json'] if isinstance(row['styling_rules_json'], str) else json.dumps(row['styling_rules_json']),
            'metadata_json': row['metadata_json'] if isinstance(row['metadata_json'], str) else json.dumps(row['metadata_json']),
            'created_at': row['created_at'].isoformat() if row['created_at'] else None,
            'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None,
            'created_by': row.get('created_by'),
            'updated_by': row.get('updated_by'),
            'storage_type': row.get('storage_type', 'jsonb'),
        }


