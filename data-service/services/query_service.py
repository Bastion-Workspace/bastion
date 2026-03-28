"""
Query Service - Execute SQL and natural language queries with security validation
"""

import logging
import uuid
import re
import json
import time
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

import sqlparse
from sqlparse.sql import Statement, IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

from db.connection_manager import DatabaseConnectionManager
from services.table_service import TableService
from services.database_service import DatabaseService

logger = logging.getLogger(__name__)


class QueryService:
    """Service for executing SQL queries with security validation"""
    
    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db = db_manager
        self.table_service = TableService(db_manager)
        self.database_service = DatabaseService(db_manager)
    
    def _extract_table_names(self, sql: str) -> List[str]:
        """
        Extract table names from SQL query using sqlparse
        
        Returns list of table identifiers found in the query
        """
        try:
            parsed = sqlparse.parse(sql)
            table_names = []
            
            for statement in parsed:
                # Get tokens from statement
                tokens = statement.tokens
                
                # Look for FROM and JOIN clauses
                from_seen = False
                for i, token in enumerate(tokens):
                    # Check if this is a FROM keyword
                    if token.ttype is Keyword and token.value.upper() == 'FROM':
                        from_seen = True
                        continue
                    
                    # Check if this is a JOIN keyword
                    if token.ttype is Keyword and 'JOIN' in token.value.upper():
                        from_seen = True
                        continue
                    
                    # After FROM/JOIN, extract identifiers
                    if from_seen and isinstance(token, Identifier):
                        # Extract table name (handle schema.table format)
                        table_name = token.get_real_name()
                        if table_name:
                            table_names.append(table_name.lower())
                    elif from_seen and isinstance(token, IdentifierList):
                        # Multiple tables in FROM clause
                        for identifier in token.get_identifiers():
                            table_name = identifier.get_real_name()
                            if table_name:
                                table_names.append(table_name.lower())
            
            # Also check for UPDATE and INSERT statements
            for statement in parsed:
                tokens = statement.tokens
                for i, token in enumerate(tokens):
                    if token.ttype is DML:
                        dml_type = token.value.upper()
                        if dml_type in ('UPDATE', 'INSERT', 'DELETE'):
                            # Next identifier should be table name
                            if i + 1 < len(tokens):
                                next_token = tokens[i + 1]
                                if isinstance(next_token, Identifier):
                                    table_name = next_token.get_real_name()
                                    if table_name:
                                        table_names.append(table_name.lower())
            
            # Remove duplicates and return
            return list(set(table_names))
            
        except Exception as e:
            logger.warning(f"Failed to parse SQL for table names: {e}")
            # Fallback: simple regex extraction
            return self._extract_table_names_regex(sql)
    
    def _extract_table_names_regex(self, sql: str) -> List[str]:
        """Fallback regex-based table name extraction"""
        table_names = []
        
        # Pattern for FROM table_name
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(from_pattern, sql, re.IGNORECASE)
        table_names.extend([m.lower() for m in matches])
        
        # Pattern for JOIN table_name
        join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(join_pattern, sql, re.IGNORECASE)
        table_names.extend([m.lower() for m in matches])
        
        # Pattern for UPDATE table_name
        update_pattern = r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(update_pattern, sql, re.IGNORECASE)
        table_names.extend([m.lower() for m in matches])
        
        # Pattern for INSERT INTO table_name
        insert_pattern = r'\bINSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(insert_pattern, sql, re.IGNORECASE)
        table_names.extend([m.lower() for m in matches])
        
        return list(set(table_names))
    
    @staticmethod
    def _workspace_schema_name(workspace_id: str) -> str:
        """Return schema name for workspace (must match TableService)."""
        return 'ws_' + re.sub(r'[^a-zA-Z0-9_]', '_', workspace_id)[:50]
    
    @staticmethod
    def _get_statement_type(sql: str) -> str:
        """Return SELECT, INSERT, UPDATE, DELETE, or DDL (CREATE/ALTER/DROP) from first token."""
        stripped = sql.strip().upper()
        for token in ('SELECT', 'INSERT', 'UPDATE', 'DELETE'):
            if stripped.startswith(token):
                return token
        for token in ('CREATE', 'ALTER', 'DROP'):
            if stripped.startswith(token):
                return token
        return 'SELECT'

    @staticmethod
    def _looks_like_sql(text: str) -> bool:
        """Heuristic: treat as SQL if the trimmed text starts with a common SQL keyword."""
        if not text or not text.strip():
            return False
        u = text.strip().upper()
        return any(
            u.startswith(p)
            for p in (
                "SELECT",
                "WITH",
                "INSERT",
                "UPDATE",
                "DELETE",
                "EXPLAIN",
                "SHOW",
                "CREATE",
                "ALTER",
                "DROP",
            )
        )
    
    @staticmethod
    def _parse_rows_affected(execute_result: str) -> int:
        """Parse asyncpg execute result string (e.g. 'UPDATE 3', 'INSERT 0 1') to get rows affected."""
        if not execute_result:
            return 0
        parts = execute_result.split()
        if len(parts) >= 2 and parts[-1].isdigit():
            return int(parts[-1])
        return 0
    
    async def _validate_table_access(
        self,
        workspace_id: str,
        table_names: List[str],
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Verify all referenced tables belong to the workspace
        
        Args:
            workspace_id: Workspace to check against
            table_names: List of table names referenced in query
            user_id: User ID for RLS context
            user_team_ids: Team IDs for RLS context
            
        Returns:
            True if all tables are accessible
            
        Raises:
            ValueError: If any table is not accessible
        """
        if not table_names:
            return True
        
        # Get all databases in workspace
        databases = await self.database_service.list_databases(
            workspace_id,
            user_id=user_id,
            user_team_ids=user_team_ids
        )
        
        # Get all tables in all databases
        all_table_names = set()
        for database in databases:
            tables = await self.table_service.list_tables(
                database['database_id'],
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            for table in tables:
                all_table_names.add(table['name'].lower())
        
        # Check if all referenced tables exist
        for table_name in table_names:
            if table_name not in all_table_names:
                raise ValueError(f"Access denied: Table '{table_name}' not found in workspace")
        
        return True
    
    async def execute_sql_query(
        self,
        workspace_id: str,
        sql_query: str,
        user_id: str,
        limit: int = 1000,
        user_team_ids: Optional[List[str]] = None,
        params: Optional[List[Any]] = None,
        read_only: bool = False,
        prefer_arrow: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute SQL query against workspace databases with security validation.
        Supports SELECT (read) and INSERT/UPDATE/DELETE (write) with parameter binding ($1, $2, ...).
        For native tables, sets search_path to workspace schema so table names resolve.
        When read_only is True, reject DDL and DML (non-SELECT) statements.
        """
        query_id = str(uuid.uuid4())
        start_time = time.time()
        param_list = list(params) if params is not None else []
        
        try:
            table_names = self._extract_table_names(sql_query)
            logger.info(f"Extracted table names from query: {table_names}")
            
            statement_type = self._get_statement_type(sql_query)
            is_ddl = statement_type in ('CREATE', 'ALTER', 'DROP')
            is_write = statement_type in ('INSERT', 'UPDATE', 'DELETE') or is_ddl

            if read_only and (is_write or is_ddl):
                execution_time_ms = int((time.time() - start_time) * 1000)
                err = (
                    "This workspace is bound as read-only. Only SELECT queries are allowed."
                )
                await self._log_query(
                    query_id=query_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    natural_language_query="",
                    generated_sql=sql_query,
                    result_count=0,
                    execution_time_ms=execution_time_ms,
                    error_message=err,
                    user_team_ids=user_team_ids,
                )
                return {
                    "query_id": query_id,
                    "column_names": [],
                    "results_json": json.dumps([]),
                    "result_count": 0,
                    "execution_time_ms": execution_time_ms,
                    "generated_sql": sql_query,
                    "error_message": err,
                    "rows_affected": 0,
                    "returning_rows_json": None,
                    "has_arrow_data": False,
                    "arrow_results": b"",
                }
            
            if not is_ddl:
                await self._validate_table_access(
                    workspace_id,
                    table_names,
                    user_id=user_id,
                    user_team_ids=user_team_ids
                )
            
            if table_names or is_ddl:
                await self.table_service._ensure_workspace_schema_exists(
                    workspace_id, user_id=user_id, user_team_ids=user_team_ids
                )
            
            statement_type = self._get_statement_type(sql_query)
            has_returning = 'RETURNING' in sql_query.upper()
            
            if not is_write:
                sql_upper = sql_query.upper().strip()
                if sql_upper.startswith('SELECT') and 'LIMIT' not in sql_upper:
                    if ';' in sql_query:
                        sql_query = sql_query.rstrip(';') + f' LIMIT {limit};'
                    else:
                        sql_query = sql_query + f' LIMIT {limit}'
            
            schema_name = self._workspace_schema_name(workspace_id)
            arrow_batch = None
            async with self.db.acquire(user_id=user_id, user_team_ids=user_team_ids) as conn:
                await conn.execute(f'SET search_path TO "{schema_name}", public')
                if is_ddl:
                    await conn.execute(sql_query, *param_list) if param_list else await conn.execute(sql_query)
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    column_names = []
                    results = []
                    rows_affected = 0
                elif is_write and has_returning:
                    rows = await conn.fetch(sql_query, *param_list) if param_list else await conn.fetch(sql_query)
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    column_names = list(rows[0].keys()) if rows else []
                    results = [dict(row) for row in rows]
                    for result in results:
                        for key, value in result.items():
                            if isinstance(value, datetime):
                                result[key] = value.isoformat()
                            elif hasattr(value, '__dict__') and not isinstance(value, (dict, list, str, int, float, bool, type(None))):
                                result[key] = str(value)
                    rows_affected = len(results)
                elif is_write:
                    result_str = await conn.execute(sql_query, *param_list) if param_list else await conn.execute(sql_query)
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    rows_affected = self._parse_rows_affected(result_str)
                    column_names = []
                    results = []
                else:
                    rows = await conn.fetch(sql_query, *param_list) if param_list else await conn.fetch(sql_query)
                    execution_time_ms = int((time.time() - start_time) * 1000)
                    if rows:
                        column_names = list(rows[0].keys())
                    else:
                        column_names = []
                    if prefer_arrow:
                        from utils.arrow_utils import asyncpg_rows_to_record_batch

                        arrow_batch = asyncpg_rows_to_record_batch(rows, column_names)
                        results = []
                    else:
                        if rows:
                            results = [dict(row) for row in rows]
                            for result in results:
                                for key, value in result.items():
                                    if isinstance(value, datetime):
                                        result[key] = value.isoformat()
                                    elif hasattr(value, '__dict__') and not isinstance(
                                        value, (dict, list, str, int, float, bool, type(None))
                                    ):
                                        result[key] = str(value)
                        else:
                            results = []
                    rows_affected = 0
            
            if arrow_batch is not None:
                from utils.arrow_utils import record_batch_to_ipc_bytes

                result_count = arrow_batch.num_rows
                results_json = ""
                arrow_bytes = record_batch_to_ipc_bytes(arrow_batch)
                has_arrow = True
            else:
                result_count = len(results)
                results_json = json.dumps(results)
                arrow_bytes = b""
                has_arrow = False

            await self._log_query(
                query_id=query_id,
                workspace_id=workspace_id,
                user_id=user_id,
                natural_language_query="",
                generated_sql=sql_query,
                result_count=result_count,
                execution_time_ms=execution_time_ms,
                error_message=None,
                user_team_ids=user_team_ids
            )
            
            return {
                'query_id': query_id,
                'column_names': column_names,
                'results_json': results_json,
                'result_count': result_count,
                'execution_time_ms': execution_time_ms,
                'generated_sql': sql_query,
                'error_message': None,
                'rows_affected': rows_affected if is_write else 0,
                'returning_rows_json': json.dumps(results) if (is_write and has_returning and results) else None,
                'has_arrow_data': has_arrow,
                'arrow_results': arrow_bytes,
            }
            
        except ValueError as e:
            # Access denied or validation error
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_message = str(e)
            
            await self._log_query(
                query_id=query_id,
                workspace_id=workspace_id,
                user_id=user_id,
                natural_language_query="",
                generated_sql=sql_query,
                result_count=0,
                execution_time_ms=execution_time_ms,
                error_message=error_message,
                user_team_ids=user_team_ids
            )
            
            return {
                'query_id': query_id,
                'column_names': [],
                'results_json': json.dumps([]),
                'result_count': 0,
                'execution_time_ms': execution_time_ms,
                'generated_sql': sql_query,
                'error_message': error_message,
                'rows_affected': 0,
                'returning_rows_json': None,
                'has_arrow_data': False,
                'arrow_results': b"",
            }
            
        except Exception as e:
            # Query execution error
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_message = str(e)
            logger.error(f"Query execution failed: {e}")
            
            await self._log_query(
                query_id=query_id,
                workspace_id=workspace_id,
                user_id=user_id,
                natural_language_query="",
                generated_sql=sql_query,
                result_count=0,
                execution_time_ms=execution_time_ms,
                error_message=error_message,
                user_team_ids=user_team_ids
            )
            
            return {
                'query_id': query_id,
                'column_names': [],
                'results_json': json.dumps([]),
                'result_count': 0,
                'execution_time_ms': execution_time_ms,
                'generated_sql': sql_query,
                'error_message': error_message,
                'rows_affected': 0,
                'returning_rows_json': None,
                'has_arrow_data': False,
                'arrow_results': b"",
            }

    async def execute_nl_query(
        self,
        workspace_id: str,
        natural_query: str,
        user_id: str,
        limit: int = 1000,
        user_team_ids: Optional[List[str]] = None,
        include_documents: bool = False,
        read_only: bool = False,
        prefer_arrow: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a natural-language query. When the text is SQL-shaped, run via execute_sql_query.
        Full LLM-based NL-to-SQL can be added later; read_only is enforced on the SQL path.
        """
        del include_documents  # Reserved for document-augmented NL
        if self._looks_like_sql(natural_query):
            return await self.execute_sql_query(
                workspace_id,
                natural_query.strip(),
                user_id,
                limit=limit,
                user_team_ids=user_team_ids,
                params=None,
                read_only=read_only,
                prefer_arrow=prefer_arrow,
            )
        query_id = str(uuid.uuid4())
        start_time = time.time()
        execution_time_ms = int((time.time() - start_time) * 1000)
        err = (
            "Natural language queries must be expressed as SQL for this service, "
            "or use query_type \"sql\" with a SELECT statement."
        )
        return {
            "query_id": query_id,
            "column_names": [],
            "results_json": json.dumps([]),
            "result_count": 0,
            "execution_time_ms": execution_time_ms,
            "generated_sql": "",
            "error_message": err,
            "rows_affected": 0,
            "returning_rows_json": None,
            "has_arrow_data": False,
            "arrow_results": b"",
        }
    
    async def _log_query(
        self,
        query_id: str,
        workspace_id: str,
        user_id: str,
        natural_language_query: str,
        generated_sql: str,
        result_count: int,
        execution_time_ms: int,
        error_message: Optional[str],
        user_team_ids: Optional[List[str]] = None
    ):
        """Log query execution to data_queries table"""
        try:
            query = """
                INSERT INTO data_queries
                (query_id, workspace_id, user_id, natural_language_query, generated_sql,
                 included_documents, result_count, execution_time_ms, error_message, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            
            await self.db.execute(
                query,
                query_id,
                workspace_id,
                user_id,
                natural_language_query,
                generated_sql,
                False,  # included_documents
                result_count,
                execution_time_ms,
                error_message,
                datetime.utcnow(),
                user_id=user_id,
                user_team_ids=user_team_ids
            )
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")


