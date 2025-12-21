"""
Query Service - Execute SQL and natural language queries with security validation
"""

import logging
import uuid
import re
import json
import time
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
        user_team_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute SQL query against workspace databases with security validation
        
        Args:
            workspace_id: Workspace containing the databases
            sql_query: SQL query to execute
            user_id: User executing the query
            limit: Maximum rows to return
            user_team_ids: User's team IDs for RLS context
            
        Returns:
            Query result with columns, rows, and metadata
        """
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Extract table names from query
            table_names = self._extract_table_names(sql_query)
            logger.info(f"Extracted table names from query: {table_names}")
            
            # Validate table access
            await self._validate_table_access(
                workspace_id,
                table_names,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            # Add LIMIT if not present (for SELECT queries)
            sql_upper = sql_query.upper().strip()
            if sql_upper.startswith('SELECT') and 'LIMIT' not in sql_upper:
                # Add LIMIT clause
                if ';' in sql_query:
                    sql_query = sql_query.rstrip(';') + f' LIMIT {limit};'
                else:
                    sql_query = sql_query + f' LIMIT {limit}'
            
            # Execute query with RLS context
            rows = await self.db.fetch(
                sql_query,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Convert rows to JSON-serializable format
            if rows:
                column_names = list(rows[0].keys())
                results = [dict(row) for row in rows]
                
                # Convert datetime and other non-serializable types
                for result in results:
                    for key, value in result.items():
                        if isinstance(value, datetime):
                            result[key] = value.isoformat()
                        elif hasattr(value, '__dict__'):
                            result[key] = str(value)
            else:
                column_names = []
                results = []
            
            # Log query to data_queries table
            await self._log_query(
                query_id=query_id,
                workspace_id=workspace_id,
                user_id=user_id,
                natural_language_query="",
                generated_sql=sql_query,
                result_count=len(results),
                execution_time_ms=execution_time_ms,
                error_message=None,
                user_team_ids=user_team_ids
            )
            
            return {
                'query_id': query_id,
                'column_names': column_names,
                'results_json': json.dumps(results),
                'result_count': len(results),
                'execution_time_ms': execution_time_ms,
                'generated_sql': sql_query,
                'error_message': None
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
                'error_message': error_message
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
                'error_message': error_message
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

