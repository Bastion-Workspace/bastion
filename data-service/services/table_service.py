import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import pandas as pd

from db.connection_manager import DatabaseConnectionManager
from services.formula_evaluator import FormulaEvaluator

logger = logging.getLogger(__name__)


class TableService:
    """Service for managing tables and their data"""
    
    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db = db_manager
        self.formula_evaluator = FormulaEvaluator()
    
    async def create_table(
        self,
        database_id: str,
        name: str,
        schema: Dict[str, Any],
        user_id: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new table"""
        try:
            table_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            query = """
                INSERT INTO custom_tables 
                (table_id, database_id, name, description, row_count, schema_json,
                 styling_rules_json, metadata_json, created_at, updated_at, created_by, updated_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING table_id, database_id, name, description, row_count, 
                          schema_json, styling_rules_json, metadata_json, created_at, updated_at, created_by, updated_by
            """
            
            row = await self.db.fetchrow(
                query,
                table_id, database_id, name, description, 0,
                json.dumps(schema), json.dumps({}), json.dumps({}), now, now, user_id, user_id,
                user_id=user_id,
                user_team_ids=None
            )
            
            logger.info(f"Created table: {table_id} in database: {database_id}")
            return self._row_to_dict(row)
            
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise
    
    async def list_tables(
        self, 
        database_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List all tables in a database"""
        try:
            query = """
                SELECT table_id, database_id, name, description, row_count,
                       schema_json, styling_rules_json, metadata_json, created_at, updated_at, created_by, updated_by
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
            return [self._row_to_dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list tables for database {database_id}: {e}")
            raise
    
    async def get_table(
        self, 
        table_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get table metadata"""
        try:
            query = """
                SELECT table_id, database_id, name, description, row_count,
                       schema_json, styling_rules_json, metadata_json, created_at, updated_at, created_by, updated_by
                FROM custom_tables
                WHERE table_id = $1
            """
            
            row = await self.db.fetchrow(
                query, 
                table_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            return self._row_to_dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Failed to get table {table_id}: {e}")
            raise
    
    async def delete_table(
        self, 
        table_id: str,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> bool:
        """Delete a table and all its data"""
        try:
            # Delete all rows first
            delete_rows_query = "DELETE FROM custom_data_rows WHERE table_id = $1"
            await self.db.execute(
                delete_rows_query, 
                table_id,
                user_id=user_id,
                user_team_ids=user_team_ids
            )
            
            # Delete table
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
    
    async def get_table_data(
        self,
        table_id: str,
        offset: int = 0,
        limit: int = 100,
        user_id: Optional[str] = None,
        user_team_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get table data with pagination and formula evaluation"""
        try:
            # Get table metadata first
            table = await self.get_table(table_id, user_id=user_id, user_team_ids=user_team_ids)
            if not table:
                return {'error': 'Table not found'}
            
            schema = json.loads(table['schema_json']) if isinstance(table['schema_json'], str) else table['schema_json']
            
            # Get all rows for formula evaluation context (needed for range functions)
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
            
            return {
                'table_id': table_id,
                'rows': data_rows,
                'total_rows': table['row_count'],
                'offset': offset,
                'limit': limit,
                'schema': schema
            }
            
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
    
    async def bulk_insert_rows(
        self,
        table_id: str,
        rows_data: List[Dict[str, Any]],
        user_id: str,
        batch_size: int = 1000
    ) -> int:
        """Bulk insert rows efficiently"""
        try:
            # Get starting row index
            max_index_query = """
                SELECT COALESCE(MAX(row_index), -1) FROM custom_data_rows WHERE table_id = $1
            """
            start_index = await self.db.fetchval(max_index_query, table_id) + 1
            
            # Prepare batch insert
            now = datetime.utcnow()
            insert_query = """
                INSERT INTO custom_data_rows 
                (row_id, table_id, row_data, row_index, created_at, updated_at, created_by, updated_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            total_inserted = 0
            for i in range(0, len(rows_data), batch_size):
                batch = rows_data[i:i + batch_size]
                args_list = []
                
                for idx, row_data in enumerate(batch):
                    row_id = str(uuid.uuid4())
                    row_index = start_index + total_inserted + idx
                    args_list.append((
                        row_id, table_id, json.dumps(row_data), row_index, now, now, user_id, user_id
                    ))
                
                await self.db.executemany(insert_query, args_list)
                total_inserted += len(batch)
                
                logger.info(f"Inserted batch: {total_inserted}/{len(rows_data)} rows into table {table_id}")
            
            # Update table row count
            await self._update_table_row_count(table_id, user_id)
            
            logger.info(f"Bulk insert complete: {total_inserted} rows into table {table_id}")
            return total_inserted
            
        except Exception as e:
            logger.error(f"Failed to bulk insert rows into table {table_id}: {e}")
            raise
    
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
            # Delete row
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
        """Update the row count for a table"""
        try:
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
            'updated_by': row.get('updated_by')
        }


