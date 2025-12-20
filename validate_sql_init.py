#!/usr/bin/env python3
"""
Validate that all SQL migration scripts are included in the init scripts.
This ensures that a fresh database will have all features available.
"""

import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Tuple

def extract_tables(sql_content: str) -> Set[str]:
    """Extract table names from CREATE TABLE statements."""
    tables = set()
    # Match CREATE TABLE IF NOT EXISTS table_name or CREATE TABLE table_name
    pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)'
    for match in re.finditer(pattern, sql_content, re.IGNORECASE):
        tables.add(match.group(1).lower())
    return tables

def extract_types(sql_content: str) -> Set[str]:
    """Extract enum/type names from CREATE TYPE statements."""
    types = set()
    # Match CREATE TYPE type_name AS ENUM
    pattern = r'CREATE\s+TYPE\s+(\w+)\s+AS\s+ENUM'
    for match in re.finditer(pattern, sql_content, re.IGNORECASE):
        types.add(match.group(1).lower())
    return types

def extract_columns(sql_content: str, table_name: str) -> Set[str]:
    """Extract column names from a specific table."""
    columns = set()
    # Find the table definition
    pattern = rf'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{re.escape(table_name)}\s*\((.*?)\)'
    match = re.search(pattern, sql_content, re.IGNORECASE | re.DOTALL)
    if match:
        table_def = match.group(1)
        # Extract column names (simple pattern - may miss some edge cases)
        col_pattern = r'^\s*(\w+)\s+[^,]+'
        for line in table_def.split('\n'):
            col_match = re.match(col_pattern, line.strip(), re.IGNORECASE)
            if col_match:
                columns.add(col_match.group(1).lower())
    return columns

def extract_alter_table_add_column(sql_content: str) -> Dict[str, Set[str]]:
    """Extract ALTER TABLE ADD COLUMN statements."""
    alterations = defaultdict(set)
    # Match ALTER TABLE table_name ADD COLUMN column_name
    pattern = r'ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)'
    for match in re.finditer(pattern, sql_content, re.IGNORECASE):
        table = match.group(1).lower()
        column = match.group(2).lower()
        alterations[table].add(column)
    return alterations

def extract_indexes(sql_content: str) -> Set[str]:
    """Extract index names from CREATE INDEX statements."""
    indexes = set()
    # Match CREATE INDEX IF NOT EXISTS index_name or CREATE INDEX index_name
    pattern = r'CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)'
    for match in re.finditer(pattern, sql_content, re.IGNORECASE):
        indexes.add(match.group(1).lower())
    return indexes

def read_sql_file(filepath: Path) -> str:
    """Read SQL file content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def main():
    base_dir = Path("/opt/bastion")
    
    # Find all SQL files
    backend_migrations = list((base_dir / "backend" / "sql" / "migrations").glob("*.sql"))
    backend_other = [base_dir / "backend" / "sql" / "12_music_tables.sql"]  # Additional SQL file
    data_service_migrations = list((base_dir / "data-service" / "sql" / "migrations").glob("*.sql"))
    
    # Read init scripts
    backend_init = base_dir / "backend" / "sql" / "01_init.sql"
    data_service_init = base_dir / "data-service" / "sql" / "01_init.sql"
    
    print("=" * 80)
    print("SQL INIT VALIDATION REPORT")
    print("=" * 80)
    print()
    
    # Validate backend migrations
    print("BACKEND DATABASE VALIDATION")
    print("-" * 80)
    
    backend_init_content = read_sql_file(backend_init)
    backend_init_tables = extract_tables(backend_init_content)
    backend_init_types = extract_types(backend_init_content)
    backend_init_alterations = extract_alter_table_add_column(backend_init_content)
    backend_init_indexes = extract_indexes(backend_init_content)
    
    print(f"Init script contains {len(backend_init_tables)} tables, {len(backend_init_types)} types")
    print()
    
    missing_items = {
        'tables': [],
        'types': [],
        'columns': [],
        'indexes': []
    }
    
    # Check all migration files
    all_migration_files = backend_migrations + backend_other
    for migration_file in sorted(all_migration_files):
        if not migration_file.exists():
            continue
            
        print(f"Checking: {migration_file.name}")
        migration_content = read_sql_file(migration_file)
        
        # Check tables
        migration_tables = extract_tables(migration_content)
        for table in migration_tables:
            if table not in backend_init_tables:
                missing_items['tables'].append((migration_file.name, table))
        
        # Check types
        migration_types = extract_types(migration_content)
        for type_name in migration_types:
            if type_name not in backend_init_types:
                missing_items['types'].append((migration_file.name, type_name))
        
        # Check ALTER TABLE ADD COLUMN
        migration_alterations = extract_alter_table_add_column(migration_content)
        for table, columns in migration_alterations.items():
            # Check if table exists
            if table in backend_init_tables:
                # Check if columns exist in init (either in CREATE TABLE or ALTER TABLE)
                init_alterations = backend_init_alterations.get(table, set())
                # Also check if columns are in the CREATE TABLE definition
                init_columns = extract_columns(backend_init_content, table)
                for column in columns:
                    if column not in init_columns and column not in init_alterations:
                        missing_items['columns'].append((migration_file.name, table, column))
            else:
                # Table doesn't exist, so columns can't exist
                for column in columns:
                    missing_items['columns'].append((migration_file.name, table, column))
        
        # Check indexes (less critical, but good to verify)
        migration_indexes = extract_indexes(migration_content)
        for index in migration_indexes:
            if index not in backend_init_indexes:
                # Index names might differ, so this is just informational
                pass
    
    # Report missing items
    print()
    print("MISSING ITEMS IN BACKEND INIT SCRIPT:")
    print("-" * 80)
    
    if missing_items['tables']:
        print("\n❌ Missing Tables:")
        for file, table in missing_items['tables']:
            print(f"  - {table} (from {file})")
    else:
        print("\n✅ All tables are present in init script")
    
    if missing_items['types']:
        print("\n❌ Missing Types/Enums:")
        for file, type_name in missing_items['types']:
            print(f"  - {type_name} (from {file})")
    else:
        print("\n✅ All types/enums are present in init script")
    
    if missing_items['columns']:
        print("\n❌ Missing Columns:")
        for file, table, column in missing_items['columns']:
            print(f"  - {table}.{column} (from {file})")
    else:
        print("\n✅ All columns are present in init script")
    
    # Validate data-service migrations
    print()
    print()
    print("DATA-SERVICE DATABASE VALIDATION")
    print("-" * 80)
    
    data_service_init_content = read_sql_file(data_service_init)
    data_service_init_tables = extract_tables(data_service_init_content)
    data_service_init_alterations = extract_alter_table_add_column(data_service_init_content)
    
    print(f"Init script contains {len(data_service_init_tables)} tables")
    print()
    
    data_service_missing = {
        'tables': [],
        'columns': []
    }
    
    for migration_file in sorted(data_service_migrations):
        if not migration_file.exists():
            continue
            
        print(f"Checking: {migration_file.name}")
        migration_content = read_sql_file(migration_file)
        
        # Check tables
        migration_tables = extract_tables(migration_content)
        for table in migration_tables:
            if table not in data_service_init_tables:
                data_service_missing['tables'].append((migration_file.name, table))
        
        # Check ALTER TABLE ADD COLUMN
        migration_alterations = extract_alter_table_add_column(migration_content)
        for table, columns in migration_alterations.items():
            if table in data_service_init_tables:
                init_alterations = data_service_init_alterations.get(table, set())
                init_columns = extract_columns(data_service_init_content, table)
                for column in columns:
                    if column not in init_columns and column not in init_alterations:
                        data_service_missing['columns'].append((migration_file.name, table, column))
            else:
                for column in columns:
                    data_service_missing['columns'].append((migration_file.name, table, column))
    
    # Report data-service missing items
    print()
    print("MISSING ITEMS IN DATA-SERVICE INIT SCRIPT:")
    print("-" * 80)
    
    if data_service_missing['tables']:
        print("\n❌ Missing Tables:")
        for file, table in data_service_missing['tables']:
            print(f"  - {table} (from {file})")
    else:
        print("\n✅ All tables are present in init script")
    
    if data_service_missing['columns']:
        print("\n❌ Missing Columns:")
        for file, table, column in data_service_missing['columns']:
            print(f"  - {table}.{column} (from {file})")
    else:
        print("\n✅ All columns are present in init script")
    
    # Final summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    backend_issues = (len(missing_items['tables']) + 
                     len(missing_items['types']) + 
                     len(missing_items['columns']))
    data_service_issues = (len(data_service_missing['tables']) + 
                           len(data_service_missing['columns']))
    
    if backend_issues == 0 and data_service_issues == 0:
        print("\n✅ SUCCESS: All SQL migrations are included in init scripts!")
        print("   Your database will come up cleanly with all features ready.")
        return 0
    else:
        print(f"\n❌ ISSUES FOUND:")
        print(f"   Backend: {backend_issues} missing items")
        print(f"   Data-Service: {data_service_issues} missing items")
        print(f"\n   Please add the missing items to the init scripts above.")
        return 1

if __name__ == "__main__":
    exit(main())






