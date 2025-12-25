# Data Workspace Database Migrations

## Purpose

These migration scripts are for **updating existing databases** that were created before certain features were added.

**For new databases:** All features are included in `../01_init.sql` - no migrations needed!

## When to Use Migrations

Use these migrations if you have an **existing database** that needs:
- User tracking columns (002)
- Row-Level Security policies (003)
- Formula support (004)

## Migration Scripts

### 002_add_user_tracking.sql
- Adds `created_by` and `updated_by` columns to all tables
- Backfills existing records with appropriate user IDs
- Creates indexes for efficient queries

### 003_enable_rls.sql
- Creates `get_accessible_workspace_ids()` helper function
- Enables Row-Level Security on all tables
- Creates RLS policies for data isolation
- Adds optimization indexes

### 004_add_formula_support.sql
- Adds `formula_data` JSONB column to `custom_data_rows`
- Creates GIN index for efficient formula queries
- Adds column documentation

## Running Migrations

Migrations are **idempotent** (safe to run multiple times) using `IF NOT EXISTS` clauses.

To run a migration manually:

```bash
docker exec bastion-postgres-data psql -U data_user -d data_workspace -f /path/to/migration.sql
```

Or connect directly:

```bash
docker exec -i bastion-postgres-data psql -U data_user -d data_workspace < migrations/002_add_user_tracking.sql
```

## Migration Status

- ✅ **002**: User tracking columns are in `01_init.sql` (columns already exist)
- ✅ **003**: RLS policies are in `01_init.sql` (full RLS setup)
- ✅ **004**: Formula support is in `01_init.sql` (column and index included)

**Note:** These migrations are kept for historical reference and updating existing databases. New databases created from `01_init.sql` already include all features.

