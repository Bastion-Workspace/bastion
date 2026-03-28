"""
Run database migrations from within the backend container.

Usage:
    python scripts/run_migration.py --migration messaging
    python scripts/run_migration.py --migration 039
    python scripts/run_migration.py --check
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_messaging_tables_exist(conn):
    """Check which messaging tables already exist"""
    tables = [
        'chat_rooms',
        'room_participants',
        'chat_messages',
        'message_reactions',
        'room_encryption_keys',
        'user_presence'
    ]
    
    existing_tables = []
    missing_tables = []
    
    for table in tables:
        result = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
            table
        )
        if result:
            existing_tables.append(table)
        else:
            missing_tables.append(table)
    
    return existing_tables, missing_tables


async def run_messaging_migration(conn):
    """Run the messaging system migration"""
    logger.info("Starting messaging system migration...")
    
    # Read the migration SQL file
    migration_path = Path(__file__).parent.parent / 'sql' / 'migrations' / '005_add_messaging_system.sql'
    
    if not migration_path.exists():
        logger.error(f"Migration file not found: {migration_path}")
        return False
    
    with open(migration_path, 'r') as f:
        migration_sql = f.read()
    
    try:
        # Execute the migration
        await conn.execute(migration_sql)
        logger.info("Migration executed successfully")
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


async def check_migration_status():
    """Check the status of messaging tables"""
    logger.info("Checking messaging system status...")
    
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        
        existing, missing = await check_messaging_tables_exist(conn)
        
        logger.info("=" * 50)
        logger.info("MESSAGING SYSTEM STATUS")
        logger.info("=" * 50)
        
        if existing:
            logger.info(f"Existing tables ({len(existing)}):")
            for table in existing:
                logger.info(f"   - {table}")
        
        if missing:
            logger.info(f"Missing tables ({len(missing)}):")
            for table in missing:
                logger.info(f"   - {table}")
        else:
            logger.info("All messaging tables exist")
        
        logger.info("=" * 50)
        
        await conn.close()
        
        return len(missing) == 0
        
    except Exception as e:
        logger.error(f"Failed to check migration status: {e}")
        return False


async def run_sql_file_migration(conn, migration_filename: str) -> bool:
    """Run a migration from backend/sql/migrations/<filename>.sql"""
    migrations_dir = Path(__file__).parent.parent / "sql" / "migrations"
    migration_path = migrations_dir / migration_filename
    if not migration_path.exists():
        logger.error(f"Migration file not found: {migration_path}")
        return False
    with open(migration_path, "r") as f:
        migration_sql = f.read()
    try:
        await conn.execute(migration_sql)
        logger.info(f"Migration {migration_filename} executed successfully")
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


async def run_migration(migration_name: str):
    """Run a specific migration"""
    migrations_dir = Path(__file__).parent.parent / "sql" / "migrations"
    sql_file = None
    if migration_name == "messaging":
        sql_file = "005_add_messaging_system.sql"
    elif migration_name in ("039", "message_attachments"):
        sql_file = "039_add_message_attachments.sql"
    elif migration_name == "042":
        sql_file = "042_add_agent_factory_tables.sql"
    elif migration_name == "043":
        sql_file = "043_add_agent_schedules.sql"
    elif migration_name == "044":
        sql_file = "044_add_agent_plugin_configs.sql"
    elif migration_name == "047":
        sql_file = "047_add_agent_team_watches.sql"
    elif migration_name == "048":
        sql_file = "048_add_agent_event_watches.sql"
    elif migration_name == "050":
        sql_file = "050_add_agent_service_bindings.sql"
    elif migration_name == "051":
        sql_file = "051_add_chat_history_config.sql"
    elif migration_name == "052":
        sql_file = "052_add_persona_enabled.sql"
    elif migration_name == "053":
        sql_file = "053_add_auto_routable.sql"
    elif migration_name == "054":
        sql_file = "054_add_document_edit_proposals.sql"
    elif migration_name == "055":
        sql_file = "055_add_user_llm_providers.sql"
    elif migration_name == "056":
        sql_file = "056_playbook_delete_set_null.sql"
    elif migration_name in ("058", "058_add_user_lock"):
        sql_file = "058_add_user_lock.sql"
    elif migration_name == "060":
        sql_file = "060_add_include_user_context.sql"
    elif migration_name == "061":
        sql_file = "061_add_include_datetime_context.sql"
    elif migration_name == "062":
        sql_file = "062_add_execution_steps_and_playbook_versions.sql"
    elif migration_name == "063":
        sql_file = "063_add_tool_call_trace_to_execution_steps.sql"
    elif migration_name == "064":
        sql_file = "064_add_user_facts.sql"
    elif migration_name == "065":
        sql_file = "065_add_include_user_facts.sql"
    elif migration_name == "066":
        sql_file = "066_enhance_user_facts.sql"
    elif migration_name == "067":
        sql_file = "067_add_episodic_memory_and_fact_history.sql"
    elif migration_name == "068":
        sql_file = "068_add_agent_skills.sql"
    elif migration_name == "069":
        sql_file = "069_add_document_versions.sql"
    elif migration_name == "072":
        sql_file = "072_add_user_control_panes.sql"
    elif migration_name == "074":
        sql_file = "074_add_control_pane_refresh_interval.sql"
    elif migration_name == "076":
        sql_file = "076_builtin_agent_profiles.sql"
    elif migration_name == "077":
        sql_file = "077_default_playbook_step_type.sql"
    elif migration_name in ("078", "078_add_device_tokens", "device_tokens"):
        sql_file = "078_add_device_tokens.sql"
    elif migration_name in ("080", "080_add_groq_provider_type"):
        sql_file = "080_add_groq_provider_type.sql"
    elif migration_name in ("081", "081_add_agent_profile_model_source"):
        sql_file = "081_add_agent_profile_model_source.sql"
    elif migration_name in ("082", "082_add_document_chunks_fulltext", "document_chunks"):
        sql_file = "082_add_document_chunks_fulltext.sql"
    elif migration_name in ("082_data_workspace", "082_add_data_workspace_config", "data_workspace_config"):
        sql_file = "082_add_data_workspace_config.sql"
    elif migration_name in ("083", "083_add_chunk_page_columns", "chunk_page_columns"):
        sql_file = "083_add_chunk_page_columns.sql"
    elif migration_name in ("083_budgets", "083_agent_budgets_and_execution_cost"):
        sql_file = "083_agent_budgets_and_execution_cost.sql"
    elif migration_name in ("084", "084_agent_approval_queue"):
        sql_file = "084_agent_approval_queue.sql"
    elif migration_name in ("085", "085_agent_memory"):
        sql_file = "085_agent_memory.sql"
    elif migration_name in ("097", "097_agent_profile_chat_visible"):
        sql_file = "097_agent_profile_chat_visible.sql"
    elif migration_name in ("098", "098_drop_agent_profile_icon"):
        sql_file = "098_drop_agent_profile_icon.sql"
    else:
        candidate = migrations_dir / f"{migration_name}.sql"
        if candidate.exists():
            sql_file = candidate.name
        else:
            logger.error(f"Unknown migration: {migration_name}")
            logger.info("Available: messaging, 039, 042, 043, 044, 047, 048, or <nnn_name>.sql (e.g. 069, 072, 074)")
            return False

    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        if migration_name == "messaging":
            existing, missing = await check_messaging_tables_exist(conn)
            if not missing:
                logger.info("All messaging tables already exist; no migration needed.")
                await conn.close()
                return True
            logger.info(f"Will create {len(missing)} missing tables...")
        success = await run_sql_file_migration(conn, sql_file)
        if success and migration_name == "messaging":
            existing_after, missing_after = await check_messaging_tables_exist(conn)
            if not missing_after:
                logger.info("Migration completed successfully")
                for t in missing:
                    logger.info(f"   - {t}")
            else:
                for t in missing_after:
                    logger.warning(f"   Still missing: {t}")
        await conn.close()
        return success
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Database migration runner")
    parser.add_argument(
        '--migration',
        type=str,
        help='Name of migration to run (e.g., messaging)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check migration status without running'
    )
    
    args = parser.parse_args()
    
    if args.check:
        # Check status only
        success = asyncio.run(check_migration_status())
        sys.exit(0 if success else 1)
    elif args.migration:
        # Run migration
        success = asyncio.run(run_migration(args.migration))
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

