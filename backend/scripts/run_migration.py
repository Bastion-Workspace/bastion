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
    else:
        candidate = migrations_dir / f"{migration_name}.sql"
        if candidate.exists():
            sql_file = candidate.name
        else:
            logger.error(f"Unknown migration: {migration_name}")
            logger.info("Available: messaging, 039, message_attachments, or <nnn_name>.sql")
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

