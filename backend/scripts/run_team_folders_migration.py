"""
Run team folders migration
Applies migration 008 to add team folder support
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
import asyncpg

async def run_migration():
    """Run the team folders migration"""
    migration_path = Path(__file__).parent.parent / 'sql' / 'migrations' / '008_add_team_folders.sql'
    
    if not migration_path.exists():
        print(f"❌ Migration file not found: {migration_path}")
        return False
    
    print("📁 Reading migration file...")
    with open(migration_path, 'r') as f:
        migration_sql = f.read()
    
    try:
        print("🔌 Connecting to database...")
        conn = await asyncpg.connect(settings.DATABASE_URL)
        
        print("🚀 Running migration...")
        await conn.execute(migration_sql)
        
        print("✅ Migration completed successfully!")
        
        # Verify team folders were created
        team_folders = await conn.fetch("""
            SELECT COUNT(*) as count FROM document_folders WHERE collection_type = 'team'
        """)
        count = team_folders[0]['count']
        print(f"📁 Found {count} team folders")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(run_migration())
    sys.exit(0 if success else 1)

