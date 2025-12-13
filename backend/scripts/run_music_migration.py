"""
Run music tables migration
Applies migration 011 to add music service tables
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
import asyncpg

async def run_migration():
    """Run the music tables migration"""
    migration_path = Path(__file__).parent.parent / 'sql' / 'migrations' / '011_add_music_tables.sql'
    
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
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('music_service_configs', 'music_cache', 'music_cache_metadata')
            ORDER BY table_name
        """)
        
        if tables:
            print(f"✅ Created {len(tables)} tables:")
            for table in tables:
                print(f"   - {table['table_name']}")
        else:
            print("⚠️ No tables found - migration may have failed")
        
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

