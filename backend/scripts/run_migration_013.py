"""
Run migration 013: Multi-Source Media Support
Adds service_name and is_active columns to music_service_configs
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
import asyncpg

async def run_migration():
    """Run migration 013"""
    migration_path = Path(__file__).parent.parent / 'sql' / 'migrations' / '013_multi_source_media_support.sql'
    
    if not migration_path.exists():
        print(f"❌ Migration file not found: {migration_path}")
        return False
    
    print("📁 Reading migration file...")
    with open(migration_path, 'r') as f:
        migration_sql = f.read()
    
    try:
        print("🔌 Connecting to database...")
        conn = await asyncpg.connect(settings.DATABASE_URL)
        
        print("🚀 Running migration 013: Multi-Source Media Support...")
        await conn.execute(migration_sql)
        
        print("✅ Migration completed successfully!")
        
        # Verify columns were added
        columns = await conn.fetch("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'music_service_configs' 
            AND column_name IN ('service_name', 'is_active', 'service_type')
            ORDER BY column_name
        """)
        
        if columns:
            print(f"✅ Verified {len(columns)} columns:")
            for col in columns:
                print(f"   - {col['column_name']} ({col['data_type']})")
        else:
            print("⚠️ Columns not found - migration may have failed")
        
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

