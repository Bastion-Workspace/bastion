"""
Run entertainment sync tables migration
Applies the entertainment_sync_config and entertainment_sync_items tables
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
import asyncpg

async def run_migration():
    """Run the entertainment sync tables migration"""
    # Extract just the entertainment sync section from 01_init.sql
    migration_sql = """
-- ========================================
-- ENTERTAINMENT SYNC TABLES
-- Sonarr/Radarr API integration for entertainment content
-- ========================================

-- Configuration for user's Radarr/Sonarr connections
CREATE TABLE IF NOT EXISTS entertainment_sync_config (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    source_type VARCHAR(50) NOT NULL, -- 'radarr' or 'sonarr'
    api_url VARCHAR(512) NOT NULL,
    api_key TEXT NOT NULL, -- Encrypted API key
    enabled BOOLEAN DEFAULT true,
    sync_frequency_minutes INTEGER DEFAULT 60, -- Default hourly
    last_sync_at TIMESTAMP WITH TIME ZONE,
    last_sync_status VARCHAR(50), -- 'success', 'failed', 'running'
    items_synced INTEGER DEFAULT 0,
    sync_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, source_type, api_url)
);

-- Track individual synced items for change detection
CREATE TABLE IF NOT EXISTS entertainment_sync_items (
    item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_id UUID NOT NULL REFERENCES entertainment_sync_config(config_id) ON DELETE CASCADE,
    external_id VARCHAR(255) NOT NULL, -- Radarr/Sonarr item ID
    external_type VARCHAR(50) NOT NULL, -- 'movie', 'series', 'episode'
    title VARCHAR(512) NOT NULL,
    tmdb_id INTEGER, -- TMDB reference
    tvdb_id INTEGER, -- TVDB reference
    season_number INTEGER, -- For episodes
    episode_number INTEGER, -- For episodes
    parent_series_id VARCHAR(255), -- Link episode to series
    metadata_hash VARCHAR(64), -- Hash of metadata to detect changes
    last_synced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    vector_document_id VARCHAR(255), -- Track generated pseudo-document ID
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(config_id, external_id, external_type)
);

-- Create indexes for entertainment sync tables
CREATE INDEX IF NOT EXISTS idx_sync_config_user ON entertainment_sync_config(user_id);
CREATE INDEX IF NOT EXISTS idx_sync_config_enabled ON entertainment_sync_config(enabled);
CREATE INDEX IF NOT EXISTS idx_sync_config_source_type ON entertainment_sync_config(source_type);
CREATE INDEX IF NOT EXISTS idx_sync_items_config ON entertainment_sync_items(config_id);
CREATE INDEX IF NOT EXISTS idx_sync_items_external ON entertainment_sync_items(external_id);
CREATE INDEX IF NOT EXISTS idx_sync_items_type ON entertainment_sync_items(external_type);
CREATE INDEX IF NOT EXISTS idx_sync_items_vector_doc ON entertainment_sync_items(vector_document_id);
CREATE INDEX IF NOT EXISTS idx_sync_items_parent_series ON entertainment_sync_items(parent_series_id);

-- Add comments for entertainment sync tables
COMMENT ON TABLE entertainment_sync_config IS 'User configurations for Radarr/Sonarr API sync';
COMMENT ON TABLE entertainment_sync_items IS 'Tracked items synced from Radarr/Sonarr for change detection';
"""
    
    try:
        print("🔌 Connecting to database...")
        conn = await asyncpg.connect(settings.DATABASE_URL)
        
        print("🚀 Running entertainment sync migration...")
        await conn.execute(migration_sql)
        
        print("✅ Migration completed successfully!")
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('entertainment_sync_config', 'entertainment_sync_items')
            ORDER BY table_name
        """)
        
        if tables:
            print(f"✅ Created {len(tables)} tables:")
            for table in tables:
                print(f"   - {table['table_name']}")
        else:
            print("⚠️ No tables found - migration may have failed")
        
        # Verify indexes
        indexes = await conn.fetch("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public' 
            AND indexname LIKE 'idx_sync_%'
            ORDER BY indexname
        """)
        
        if indexes:
            print(f"✅ Created {len(indexes)} indexes:")
            for idx in indexes:
                print(f"   - {idx['indexname']}")
        
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







