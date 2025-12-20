-- ========================================
-- ADD MUSIC RLS POLICIES
-- Enable Row-Level Security for music tables
-- ========================================
-- This migration adds RLS policies to music_service_configs,
-- music_cache, and music_cache_metadata tables to ensure
-- users can only access their own music data.
--
-- Usage:
-- docker exec -i <postgres-container> psql -U bastion_user -d bastion_knowledge_base < backend/sql/migrations/015_add_music_rls_policies.sql
-- Or from within container:
-- psql -U bastion_user -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/015_add_music_rls_policies.sql
-- ========================================

-- Enable RLS on music tables
ALTER TABLE music_service_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE music_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE music_cache_metadata ENABLE ROW LEVEL SECURITY;

-- ========================================
-- MUSIC_SERVICE_CONFIGS POLICIES
-- ========================================

-- SELECT: users see only their own configurations
CREATE POLICY music_configs_select_policy ON music_service_configs
FOR SELECT
USING (user_id = current_setting('app.current_user_id', true));

-- INSERT: users insert only their own configurations
CREATE POLICY music_configs_insert_policy ON music_service_configs
FOR INSERT
WITH CHECK (user_id = current_setting('app.current_user_id', true));

-- UPDATE: users update only their own configurations
CREATE POLICY music_configs_update_policy ON music_service_configs
FOR UPDATE
USING (user_id = current_setting('app.current_user_id', true))
WITH CHECK (user_id = current_setting('app.current_user_id', true));

-- DELETE: users delete only their own configurations
CREATE POLICY music_configs_delete_policy ON music_service_configs
FOR DELETE
USING (user_id = current_setting('app.current_user_id', true));

-- ========================================
-- MUSIC_CACHE POLICIES
-- ========================================

-- SELECT: users see only their own cached data
CREATE POLICY music_cache_select_policy ON music_cache
FOR SELECT
USING (user_id = current_setting('app.current_user_id', true));

-- INSERT: users insert only their own cached data
CREATE POLICY music_cache_insert_policy ON music_cache
FOR INSERT
WITH CHECK (user_id = current_setting('app.current_user_id', true));

-- UPDATE: users update only their own cached data
CREATE POLICY music_cache_update_policy ON music_cache
FOR UPDATE
USING (user_id = current_setting('app.current_user_id', true))
WITH CHECK (user_id = current_setting('app.current_user_id', true));

-- DELETE: users delete only their own cached data
CREATE POLICY music_cache_delete_policy ON music_cache
FOR DELETE
USING (user_id = current_setting('app.current_user_id', true));

-- ========================================
-- MUSIC_CACHE_METADATA POLICIES
-- ========================================

-- SELECT: users see only their own metadata
CREATE POLICY music_metadata_select_policy ON music_cache_metadata
FOR SELECT
USING (user_id = current_setting('app.current_user_id', true));

-- INSERT: users insert only their own metadata
CREATE POLICY music_metadata_insert_policy ON music_cache_metadata
FOR INSERT
WITH CHECK (user_id = current_setting('app.current_user_id', true));

-- UPDATE: users update only their own metadata
CREATE POLICY music_metadata_update_policy ON music_cache_metadata
FOR UPDATE
USING (user_id = current_setting('app.current_user_id', true))
WITH CHECK (user_id = current_setting('app.current_user_id', true));

-- DELETE: users delete only their own metadata
CREATE POLICY music_metadata_delete_policy ON music_cache_metadata
FOR DELETE
USING (user_id = current_setting('app.current_user_id', true));



