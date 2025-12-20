-- ========================================
-- FIX ROOM UPDATE RLS POLICY
-- Fix chat_rooms_update_policy to use false parameter
-- ========================================
-- This migration fixes the chat_rooms_update_policy to use
-- false instead of true for current_setting, which prevents
-- errors when the setting doesn't exist and ensures consistency.
--
-- Usage:
-- docker exec -i <postgres-container> psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/019_fix_room_update_rls_policy.sql
-- Or from within container:
-- psql -U postgres -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/019_fix_room_update_rls_policy.sql
-- ========================================

-- Drop the old policy
DROP POLICY IF EXISTS chat_rooms_update_policy ON chat_rooms;

-- Create the updated policy with false parameter
CREATE POLICY chat_rooms_update_policy ON chat_rooms
    FOR UPDATE USING (
        -- Allow if user is a participant in the room
        room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', false)::varchar
        )
        -- OR if user is admin
        OR current_setting('app.current_user_role', false) = 'admin'
    );



