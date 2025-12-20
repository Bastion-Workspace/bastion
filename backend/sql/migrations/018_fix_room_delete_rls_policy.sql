-- ========================================
-- FIX ROOM DELETE RLS POLICY
-- Allow participants to delete rooms, not just creators
-- ========================================
-- This migration fixes the chat_rooms_delete_policy to allow
-- any participant to delete a room, matching the service behavior.
-- Previously only room creators could delete.
--
-- Usage:
-- docker exec -i <postgres-container> psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/018_fix_room_delete_rls_policy.sql
-- Or from within container:
-- psql -U postgres -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/018_fix_room_delete_rls_policy.sql
-- ========================================

-- Drop the old restrictive policy
DROP POLICY IF EXISTS chat_rooms_delete_policy ON chat_rooms;

-- Create the updated policy: allow room creator OR any participant to delete
CREATE POLICY chat_rooms_delete_policy ON chat_rooms
    FOR DELETE USING (
        -- Allow room creator to delete
        created_by = current_setting('app.current_user_id', false)::varchar
        -- OR any participant can delete
        OR room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', false)::varchar
        )
        -- OR admin can delete
        OR current_setting('app.current_user_role', false) = 'admin'
    );



