-- Migration 025: Add UPDATE RLS policy for room_participants
-- 
-- Problem: Users were unable to mark messages as read because there was no 
-- UPDATE policy for the room_participants table. The update to last_read_at
-- would fail silently or be blocked, causing unread counts to persist after refresh.
--
-- Solution: Add an UPDATE policy allowing users to update their own 
-- participation records (like last_read_at and notification_settings).

-- Ensure RLS is enabled (it should be, but let's be sure)
ALTER TABLE room_participants ENABLE ROW LEVEL SECURITY;

-- Drop existing update policy if it somehow exists
DROP POLICY IF EXISTS room_participants_update_policy ON room_participants;

-- Create the update policy: users can update their own records
CREATE POLICY room_participants_update_policy ON room_participants
    FOR UPDATE USING (
        -- User can only update their own participation records
        user_id = current_setting('app.current_user_id', false)::varchar
        OR current_setting('app.current_user_role', false) = 'admin'
    )
    WITH CHECK (
        -- Ensure they don't try to change the record to belong to someone else
        user_id = current_setting('app.current_user_id', false)::varchar
        OR current_setting('app.current_user_role', false) = 'admin'
    );


