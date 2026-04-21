-- Migration 022: Fix room_participants SELECT RLS policy to allow seeing other participants in the same room
-- 
-- Problem: The previous policy only allowed users to see their own participation records.
-- This prevented the messaging service from correctly identifying other participants
-- to set the room's display name, resulting in "Unnamed Room".
--
-- Solution: Allow users to see any participation record for a room they are a member of.

-- Drop the restrictive policy
DROP POLICY IF EXISTS room_participants_select_policy ON room_participants;

-- Create the correct policy: users can see participants in rooms they belong to
CREATE POLICY room_participants_select_policy ON room_participants
    FOR SELECT USING (
        -- User can see participants in any room they are also a member of
        room_id IN (
            SELECT rp.room_id FROM room_participants rp
            WHERE rp.user_id = current_setting('app.current_user_id', false)::varchar
        )
        -- OR if user is admin
        OR current_setting('app.current_user_role', false) = 'admin'
    );


