-- Migration 023: Fix circular dependency in room_participants SELECT policy
-- 
-- Problem: The previous policy created a circular dependency by querying 
-- room_participants within its own SELECT policy. This caused RLS to fail
-- and return no records on initial load.
--
-- Solution: Simplify the policy. A user should be able to see:
-- 1. Their own participation records (always safe)
-- 2. Participation records for rooms where they are a member (using a join-free check)
-- 3. Any record if they are an admin

-- Drop the circular policy
DROP POLICY IF EXISTS room_participants_select_policy ON room_participants;

-- Create a robust policy without circular subqueries on the same table
CREATE POLICY room_participants_select_policy ON room_participants
    FOR SELECT USING (
        -- 1. User can ALWAYS see their own participation records
        user_id = current_setting('app.current_user_id', false)::varchar
        
        -- 2. User can see other participants in rooms they are members of
        -- We use a subquery on chat_rooms instead, which then checks membership
        OR room_id IN (
            SELECT r.room_id FROM chat_rooms r
            WHERE r.room_id = room_participants.room_id
            -- chat_rooms_select_policy will handle the membership check
        )
        
        -- 3. Admin can see everything
        OR current_setting('app.current_user_role', false) = 'admin'
    );


