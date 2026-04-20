-- Allow non-admin participants of a federated room to read the linked federation peer row
-- (for peer status in room list / federation UI). Admin policy remains.

CREATE POLICY federation_peers_select_for_federated_room_member ON federation_peers
    FOR SELECT USING (
        EXISTS (
            SELECT 1
            FROM chat_rooms r
            JOIN room_participants rp ON rp.room_id = r.room_id
            WHERE r.room_type = 'federated'
              AND (r.federation_metadata->>'peer_id')::uuid = federation_peers.peer_id
              AND rp.user_id = current_setting('app.current_user_id', true)::varchar
        )
    );
