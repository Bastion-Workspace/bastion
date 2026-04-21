-- User-controlled lock to prevent editing/deletion of connections, playbooks, connectors, and agents.
-- Only the owner can set/clear the lock. When locked, edit and delete are blocked; for agents, pause/resume (is_active) remains allowed.

ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS is_locked BOOLEAN DEFAULT false;
ALTER TABLE custom_playbooks ADD COLUMN IF NOT EXISTS is_locked BOOLEAN DEFAULT false;
ALTER TABLE data_source_connectors ADD COLUMN IF NOT EXISTS is_locked BOOLEAN DEFAULT false;
ALTER TABLE external_connections ADD COLUMN IF NOT EXISTS is_locked BOOLEAN DEFAULT false;
