-- ========================================
-- Agent Factory UX Cleanup
-- Promote default_playbook_id, drop output_config, default_execution_mode, agent_skills
-- ========================================
-- Run: docker exec -i bastion-postgres psql -U bastion_user -d bastion_knowledge_base < backend/sql/migrations/049_agent_factory_ux_cleanup.sql
-- ========================================

-- Add default_playbook_id column to agent_profiles
ALTER TABLE agent_profiles
    ADD COLUMN IF NOT EXISTS default_playbook_id UUID REFERENCES custom_playbooks(id) ON DELETE SET NULL;

-- Migrate existing default_playbook_id from output_config JSONB
UPDATE agent_profiles
SET default_playbook_id = (output_config->>'default_playbook_id')::uuid
WHERE output_config IS NOT NULL
  AND output_config->>'default_playbook_id' IS NOT NULL
  AND (output_config->>'default_playbook_id') ~ '^[0-9a-fA-F-]{36}$';

-- Drop output_config and default_execution_mode
ALTER TABLE agent_profiles DROP COLUMN IF EXISTS output_config;
ALTER TABLE agent_profiles DROP COLUMN IF EXISTS default_execution_mode;

-- Drop unused agent_skills table
DROP TABLE IF EXISTS agent_skills;
