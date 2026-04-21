-- ========================================
-- Remove playbook_id from agent_schedules
-- ========================================
-- Schedules run the agent with its profile default playbook only.
-- Run: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/postgres_init/migrations/046_remove_schedule_playbook_id.sql
-- ========================================

ALTER TABLE agent_schedules DROP COLUMN IF EXISTS playbook_id;
