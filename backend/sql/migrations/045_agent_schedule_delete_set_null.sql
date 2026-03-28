-- ========================================
-- Allow deleting agent_schedules when agent_execution_log references exist
-- ========================================
-- Run: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/045_agent_schedule_delete_set_null.sql
-- ========================================

ALTER TABLE agent_execution_log
    DROP CONSTRAINT IF EXISTS agent_execution_log_schedule_id_fkey;

ALTER TABLE agent_execution_log
    ADD CONSTRAINT agent_execution_log_schedule_id_fkey
    FOREIGN KEY (schedule_id) REFERENCES agent_schedules(id) ON DELETE SET NULL;
