-- ========================================
-- Allow deleting custom_playbooks when agent_execution_log references exist
-- ========================================
-- Run: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/056_playbook_delete_set_null.sql
-- ========================================

ALTER TABLE agent_execution_log
    DROP CONSTRAINT IF EXISTS agent_execution_log_playbook_id_fkey;

ALTER TABLE agent_execution_log
    ADD CONSTRAINT agent_execution_log_playbook_id_fkey
    FOREIGN KEY (playbook_id) REFERENCES custom_playbooks(id) ON DELETE SET NULL;
