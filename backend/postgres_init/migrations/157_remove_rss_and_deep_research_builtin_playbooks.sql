-- Remove built-in RSS Manager and Deep Research template playbooks and their built-in agent profiles.
-- Greenfield: 01_init.sql no longer inserts these. This migration is idempotent for existing databases.

DELETE FROM agent_profiles
WHERE is_builtin = true
  AND handle IN ('rss-manager', 'devops-advisor');

DELETE FROM custom_playbooks
WHERE id IN (
    '00000000-0001-4000-8000-000000000002'::uuid,
    '00000000-0001-4000-8000-000000000003'::uuid
)
  AND user_id IS NULL
  AND is_builtin = true;
