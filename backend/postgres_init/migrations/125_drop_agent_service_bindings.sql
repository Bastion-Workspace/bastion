-- Remove profile-level service bindings; account selection is per playbook step (tool_packs).
DROP TABLE IF EXISTS agent_service_bindings CASCADE;
