-- Unlock built-in agent profiles so users can customize name, playbook assignment, persona, etc.
UPDATE agent_profiles SET is_locked = false WHERE is_builtin = true;
