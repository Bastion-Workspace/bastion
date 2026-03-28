-- Shared built-in playbook for RSS Manager (tool pack rss only).
-- Per-user agent profile handle rss-manager (second builtin; independent of Bastion Assistant).

INSERT INTO custom_playbooks (
    id,
    user_id,
    name,
    description,
    version,
    definition,
    triggers,
    is_template,
    is_locked,
    is_builtin,
    category,
    tags,
    required_connectors
) VALUES (
    '00000000-0001-4000-8000-000000000002'::uuid,
    NULL,
    'RSS Manager Playbook',
    'Manage RSS feeds: list, add, refresh, search, delete, mark read, unread counts, pause/resume polling',
    '1.0',
    '{"steps": [{"step_type": "llm_agent", "name": "rss_manager", "prompt_template": "{query}", "tool_packs": ["rss"], "auto_discover_skills": false, "max_auto_skills": 0, "max_iterations": 20, "available_tools": []}]}'::jsonb,
    '[]'::jsonb,
    true,
    true,
    true,
    'rss',
    '{}',
    '{}'
) ON CONFLICT (id) DO NOTHING;

INSERT INTO agent_profiles (
    user_id,
    name,
    handle,
    description,
    is_active,
    is_locked,
    is_builtin,
    default_playbook_id,
    chat_history_enabled,
    chat_history_lookback,
    persona_mode,
    include_user_facts,
    include_datetime_context,
    include_user_context
)
SELECT
    u.user_id,
    'RSS Manager',
    'rss-manager',
    'Manage monitored RSS feeds: add/remove feeds, refresh, search articles, mark read, unread counts, enable or disable polling',
    true,
    true,
    true,
    '00000000-0001-4000-8000-000000000002'::uuid,
    true,
    10,
    'default',
    true,
    true,
    true
FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM agent_profiles p
    WHERE p.user_id = u.user_id AND p.handle = 'rss-manager'
);
