-- Fix default playbook step definition: use step_type (not type) so orchestrator and UI treat it as llm_agent.
UPDATE custom_playbooks
SET definition = jsonb_set(
    definition,
    '{steps,0}',
    (
        (definition->'steps'->0)
        - 'type'
        || jsonb_build_object('step_type', coalesce(definition->'steps'->0->>'type', 'llm_agent'))
    )
)
WHERE id = '00000000-0001-4000-8000-000000000001'::uuid
  AND definition->'steps'->0 ? 'type'
  AND NOT (definition->'steps'->0 ? 'step_type');
