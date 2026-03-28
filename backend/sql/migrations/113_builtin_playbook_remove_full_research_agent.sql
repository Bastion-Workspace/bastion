-- Replace escalate_to_research with direct Agent Factory tools on the default playbook.
-- Optional: seed a multi-step "deep research" template playbook (no FullResearchAgent).

UPDATE custom_playbooks
SET
    description = 'Built-in single-step ReAct agent with skill search and document/web research tools',
    version = '1.1',
    definition = $playbook$
{
  "steps": [
    {
      "type": "llm_agent",
      "name": "main",
      "prompt_template": "{query}",
      "available_tools": [
        "search_documents_tool",
        "search_web_tool",
        "enhance_query_tool",
        "search_conversation_cache_tool",
        "search_segments_across_documents_tool",
        "crawl_web_content_tool",
        "get_document_content_tool",
        "get_document_metadata_tool",
        "search_images_tool"
      ],
      "auto_discover_skills": true,
      "max_auto_skills": 5,
      "max_iterations": 15
    }
  ]
}
$playbook$::jsonb
WHERE id = '00000000-0001-4000-8000-000000000001'::uuid;

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
    '00000000-0001-4000-8000-000000000003'::uuid,
    NULL,
    'Deep research (template)',
    'Multi-step template: outline, tool-backed research, then synthesis. Copy or assign as needed.',
    '1.0',
    $deep$
{
  "steps": [
    {
      "type": "llm_task",
      "name": "research_outline",
      "output_key": "research_outline",
      "prompt_template": "The user asked: {query}\n\nProduce a short research outline: 3-6 focused sub-questions and note whether each is best answered from local documents, the web, or both. Be concise.",
      "auto_discover_skills": false,
      "max_auto_skills": 0
    },
    {
      "type": "llm_agent",
      "name": "research_execute",
      "output_key": "research_execute",
      "prompt_template": "Use the available tools to investigate the user request. Follow this outline:\n\n{research_outline.formatted}\n\nOriginal user request: {query}",
      "available_tools": [
        "search_documents_tool",
        "search_web_tool",
        "enhance_query_tool",
        "search_conversation_cache_tool",
        "search_segments_across_documents_tool",
        "crawl_web_content_tool",
        "get_document_content_tool",
        "get_document_metadata_tool",
        "search_images_tool"
      ],
      "auto_discover_skills": true,
      "max_auto_skills": 5,
      "max_iterations": 20
    },
    {
      "type": "llm_task",
      "name": "research_synthesize",
      "output_key": "research_synthesize",
      "prompt_template": "Original user request: {query}\n\nResearch outline:\n{research_outline.formatted}\n\nTool-backed findings and notes:\n{research_execute.formatted}\n\nWrite a clear, well-structured final answer for the user. Cite whether claims came from local docs or the web when relevant.",
      "auto_discover_skills": false,
      "max_auto_skills": 0
    }
  ]
}
$deep$::jsonb,
    '[]'::jsonb,
    true,
    true,
    true,
    'default',
    '{}',
    '{}'
) ON CONFLICT (id) DO NOTHING;
