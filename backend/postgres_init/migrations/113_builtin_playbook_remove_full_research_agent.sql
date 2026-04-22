-- Replace escalate_to_research with direct Agent Factory tools on the default playbook.
-- (Deep research template at ...000003 was removed from product defaults; do not re-seed it here.)

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
