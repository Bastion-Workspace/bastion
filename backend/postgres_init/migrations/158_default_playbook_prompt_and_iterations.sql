-- Default Agent Playbook: richer step prompt (query + optional history + optional editor) and max_iterations 20.
-- Idempotent. Greenfield has the same in 01_init.sql; this updates existing DBs that already have ...000001.

UPDATE custom_playbooks
SET
    version = '1.2',
    description = 'Built-in single-step ReAct agent: general assistant with tools; prompt includes current message, recent conversation, and open editor when present',
    definition = $def$
{
  "steps": [
    {
      "type": "llm_agent",
      "name": "main",
      "prompt_template": "You are a helpful general assistant for this workspace. Use the available tools when they improve the answer: search the knowledge base, the web, prior conversation, and document content as needed. Be direct; add detail when the user asks for it.\n\n{{#history}}\nRecent conversation (for context; the latest user message is repeated below):\n{history}\n{{/history}}\n\nThe user's current message:\n{query}\n\n{{#editor}}\nThe user has a document open in the editor (they may say \"this file\", the selection, or a heading). Use it when it helps:\n{editor}\n{{/editor}}",
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
    }
  ]
}
$def$::jsonb
WHERE id = '00000000-0001-4000-8000-000000000001'::uuid;
