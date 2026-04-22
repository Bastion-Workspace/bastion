-- Runs after 01_init.sql on fresh PostgreSQL containers (docker-entrypoint-initdb.d).
-- Closes greenfield gaps: AI convo attachments, event watches, edit proposals, browser sessions,
-- chunk page columns + metadata FTS, agent line watches + workspace (+ grants/RLS in 156).

\c bastion_knowledge_base

\ir migrations/039_add_message_attachments.sql
\ir migrations/048_add_agent_event_watches.sql
\ir migrations/054_add_document_edit_proposals.sql
\ir migrations/071_add_browser_session_states.sql
\ir migrations/083_add_chunk_page_columns.sql
\ir migrations/156_greenfield_agent_line_watches_workspace.sql
