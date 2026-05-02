-- Runs after 01_init.sql on fresh PostgreSQL containers (docker-entrypoint-initdb.d).
-- Closes greenfield gaps: AI convo attachments, event watches, edit proposals, browser sessions,
-- chunk page columns + metadata FTS, agent line watches + workspace (+ grants/RLS in 156),
-- legacy admin TTS prefs (160), voice provider enum parity with brownfield migrations (161),
-- Zettelkasten user settings table (162; idempotent with 01_init.sql merge).
-- Code workspace chunk index for semantic search (163; idempotent with 01_init.sql merge).
-- Per-user shell command policy (164; idempotent with 01_init.sql merge).
-- Code workspace RLS (165; idempotent with 01_init.sql merge).
-- device_tokens capabilities_policy (167; idempotent for DBs created before column merge).
-- Music cover art disk cache index (168; idempotent with 01_init.sql merge).
-- Widen cover_art_id to TEXT (169; idempotent).

\c bastion_knowledge_base

\ir migrations/168_music_cover_cache_index.sql
\ir migrations/169_music_cover_art_id_text.sql
\ir migrations/167_device_capabilities_policy.sql
\ir migrations/162_add_zettelkasten_settings.sql
\ir migrations/163_code_chunks.sql
\ir migrations/164_user_shell_policy.sql
\ir migrations/165_code_workspace_rls.sql
\ir migrations/039_add_message_attachments.sql
\ir migrations/048_add_agent_event_watches.sql
\ir migrations/054_add_document_edit_proposals.sql
\ir migrations/071_add_browser_session_states.sql
\ir migrations/083_add_chunk_page_columns.sql
\ir migrations/156_greenfield_agent_line_watches_workspace.sql
\ir migrations/157_remove_rss_and_deep_research_builtin_playbooks.sql
\ir migrations/160_admin_tts_empty_means_browser.sql
\ir migrations/161_add_openrouter_voice_provider.sql
