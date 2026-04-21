-- Runs after 06_learning_and_message_branching.sql on fresh PostgreSQL (docker-entrypoint-initdb.d).
-- Applies migration 130: bot users (users.is_bot, agent_profiles.bot_user_id), reply-to and edit columns on chat_messages.
-- Idempotent per-file (IF NOT EXISTS on columns and indexes in 130).
--
-- Each init file runs in a new psql session (default DB = POSTGRES_DB, usually "postgres").
\c bastion_knowledge_base

\ir migrations/130_messaging_improvements.sql
