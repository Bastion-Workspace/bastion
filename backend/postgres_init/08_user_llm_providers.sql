-- Runs after 01_init.sql on fresh PostgreSQL containers (docker-entrypoint-initdb.d).
-- Per-user LLM credentials and enabled model rows (Settings → Models).
-- Keep in sync with migrations/055_add_user_llm_providers.sql and 080_add_groq_provider_type.sql.

\c bastion_knowledge_base

\ir migrations/055_add_user_llm_providers.sql
\ir migrations/080_add_groq_provider_type.sql
