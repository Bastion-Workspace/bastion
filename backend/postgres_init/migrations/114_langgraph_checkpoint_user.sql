-- Dedicated PostgreSQL role for LangGraph AsyncPostgresSaver (checkpoint tables only).
-- Idempotent. Requires privileges to CREATE ROLE (typically run as postgres superuser).
-- Example: psql -U postgres -d bastion_knowledge_base -f 114_langgraph_checkpoint_user.sql

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'langgraph_checkpoint_user') THEN
        CREATE USER langgraph_checkpoint_user WITH PASSWORD 'langgraph_checkpoint_secure_password';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE bastion_knowledge_base TO langgraph_checkpoint_user;
GRANT USAGE ON SCHEMA public TO langgraph_checkpoint_user;

GRANT SELECT, INSERT, UPDATE, DELETE ON checkpoints TO langgraph_checkpoint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON checkpoint_blobs TO langgraph_checkpoint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON checkpoint_writes TO langgraph_checkpoint_user;

ALTER ROLE langgraph_checkpoint_user BYPASSRLS;
