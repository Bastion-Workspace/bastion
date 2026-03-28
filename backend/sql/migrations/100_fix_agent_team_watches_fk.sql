-- Fix agent_team_watches.team_id to reference agent_teams(id) instead of teams(team_id).
-- Run: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/100_fix_agent_team_watches_fk.sql

ALTER TABLE agent_team_watches
    DROP CONSTRAINT IF EXISTS agent_team_watches_team_id_fkey;

ALTER TABLE agent_team_watches
    ADD CONSTRAINT agent_team_watches_team_id_fkey
    FOREIGN KEY (team_id) REFERENCES agent_teams(id) ON DELETE CASCADE;
