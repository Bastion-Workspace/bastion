-- Add auto_routable flag so custom agents can opt in to skill discovery auto-routing.
-- Default false: agents are only reachable via @handle or sticky routing unless opted in.
ALTER TABLE agent_profiles
    ADD COLUMN IF NOT EXISTS auto_routable BOOLEAN DEFAULT false;
