-- Agent line governance mode: hierarchical (default), committee, round_robin, consensus
ALTER TABLE agent_lines
  ADD COLUMN IF NOT EXISTS governance_mode VARCHAR(30) NOT NULL DEFAULT 'hierarchical'
    CHECK (governance_mode IN ('hierarchical', 'committee', 'round_robin', 'consensus'));
