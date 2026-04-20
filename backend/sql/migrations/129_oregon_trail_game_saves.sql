-- Oregon Trail game saves table
CREATE TABLE IF NOT EXISTS oregon_trail_saves (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    game_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT true,
    final_score INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_oregon_trail_saves_user_id ON oregon_trail_saves(user_id);
CREATE INDEX IF NOT EXISTS idx_oregon_trail_saves_active ON oregon_trail_saves(user_id, is_active) WHERE is_active = true;

COMMENT ON TABLE oregon_trail_saves IS 'Persisted Oregon Trail game states';
