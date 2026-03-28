-- User-designed Home dashboards (first-class rows; layout stored as JSONB)

CREATE TABLE IF NOT EXISTS user_home_dashboards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT false,
    layout_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_home_dashboards_user
    ON user_home_dashboards(user_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_user_home_dashboard_default
    ON user_home_dashboards(user_id)
    WHERE is_default;
