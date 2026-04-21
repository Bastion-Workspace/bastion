-- Browser session states for persistent Playwright sessions (cookies, localStorage).
-- Used by granular browser automation tools; state is encrypted at rest.
CREATE TABLE IF NOT EXISTS browser_session_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    site_domain TEXT NOT NULL,
    encrypted_state_blob TEXT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    is_valid BOOLEAN NOT NULL DEFAULT TRUE,
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, site_domain)
);

CREATE INDEX IF NOT EXISTS idx_browser_session_states_user_domain
    ON browser_session_states(user_id, site_domain);
CREATE INDEX IF NOT EXISTS idx_browser_session_states_valid
    ON browser_session_states(is_valid) WHERE is_valid = TRUE;
