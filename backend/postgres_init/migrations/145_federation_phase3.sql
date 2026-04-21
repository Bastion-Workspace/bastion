-- ========================================
-- FEDERATION PHASE 3 — user discovery opt-in
-- Idempotent: safe to run multiple times.
-- ========================================

ALTER TABLE users ADD COLUMN IF NOT EXISTS federation_discoverable BOOLEAN NOT NULL DEFAULT FALSE;
