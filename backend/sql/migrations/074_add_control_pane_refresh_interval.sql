-- Add refresh_interval to user_control_panes (Control Pane V2: configurable polling).
-- Run on existing DB: docker exec -i bastion-backend python scripts/run_migration.py --migration 074
-- Or: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/074_add_control_pane_refresh_interval.sql

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'user_control_panes' AND column_name = 'refresh_interval'
  ) THEN
    ALTER TABLE user_control_panes ADD COLUMN refresh_interval INTEGER NOT NULL DEFAULT 0;
  END IF;
END $$;
