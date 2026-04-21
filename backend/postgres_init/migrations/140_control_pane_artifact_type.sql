-- Control panes: optional artifact embeds (saved_artifacts) in addition to connector-backed panes.
-- Makes connector_id nullable for artifact panes.

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'user_control_panes' AND column_name = 'pane_type'
  ) THEN
    ALTER TABLE user_control_panes ADD COLUMN pane_type VARCHAR(20) NOT NULL DEFAULT 'connector';
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'user_control_panes' AND column_name = 'artifact_id'
  ) THEN
    ALTER TABLE user_control_panes
      ADD COLUMN artifact_id UUID REFERENCES saved_artifacts(id) ON DELETE SET NULL;
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'user_control_panes' AND column_name = 'artifact_popover_width'
  ) THEN
    ALTER TABLE user_control_panes ADD COLUMN artifact_popover_width INTEGER;
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'user_control_panes' AND column_name = 'artifact_popover_height'
  ) THEN
    ALTER TABLE user_control_panes ADD COLUMN artifact_popover_height INTEGER;
  END IF;
END $$;

-- Allow connector-backed panes to keep FK; artifact panes use NULL connector_id.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'user_control_panes'
      AND column_name = 'connector_id' AND is_nullable = 'NO'
  ) THEN
    ALTER TABLE user_control_panes ALTER COLUMN connector_id DROP NOT NULL;
  END IF;
END $$;
