-- Stable device id for code workspaces (daemon registration id)

ALTER TABLE code_workspaces
    ADD COLUMN IF NOT EXISTS device_id TEXT;

UPDATE code_workspaces
SET device_id = device_name
WHERE device_id IS NULL AND device_name IS NOT NULL AND btrim(device_name) <> '';

CREATE INDEX IF NOT EXISTS idx_code_workspaces_user_device
    ON code_workspaces (user_id, device_id);
