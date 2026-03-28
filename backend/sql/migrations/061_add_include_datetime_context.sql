-- Add include_datetime_context flag for Agent Factory (current date/time in user timezone)
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS include_datetime_context BOOLEAN NOT NULL DEFAULT true;
