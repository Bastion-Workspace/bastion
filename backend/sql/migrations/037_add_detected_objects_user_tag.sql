-- Add user_tag to detected_objects for refined labels (e.g. "Car" -> "BMW i3")
-- Rejected objects are already supported; filter them in API when include_rejected=false.

ALTER TABLE detected_objects
ADD COLUMN IF NOT EXISTS user_tag TEXT;

COMMENT ON COLUMN detected_objects.user_tag IS 'User-refined label for search/display (e.g. YOLO class_name "car" -> "BMW i3")';
