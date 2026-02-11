-- Preserve original YOLO/CLIP detection label when user corrects with user_tag.
-- Not vectorized; used for audit and display of "was detected as X, corrected to Y".

ALTER TABLE detected_objects
ADD COLUMN IF NOT EXISTS original_class_name TEXT;

COMMENT ON COLUMN detected_objects.original_class_name IS 'Original detection label (YOLO/CLIP) before user correction; not vectorized';
