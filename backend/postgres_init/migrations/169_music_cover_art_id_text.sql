-- Navidrome / OpenSubsonic cover IDs can exceed VARCHAR(255); widening avoids INSERT errors -> HTTP 500 on cover proxy.

ALTER TABLE music_cover_cache_index
  ALTER COLUMN cover_art_id TYPE TEXT;

ALTER TABLE music_cache
  ALTER COLUMN cover_art_id TYPE TEXT;
