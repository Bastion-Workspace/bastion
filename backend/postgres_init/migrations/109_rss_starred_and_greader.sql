-- RSS starred flag and Google Reader API integer item IDs

ALTER TABLE rss_articles ADD COLUMN IF NOT EXISTS is_starred BOOLEAN DEFAULT FALSE;
UPDATE rss_articles SET is_starred = FALSE WHERE is_starred IS NULL;
ALTER TABLE rss_articles ALTER COLUMN is_starred SET DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_rss_articles_is_starred ON rss_articles(is_starred);

DO $mig$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'rss_articles'
      AND column_name = 'greader_id'
  ) THEN
    ALTER TABLE rss_articles ADD COLUMN greader_id BIGSERIAL;
  END IF;
END
$mig$;

CREATE UNIQUE INDEX IF NOT EXISTS idx_rss_articles_greader_id ON rss_articles(greader_id);

DO $grant_seq$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_sequences WHERE schemaname = 'public' AND sequencename = 'rss_articles_greader_id_seq'
  ) THEN
    EXECUTE 'GRANT USAGE, SELECT ON SEQUENCE rss_articles_greader_id_seq TO bastion_user';
  END IF;
END
$grant_seq$;
