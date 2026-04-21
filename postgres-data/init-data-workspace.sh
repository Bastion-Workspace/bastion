#!/bin/sh
# Docker entrypoint runs only top-level /docker-entrypoint-initdb.d/* files; subdirectories
# are ignored. This script applies the bundled data-workspace schema with psql -v ON_ERROR_STOP=1.
set -eu
SQL_DIR="/var/lib/bastion-data-workspace-sql"
if [ ! -r "$SQL_DIR/01_init.sql" ]; then
  echo "postgres-data: missing $SQL_DIR/01_init.sql (image build error)" >&2
  exit 1
fi
psql -v ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --no-password \
  --no-psqlrc \
  --dbname "$POSTGRES_DB" \
  -f "$SQL_DIR/01_init.sql"
echo "postgres-data: applied data-service/sql/01_init.sql to database $POSTGRES_DB"
