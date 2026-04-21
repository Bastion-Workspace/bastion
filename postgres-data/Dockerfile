# Data workspace Postgres — bundles data-service/sql for image-only deployments.
FROM postgres:15-alpine
COPY data-service/sql/ /docker-entrypoint-initdb.d/
