# Data workspace Postgres — bundles data-service/sql for image-only deployments.
# SQL lives under /var/lib/bastion-data-workspace-sql/; only this shell is placed in
# docker-entrypoint-initdb.d/ because nested paths under initdb.d are not executed as SQL.
FROM postgres:15-alpine
COPY data-service/sql/ /var/lib/bastion-data-workspace-sql/
COPY postgres-data/init-data-workspace.sh /docker-entrypoint-initdb.d/00_init_data_workspace.sh
RUN chmod +x /docker-entrypoint-initdb.d/00_init_data_workspace.sh
