# Bastion main DB — bundles backend/postgres_init so stacks work without bind-mounting the monorepo.
# Published as ghcr.io/<org>/bastion-postgres (or bastion-dev-postgres) alongside other images.
FROM postgres:15-alpine
COPY backend/postgres_init/ /docker-entrypoint-initdb.d/
