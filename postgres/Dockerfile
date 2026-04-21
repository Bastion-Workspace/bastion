# Bastion main DB — bundles backend/sql so stacks work without bind-mounting the monorepo.
# Published as ghcr.io/<org>/bastion-postgres (or bastion-dev-postgres) alongside other images.
FROM postgres:15-alpine
COPY backend/sql/ /docker-entrypoint-initdb.d/
