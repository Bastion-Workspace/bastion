# OSRM routing + TileServer GL in one container (same OSM data volume)
FROM ghcr.io/project-osrm/osrm-backend:latest

USER root

# Alpine: install Node so we can run tileserver-gl
RUN apk add --no-cache nodejs npm

# TileServer GL serves .mbtiles from the same volume as OSRM
RUN npm install -g tileserver-gl && npm cache clean --force

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000 8080

ENTRYPOINT ["/entrypoint.sh"]
