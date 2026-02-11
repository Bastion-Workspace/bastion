#!/bin/bash
# Download and preprocess OSM data for OSRM
# Default: one state (fits ~32GB VM). Full US needs 48–64GB+ RAM and will OOM on 32GB.
# See https://download.geofabrik.de/ (e.g. north-america/us/new-york, or north-america/us for full US).

set -e

# Geofabrik: one state = north-america/us/<state>; full US = north-america, REGION_NAME=us
GEOFABRIK_BASE="https://download.geofabrik.de"
REGION_PATH="north-america/us"
REGION_NAME="us"
PBF_URL="${GEOFABRIK_BASE}/${REGION_PATH}/${REGION_NAME}-latest.osm.pbf"
OSRM_DATA_DIR="./osrm-data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Setting up OSRM data for $REGION_NAME..."
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"
mkdir -p "$OSRM_DATA_DIR"
cd "$OSRM_DATA_DIR"

# Download OSM extract if not exists (skip if file exists but is tiny = redirect/html)
if [ ! -f "${REGION_NAME}-latest.osm.pbf" ] || [ "$(stat -c%s "${REGION_NAME}-latest.osm.pbf" 2>/dev/null || echo 0)" -lt 1000000 ]; then
  echo "Downloading OSM data for $REGION_NAME..."
  rm -f "${REGION_NAME}-latest.osm.pbf"
  wget -c -O "${REGION_NAME}-latest.osm.pbf" "$PBF_URL"
fi

# If pipeline completed (customize done) but rename was never run, rename to map.osrm* so the container finds them.
# Only rename when customize output exists; otherwise resume logic below will run partition/customize on us-latest.*
if [ ! -f "map.osrm" ] && [ ! -f "map.osrm.cnbg" ] && [ -f "${REGION_NAME}-latest.osrm.nbg_nodes" ]; then
  echo "Renaming ${REGION_NAME}-latest.osrm* to map.osrm* for container..."
  for f in ${REGION_NAME}-latest.osrm*; do
    [ -e "$f" ] || continue
    newname="map.${f#${REGION_NAME}-latest.}"
    mv "$f" "$newname"
  done
  echo "Rename done."
fi

# Preprocess with resume: run only the steps that have not completed.
# Stages: extract (.osrm + .osrm.names) -> partition (.osrm.cnbg) -> customize (.osrm.nbg_nodes, .osrm.properties) -> rename (map.*)
if [ -f "map.osrm" ] || [ -f "map.osrm.cnbg" ]; then
  echo "OSRM preprocessing already complete (map.osrm* present). Delete to regenerate."
elif [ -f "${REGION_NAME}-latest.osrm.nbg_nodes" ]; then
  echo "Resume: customize done, renaming to map.osrm*..."
  for f in ${REGION_NAME}-latest.osrm*; do
    [ -e "$f" ] || continue
    newname="map.${f#${REGION_NAME}-latest.}"
    mv "$f" "$newname"
  done
  echo "OSRM preprocessing done."
elif [ -f "${REGION_NAME}-latest.osrm.cnbg" ]; then
  echo "Resume: partition done, running customize then rename..."
  docker run --rm -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend:latest \
    osrm-customize /data/${REGION_NAME}-latest.osrm
  for f in ${REGION_NAME}-latest.osrm*; do
    [ -e "$f" ] || continue
    newname="map.${f#${REGION_NAME}-latest.}"
    mv "$f" "$newname"
  done
  echo "OSRM preprocessing done."
elif [ -f "${REGION_NAME}-latest.osrm" ] && [ -f "${REGION_NAME}-latest.osrm.names" ]; then
  echo "Resume: extract done, running partition then customize then rename..."
  docker run --rm -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend:latest \
    osrm-partition /data/${REGION_NAME}-latest.osrm
  docker run --rm -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend:latest \
    osrm-customize /data/${REGION_NAME}-latest.osrm
  for f in ${REGION_NAME}-latest.osrm*; do
    [ -e "$f" ] || continue
    newname="map.${f#${REGION_NAME}-latest.}"
    mv "$f" "$newname"
  done
  echo "OSRM preprocessing done."
else
  echo "Running full pipeline (extract -> partition -> customize -> rename). Extract is the heaviest (RAM); swap helps."
  # Remove any partial output from a previous failed run so extract starts clean
  rm -f ${REGION_NAME}-latest.osrm*
  docker run --rm -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend:latest \
    osrm-extract -p /opt/car.lua /data/${REGION_NAME}-latest.osm.pbf

  docker run --rm -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend:latest \
    osrm-partition /data/${REGION_NAME}-latest.osrm

  docker run --rm -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend:latest \
    osrm-customize /data/${REGION_NAME}-latest.osrm

  for f in ${REGION_NAME}-latest.osrm*; do
    [ -e "$f" ] || continue
    newname="map.${f#${REGION_NAME}-latest.}"
    mv "$f" "$newname"
  done
  echo "OSRM preprocessing done."
fi

# Generate vector tiles (map.mbtiles) from same .pbf for offline map display
echo "Building vector tiles (tilemaker)..."
TILEMAKER_IMAGE="bastion-tilemaker"
if ! docker image inspect "$TILEMAKER_IMAGE" >/dev/null 2>&1; then
  echo "Building tilemaker image (one-time; from github.com/systemed/tilemaker)..."
  TMP_TM=$(mktemp -d)
  git clone --depth 1 https://github.com/systemed/tilemaker.git "$TMP_TM"
  docker build -t "$TILEMAKER_IMAGE" "$TMP_TM"
  rm -rf "$TMP_TM"
fi

if [ ! -f "map.mbtiles" ]; then
  docker run --rm -v "${PWD}:/data" "$TILEMAKER_IMAGE" \
    --input "/data/${REGION_NAME}-latest.osm.pbf" \
    --output "/data/map.mbtiles" \
    --config /usr/src/app/resources/config-openmaptiles.json \
    --process /usr/src/app/resources/process-openmaptiles.lua \
    --store /tmp/tilemaker-store
  echo "Vector tiles written to map.mbtiles"
else
  echo "map.mbtiles already exists; skip tilemaker (delete it to regenerate)"
fi

echo "OSRM data ready in $PROJECT_ROOT/$OSRM_DATA_DIR"
echo "  - Routing: map.osrm* (osrm service, port 5000)"
echo "  - Display: map.mbtiles (map-tiles service, port 8080)"
echo "Run: docker compose up -d osrm map-tiles"
