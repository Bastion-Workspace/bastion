#!/bin/bash
# Download OSM data for Valhalla routing (US extract).
# Valhalla container auto-builds tiles on first start from PBF files in /custom_files.
# Needs ~16GB RAM for US build (vs 48-64GB for OSRM). See https://valhalla.github.io/valhalla/

set -e

GEOFABRIK_BASE="https://download.geofabrik.de"
REGION_PATH="north-america/us"
REGION_NAME="us"
PBF_URL="${GEOFABRIK_BASE}/${REGION_PATH}/${REGION_NAME}-latest.osm.pbf"
VALHALLA_DATA_DIR="${VALHALLA_DATA_PATH:-./valhalla-data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Setting up Valhalla data for $REGION_NAME..."
echo "Project root: $PROJECT_ROOT"
echo "Data dir: $VALHALLA_DATA_DIR"

cd "$PROJECT_ROOT"
mkdir -p "$VALHALLA_DATA_DIR"
cd "$VALHALLA_DATA_DIR"

if [ ! -f "${REGION_NAME}-latest.osm.pbf" ] || [ "$(stat -c%s "${REGION_NAME}-latest.osm.pbf" 2>/dev/null || echo 0)" -lt 1000000 ]; then
  echo "Downloading OSM data for $REGION_NAME..."
  rm -f "${REGION_NAME}-latest.osm.pbf"
  if command -v curl &>/dev/null; then
    curl -L -o "${REGION_NAME}-latest.osm.pbf" "$PBF_URL"
  else
    wget -c -O "${REGION_NAME}-latest.osm.pbf" "$PBF_URL"
  fi
else
  echo "${REGION_NAME}-latest.osm.pbf already exists."
fi

echo "Valhalla data ready in $PROJECT_ROOT/$VALHALLA_DATA_DIR"
echo "  - Place this directory in VALHALLA_DATA_PATH and mount it as /custom_files in the Valhalla container"
echo "  - On first start Valhalla will build tiles (allow ~1-2 hours for US, ~16GB RAM)"
echo "Run: docker compose up -d valhalla"
