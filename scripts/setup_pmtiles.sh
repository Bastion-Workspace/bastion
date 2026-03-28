#!/bin/bash
# Download PMTiles basemap for world vector map display.
# Serves via nginx (no tile server container). See frontend/nginx.conf /pmtiles/ location.
# Optional: PROTOMAPS_PMTILES_URL to override the default build URL.

set -e

PROTOMAPS_BUILDS="${PROTOMAPS_PMTILES_URL:-https://build.protomaps.com/20241020.pmtiles}"
PMTILES_DATA_DIR="${PMTILES_DATA_PATH:-./pmtiles-data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Setting up PMTiles data..."
echo "Project root: $PROJECT_ROOT"
echo "Data dir: $PMTILES_DATA_DIR"
echo "Source: $PROTOMAPS_BUILDS"

cd "$PROJECT_ROOT"
mkdir -p "$PMTILES_DATA_DIR"
cd "$PMTILES_DATA_DIR"

# Default filename for single-file setup (used by nginx alias)
OUTPUT_FILE="world.pmtiles"

if [ -f "$OUTPUT_FILE" ]; then
  echo "$OUTPUT_FILE already exists. Delete it to re-download."
  echo "PMTiles data ready in $PROJECT_ROOT/$PMTILES_DATA_DIR"
  echo "  - Mount this path as /data/pmtiles in the frontend container"
  echo "  - Set VITE_PMTILES_URL to the URL where the file is served (e.g. /pmtiles/world.pmtiles)"
  exit 0
fi

echo "Downloading PMTiles (this may be large, 70-80GB for planet)..."
if command -v curl &>/dev/null; then
  curl -L -o "$OUTPUT_FILE" "$PROTOMAPS_BUILDS"
else
  wget -c -O "$OUTPUT_FILE" "$PROTOMAPS_BUILDS"
fi

echo "PMTiles data ready in $PROJECT_ROOT/$PMTILES_DATA_DIR"
echo "  - File: $OUTPUT_FILE"
echo "  - Mount this directory as /data/pmtiles in the frontend container"
echo "  - Set VITE_PMTILES_URL to the URL where the file is served (e.g. /pmtiles/world.pmtiles)"
