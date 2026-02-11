#!/bin/sh
set -e

# Start OSRM routing engine in background (port 5000)
osrm-routed --algorithm mld --max-table-size 10000 /data/map.osrm &

# Serve map tiles from same volume if mbtiles exists (port 8080); otherwise keep container alive
if [ -f /data/map.mbtiles ]; then
  exec tileserver-gl --file /data/map.mbtiles --port 8080 --bind 0.0.0.0 --verbose
else
  wait
fi
