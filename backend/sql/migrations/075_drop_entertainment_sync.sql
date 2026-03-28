-- Drop entertainment sync tables (Sonarr/Radarr bespoke sync removed in favor of Agent Factory data connectors)
DROP TABLE IF EXISTS entertainment_sync_items;
DROP TABLE IF EXISTS entertainment_sync_config;
