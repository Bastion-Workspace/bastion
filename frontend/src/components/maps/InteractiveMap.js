import React, { useEffect, useState } from 'react';
import { Box, Typography } from '@mui/material';
import maplibregl from 'maplibre-gl';
import { useTheme } from '../../contexts/ThemeContext';
import { Protocol } from 'pmtiles';
import { getMapStyle } from './mapStyles';
import MapWidget from './MapWidget';

const TILE_BASE = import.meta.env.VITE_MAP_TILE_URL || '';
const PMTILES_URL = import.meta.env.VITE_PMTILES_URL || '';
const STYLE_PATH = '/styles/basic-preview/style.json';
const DARK_STYLE_PATH = import.meta.env.VITE_MAP_DARK_STYLE || '';

/**
 * InteractiveMap - backward-compatible wrapper around MapWidget.
 * Resolves basemap (PMTiles > tileserver-gl > Leaflet fallback) and registers PMTiles protocol.
 * Preserves existing props: locations, routeGeometry, onMapDoubleClick, onLocationClick, searchCenter, darkMap, stylePreference.
 */
const InteractiveMap = ({
  locations = [],
  onLocationClick,
  onMapDoubleClick,
  routeGeometry,
  searchCenter,
  onSearchCenterApplied,
  darkMap,
  stylePreference: stylePreferenceProp
}) => {
  const { darkMode: appDarkMode } = useTheme();
  const darkMode = darkMap !== undefined ? darkMap : appDarkMode;
  const routePositions = routeGeometry?.coordinates
    ? routeGeometry.coordinates.map((c) => [c[1], c[0]])
    : [];
  const center =
    locations.length > 0
      ? [
          locations.reduce((s, l) => s + parseFloat(l.latitude), 0) / locations.length,
          locations.reduce((s, l) => s + parseFloat(l.longitude), 0) / locations.length
        ]
      : routePositions.length > 0
        ? [
            routePositions.reduce((s, p) => s + p[0], 0) / routePositions.length,
            routePositions.reduce((s, p) => s + p[1], 0) / routePositions.length
          ]
        : [34.0522, -118.2437];
  const zoom = locations.length > 0 || routePositions.length > 0 ? 10 : 8;

  const [localTilesReady, setLocalTilesReady] = useState(null);
  const [pmtilesStyleObject, setPmtilesStyleObject] = useState(null);

  useEffect(() => {
    const protocol = new Protocol();
    maplibregl.addProtocol('pmtiles', protocol.tile);
    return () => maplibregl.removeProtocol('pmtiles');
  }, []);

  useEffect(() => {
    if (PMTILES_URL && PMTILES_URL.trim() !== '') {
      const origin = typeof window !== 'undefined' ? window.location.origin : '';
      const path = PMTILES_URL.startsWith('/') ? PMTILES_URL : `/${PMTILES_URL}`;
      const fullPmtilesUrl = `pmtiles://${origin}${path}`;
      const theme = stylePreferenceProp === 'auto' ? (darkMode ? 'dark' : 'light') : (stylePreferenceProp || (darkMode ? 'dark' : 'light'));
      setPmtilesStyleObject(getMapStyle(fullPmtilesUrl, theme));
      setLocalTilesReady(true);
      return;
    }
    if (!TILE_BASE || TILE_BASE.trim() === '') {
      setLocalTilesReady(false);
      return;
    }
    const base = TILE_BASE.replace(/\/$/, '');
    const styleUrl = `${base}${STYLE_PATH}`;
    fetch(styleUrl)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then(() => setLocalTilesReady(true))
      .catch(() => setLocalTilesReady(false));
  }, [PMTILES_URL, darkMode, stylePreferenceProp]);

  const useLeaflet = localTilesReady === false || (localTilesReady === null && !TILE_BASE && !PMTILES_URL);
  const baseUrl = TILE_BASE.replace(/\/$/, '');
  const darkStyleUrl =
    darkMode && DARK_STYLE_PATH ? `${baseUrl}${DARK_STYLE_PATH.startsWith('/') ? '' : '/'}${DARK_STYLE_PATH}` : null;

  return (
    <Box sx={{ height: '100%', minHeight: 400, width: '100%', borderRadius: 1, overflow: 'hidden', position: 'relative' }}>
      <Box
        sx={{
          position: 'absolute',
          top: 8,
          left: 8,
          zIndex: 1000,
          backgroundColor: darkMode ? 'rgba(30, 30, 30, 0.9)' : 'rgba(255, 255, 255, 0.9)',
          color: darkMode ? 'rgba(255, 255, 255, 0.9)' : 'inherit',
          padding: '4px 8px',
          borderRadius: 1,
          fontSize: '0.75rem',
          pointerEvents: 'none',
          boxShadow: darkMode ? '0 2px 4px rgba(0,0,0,0.4)' : '0 2px 4px rgba(0,0,0,0.2)'
        }}
      >
        Double-click map to add location
      </Box>
      {localTilesReady === null && (TILE_BASE || PMTILES_URL) ? (
        <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography color="text.secondary">Loading map…</Typography>
        </Box>
      ) : (
        <MapWidget
          useLeaflet={useLeaflet}
          styleUrl={PMTILES_URL ? undefined : `${baseUrl}${STYLE_PATH}`}
          darkStyleUrl={PMTILES_URL ? undefined : darkStyleUrl}
          styleObject={pmtilesStyleObject}
          locations={locations}
          onLocationClick={onLocationClick}
          onMapDoubleClick={onMapDoubleClick}
          routeGeometry={routeGeometry}
          center={center}
          zoom={zoom}
          searchCenter={searchCenter}
          onSearchCenterApplied={onSearchCenterApplied}
          darkMode={darkMode}
          showControls={true}
          height="100%"
        />
      )}
    </Box>
  );
};

export default InteractiveMap;
