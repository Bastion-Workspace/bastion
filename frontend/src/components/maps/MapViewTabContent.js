/**
 * Tab content for map-view tabs. Registers PMTiles protocol and renders MapWidget
 * with config from the tab (center, zoom, style). Optional layers for future GeoJSON overlays.
 */

import React, { useEffect, useState } from 'react';
import maplibregl from 'maplibre-gl';
import { Protocol } from 'pmtiles';
import { getMapStyle } from './mapStyles';
import MapWidget from './MapWidget';

const PMTILES_URL = import.meta.env.VITE_PMTILES_URL || '';
const DEFAULT_CENTER = [34.0522, -118.2437];
const DEFAULT_ZOOM = 8;

export default function MapViewTabContent({ tab, darkMode = false }) {
  const [styleObject, setStyleObject] = useState(null);
  const styleName = tab.style || 'light';
  const theme = styleName === 'auto' ? (darkMode ? 'dark' : 'light') : styleName;
  const center = Array.isArray(tab.center) && tab.center.length >= 2 ? tab.center : DEFAULT_CENTER;
  const zoom = typeof tab.zoom === 'number' ? tab.zoom : DEFAULT_ZOOM;

  useEffect(() => {
    const protocol = new Protocol();
    maplibregl.addProtocol('pmtiles', protocol.tile);
    return () => maplibregl.removeProtocol('pmtiles');
  }, []);

  useEffect(() => {
    if (!PMTILES_URL || PMTILES_URL.trim() === '') {
      setStyleObject(null);
      return;
    }
    const origin = typeof window !== 'undefined' ? window.location.origin : '';
    const path = PMTILES_URL.startsWith('/') ? PMTILES_URL : `/${PMTILES_URL}`;
    const fullPmtilesUrl = `pmtiles://${origin}${path}`;
    setStyleObject(getMapStyle(fullPmtilesUrl, theme));
  }, [theme]);

  if (!styleObject) {
    return (
      <div style={{ padding: 24, textAlign: 'center', color: darkMode ? '#b3b3b3' : '#666' }}>
        Map tiles not configured. Set VITE_PMTILES_URL to enable map view.
      </div>
    );
  }

  return (
    <div style={{ height: '100%', width: '100%', minHeight: 300 }}>
      <MapWidget
        useLeaflet={false}
        styleObject={styleObject}
        center={center}
        zoom={zoom}
        darkMode={darkMode}
        showControls={true}
        height="100%"
      />
    </div>
  );
}
