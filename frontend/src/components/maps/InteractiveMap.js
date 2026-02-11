import React, { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMapEvents, useMap } from 'react-leaflet';
import { Box, Typography, Chip } from '@mui/material';
import L from 'leaflet';
import maplibregl from 'maplibre-gl';
import 'leaflet/dist/leaflet.css';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useTheme } from '../../contexts/ThemeContext';

// Fix Leaflet default icon issue
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
const DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

const TILE_BASE = process.env.REACT_APP_MAP_TILE_URL || '';
const STYLE_PATH = '/styles/basic-preview/style.json';
const DARK_STYLE_PATH = process.env.REACT_APP_MAP_DARK_STYLE || '';

const OSM_TILES = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const CARTO_DARK_TILES = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

function MapDoubleClickHandler({ onMapDoubleClick }) {
  useMapEvents({
    dblclick: (e) => {
      const { lat, lng } = e.latlng;
      onMapDoubleClick && onMapDoubleClick(lat, lng);
    }
  });
  return null;
}

function MapBoundsFitter({ routeGeometry, locations }) {
  const map = useMap();
  useEffect(() => {
    const coords = [];
    if (routeGeometry?.coordinates?.length > 0) {
      routeGeometry.coordinates.forEach((c) => coords.push([c[1], c[0]]));
    }
    if (locations?.length > 0) {
      locations.forEach((loc) => coords.push([parseFloat(loc.latitude), parseFloat(loc.longitude)]));
    }
    if (coords.length >= 2) {
      try {
        map.fitBounds(coords, { padding: [30, 30], maxZoom: 14 });
      } catch (e) {
        // ignore
      }
    }
  }, [map, routeGeometry, locations]);
  return null;
}

function SearchCenterFly({ searchCenter, onSearchCenterApplied }) {
  const map = useMap();
  useEffect(() => {
    if (!searchCenter) return;
    map.flyTo([searchCenter.lat, searchCenter.lng], 17);
    onSearchCenterApplied?.();
  }, [map, searchCenter, onSearchCenterApplied]);
  return null;
}

function LeafletMap({ locations, onLocationClick, onMapDoubleClick, routeGeometry, center, zoom, routePositions, searchCenter, onSearchCenterApplied, darkMode }) {
  return (
    <Box sx={{ height: '100%', width: '100%', ...(darkMode && { filter: 'contrast(1.06) brightness(1.04)' }) }}>
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: '100%', width: '100%' }}
        doubleClickZoom={false}
      >
      {searchCenter && <SearchCenterFly searchCenter={searchCenter} onSearchCenterApplied={onSearchCenterApplied} />}
      <TileLayer
        attribution={darkMode ? '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>' : '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}
        url={darkMode ? CARTO_DARK_TILES : OSM_TILES}
      />
      <MapDoubleClickHandler onMapDoubleClick={onMapDoubleClick} />
      {routePositions.length > 0 && (
        <Polyline positions={routePositions} color="#1976d2" weight={4} opacity={0.8} />
      )}
      <MapBoundsFitter routeGeometry={routeGeometry} locations={locations} />
      {locations.map((location) => (
        <Marker
          key={location.location_id}
          position={[parseFloat(location.latitude), parseFloat(location.longitude)]}
          eventHandlers={{ click: () => onLocationClick && onLocationClick(location) }}
        >
          <Popup>
            <Box>
              <Typography variant="subtitle1" fontWeight="bold">
                {location.name}
                {location.is_global && <Chip label="Global" size="small" sx={{ ml: 1 }} />}
              </Typography>
              {location.address && (
                <Typography variant="body2" color="text.secondary">{location.address}</Typography>
              )}
              {location.notes && <Typography variant="body2" sx={{ mt: 1 }}>{location.notes}</Typography>}
            </Box>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
    </Box>
  );
}

function MapLibreMap({ styleUrl, locations, onLocationClick, onMapDoubleClick, routeGeometry, center, zoom, searchCenter, onSearchCenterApplied, darkStyleUrl }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const [popupLocation, setPopupLocation] = useState(null);
  const locationsRef = useRef(locations);

  useEffect(() => {
    locationsRef.current = locations;
  }, [locations]);

  useEffect(() => {
    const effectiveStyleUrl = darkStyleUrl || styleUrl;
    if (!containerRef.current || !effectiveStyleUrl) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: effectiveStyleUrl,
      center: [center[1], center[0]],
      zoom
    });
    map.addControl(new maplibregl.NavigationControl(), 'top-right');
    mapRef.current = map;

    const dblClick = (e) => {
      const { lng, lat } = e.lngLat;
      onMapDoubleClick && onMapDoubleClick(lat, lng);
    };
    map.on('dblclick', dblClick);

    const onLoad = () => {
      map.addSource('route', {
        type: 'geojson',
        data: {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'LineString',
            coordinates: routeGeometry?.coordinates || []
          }
        }
      });
      map.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route',
        layout: {},
        paint: { 'line-color': '#1976d2', 'line-width': 4, 'line-opacity': 0.8 }
      });
      const features = (locations || []).map((loc) => ({
        type: 'Feature',
        properties: { id: loc.location_id },
        geometry: { type: 'Point', coordinates: [parseFloat(loc.longitude), parseFloat(loc.latitude)] }
      }));
      map.addSource('markers', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features }
      });
      map.addLayer({
        id: 'markers-circle',
        type: 'circle',
        source: 'markers',
        paint: { 'circle-radius': 8, 'circle-color': '#1976d2', 'circle-stroke-width': 2, 'circle-stroke-color': '#fff' }
      });
      map.on('click', 'markers-circle', (e) => {
        const id = e.features?.[0]?.properties?.id;
        const loc = locationsRef.current?.find((l) => l.location_id === id);
        if (loc) setPopupLocation(loc);
      });
    };
    map.on('load', onLoad);

    return () => {
      map.off('dblclick', dblClick);
      map.off('load', onLoad);
      map.off('click', 'markers-circle');
      map.remove();
      mapRef.current = null;
    };
  }, [styleUrl, darkStyleUrl]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getSource) return;

    const routeCoords = routeGeometry?.coordinates || [];
    const routeSource = map.getSource('route');
    if (routeSource) {
      routeSource.setData({
        type: 'Feature',
        properties: {},
        geometry: { type: 'LineString', coordinates: routeCoords }
      });
    }

    const features = (locations || []).map((loc) => ({
      type: 'Feature',
      properties: { id: loc.location_id },
      geometry: { type: 'Point', coordinates: [parseFloat(loc.longitude), parseFloat(loc.latitude)] }
    }));
    const markersSource = map.getSource('markers');
    if (markersSource) {
      markersSource.setData({ type: 'FeatureCollection', features });
    }
  }, [routeGeometry, locations]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const coords = [];
    if (routeGeometry?.coordinates?.length > 0) {
      routeGeometry.coordinates.forEach((c) => coords.push([c[1], c[0]]));
    }
    if (locations?.length > 0) {
      locations.forEach((loc) => coords.push([parseFloat(loc.latitude), parseFloat(loc.longitude)]));
    }
    if (coords.length >= 2) {
      try {
        const lngLats = coords.map(([lat, lng]) => [lng, lat]);
        let minLng = lngLats[0][0], maxLng = lngLats[0][0], minLat = lngLats[0][1], maxLat = lngLats[0][1];
        lngLats.forEach(([lng, lat]) => {
          minLng = Math.min(minLng, lng); maxLng = Math.max(maxLng, lng);
          minLat = Math.min(minLat, lat); maxLat = Math.max(maxLat, lat);
        });
        map.fitBounds([[minLng, minLat], [maxLng, maxLat]], { padding: 30, maxZoom: 14 });
      } catch (e) {
        // ignore
      }
    }
  }, [routeGeometry, locations]);

  useEffect(() => {
    if (!searchCenter || !mapRef.current) return;
    mapRef.current.flyTo({ center: [searchCenter.lng, searchCenter.lat], zoom: 17 });
    onSearchCenterApplied?.();
  }, [searchCenter, onSearchCenterApplied]);

  return (
    <>
      <div ref={containerRef} style={{ height: '100%', width: '100%' }} />
      {popupLocation && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            right: 16,
            maxWidth: 320,
            backgroundColor: 'rgba(255,255,255,0.95)',
            p: 1.5,
            borderRadius: 1,
            boxShadow: 2,
            zIndex: 1000
          }}
        >
          <Typography variant="subtitle1" fontWeight="bold">
            {popupLocation.name}
            {popupLocation.is_global && <Chip label="Global" size="small" sx={{ ml: 1 }} />}
          </Typography>
          {popupLocation.address && (
            <Typography variant="body2" color="text.secondary">{popupLocation.address}</Typography>
          )}
          {popupLocation.notes && <Typography variant="body2" sx={{ mt: 1 }}>{popupLocation.notes}</Typography>}
          <Typography
            component="button"
            variant="caption"
            onClick={() => setPopupLocation(null)}
            sx={{ mt: 1, cursor: 'pointer', border: 0, background: 'none' }}
          >
            Close
          </Typography>
        </Box>
      )}
    </>
  );
}

const InteractiveMap = ({ locations = [], onLocationClick, onMapDoubleClick, routeGeometry, searchCenter, onSearchCenterApplied, darkMap }) => {
  const { darkMode: appDarkMode } = useTheme();
  const darkMode = darkMap !== undefined ? darkMap : appDarkMode;
  const routePositions = routeGeometry?.coordinates
    ? routeGeometry.coordinates.map((c) => [c[1], c[0]])
    : [];
  const center = locations.length > 0
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

  useEffect(() => {
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
  }, []);

  const useLeaflet = localTilesReady === false || (localTilesReady === null && !TILE_BASE);
  const baseUrl = TILE_BASE.replace(/\/$/, '');
  const darkStyleUrl = darkMode && DARK_STYLE_PATH ? `${baseUrl}${DARK_STYLE_PATH.startsWith('/') ? '' : '/'}${DARK_STYLE_PATH}` : null;

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
      {localTilesReady === null && TILE_BASE ? (
        <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography color="text.secondary">Loading map…</Typography>
        </Box>
      ) : useLeaflet ? (
        <LeafletMap
          locations={locations}
          onLocationClick={onLocationClick}
          onMapDoubleClick={onMapDoubleClick}
          routeGeometry={routeGeometry}
          center={center}
          zoom={zoom}
          routePositions={routePositions}
          searchCenter={searchCenter}
          onSearchCenterApplied={onSearchCenterApplied}
          darkMode={darkMode}
        />
      ) : (
        <MapLibreMap
          styleUrl={`${baseUrl}${STYLE_PATH}`}
          darkStyleUrl={darkStyleUrl}
          locations={locations}
          onLocationClick={onLocationClick}
          onMapDoubleClick={onMapDoubleClick}
          routeGeometry={routeGeometry}
          center={center}
          zoom={zoom}
          searchCenter={searchCenter}
          onSearchCenterApplied={onSearchCenterApplied}
        />
      )}
    </Box>
  );
};

export default InteractiveMap;
