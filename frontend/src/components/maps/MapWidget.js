/**
 * Composable map widget: renders basemap + data layers (locations, route, optional GeoJSON layers).
 * Used by InteractiveMap and by map-view tabs. Does not resolve basemap source; receives resolved style.
 */

import React, { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMapEvents, useMap } from 'react-leaflet';
import { Box, Typography, Chip } from '@mui/material';
import L from 'leaflet';
import maplibregl from 'maplibre-gl';
import 'leaflet/dist/leaflet.css';
import 'maplibre-gl/dist/maplibre-gl.css';

import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

const DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

const OSM_TILES = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const CARTO_DARK_TILES = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

function MapDoubleClickHandler({ onMapDoubleClick }) {
  useMapEvents({
    dblclick: (e) => {
      const { lat, lng } = e.latlng;
      onMapDoubleClick?.(lat, lng);
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
      <MapContainer center={center} zoom={zoom} style={{ height: '100%', width: '100%' }} doubleClickZoom={false}>
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
            eventHandlers={{ click: () => onLocationClick?.(location) }}
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

function MapLibreMap({ styleUrl, darkStyleUrl, styleObject, locations, onLocationClick, onMapDoubleClick, routeGeometry, center, zoom, searchCenter, onSearchCenterApplied, showControls = true }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const [popupLocation, setPopupLocation] = useState(null);
  const locationsRef = useRef(locations);

  useEffect(() => {
    locationsRef.current = locations;
  }, [locations]);

  useEffect(() => {
    const effectiveStyle = styleObject || darkStyleUrl || styleUrl;
    if (!containerRef.current || !effectiveStyle) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: effectiveStyle,
      center: [center[1], center[0]],
      zoom
    });
    if (showControls) {
      map.addControl(new maplibregl.NavigationControl(), 'top-right');
    }
    mapRef.current = map;

    const dblClick = (e) => {
      const { lng, lat } = e.lngLat;
      onMapDoubleClick?.(lat, lng);
    };
    map.on('dblclick', dblClick);

    const onLoad = () => {
      map.addSource('route', {
        type: 'geojson',
        data: {
          type: 'Feature',
          properties: {},
          geometry: { type: 'LineString', coordinates: routeGeometry?.coordinates || [] }
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
  }, [styleUrl, darkStyleUrl, styleObject]);

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
          minLng = Math.min(minLng, lng);
          maxLng = Math.max(maxLng, lng);
          minLat = Math.min(minLat, lat);
          maxLat = Math.max(maxLat, lat);
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

/**
 * MapWidget - composable map with resolved basemap and data layers.
 * Props: useLeaflet, styleUrl, darkStyleUrl, styleObject, locations, routeGeometry, center, zoom,
 *        searchCenter, onSearchCenterApplied, onMapDoubleClick, onLocationClick, darkMode, showControls, height.
 */
function MapWidget({
  useLeaflet = false,
  styleUrl,
  darkStyleUrl,
  styleObject,
  locations = [],
  routeGeometry,
  center = [34.0522, -118.2437],
  zoom = 8,
  searchCenter,
  onSearchCenterApplied,
  onMapDoubleClick,
  onLocationClick,
  darkMode = false,
  showControls = true,
  height = '100%'
}) {
  const routePositions = routeGeometry?.coordinates
    ? routeGeometry.coordinates.map((c) => [c[1], c[0]])
    : [];

  return (
    <Box sx={{ height, width: '100%', minHeight: 200 }}>
      {useLeaflet ? (
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
          styleUrl={styleUrl}
          darkStyleUrl={darkStyleUrl}
          styleObject={styleObject}
          locations={locations}
          onLocationClick={onLocationClick}
          onMapDoubleClick={onMapDoubleClick}
          routeGeometry={routeGeometry}
          center={center}
          zoom={zoom}
          searchCenter={searchCenter}
          onSearchCenterApplied={onSearchCenterApplied}
          showControls={showControls}
        />
      )}
    </Box>
  );
}

export default MapWidget;
export { LeafletMap, MapLibreMap };
