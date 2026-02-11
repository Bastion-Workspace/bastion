import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  Tabs,
  Tab,
  FormControl,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  IconButton,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Chip,
  Slider,
  Paper,
  Autocomplete,
} from '@mui/material';
import { Add, Public, Lock, Route, DirectionsCar, Delete, PlayArrow, Search, MyLocation, Place } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import { useTheme } from '../../contexts/ThemeContext';
import LocationsList from './LocationsList';
import InteractiveMap from './InteractiveMap';
import LocationDialog from './LocationDialog';

const MAP_STYLE_KEY = 'mapStyle';

const mapStyleFromValue = (v) => (v === 0 ? 'light' : v === 1 ? 'auto' : 'dark');
const valueFromMapStyle = (s) => (s === 'light' ? 0 : s === 'auto' ? 1 : 2);

function flattenSteps(legs) {
  if (!legs || !Array.isArray(legs)) return [];
  const steps = [];
  legs.forEach((leg) => {
    (leg.steps || []).forEach((s) => steps.push(s));
  });
  return steps;
}

/** Normalize road name for merging: same road can be "State Highway 96" vs "State Route 96" in OSM. */
function roadMergeKey(name) {
  if (!name || typeof name !== 'string') return '';
  const t = name.trim();
  const m = t.match(/(?:State|US|County)?\s*(?:Route|Highway|Road|SR|CR|HWY)\s*(\d+)/i)
    || t.match(/(?:Route|Highway)\s*(\d+)/i);
  if (m) return String(m[1]);
  return t.toLowerCase();
}

/** Convert meters to human-readable distance (miles or feet). */
function formatDistance(meters) {
  if (meters < 100) {
    return `${Math.round(meters * 3.28084)} ft`;
  }
  const miles = meters / 1609.34;
  if (miles < 0.1) {
    return `${Math.round(meters * 3.28084)} ft`;
  }
  return `${miles.toFixed(1)} mi`;
}

/** Format turn instruction with action and road name. */
function formatTurnInstruction(step) {
  const type = step.maneuver?.type;
  const modifier = step.maneuver?.modifier || '';
  const roadName = (step.name || '').trim();
  
  if (type === 'depart') {
    return roadName ? `Depart on ${roadName}` : 'Depart';
  }
  if (type === 'arrive') {
    return 'Arrive at destination';
  }
  
  // Format turn/direction with road name
  const actionMap = {
    'turn': modifier ? `Turn ${modifier}` : 'Turn',
    'sharp left': 'Sharp left',
    'sharp right': 'Sharp right',
    'slight left': 'Slight left',
    'slight right': 'Slight right',
    'left': 'Turn left',
    'right': 'Turn right',
    'straight': 'Continue straight',
    'uturn': 'Make U-turn',
    'fork': modifier ? `Take ${modifier} fork` : 'At fork',
    'merge': modifier ? `Merge ${modifier}` : 'Merge',
    'ramp': modifier ? `Take ramp ${modifier}` : 'Take ramp',
    'roundabout': 'Enter roundabout',
    'exit roundabout': 'Exit roundabout',
  };
  
  let action = actionMap[type] || actionMap[modifier] || modifier || 'Continue';
  
  // Add road name if available
  if (roadName && type !== 'arrive') {
    if (action.toLowerCase().includes('onto') || action.toLowerCase().includes('on ')) {
      return `${action} ${roadName}`;
    }
    return `${action} onto ${roadName}`;
  }
  
  return action;
}

/** Merge consecutive steps on the same road so turn-by-turn is less fragmented. */
function mergeConsecutiveSteps(steps) {
  if (!steps?.length) return [];
  const out = [];
  let prev = null;
  for (const s of steps) {
    const type = s.maneuver?.type;
    const name = (s.name || '').trim() || (s.maneuver?.modifier || '—');
    const key = roadMergeKey(s.name || '');
    const dist = typeof s.distance === 'number' ? s.distance : 0;
    if (type === 'depart') {
      prev = { ...s, distance: dist, name: 'Depart' };
      out.push(prev);
      continue;
    }
    if (type === 'arrive') {
      out.push({ ...s, name: 'Arrive', distance: dist });
      prev = null;
      continue;
    }
    const prevKey = prev?.roadNameRaw != null ? roadMergeKey(prev.roadNameRaw) : '';
    const sameRoad = key && prevKey && key === prevKey && prev.name !== 'Depart';
    if (sameRoad && prev) {
      prev.distance += dist;
    } else {
      prev = { ...s, distance: dist, name, roadNameRaw: name };
      out.push(prev);
    }
  }
  return out.map(({ roadNameRaw, ...rest }) => ({ ...rest }));
}

const MapPage = () => {
  const queryClient = useQueryClient();
  const { darkMode: appDarkMode } = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingLocation, setEditingLocation] = useState(null);
  const [mapStylePreference, setMapStylePreference] = useState(() => {
    try {
      const saved = localStorage.getItem(MAP_STYLE_KEY);
      return saved === 'light' || saved === 'auto' || saved === 'dark' ? saved : 'auto';
    } catch {
      return 'auto';
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(MAP_STYLE_KEY, mapStylePreference);
    } catch {
      // ignore
    }
  }, [mapStylePreference]);

  const [fromLocationId, setFromLocationId] = useState('');
  const [toLocationId, setToLocationId] = useState('');
  const [fromManual, setFromManual] = useState(null);
  const [toManual, setToManual] = useState(null);
  const [currentRoute, setCurrentRoute] = useState(null);
  const [routeError, setRouteError] = useState(null);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveRouteName, setSaveRouteName] = useState('');
  const [addressLookup, setAddressLookup] = useState('');
  const [addressLookupLoading, setAddressLookupLoading] = useState(false);
  const [addressLookupError, setAddressLookupError] = useState('');
  const [addressSuggestions, setAddressSuggestions] = useState([]);
  const [addressSuggestionsLoading, setAddressSuggestionsLoading] = useState(false);
  const [lastGeocoded, setLastGeocoded] = useState(null);
  const [searchCenter, setSearchCenter] = useState(null);
  const addressSuggestionsDebounceRef = useRef(null);
  const lastNominatimReqRef = useRef(0);

  const { data: locationsData, isLoading } = useQuery(
    'locations',
    () => apiService.get('/api/locations')
  );

  const { data: savedRoutesData, isLoading: savedRoutesLoading } = useQuery(
    'savedRoutes',
    () => apiService.get('/api/routes'),
    { enabled: activeTab === 3 }
  );

  const routeMutation = useMutation(
    (params) => {
      const q = new URLSearchParams(params);
      return apiService.get(`/api/routes/route?${q}`);
    },
    {
      onSuccess: (data) => {
        setCurrentRoute(data);
        setRouteError(null);
      },
      onError: (err) => {
        setCurrentRoute(null);
        const status = err.response?.status;
        const detail = err.response?.data?.detail ?? err.message ?? 'Route failed';
        const routingUnavailable =
          status === 503 ||
          (typeof detail === 'string' && (
            detail.includes('Routing service') ||
            detail.includes('OSRM') ||
            detail.includes('InvalidUrl') ||
            detail.toLowerCase().includes('connection') ||
            detail.toLowerCase().includes('network')
          ));
        setRouteError(
          routingUnavailable
            ? 'Directions unavailable. Start the OSRM service (e.g. docker compose up -d osrm) for routing. You can still view the map and locations.'
            : detail
        );
      },
    }
  );

  const saveRouteMutation = useMutation(
    (body) => apiService.post('/api/routes', body),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('savedRoutes');
        setSaveDialogOpen(false);
        setSaveRouteName('');
      },
    }
  );

  const deleteRouteMutation = useMutation(
    (id) => apiService.delete(`/api/routes/${id}`),
    {
      onSuccess: () => queryClient.invalidateQueries('savedRoutes'),
    }
  );

  const createMutation = useMutation(
    (locationData) => apiService.post('/api/locations', locationData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('locations');
        setDialogOpen(false);
      },
    }
  );

  const updateMutation = useMutation(
    ({ id, data }) => apiService.put(`/api/locations/${id}`, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('locations');
        setDialogOpen(false);
        setEditingLocation(null);
      },
    }
  );

  const deleteMutation = useMutation(
    (id) => apiService.delete(`/api/locations/${id}`),
    {
      onSuccess: () => queryClient.invalidateQueries('locations'),
    }
  );

  const locations = locationsData?.locations || [];
  const myLocations = locations.filter((loc) => !loc.is_global);
  const globalLocations = locations.filter((loc) => loc.is_global);
  const savedRoutes = savedRoutesData?.routes || [];

  // Autocomplete: debounced Nominatim search (max 1 req/s per Nominatim policy)
  useEffect(() => {
    const q = addressLookup.trim();
    if (!q || q.length < 2) {
      setAddressSuggestions([]);
      return;
    }
    if (addressSuggestionsDebounceRef.current) clearTimeout(addressSuggestionsDebounceRef.current);
    addressSuggestionsDebounceRef.current = setTimeout(async () => {
      const now = Date.now();
      const wait = Math.max(0, 1000 - (now - lastNominatimReqRef.current));
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      lastNominatimReqRef.current = Date.now();
      setAddressSuggestionsLoading(true);
      try {
        const res = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&limit=6`,
          { headers: { Accept: 'application/json', 'User-Agent': 'Bastion-Map/1.0' } }
        );
        const data = await res.json();
        setAddressSuggestions(Array.isArray(data) ? data.map((r) => ({ lat: parseFloat(r.lat), lng: parseFloat(r.lon), display_name: r.display_name })) : []);
      } catch {
        setAddressSuggestions([]);
      } finally {
        setAddressSuggestionsLoading(false);
      }
    }, 400);
    return () => {
      if (addressSuggestionsDebounceRef.current) clearTimeout(addressSuggestionsDebounceRef.current);
    };
  }, [addressLookup]);

  const handleGeocode = async () => {
    const q = addressLookup.trim();
    if (!q) {
      setAddressLookupError('Enter an address');
      return;
    }
    setAddressLookupError('');
    setAddressLookupLoading(true);
    try {
      const now = Date.now();
      const wait = Math.max(0, 1000 - (now - lastNominatimReqRef.current));
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      lastNominatimReqRef.current = Date.now();
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&limit=1`,
        { headers: { Accept: 'application/json', 'User-Agent': 'Bastion-Map/1.0' } }
      );
      const data = await res.json();
      if (!data || data.length === 0) {
        setAddressLookupError('Address not found');
        setLastGeocoded(null);
        return;
      }
      const { lat, lon, display_name } = data[0];
      const point = { lat: parseFloat(lat), lng: parseFloat(lon), displayName: display_name };
      setLastGeocoded(point);
      setSearchCenter(point);
    } catch (e) {
      setAddressLookupError('Lookup failed');
      setLastGeocoded(null);
    } finally {
      setAddressLookupLoading(false);
    }
  };

  const handleSelectAddressSuggestion = useCallback((option) => {
    if (!option?.display_name) return;
    const point = { lat: option.lat, lng: option.lng, displayName: option.display_name };
    setAddressLookup(option.display_name);
    setAddressSuggestions([]);
    setAddressLookupError('');
    setLastGeocoded(point);
    setSearchCenter(point);
  }, []);

  const handleSearchCenter = () => {
    if (!lastGeocoded) return;
    setSearchCenter(lastGeocoded);
  };

  const handleSearchCenterApplied = () => {
    setSearchCenter(null);
  };

  const handleSetAsFrom = () => {
    if (!lastGeocoded) return;
    setFromManual(lastGeocoded);
    setFromLocationId('');
  };

  const handleSetAsTo = () => {
    if (!lastGeocoded) return;
    setToManual(lastGeocoded);
    setToLocationId('');
  };

  const fromPoint = fromManual
    ? { lat: fromManual.lat, lng: fromManual.lng }
    : fromLocationId
      ? (() => {
          const loc = locations.find((l) => l.location_id === fromLocationId);
          return loc ? { lat: parseFloat(loc.latitude), lng: parseFloat(loc.longitude) } : null;
        })()
      : null;
  const toPoint = toManual
    ? { lat: toManual.lat, lng: toManual.lng }
    : toLocationId
      ? (() => {
          const loc = locations.find((l) => l.location_id === toLocationId);
          return loc ? { lat: parseFloat(loc.latitude), lng: parseFloat(loc.longitude) } : null;
        })()
      : null;
  const hasFromAndTo = Boolean(fromPoint && toPoint);

  const handleGetRoute = () => {
    if (!hasFromAndTo) return;
    const coords = `${fromPoint.lat},${fromPoint.lng};${toPoint.lat},${toPoint.lng}`;
    routeMutation.mutate({ coordinates: coords });
  };

  const handleClearRoute = () => {
    setCurrentRoute(null);
    setRouteError(null);
  };

  const handleSaveRoute = () => {
    if (!currentRoute || !saveRouteName.trim()) return;
    const waypoints = [
      fromManual
        ? { latitude: fromManual.lat, longitude: fromManual.lng, name: fromManual.displayName || 'From (search)' }
        : fromLocationId
          ? (() => {
              const loc = locations.find((l) => l.location_id === fromLocationId);
              return loc ? { location_id: fromLocationId, latitude: loc.latitude, longitude: loc.longitude, name: loc.name } : null;
            })()
          : null,
      toManual
        ? { latitude: toManual.lat, longitude: toManual.lng, name: toManual.displayName || 'To (search)' }
        : toLocationId
          ? (() => {
              const loc = locations.find((l) => l.location_id === toLocationId);
              return loc ? { location_id: toLocationId, latitude: loc.latitude, longitude: loc.longitude, name: loc.name } : null;
            })()
          : null,
    ].filter(Boolean);
    const steps = flattenSteps(currentRoute.legs);
    saveRouteMutation.mutate({
      name: saveRouteName.trim(),
      waypoints,
      geometry: currentRoute.geometry,
      steps,
      distance_meters: currentRoute.distance,
      duration_seconds: currentRoute.duration,
      profile: 'driving',
    });
  };

  const clearFrom = () => {
    setFromManual(null);
    setFromLocationId('');
  };
  const clearTo = () => {
    setToManual(null);
    setToLocationId('');
  };

  const handleLoadSavedRoute = (route) => {
    setCurrentRoute({
      geometry: route.geometry,
      legs: route.steps && route.steps.length ? [{ steps: route.steps }] : [],
      distance: route.distance_meters,
      duration: route.duration_seconds,
    });
    setActiveTab(0);
  };

  const handleAddLocation = () => {
    setEditingLocation(null);
    setDialogOpen(true);
  };

  const handleMapDoubleClick = (lat, lng) => {
    setEditingLocation({
      latitude: lat.toString(),
      longitude: lng.toString(),
      name: '',
      address: '',
      notes: '',
      is_global: false,
    });
    setDialogOpen(true);
  };

  const handleEditLocation = (location) => {
    setEditingLocation(location);
    setDialogOpen(true);
  };

  const handleSaveLocation = (locationData) => {
    if (editingLocation?.location_id) {
      updateMutation.mutate({ id: editingLocation.location_id, data: locationData });
    } else {
      createMutation.mutate(locationData);
    }
  };

  const handleDeleteLocation = (locationId) => {
    if (window.confirm('Are you sure you want to delete this location?')) {
      deleteMutation.mutate(locationId);
    }
  };

  const routeSteps = currentRoute ? mergeConsecutiveSteps(flattenSteps(currentRoute.legs)) : [];
  const routeGeometry = currentRoute?.geometry || null;

  return (
    <Box sx={{ p: 3, display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3, flexShrink: 0 }}>
        <Typography variant="h4">Map & Locations</Typography>
        <Button variant="contained" startIcon={<Add />} onClick={handleAddLocation}>
          Add Location
        </Button>
      </Box>

      <Card sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)} sx={{ flexShrink: 0 }}>
          <Tab label="Map View" />
          <Tab label={`My Locations (${myLocations.length})`} icon={<Lock />} iconPosition="end" />
          <Tab label={`Global Locations (${globalLocations.length})`} icon={<Public />} iconPosition="end" />
          <Tab label={`Saved Routes (${savedRoutes.length})`} icon={<Route />} iconPosition="end" />
        </Tabs>

        <Box sx={{ p: 2, flex: 1, minHeight: 0, overflow: 'auto' }}>
          {activeTab === 0 && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, height: '100%', minHeight: 0 }}>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'flex-start', flexShrink: 0 }}>
                <Card variant="outlined" sx={{ p: 2, flex: '1 1 320px' }}>
                  <Typography variant="subtitle1" sx={{ mb: 2 }} fontWeight="bold">
                    Directions
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>From</Typography>
                      {fromManual ? (
                        <Chip
                          label={fromManual.displayName?.slice(0, 40) + (fromManual.displayName?.length > 40 ? '…' : '')}
                          onDelete={clearFrom}
                          size="small"
                          sx={{ maxWidth: '100%' }}
                        />
                      ) : (
                        <FormControl size="small" fullWidth>
                          <Select
                            value={fromLocationId}
                            displayEmpty
                            onChange={(e) => setFromLocationId(e.target.value)}
                          >
                            <MenuItem value="">Select location</MenuItem>
                            {locations.map((loc) => (
                              <MenuItem key={loc.location_id} value={loc.location_id}>{loc.name}</MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      )}
                    </Box>
                    <Box>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>To</Typography>
                      {toManual ? (
                        <Chip
                          label={toManual.displayName?.slice(0, 40) + (toManual.displayName?.length > 40 ? '…' : '')}
                          onDelete={clearTo}
                          size="small"
                          sx={{ maxWidth: '100%' }}
                        />
                      ) : (
                        <FormControl size="small" fullWidth>
                          <Select
                            value={toLocationId}
                            displayEmpty
                            onChange={(e) => setToLocationId(e.target.value)}
                          >
                            <MenuItem value="">Select location</MenuItem>
                            {locations.map((loc) => (
                              <MenuItem key={loc.location_id} value={loc.location_id}>{loc.name}</MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      )}
                    </Box>
                    <Button
                      variant="contained"
                      startIcon={routeMutation.isLoading ? <CircularProgress size={20} /> : <DirectionsCar />}
                      onClick={handleGetRoute}
                      disabled={!hasFromAndTo || routeMutation.isLoading}
                    >
                      Get route
                    </Button>
                    {currentRoute && (
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        <Button variant="outlined" size="small" onClick={() => setSaveDialogOpen(true)}>
                          Save route
                        </Button>
                        <Button variant="outlined" size="small" color="secondary" onClick={handleClearRoute}>
                          Clear route
                        </Button>
                      </Box>
                    )}
                    {routeError && (
                      <Typography color="error" variant="body2">{routeError}</Typography>
                    )}
                    {currentRoute && (
                      <Typography variant="body2" color="text.secondary">
                        {(currentRoute.distance / 1609.34).toFixed(1)} mi · {Math.round(currentRoute.duration / 60)} min
                      </Typography>
                    )}
                  </Box>
                </Card>

                <Card variant="outlined" sx={{ p: 2, flex: '1 1 280px', minWidth: 280 }}>
                  <Typography variant="subtitle1" sx={{ mb: 2 }} fontWeight="bold">
                    Address lookup
                  </Typography>
                  <Autocomplete
                    freeSolo
                    size="small"
                    options={addressSuggestions}
                    getOptionLabel={(opt) => (typeof opt === 'string' ? opt : opt?.display_name ?? '')}
                    filterOptions={(x) => x}
                    loading={addressSuggestionsLoading}
                    inputValue={addressLookup}
                    onInputChange={(_, v) => setAddressLookup(v ?? '')}
                    onChange={(_, v) => {
                      if (v && typeof v === 'object' && v.display_name) handleSelectAddressSuggestion(v);
                    }}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        placeholder="Search address (e.g. city, street)"
                        onKeyDown={(e) => e.key === 'Enter' && handleGeocode()}
                        error={Boolean(addressLookupError)}
                        helperText={addressLookupError}
                      />
                    )}
                    renderOption={(props, option) => (
                      <li {...props} key={`${option.lat},${option.lng}`}>
                        <Typography variant="body2" noWrap sx={{ maxWidth: '100%' }}>
                          {option.display_name}
                        </Typography>
                      </li>
                    )}
                    sx={{ mb: 1.5 }}
                  />
                  <Button
                    fullWidth
                    variant="outlined"
                    size="small"
                    startIcon={addressLookupLoading ? <CircularProgress size={16} /> : <Search />}
                    onClick={handleGeocode}
                    disabled={addressLookupLoading || !addressLookup.trim()}
                    sx={{ mb: 1 }}
                  >
                    Look up
                  </Button>
                  {lastGeocoded && (
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Button
                        size="small"
                        startIcon={<MyLocation />}
                        onClick={handleSearchCenter}
                      >
                        Center on map
                      </Button>
                      <Button
                        size="small"
                        startIcon={<Place />}
                        onClick={handleSetAsFrom}
                      >
                        Set as From
                      </Button>
                      <Button
                        size="small"
                        startIcon={<Place />}
                        onClick={handleSetAsTo}
                      >
                        Set as To
                      </Button>
                    </Box>
                  )}
                </Card>
              </Box>

              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', flex: 1, minHeight: 0 }}>
                <Box sx={{ flex: '1 1 400px', minWidth: 0, minHeight: 400, position: 'relative' }}>
                  <InteractiveMap
                    locations={locations}
                    onLocationClick={handleEditLocation}
                    onMapDoubleClick={handleMapDoubleClick}
                    routeGeometry={routeGeometry}
                    searchCenter={searchCenter}
                    onSearchCenterApplied={handleSearchCenterApplied}
                    darkMap={mapStylePreference === 'auto' ? appDarkMode : mapStylePreference === 'dark'}
                  />
                  <Paper
                    elevation={2}
                    sx={{
                      position: 'absolute',
                      top: 8,
                      right: 8,
                      zIndex: 1000,
                      px: 1.5,
                      py: 1,
                      borderRadius: 1,
                      minWidth: 200,
                      overflow: 'visible',
                    }}
                  >
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      Map style
                    </Typography>
                    <Slider
                      value={valueFromMapStyle(mapStylePreference)}
                      onChange={(_, v) => setMapStylePreference(mapStyleFromValue(v))}
                      min={0}
                      max={2}
                      step={1}
                      marks={[
                        { value: 0, label: 'Light' },
                        { value: 1, label: 'Auto' },
                        { value: 2, label: 'Dark' },
                      ]}
                      valueLabelDisplay="off"
                      size="small"
                      sx={{
                        mt: 0.5,
                        width: 'calc(100% - 16px)',
                        mx: 1,
                        '& .MuiSlider-markLabel': { fontSize: '0.7rem', maxWidth: 48 },
                      }}
                    />
                  </Paper>
                </Box>
                {currentRoute && routeSteps.length > 0 && (
                  <Card variant="outlined" sx={{ width: 320, maxHeight: 500, overflow: 'auto' }}>
                    <Typography variant="subtitle2" sx={{ p: 1.5, fontWeight: 'bold' }}>
                      Turn-by-turn
                    </Typography>
                    <List dense>
                      {routeSteps.map((step, idx) => (
                        <ListItem key={idx}>
                          <ListItemText
                            primary={formatTurnInstruction(step)}
                            secondary={
                              step.distance > 0
                                ? formatDistance(step.distance)
                                : null
                            }
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Card>
                )}
              </Box>
            </Box>
          )}
          {activeTab === 1 && (
            <LocationsList
              locations={myLocations}
              onEdit={handleEditLocation}
              onDelete={handleDeleteLocation}
              isLoading={isLoading}
            />
          )}
          {activeTab === 2 && (
            <LocationsList
              locations={globalLocations}
              onEdit={handleEditLocation}
              onDelete={handleDeleteLocation}
              isLoading={isLoading}
              readOnly
            />
          )}
          {activeTab === 3 && (
            <Box>
              {savedRoutesLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                  <CircularProgress />
                </Box>
              ) : savedRoutes.length === 0 ? (
                <Typography color="text.secondary" sx={{ p: 4, textAlign: 'center' }}>
                  No saved routes. Get a route on the map and click "Save route".
                </Typography>
              ) : (
                <List>
                  {savedRoutes.map((route) => (
                    <ListItem
                      key={route.route_id}
                      secondaryAction={
                        <Box>
                          <IconButton
                            edge="end"
                            onClick={() => handleLoadSavedRoute(route)}
                            title="Load on map"
                          >
                            <PlayArrow />
                          </IconButton>
                          <IconButton
                            edge="end"
                            onClick={() => {
                              if (window.confirm('Delete this route?')) {
                                deleteRouteMutation.mutate(route.route_id);
                              }
                            }}
                          >
                            <Delete />
                          </IconButton>
                        </Box>
                      }
                    >
                      <ListItemText
                        primary={route.name}
                        secondary={`${(route.distance_meters / 1609.34).toFixed(1)} mi · ${Math.round(route.duration_seconds / 60)} min`}
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </Box>
          )}
        </Box>
      </Card>

      <LocationDialog
        open={dialogOpen}
        onClose={() => {
          setDialogOpen(false);
          setEditingLocation(null);
        }}
        onSave={handleSaveLocation}
        location={editingLocation}
      />

      <Dialog open={saveDialogOpen} onClose={() => setSaveDialogOpen(false)}>
        <DialogTitle>Save route</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Route name"
            fullWidth
            value={saveRouteName}
            onChange={(e) => setSaveRouteName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleSaveRoute}
            disabled={!saveRouteName.trim() || saveRouteMutation.isLoading}
            variant="contained"
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MapPage;
