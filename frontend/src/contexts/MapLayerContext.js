/**
 * Map layer and style context for composable map views.
 * Holds active style preference (persisted to localStorage) and optional extra data layers.
 */

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';

const STORAGE_KEY = 'mapStyle';

const THEMES = ['light', 'dark', 'auto', 'minimal', 'white', 'black'];

const MapLayerContext = createContext(null);

export function MapLayerProvider({ children, defaultStyle = 'auto' }) {
  const [stylePreference, setStylePreferenceState] = useState(defaultStyle);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored && THEMES.includes(stored)) setStylePreferenceState(stored);
    } catch (e) {
      // ignore
    }
  }, []);

  const setStylePreference = useCallback((value) => {
    setStylePreferenceState(value);
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch (e) {
      // ignore
    }
  }, []);

  const [extraLayers, setExtraLayers] = useState([]);
  const registerLayer = useCallback((id, config) => {
    setExtraLayers((prev) => {
      const next = prev.filter((l) => l.id !== id);
      if (config) next.push({ id, ...config });
      return next;
    });
  }, []);
  const unregisterLayer = useCallback((id) => {
    setExtraLayers((prev) => prev.filter((l) => l.id !== id));
  }, []);

  const value = {
    stylePreference,
    setStylePreference,
    extraLayers,
    registerLayer,
    unregisterLayer,
    themes: THEMES
  };

  return (
    <MapLayerContext.Provider value={value}>
      {children}
    </MapLayerContext.Provider>
  );
}

export function useMapLayer() {
  const ctx = useContext(MapLayerContext);
  return ctx || {
    stylePreference: 'auto',
    setStylePreference: () => {},
    extraLayers: [],
    registerLayer: () => {},
    unregisterLayer: () => {},
    themes: THEMES
  };
}
