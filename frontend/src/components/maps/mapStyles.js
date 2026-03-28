/**
 * MapLibre GL style definitions for PMTiles basemap.
 * Uses @protomaps/basemaps for layer definitions (light, dark, minimal).
 * All styles point at the same PMTiles source URL.
 */

import { layers, namedFlavor } from '@protomaps/basemaps';

const GLYPHS = 'https://protomaps.github.io/basemaps-assets/fonts/{fontstack}/{range}.pbf';
const SPRITE_BASE = 'https://protomaps.github.io/basemaps-assets/sprites/v4';

const FLAVORS = {
  light: 'light',
  dark: 'dark',
  minimal: 'grayscale',
  white: 'white',
  black: 'black'
};

/**
 * Build a MapLibre GL style object for the given PMTiles URL and theme.
 * @param {string} pmtilesUrl - Full URL to the PMTiles file (e.g. pmtiles://https://example.com/pmtiles/world.pmtiles)
 * @param {string} theme - One of: light, dark, minimal, white, black
 * @returns {object} MapLibre GL style spec (version 8)
 */
export function getMapStyle(pmtilesUrl, theme = 'light') {
  const flavorName = FLAVORS[theme] || 'light';
  const flavor = namedFlavor(flavorName);
  const spriteUrl = `${SPRITE_BASE}/${flavorName}`;

  return {
    version: 8,
    glyphs: GLYPHS,
    sprite: spriteUrl,
    sources: {
      protomaps: {
        type: 'vector',
        url: pmtilesUrl.startsWith('pmtiles://') ? pmtilesUrl : `pmtiles://${pmtilesUrl}`,
        attribution: '<a href="https://protomaps.com">Protomaps</a> © <a href="https://openstreetmap.org">OpenStreetMap</a>'
      }
    },
    layers: layers('protomaps', flavor, { lang: 'en' })
  };
}

/**
 * Get the list of available style theme names.
 */
export function getMapStyleThemes() {
  return Object.keys(FLAVORS);
}
