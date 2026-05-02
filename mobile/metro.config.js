const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

const projectRoot = __dirname;

const config = getDefaultConfig(projectRoot);

// Allow bundling shared web PWA marks (splash / branding) from the monorepo frontend tree.
config.watchFolders = [
  ...(config.watchFolders || []),
  path.resolve(projectRoot, '..', 'frontend', 'public', 'images'),
];

module.exports = config;
