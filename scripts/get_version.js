/**
 * Read version from root VERSION file
 * Used by frontend build process
 */

const fs = require('fs');
const path = require('path');

function getVersion() {
  const versionFile = path.join(__dirname, '..', 'VERSION');
  try {
    const version = fs.readFileSync(versionFile, 'utf8').trim();
    return version;
  } catch (error) {
    console.warn('Could not read VERSION file, using default');
    return '0.11.0';
  }
}

module.exports = getVersion;

