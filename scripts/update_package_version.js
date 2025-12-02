/**
 * Update package.json version from VERSION file
 * Run this before building frontend
 */

const fs = require('fs');
const path = require('path');

const getVersion = require('./get_version');

const packageJsonPath = path.join(__dirname, '..', 'frontend', 'package.json');

try {
  const version = getVersion();
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  
  packageJson.version = version;
  
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2) + '\n');
  console.log(`Updated frontend/package.json version to ${version}`);
} catch (error) {
  console.error('Error updating package.json version:', error);
  process.exit(1);
}

