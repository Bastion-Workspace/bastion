/**
 * Update package.json version from VERSION file if scripts are available
 * Gracefully skips if scripts directory is not available (e.g., in Docker build)
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const scriptsPath = path.join(__dirname, '..', '..', 'scripts', 'update_package_version.js');

if (fs.existsSync(scriptsPath)) {
  try {
    // Execute the script using node
    execSync(`node "${scriptsPath}"`, { stdio: 'inherit' });
  } catch (error) {
    console.warn('Could not run version update script:', error.message);
    process.exit(0); // Don't fail the build
  }
} else {
  console.log('Version update script not available (expected in Docker builds), skipping...');
  process.exit(0);
}

