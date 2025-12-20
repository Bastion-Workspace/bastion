# Version Management

## Central Version Control

The project uses a single source of truth for version numbers: the `VERSION` file in the root directory.

## How It Works

### Root VERSION File
- Location: `/VERSION`
- Contains: Single line with version number (e.g., `0.11.0`)
- **This is the only file you need to edit to update the version**

### Backend (Python)
All Python modules import the version from `backend/version.py`:
- `backend/version.py` - Reads from root `VERSION` file
- `backend/main.py` - FastAPI app version
- `backend/api/orchestrator_chat_api.py` - API version
- `backend/mcp/__init__.py` - MCP module version
- `backend/webdav/__init__.py` - WebDAV module version

**Usage in Python:**
```python
from version import __version__
# Use __version__ anywhere
```

### Frontend (JavaScript/React)
The frontend `package.json` version is automatically updated during build:
- `scripts/get_version.js` - Reads from root `VERSION` file
- `scripts/update_package_version.js` - Updates `package.json` before build
- `frontend/package.json` - Automatically synced via `prebuild` script

**Usage:**
- Version is automatically synced when running `npm run build`
- Or manually run: `node scripts/update_package_version.js`

## Updating the Version

**To update the version for the entire project:**

1. Edit `/VERSION` file:
   ```
   0.11.0  →  0.12.0
   ```

2. For frontend, the version will auto-update on next build, or run:
   ```bash
   node scripts/update_package_version.js
   ```

3. Backend will automatically use the new version (no rebuild needed for Python)

## Benefits

- ✅ Single source of truth
- ✅ No version drift between components
- ✅ Easy to update (one file)
- ✅ Automatic synchronization
- ✅ Consistent versioning across all services

