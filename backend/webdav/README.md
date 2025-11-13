# WebDAV Service for Document Sync

## Overview

Provides WebDAV access to files in the `uploads/` directory, enabling mobile synchronization and remote file editing via WebDAV clients.

## Features

- **Simple Filesystem Access**: Serves files directly from `uploads/` directory
- **Actual Folder Structure**: Preserves real file/folder hierarchy (no virtual categories)
- **User Authentication**: Integrates with Bastion's user authentication system  
- **All File Types**: Works with any document type, not just .org files
- **Mobile Compatible**: Works with mobile apps that support WebDAV (OrgMode apps, file managers, etc.)

## Architecture

### Components

1. **`webdav_server.py`**: Main entry point, starts the WsgiDAV server
2. **`simple_filesystem_provider.py`**: DAV provider using WsgiDAV's built-in FilesystemProvider
3. **`auth_provider.py`**: Custom authentication using Bastion user database (psycopg2)
4. **`config.py`**: WsgiDAV server configuration

### URL Structure

```
https://bastion.pilbeams.net/dav/                         <- Root (uploads/ directory)
https://bastion.pilbeams.net/dav/my_folder/              <- Actual folder
https://bastion.pilbeams.net/dav/my_folder/document.org  <- Actual file
https://bastion.pilbeams.net/dav/web_sources/            <- Another folder
```

## Client Configuration

### OrgMode Mobile Apps

**URL:** `https://bastion.pilbeams.net/dav/`  
**Username:** Your Bastion username  
**Password:** Your Bastion password  
**Protocol:** HTTPS with Basic Authentication

### File Managers

Most mobile/desktop file managers with WebDAV support can connect using the same credentials.

## Docker Integration

The WebDAV service runs as a separate container:
- **Service:** `webdav`
- **Internal Port:** `8001`
- **External Access:** Via nginx reverse proxy at `/dav/`
- **Dependencies:** PostgreSQL database for authentication

## Authentication

Uses **synchronous psycopg2** connections to avoid event loop conflicts with WsgiDAV's synchronous architecture.

Each request authenticates against the `users` table in PostgreSQL using bcrypt password hashing.

## File Access

Currently serves ALL files in `uploads/` directory. Future enhancement could filter by `user_id` for multi-tenant isolation.
