"""
OrgMode WebDAV Provider - Roosevelt's Mobile Sync File Provider

Provides WebDAV access to OrgMode files stored in Plato's database.
Supports read/write operations for mobile sync with Orgzly and beorg.
"""

import logging
import asyncio
import os
import io
from datetime import datetime
from typing import List, Optional
from wsgidav.dav_provider import DAVProvider, DAVCollection, DAVNonCollection
from wsgidav.dav_error import DAVError, HTTP_FORBIDDEN, HTTP_NOT_FOUND

logger = logging.getLogger(__name__)


class OrgModeResource(DAVNonCollection):
    """
    Represents a single org-mode file in the WebDAV hierarchy.
    Maps to a free_form_note in the database.
    """
    
    def __init__(self, path, environ, note_data, db_pool):
        super().__init__(path, environ)
        self.note_data = note_data
        self.db_pool = db_pool
        self.note_id = note_data['note_id']
        
    def get_content_length(self):
        """Return the byte length of the file content"""
        content = self.note_data.get('content', '')
        return len(content.encode('utf-8'))
    
    def get_content_type(self):
        """Return the MIME type"""
        return "text/x-org"
    
    def get_creation_date(self):
        """Return the creation date as Unix timestamp"""
        created_at = self.note_data.get('created_at')
        if created_at:
            return created_at.timestamp()
        return None
    
    def get_last_modified(self):
        """Return last modified date as Unix timestamp"""
        updated_at = self.note_data.get('updated_at')
        if updated_at:
            return updated_at.timestamp()
        return None
    
    def get_content(self):
        """Return file content as a file-like object"""
        content = self.note_data.get('content', '')
        return io.BytesIO(content.encode('utf-8'))
    
    def begin_write(self, content_type=None):
        """Begin writing to the file - returns a file-like object"""
        return io.BytesIO()
    
    def end_write(self, with_errors):
        """Called when writing is complete"""
        if with_errors:
            logger.error(f"❌ Errors writing to org file: {self.path}")
        else:
            logger.info(f"✅ Successfully wrote to org file: {self.path}")


class OrgModeFolderCollection(DAVCollection):
    """
    Represents a folder/category of org-mode files.
    """
    
    def __init__(self, path, environ, folder_name, db_pool, user_id):
        super().__init__(path, environ)
        self.folder_name = folder_name
        self.db_pool = db_pool
        self.user_id = user_id
    
    def get_member_names(self):
        """Return list of org files in this folder"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._get_member_names())
        finally:
            loop.close()
    
    async def _get_member_names(self):
        """Async method to get org file names from database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Query notes for this category/folder
                rows = await conn.fetch("""
                    SELECT note_id, title
                    FROM free_form_notes
                    WHERE user_id = $1
                      AND (category = $2 OR ($2 = 'inbox' AND category IS NULL))
                    ORDER BY title
                """, self.user_id, self.folder_name)
                
                # Return filenames (title + .org extension)
                return [self._sanitize_filename(row['title']) + '.org' for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error getting member names: {e}")
            return []
    
    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title to be a valid filename"""
        # Remove/replace invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            title = title.replace(char, '_')
        return title.strip() or "untitled"
    
    def get_member(self, name):
        """Get a specific org file by name"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._get_member(name))
        finally:
            loop.close()
    
    async def _get_member(self, name):
        """Async method to get a specific org file"""
        try:
            # Extract title from filename (remove .org extension)
            if not name.endswith('.org'):
                raise DAVError(HTTP_NOT_FOUND, f"File must have .org extension: {name}")
            
            title = name[:-4]  # Remove .org extension
            
            async with self.db_pool.acquire() as conn:
                # Query note by title and category
                row = await conn.fetchrow("""
                    SELECT note_id, title, content, created_at, updated_at,
                           note_date, tags, category, metadata_json
                    FROM free_form_notes
                    WHERE user_id = $1
                      AND title = $2
                      AND (category = $3 OR ($3 = 'inbox' AND category IS NULL))
                """, self.user_id, title, self.folder_name)
                
                if not row:
                    raise DAVError(HTTP_NOT_FOUND, f"Org file not found: {name}")
                
                # Return OrgModeResource
                note_data = dict(row)
                resource_path = os.path.join(self.path, name)
                return OrgModeResource(resource_path, self.environ, note_data, self.db_pool)
                
        except DAVError:
            raise
        except Exception as e:
            logger.error(f"❌ Error getting member {name}: {e}")
            raise DAVError(HTTP_NOT_FOUND, f"Error accessing file: {name}")


class OrgModeRootCollection(DAVCollection):
    """
    Root collection that shows folders/categories of org files.
    """
    
    def __init__(self, path, environ, db_pool):
        super().__init__(path, environ)
        self.db_pool = db_pool
        # Get user_id from authentication
        self.user_id = environ.get('webdav.auth.user_name', 'unknown')
    
    def get_member_names(self):
        """Return list of folders/categories"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._get_member_names())
        finally:
            loop.close()
    
    async def _get_member_names(self):
        """Async method to get folder/category names"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get distinct categories for this user
                rows = await conn.fetch("""
                    SELECT DISTINCT 
                        COALESCE(category, 'inbox') as folder_name
                    FROM free_form_notes
                    WHERE user_id = $1
                    ORDER BY folder_name
                """, self.user_id)
                
                return [row['folder_name'] for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error getting folder names: {e}")
            return ['inbox']  # At least return inbox
    
    def get_member(self, name):
        """Get a folder collection"""
        folder_path = os.path.join(self.path, name)
        return OrgModeFolderCollection(folder_path, self.environ, name, self.db_pool, self.user_id)


class OrgModeDAVProvider(DAVProvider):
    """
    WebDAV provider for OrgMode files stored in Plato's database.
    
    Hierarchy:
    /orgmode/              <- Root
    /orgmode/inbox/        <- Category folder
    /orgmode/inbox/todo.org  <- Org file
    """
    
    def __init__(self, db_pool):
        """
        Initialize the provider.
        
        Args:
            db_pool: Database connection pool
        """
        super().__init__()
        self.db_pool = db_pool
        logger.info("📁 BULLY! OrgModeDAVProvider initialized")
    
    def get_resource_inst(self, path, environ):
        """
        Return a DAVResource object for the given path.
        
        Args:
            path: The requested path
            environ: WSGI environment
            
        Returns:
            DAVResource: The appropriate resource object
        """
        # Normalize path
        path = path.rstrip('/')
        
        # Root collection
        if path == '' or path == '/':
            return OrgModeRootCollection('/', environ, self.db_pool)
        
        # Split path into parts
        parts = [p for p in path.split('/') if p]
        
        if len(parts) == 1:
            # Folder/category level
            return OrgModeFolderCollection(path, environ, parts[0], self.db_pool, 
                                          environ.get('webdav.auth.user_name', 'unknown'))
        elif len(parts) == 2:
            # File level - handled by folder collection
            folder = OrgModeFolderCollection(f"/{parts[0]}", environ, parts[0], 
                                            self.db_pool, environ.get('webdav.auth.user_name', 'unknown'))
            return folder.get_member(parts[1])
        else:
            raise DAVError(HTTP_NOT_FOUND, f"Path not found: {path}")



