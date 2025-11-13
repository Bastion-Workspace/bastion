"""
Filesystem-Based OrgMode WebDAV Provider - Document Integration (Synchronous)

Provides WebDAV access to .org files stored in the uploads/ directory,
using document_metadata table for organization and user isolation.

Uses psycopg2 (synchronous) instead of asyncpg to avoid event loop conflicts.
"""

import logging
import os
import io
from datetime import datetime
from typing import List, Optional, Dict, Any
import psycopg2
from wsgidav.dav_provider import DAVProvider, DAVCollection, DAVNonCollection
from wsgidav.dav_error import DAVError, HTTP_FORBIDDEN, HTTP_NOT_FOUND, HTTP_CONFLICT

logger = logging.getLogger(__name__)


class OrgFileResource(DAVNonCollection):
    """
    Represents a single .org file from the uploads directory.
    """
    
    def __init__(self, path, environ, document_info, file_path):
        super().__init__(path, environ)
        self.document_info = document_info
        self.file_path = file_path
        self.document_id = document_info['document_id']
        
    def get_content_length(self):
        """Return file size from filesystem"""
        try:
            if os.path.exists(self.file_path):
                return os.path.getsize(self.file_path)
        except Exception as e:
            logger.error(f"❌ Error getting file size: {e}")
        return 0
    
    def get_content_type(self):
        """Return MIME type for org files"""
        return "text/x-org"
    
    def get_creation_date(self):
        """Return creation date from database"""
        upload_date = self.document_info.get('upload_date')
        if upload_date:
            return upload_date.timestamp()
        return None
    
    def get_last_modified(self):
        """Return last modified date from filesystem"""
        try:
            if os.path.exists(self.file_path):
                return os.path.getmtime(self.file_path)
        except Exception as e:
            logger.error(f"❌ Error getting modified time: {e}")
        return None
    
    def get_content(self):
        """Return file content as file-like object"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'rb') as f:
                    content = f.read()
                return io.BytesIO(content)
            else:
                logger.error(f"❌ File not found: {self.file_path}")
                return io.BytesIO(b"")
        except Exception as e:
            logger.error(f"❌ Error reading file: {e}")
            return io.BytesIO(b"")


class OrgModeFolderCollection(DAVCollection):
    """
    Represents a virtual folder/category of org files.
    Groups documents by category from document_metadata.
    """
    
    def __init__(self, path, environ, category, db_config, user_id, uploads_dir):
        super().__init__(path, environ)
        self.category = category
        self.db_config = db_config
        self.user_id = user_id
        self.uploads_dir = uploads_dir
    
    def get_member_names(self):
        """Return list of org filenames in this category"""
        conn = None
        try:
            logger.info(f"📂 Listing files for category: '{self.category}', user_id: '{self.user_id}'")
            
            # Synchronous database connection
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            
            cur = conn.cursor()
            
            # Query documents with .org extension
            query = """
                SELECT filename, category
                FROM document_metadata
                WHERE user_id = %s
                  AND filename LIKE '%.org'
                  AND (category = %s OR (%s = 'uncategorized' AND category IS NULL))
                ORDER BY filename
            """
            cur.execute(query, (self.user_id, self.category, self.category))
            rows = cur.fetchall()
            cur.close()
            
            logger.info(f"📂 Found {len(rows)} org files in category '{self.category}'")
            for row in rows:
                logger.info(f"   - {row[0]} (category: {row[1]})")
            
            return [row[0] for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error getting org file names: {e}")
            logger.exception("Full traceback:")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_member(self, name):
        """Get a specific org file by filename"""
        conn = None
        try:
            if not name.endswith('.org'):
                raise DAVError(HTTP_NOT_FOUND, f"Not an org file: {name}")
            
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            
            cur = conn.cursor()
            
            # Query document by filename
            query = """
                SELECT document_id, filename, title, category, 
                       file_path, upload_date, user_id
                FROM document_metadata
                WHERE user_id = %s
                  AND filename = %s
                  AND filename LIKE '%.org'
            """
            cur.execute(query, (self.user_id, name))
            row = cur.fetchone()
            cur.close()
            
            if not row:
                raise DAVError(HTTP_NOT_FOUND, f"Org file not found: {name}")
            
            # Convert tuple to dict
            doc_info = {
                'document_id': row[0],
                'filename': row[1],
                'title': row[2],
                'category': row[3],
                'file_path': row[4],
                'upload_date': row[5],
                'user_id': row[6]
            }
            
            # Construct file path in uploads directory
            file_path = os.path.join(self.uploads_dir, doc_info['filename'])
            
            if not os.path.exists(file_path):
                logger.error(f"❌ File exists in DB but not on disk: {file_path}")
                raise DAVError(HTTP_NOT_FOUND, f"File not found on disk: {name}")
            
            # Return resource
            resource_path = os.path.join(self.path, name)
            return OrgFileResource(resource_path, self.environ, doc_info, file_path)
                
        except DAVError:
            raise
        except Exception as e:
            logger.error(f"❌ Error getting org file {name}: {e}")
            raise DAVError(HTTP_NOT_FOUND, f"Error accessing file: {name}")
        finally:
            if conn:
                conn.close()


class OrgModeRootCollection(DAVCollection):
    """
    Root collection showing categories/folders of org files.
    """
    
    def __init__(self, path, environ, db_config, uploads_dir):
        super().__init__(path, environ)
        self.db_config = db_config
        self.uploads_dir = uploads_dir
        # Get authenticated user_id (set by auth controller)
        self.user_id = environ.get('webdav.auth.user_id', 'unknown')
    
    def get_member_names(self):
        """Return list of category folders"""
        conn = None
        try:
            logger.info(f"📂 ROOT: Listing categories for user_id: '{self.user_id}'")
            
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            
            cur = conn.cursor()
            
            # Get distinct categories for this user's org files
            query = """
                SELECT DISTINCT 
                    COALESCE(category, 'uncategorized') as folder_name
                FROM document_metadata
                WHERE user_id = %s
                  AND filename LIKE '%.org'
                ORDER BY folder_name
            """
            cur.execute(query, (self.user_id,))
            rows = cur.fetchall()
            cur.close()
            
            logger.info(f"📂 ROOT: Found {len(rows)} categories")
            for row in rows:
                logger.info(f"   - Category: {row[0]}")
            
            if not rows:
                # Return at least uncategorized folder
                logger.info(f"📂 ROOT: No categories found, returning default 'uncategorized'")
                return ['uncategorized']
            
            return [row[0] for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error getting category names: {e}")
            logger.exception("Full traceback:")
            return ['uncategorized']
        finally:
            if conn:
                conn.close()
    
    def get_member(self, name):
        """Get a category/folder collection"""
        folder_path = os.path.join(self.path, name)
        return OrgModeFolderCollection(
            folder_path, self.environ, name, 
            self.db_config, self.user_id, self.uploads_dir
        )


class FilesystemOrgModeDAVProvider(DAVProvider):
    """
    WebDAV provider for OrgMode files stored in uploads/ directory.
    Reads metadata from document_metadata table.
    """
    
    def __init__(self, db_config, uploads_dir):
        """
        Initialize provider.
        
        Args:
            db_config: Database connection config dict
            uploads_dir: Path to uploads directory (e.g., '/app/uploads')
        """
        super().__init__()
        self.db_config = db_config
        self.uploads_dir = uploads_dir
        logger.info(f"📁 FilesystemOrgModeDAVProvider initialized")
        logger.info(f"📂 Serving org files from: {uploads_dir}")
    
    def get_resource_inst(self, path, environ):
        """
        Return DAVResource for the given path.
        """
        logger.info(f"📂 get_resource_inst called with path: '{path}'")
        logger.info(f"📂 user_id from environ: {environ.get('webdav.auth.user_id', 'NOT SET')}")
        
        # Strip /orgmode prefix if present (from nginx routing)
        if path.startswith('/orgmode'):
            path = path[8:]  # Remove '/orgmode'
            logger.info(f"📂 Stripped /orgmode prefix, new path: '{path}'")
        
        # Normalize path
        path = path.rstrip('/')
        logger.info(f"📂 Normalized path: '{path}'")
        
        # Root collection
        if path == '' or path == '/':
            logger.info(f"📂 Returning root collection for path: '{path}'")
            return OrgModeRootCollection('/', environ, self.db_config, self.uploads_dir)
        
        # Split path into parts
        parts = [p for p in path.split('/') if p]
        
        if len(parts) == 1:
            # Category/folder level
            return OrgModeFolderCollection(
                path, environ, parts[0], self.db_config, 
                environ.get('webdav.auth.user_id', 'unknown'),
                self.uploads_dir
            )
        elif len(parts) == 2:
            # File level - delegate to folder collection
            folder = OrgModeFolderCollection(
                f"/{parts[0]}", environ, parts[0], 
                self.db_config,
                environ.get('webdav.auth.user_id', 'unknown'),
                self.uploads_dir
            )
            return folder.get_member(parts[1])
        else:
            raise DAVError(HTTP_NOT_FOUND, f"Path not found: {path}")

