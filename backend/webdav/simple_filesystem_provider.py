"""
Simple Filesystem WebDAV Provider

Serves user-specific subdirectories from uploads/ based on authenticated user.
Structure: uploads/Users/<username>/ for each user
"""

import logging
import os
import hashlib
from pathlib import Path
from wsgidav.dav_provider import DAVProvider
from wsgidav.fs_dav_provider import FilesystemProvider, FolderResource, FileResource

logger = logging.getLogger(__name__)


class MD5FileResource(FileResource):
    """
    Content-hash based WebDAV file resource
    
    Custom FileResource that generates ETags based on MD5 content hash
    instead of mtime+size. This provides more reliable sync conflict detection.
    
    Benefits:
    - Accurate change detection (content-based, not timestamp-based)
    - Prevents false "file unchanged" when content actually changed
    - Better conflict resolution for sync clients
    
    Trade-offs:
    - Slower ETag generation (must read entire file)
    - Cached after first read for performance
    """
    
    def __init__(self, path, environ, file_path):
        super().__init__(path, environ, file_path)
        # **CRITICAL:** Parent class doesn't store file_path, so we must!
        self.file_path = file_path
        self._etag_cache = None
        self._etag_cache_mtime = None
    
    def get_etag(self):
        """
        Generate ETag based on MD5 hash of file content.
        
        Falls back to mtime-based ETag if:
        - File is too large (> 50MB) - avoid performance hit
        - File read fails
        
        Content hashing for reliable sync.
        """
        try:
            file_stat = os.stat(self.file_path)
            file_mtime = file_stat.st_mtime
            file_size = file_stat.st_size
            
            # Check cache - reuse if file hasn't been modified
            if self._etag_cache and self._etag_cache_mtime == file_mtime:
                logger.debug(f"📦 ETag cache hit for: {self.name}")
                return self._etag_cache
            
            # Large files (>100MB): prefer mtime-based ETag for performance
            # Still use MD5 for files up to 100MB to maintain consistency
            MAX_HASH_SIZE = 100 * 1024 * 1024  # 100 MB (increased from 50MB)
            if file_size > MAX_HASH_SIZE:
                logger.debug(f"📦 Large file ({file_size} bytes), using mtime-size ETag: {self.name}")
                # **CRITICAL:** Return ETag WITHOUT quotes - WsgiDAV adds them!
                etag = f"{int(file_mtime * 1000)}-{file_size}"
                # Cache it
                self._etag_cache = etag
                self._etag_cache_mtime = file_mtime
                return etag
            
            # Calculate MD5 hash of file content
            logger.debug(f"🔐 Calculating MD5 ETag for: {self.name} ({file_size} bytes)")
            md5_hash = hashlib.md5()
            
            with open(self.file_path, 'rb') as f:
                # Read in chunks to avoid memory issues
                for chunk in iter(lambda: f.read(8192), b''):
                    md5_hash.update(chunk)
            
            # **CRITICAL:** Return MD5 hash WITHOUT quotes - WsgiDAV adds them!
            etag = md5_hash.hexdigest()
            
            # Cache the result
            self._etag_cache = etag
            self._etag_cache_mtime = file_mtime
            
            logger.debug(f"✅ MD5 ETag generated: {self.name} → {etag}")
            return etag
            
        except Exception as e:
            logger.error(f"❌ Failed to generate MD5 ETag for {self.name}: {e}")
            logger.error(f"❌ Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            # **CRITICAL:** Use parent's default ETag method as fallback
            # This ensures we always return a valid, properly-formatted ETag
            try:
                default_etag = super().get_etag()
                logger.warning(f"⚠️ Using default ETag fallback: {self.name} → {default_etag}")
                return default_etag
            except Exception as fallback_error:
                # Absolute last resort: return None (no ETag)
                logger.error(f"❌ Even fallback ETag failed: {fallback_error}")
                return None


class MD5FilesystemProvider(FilesystemProvider):
    """
    Custom FilesystemProvider that uses MD5-based ETags for files.
    
    Use content hashing instead of timestamp-only heuristics.
    
    Use MD5-based ETags consistently for this provider
    We pass fs_opts to ensure proper configuration.
    """
    
    def __init__(self, root_folder_path):
        # **CRITICAL:** Pass fs_opts to configure the provider
        # This ensures our ETag customization is respected
        fs_opts = {
            "follow_symlinks": False,
            "re_encode_path": None,
        }
        super().__init__(root_folder_path, fs_opts=fs_opts)
    
    def _loc_to_file_path(self, path, environ=None):
        """Convert WebDAV path to filesystem path"""
        return super()._loc_to_file_path(path, environ)
    
    def get_resource_inst(self, path, environ):
        """
        Return a MD5FileResource for files, FolderResource for folders.
        
        This is where we inject our custom MD5-based ETag logic!
        """
        # Get the filesystem path
        file_path = self._loc_to_file_path(path, environ)
        
        # Verbose logging for resource resolution (mobile clients)
        logger.debug(f"🔍 MD5Provider.get_resource_inst: path='{path}', file_path='{file_path}'")
        
        # Check if path exists
        if not os.path.exists(file_path):
            logger.debug(f"⚠️ Path does not exist: {file_path}")
            return None
        
        # Return folder or file resource
        if os.path.isdir(file_path):
            logger.debug(f"📁 Returning FolderResource for: {path}")
            return FolderResource(path, environ, file_path)
        else:
            # Use MD5FileResource instead of default FileResource
            logger.debug(f"📄 Returning MD5FileResource for: {path}")
            return MD5FileResource(path, environ, file_path)


class UserFilteredFilesystemProvider(DAVProvider):
    """
    WebDAV provider that serves user-specific subdirectories.
    
    Each user sees their own directory: uploads/Users/<username>/
    Admins additionally see: uploads/Global/
    
    Authentication provides user_id which we use to determine the username
    and serve the appropriate directory.
    """
    
    def __init__(self, root_path, db_config):
        """
        Initialize provider.
        
        Args:
            root_path: Path to uploads directory (e.g., '/app/uploads')
            db_config: Database config for user lookup
        """
        # CRITICAL: Initialize our attributes FIRST, before calling super().__init__()
        # because the parent class will call our mount_path setter which needs _provider_cache
        self._mount_path = "/"  # Mount at root since nginx strips /dav prefix
        self._provider_cache = {}  # MUST be initialized before super().__init__()
        self.base_root = Path(root_path)
        self.db_config = db_config
        
        # Now initialize DAVProvider parent (this will call mount_path setter)
        super().__init__()
        
        # Set other required attributes
        self.readonly = False
        # share_path must be empty to avoid double slashes in URLs
        # mount_path is "/" so share_path should be ""
        self.share_path = ""
        
        logger.info(f"📁 UserFilteredFilesystemProvider initialized")
        logger.debug(f"📂 Base path: {root_path}")
    
    @property
    def mount_path(self):
        """Get mount_path - ensures it's never None or empty"""
        # Treat None and empty string as root path "/"
        if not self._mount_path:  # Catches None, "", etc.
            return "/"
        return self._mount_path
    
    @mount_path.setter
    def mount_path(self, value):
        """
        Set mount_path and propagate to all cached child providers.
        
        WsgiDAV calls this after instantiation to set the mount path.
        We need to propagate it to all child FilesystemProvider instances.
        
        Treat None and empty string as equivalent where noted
        """
        # **TRUST BUST!** WsgiDAV might set this to "" (empty string) OR None!
        # We want "/" in either case (nginx strips /dav prefix)
        self._mount_path = value if value else "/"
        logger.debug(f"📂 Mount path set to: '{self._mount_path}' (received: '{value}')")
        
        # Propagate to all cached child providers (if they exist yet)
        # Use getattr for defensive initialization handling
        provider_cache = getattr(self, '_provider_cache', {})
        if provider_cache:
            for user_id, provider in provider_cache.items():
                provider.mount_path = self._mount_path
                logger.debug(f"📂 Updated mount_path for cached provider (user: {user_id})")
    
    def _get_username_from_db_sync(self, user_id: str) -> str:
        """Get username from database (synchronous)"""
        import psycopg2
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            cur = conn.cursor()
            cur.execute("SELECT username FROM users WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            cur.close()
            username = row[0] if row else user_id
            logger.debug(f"📂 Resolved user_id {user_id} to username: {username}")
            return username
        except Exception as e:
            logger.error(f"❌ Failed to get username for {user_id}: {e}")
            return user_id
        finally:
            if conn:
                conn.close()
    
    def _get_user_provider(self, user_id: str, environ=None) -> FilesystemProvider:
        """
        Get or create a FilesystemProvider for a specific user.
        
        Args:
            user_id: The user identifier
            environ: WSGI environment dict (optional, for logging)
            
        Returns:
            FilesystemProvider instance scoped to user's directory
        """
        cache_key = user_id
        
        if cache_key not in self._provider_cache:
            # Get username from database
            username = self._get_username_from_db_sync(user_id)
            
            # Construct user-specific root path
            user_root = self.base_root / "Users" / username
            
            # Ensure user directory exists
            user_root.mkdir(parents=True, exist_ok=True)
            
            user_root_str = str(user_root)
            logger.debug(f"📂 Creating NEW MD5FilesystemProvider for user '{username}'")
            logger.debug(f"📂 User root directory: {user_root_str}")
            
            # Create a fresh MD5FilesystemProvider instance for this user
            # This provider will handle ALL filesystem operations for this user
            # MD5-based ETags for sync conflict detection
            provider = MD5FilesystemProvider(user_root_str)

            # Child providers use empty mount_path
            # nginx strips /dav prefix, provider at "/" (parent), child at "" (empty)
            # Child: mount_path="" + resource path "/OrgMode/file.org" = "/OrgMode/file.org" ✅
            # Clients add their base URL (/dav) back: /dav + /OrgMode/file.org = /dav/OrgMode/file.org
            # NOTE: mount_path="/" would cause DOUBLE SLASH: "/" + "/OrgMode" = "//OrgMode" ❌
            provider.mount_path = ""
            provider.share_path = ""
            logger.debug(f"📂 Set child provider mount_path='' (empty) and share_path='' (clean hrefs)")
            
            # Store in cache
            self._provider_cache[cache_key] = provider
            logger.debug(f"✅ Provider cached for user: {username}")
        
        # Always return the cached provider
        provider = self._provider_cache[cache_key]
        
        # Ensure child provider keeps '' (empty) as mount path
        if provider.mount_path != "":
            logger.warning(f"⚠️ Provider mount_path changed! Restoring to '' (was: '{provider.mount_path}')")
            provider.mount_path = ""
            
        return provider
    
    def get_resource_inst(self, path, environ):
        """
        Return DAVResource for the given path, scoped to user's directory.
        
        This is the key method that WsgiDAV calls for every request.
        We delegate to a user-specific FilesystemProvider instance.
        
        CRITICAL: Resources MUST have the correct provider reference
        so that operations like create_collection() work properly.
        """
        user_id = environ.get("webdav.auth.user_id", "unknown")

        # nginx strips /dav; mount provider at /
        # nginx: /dav/OrgMode → (strips /dav) → WsgiDAV: /OrgMode
        # Provider at "/" receives clean paths like /OrgMode
        # hrefs generated: /OrgMode/file.org (clients add /dav base back)
        
        logger.debug(f"📂 ========== WebDAV Request ==========")
        logger.debug(f"📂 User: {user_id}")
        logger.debug(f"📂 Path received: '{path}'")
        logger.debug(f"📂 HTTP Method: {environ.get('REQUEST_METHOD', 'N/A')}")
        logger.debug(f"📂 User-Agent: {environ.get('HTTP_USER_AGENT', 'N/A')}")
        logger.debug(f"📂 Depth: {environ.get('HTTP_DEPTH', 'N/A')}")
        
        # Get user-specific provider
        provider = self._get_user_provider(user_id, environ)
        
        logger.debug(f"📂 Provider root_folder_path: {provider.root_folder_path}")
        logger.debug(f"📂 Provider mount_path: {provider.mount_path}")
        logger.debug(f"📂 Provider share_path: {provider.share_path}")
        
        # Adjust WSGI environ so resources resolve under child provider
        original_provider = environ.get('wsgidav.provider')
        environ['wsgidav.provider'] = provider
        
        try:
            # Get resource from child (child has mount_path="" and share_path="")
            resource = provider.get_resource_inst(path, environ)
        finally:
            # Restore original provider
            if original_provider is not None:
                environ['wsgidav.provider'] = original_provider
            else:
                environ.pop('wsgidav.provider', None)
        
        if resource:
            # Do not modify resource.path; paths must start with /
            # Child provider has mount_path="" so: "" + "" + "/OrgMode" = "/OrgMode" ✅
            # 
            # Log resolved href for XML response debugging
            try:
                ref_url = resource.get_ref_url() if hasattr(resource, 'get_ref_url') else 'N/A'
                href = resource.get_href() if hasattr(resource, 'get_href') else 'N/A'
                logger.debug(f"✅ Resource found: ref_url={ref_url}, href={href}")
            except Exception as e:
                logger.debug(f"✅ Resource found (href error: {e})")
                
            logger.debug(f"📂 Resource path: {resource.path if hasattr(resource, 'path') else 'N/A'}")
            logger.debug(f"📂 Resource provider type: {type(resource.provider).__name__ if hasattr(resource, 'provider') else 'N/A'}")
            logger.debug(f"📂 Resource is_collection: {resource.is_collection if hasattr(resource, 'is_collection') else 'N/A'}")
        else:
            logger.debug(f"❌ Resource not found for path: {path}")
        
        logger.debug(f"📂 =====================================")
        
        return resource

