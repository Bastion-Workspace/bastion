"""
WebDAV Authentication Provider - Plato User Database Integration

Implements WsgiDAV authentication using Plato's existing user database
and authentication service.
"""

import logging
from typing import Optional, Dict, Any
from wsgidav.dc.base_dc import BaseDomainController
import psycopg2
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# Password context for verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PlatoAuthController(BaseDomainController):
    """
    Custom authentication controller for WsgiDAV that uses Plato's
    existing user database and password hashing.
    
    Integrates with services.auth_service for consistent authentication.
    """
    
    def __init__(self, wsgidav_app, config):
        """
        Initialize authentication controller.
        
        Args:
            wsgidav_app: WsgiDAV application instance (or None during initial config)
            config: WsgiDAV configuration dict
        """
        # Pass to BaseDomainController
        super().__init__(wsgidav_app, config)
        # Store database connection info from config (not async pool)
        self.db_config = config.get("_plato_db_config")
        logger.info("🔐 PlatoAuthController initialized for WebDAV")
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def get_domain_realm(self, path_info, environ):
        """
        Return the authentication realm for a given path.
        
        Args:
            path_info: The path being accessed
            environ: WSGI environment
            
        Returns:
            str: Authentication realm name
        """
        return "Plato OrgMode WebDAV"
    
    def require_authentication(self, realm, environ):
        """
        Determine if authentication is required.
        
        Args:
            realm: The authentication realm
            environ: WSGI environment
            
        Returns:
            bool: Always True - we require authentication for all WebDAV access
        """
        return True
    
    def basic_auth_user(self, realm, user_name, password, environ):
        """
        Validate user credentials using Plato's authentication service.
        
        This is called by WsgiDAV for HTTP Basic authentication.
        Uses synchronous database connection to avoid event loop conflicts.
        
        Args:
            realm: The authentication realm
            user_name: Username provided
            password: Password provided
            environ: WSGI environment
            
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            logger.debug(f"🔐 WebDAV auth attempt for user: {user_name}")
            
            # Verify credentials synchronously
            user_info = self._verify_credentials_sync(user_name, password)
            
            if user_info:
                logger.debug(f"✅ WebDAV auth SUCCESS for user: {user_name}")
                # Store user info in environ for later use
                environ["webdav.auth.user_name"] = user_name
                environ["webdav.auth.user_id"] = user_info['user_id']
                return True
            else:
                logger.warning(f"❌ WebDAV auth FAILED for user: {user_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ WebDAV auth error for user {user_name}: {e}")
            logger.exception("Full auth error traceback:")
            return False
    
    def _verify_credentials_sync(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Synchronous method to verify credentials against database.
        Uses psycopg2 instead of asyncpg to avoid event loop issues.
        
        Args:
            username: Username to verify
            password: Password to verify
            
        Returns:
            dict: User info (user_id, username) if valid, None otherwise
        """
        conn = None
        try:
            logger.debug(f"🔍 Connecting to database: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
            
            # Create synchronous database connection
            conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database']
            )
            
            logger.debug(f"✅ Database connected")
            
            cur = conn.cursor()
            
            # Query database for user
            logger.debug(f"🔍 Querying for user: {username}")
            cur.execute("""
                SELECT user_id, username, password_hash, salt, is_active
                FROM users
                WHERE username = %s OR email = %s
            """, (username, username))
            
            user_row = cur.fetchone()
            cur.close()
            
            if not user_row:
                logger.warning(f"🔍 User not found in database: {username}")
                return None
            
            user_id, db_username, password_hash, salt, is_active = user_row
            logger.debug(f"✅ User found: {db_username}, active: {is_active}")
            
            if not is_active:
                logger.warning(f"🔍 User inactive: {username}")
                return None
            
            # Verify password using passlib (same as main auth)
            logger.debug(f"🔍 Verifying password for user: {db_username}")

            is_valid = pwd_context.verify(password, password_hash)
            logger.debug(f"🔍 Password verification result: {is_valid}")
            
            if is_valid:
                return {
                    'user_id': user_id,
                    'username': db_username
                }
            else:
                logger.warning(f"❌ Password verification failed for user: {db_username}")
                return None
            
        except Exception as e:
            logger.error(f"❌ Credential verification error: {e}")
            logger.exception("Full credential verification traceback:")
            return None
        finally:
            if conn:
                conn.close()
    
    def supports_http_digest_auth(self):
        """
        Indicate if HTTP Digest authentication is supported.
        
        Returns:
            bool: False - we only support Basic auth for simplicity
        """
        return False
    
    def is_realm_user(self, realm, user_name, environ):
        """
        Check if user exists in realm.
        
        Args:
            realm: Authentication realm
            user_name: Username to check
            environ: WSGI environment
            
        Returns:
            bool: True if user exists (actual validation in basic_auth_user)
        """
        return True
    
    def get_realm_user_password(self, realm, user_name, environ):
        """
        Not used for our authentication - we handle it in basic_auth_user.
        
        Returns:
            None
        """
        return None

