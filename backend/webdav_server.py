"""
WebDAV Server Startup - OrgMode Mobile Sync Server

Main entry point for the WsgiDAV server that provides WebDAV access
to OrgMode files for mobile synchronization.

Run with:
    python webdav_server.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
import logging.config

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from webdav.auth_provider import PlatoAuthController
from webdav.orgmode_provider import OrgModeDAVProvider
from webdav.config import create_webdav_config, get_logging_config
from config import settings

# **ROOSEVELT'S LOGGING COUP D'ÉTAT!**
# Forcefully configure logging at startup to ensure WsgiDAV's verbose XML
# logging is enabled. This overrides any default logging.
logging.config.dictConfig(get_logging_config(verbose_level=3))

logger = logging.getLogger(__name__)


async def initialize_database_pool():
    """Initialize database connection pool for WebDAV provider"""
    import asyncpg
    
    logger.info("🔗 Connecting to PostgreSQL database...")
    
    pool = await asyncpg.create_pool(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        database=settings.POSTGRES_DB,
        min_size=2,
        max_size=10,
        command_timeout=60,
    )
    
    logger.info("✅ Database connection pool established")
    return pool


async def initialize_auth_service(db_pool):
    """Initialize authentication service for WebDAV"""
    from services.auth_service import AuthenticationService
    
    logger.info("🔐 Initializing authentication service...")
    
    auth_service = AuthenticationService()
    await auth_service.initialize(shared_db_pool=db_pool)
    
    logger.info("✅ Authentication service initialized")
    return auth_service


def create_wsgi_app(auth_service, db_pool):
    """Create the WsgiDAV application"""
    from wsgidav.wsgidav_app import WsgiDAVApp
    from webdav.simple_filesystem_provider import UserFilteredFilesystemProvider
    
    logger.info("📝 Creating WebDAV application...")
    
    # Determine uploads directory path
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    if not os.path.exists(uploads_dir):
        logger.warning(f"⚠️  Uploads directory not found: {uploads_dir}")
        # Try absolute path for Docker
        uploads_dir = '/app/uploads'
    
    logger.info(f"📂 Using uploads directory: {uploads_dir}")
    
    # Create database config for user lookups
    db_config = {
        'host': settings.POSTGRES_HOST,
        'port': settings.POSTGRES_PORT,
        'user': settings.POSTGRES_USER,
        'password': settings.POSTGRES_PASSWORD,
        'database': settings.POSTGRES_DB
    }
    
    # Create simple filesystem provider - serves per-user subdirectories
    filesystem_provider = UserFilteredFilesystemProvider(uploads_dir, db_config)
    
    # Create configuration with database config passed
    # WsgiDAV will instantiate PlatoAuthController with (wsgidav_app, config)
    config = create_webdav_config(
        auth_controller=PlatoAuthController,  # Pass the CLASS, not an instance
        filesystem_provider=filesystem_provider,
        host=settings.WEBDAV_HOST if hasattr(settings, 'WEBDAV_HOST') else "0.0.0.0",
        port=settings.WEBDAV_PORT if hasattr(settings, 'WEBDAV_PORT') else 8001,
        verbose=4,  # **BULLY!** Level 4 is needed for DEBUG and XML body logging!
    )
    
    # Store database config so PlatoAuthController can create sync connections
    config["_plato_db_config"] = {
        'host': settings.POSTGRES_HOST,
        'port': settings.POSTGRES_PORT,
        'user': settings.POSTGRES_USER,
        'password': settings.POSTGRES_PASSWORD,
        'database': settings.POSTGRES_DB
    }
    
    # Create WsgiDAV app (it will instantiate PlatoAuthController)
    app = WsgiDAVApp(config)
    
    logger.info("✅ WebDAV application created")
    return app


 


def run_server(app, host="0.0.0.0", port=8001):
    """Run the WsgiDAV server using Cheroot WSGI server"""
    from cheroot import wsgi
    
    class KeepAliveHeadersMiddleware:
        """WSGI middleware to enforce keep-alive response headers."""
        def __init__(self, next_app):
            self.next_app = next_app

        def __call__(self, environ, start_response):
            def wrapped_start_response(status, headers, exc_info=None):
                # Ensure Connection: keep-alive and Keep-Alive hints are present
                header_names = {k.lower() for k, _ in headers}
                if "connection" not in header_names:
                    headers.append(("Connection", "keep-alive"))
                if "keep-alive" not in header_names:
                    headers.append(("Keep-Alive", "timeout=10, max=100"))
                return start_response(status, headers, exc_info)

            return self.next_app(environ, wrapped_start_response)

    logger.info(f"🚀 Starting WebDAV server on {host}:{port}")
    logger.info(f"📱 Mobile clients can connect to: http://{host}:{port}/")
    logger.info(f"📂 Serving: uploads/ directory with actual file/folder structure")
    logger.info(f"🔐 Authentication required using Bastion user credentials")
    
    # Create Cheroot WSGI server
    app = KeepAliveHeadersMiddleware(app)
    server = wsgi.Server(
        bind_addr=(host, port),
        wsgi_app=app,
        numthreads=10,
        server_name="Plato-WebDAV",
        # **ROOSEVELT'S RFC 4918 COMPLIANCE FIX:** Keep-alive is the default in Cheroot 10.
        # No explicit parameter is needed. The `connection: close` from 1Writer
        # was a red herring; the real issue was the href generation.
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down WebDAV server...")
        server.stop()
        logger.info("✅ WebDAV server stopped")


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("🔐 Plato WebDAV Server - OrgMode Mobile Sync")
    logger.info("=" * 60)
    
    # Initialize async components
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Initialize database and auth service
        db_pool = loop.run_until_complete(initialize_database_pool())
        auth_service = loop.run_until_complete(initialize_auth_service(db_pool))
        
        # Create WSGI app
        app = create_wsgi_app(auth_service, db_pool)
        
        # Get host and port from settings
        host = settings.WEBDAV_HOST if hasattr(settings, 'WEBDAV_HOST') else "0.0.0.0"
        port = settings.WEBDAV_PORT if hasattr(settings, 'WEBDAV_PORT') else 8001
        
        # Run server (blocking)
        run_server(app, host=host, port=port)
        
    except Exception as e:
        logger.error(f"❌ Fatal error starting WebDAV server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if 'db_pool' in locals():
            loop.run_until_complete(db_pool.close())
        loop.close()


if __name__ == "__main__":
    main()

