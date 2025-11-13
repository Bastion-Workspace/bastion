"""
WebDAV Configuration - OrgMode Sync Server Config

Configures the WsgiDAV server for OrgMode file synchronization.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_webdav_config(auth_controller, filesystem_provider, host="0.0.0.0", port=8001, verbose=1) -> Dict[str, Any]:
    """
    Create WsgiDAV configuration dictionary.
    
    Args:
        auth_controller: PlatoAuthController class (not instance)
        filesystem_provider: Instance of UserFilteredFilesystemProvider
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8001)
        verbose: Logging verbosity level (0-5)
        
    Returns:
        dict: WsgiDAV configuration
    """
    config = {
        # Server settings
        "host": host,
        "port": port,
        
        # Pass verbose logging configuration
        "logging": get_logging_config(verbose),
        
        # Provider mapping - map URL path to provider
        # **ROOSEVELT'S CLEAN FIX:** Mount at "/" because nginx strips /dav prefix
        # nginx: /dav/OrgMode → (strips /dav) → WsgiDAV: /OrgMode
        # Provider at "/" handles all paths, generates clean hrefs
        # Clients add their base URL (/dav) back when making requests
        "provider_mapping": {
            "/": filesystem_provider,
        },
        
        # Authentication settings
        "http_authenticator": {
            "domain_controller": auth_controller,
            "accept_basic": True,  # Enable HTTP Basic auth
            "accept_digest": False,  # Disable Digest auth for simplicity
            "default_to_digest": False,
        },
        
        # Simple domain controller settings (not used with custom auth)
        "simple_dc": {
            "user_mapping": {}  # Empty - we use custom authentication
        },
        
        # Verbose logging
        "verbose": verbose,
        
        # **ROOSEVELT DEBUG:** Log full request and response bodies to see XML
        "enable_log_request_body": True,
        "enable_log_response_body": True,
        
        # Enable directory browsing
        "dir_browser": {
            "enable": True,
            "response_trailer": "Plato OrgMode WebDAV Server",
            "davmount": False,  # Disable MS Office Web Folder feature
        },
        
        # Hotfixes (WsgiDAV 4.3 moved these options here)
        "hotfixes": {
            "unquote_path_info": False,
        },
        
        # Lock manager (for concurrent access protection)
        "lock_storage": True,
        
        # Property manager
        "property_manager": True,
        
        # Additional settings
        "add_header_MS_Author_Via": False,
        
        # CORS headers (if needed for web access)
        "cors": {
            "enable": False,  # Disable CORS by default
        },
        
        # Middleware stack (use default by omitting this key)
        # WsgiDAV 4.3+ uses a different middleware system
    }
    
    logger.info(f"📋 WebDAV config created: host={host}, port={port}")
    logger.info(f"📁 Provider mapping: / -> FilesystemProvider (uploads/ directory)")
    logger.info(f"📁 Note: nginx strips /dav prefix, WsgiDAV receives clean paths")
    
    return config


def get_logging_config(verbose_level=1) -> Dict[str, Any]:
    """
    Create logging configuration for WsgiDAV.
    
    Args:
        verbose_level: Verbosity level (0-5)
        
    Returns:
        dict: Logging configuration
    """
    # Map verbose level to Python logging level
    level_map = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
        4: logging.DEBUG,
        5: logging.DEBUG,
    }
    
    level = level_map.get(verbose_level, logging.INFO)
    
    # **ROOSEVELT'S BIG STICK:** Force DEBUG logging to see the XML!
    # We are setting all levels to DEBUG to override any stubborn defaults.
    debug_level = logging.DEBUG
    
    return {
        "enable": True,
        "version": 1,
        "disable_existing_loggers": True,  # **BULLY!** The Big Stick! Disable all other loggers!
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": debug_level,
            },
        },
        "loggers": {
            "wsgidav": {
                "handlers": ["console"],
                "level": debug_level,
                "propagate": True,  # **BULLY!** Propagate to see ALL sub-logger messages!
            },
            "webdav": {
                "handlers": ["console"],
                "level": debug_level,  # Force our logger to DEBUG
                "propagate": False,
            },
        },
        "root": {
            "level": debug_level,  # Force root logger to DEBUG
            "handlers": ["console"],
        },
    }

