"""
Music Client Factory
Factory for creating music service client instances
"""

import logging
from typing import Optional

from .base_client import BaseMusicClient
from .subsonic_client import SubSonicClient
from .audiobookshelf_client import AudiobookshelfClient
from .deezer_client import DeezerClient

logger = logging.getLogger(__name__)


class MusicClientFactory:
    """Factory for creating music service clients"""
    
    _client_classes = {
        "subsonic": SubSonicClient,
        "audiobookshelf": AudiobookshelfClient,
        "deezer": DeezerClient,
        # Future clients:
        # "plex": PlexClient,
        # "emby": EmbyClient,
        # "jellyfin": JellyfinClient,
    }
    
    @classmethod
    def create_client(
        cls,
        service_type: str,
        server_url: str,
        username: str,
        password: str,
        **kwargs
    ) -> Optional[BaseMusicClient]:
        """
        Create a music service client instance
        
        Args:
            service_type: Type of music service ('subsonic', 'plex', 'emby', etc.)
            server_url: Music server URL
            username: Username for authentication
            password: Password or token for authentication
            **kwargs: Additional service-specific parameters
            
        Returns:
            Music client instance or None if service_type not supported
        """
        service_type = service_type.lower()
        
        client_class = cls._client_classes.get(service_type)
        if not client_class:
            logger.error(f"Unsupported music service type: {service_type}")
            return None
        
        try:
            return client_class(server_url, username, password, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create {service_type} client: {e}")
            return None
    
    @classmethod
    def get_supported_services(cls) -> list:
        """Get list of supported music service types"""
        return list(cls._client_classes.keys())
    
    @classmethod
    def register_client(cls, service_type: str, client_class: type):
        """
        Register a new music client class
        
        Args:
            service_type: Service type identifier
            client_class: Client class that extends BaseMusicClient
        """
        if not issubclass(client_class, BaseMusicClient):
            raise ValueError(f"Client class must extend BaseMusicClient")
        cls._client_classes[service_type.lower()] = client_class
        logger.info(f"Registered music client: {service_type}")

