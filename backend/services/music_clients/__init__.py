"""
Music Service Clients
Client implementations for different music services (SubSonic, Audiobookshelf, Plex, Emby, etc.)
"""

from .base_client import BaseMusicClient
from .subsonic_client import SubSonicClient
from .audiobookshelf_client import AudiobookshelfClient

__all__ = ["BaseMusicClient", "SubSonicClient", "AudiobookshelfClient"]

