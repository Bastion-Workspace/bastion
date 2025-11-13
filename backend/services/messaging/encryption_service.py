"""
Roosevelt's Encryption Service
Handles at-rest message encryption with master key management

BULLY! Keep your messages secure like a well-guarded cavalry fort!
"""

import logging
import base64
import secrets
from typing import Optional, Tuple
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from config import settings

logger = logging.getLogger(__name__)


class EncryptionService:
    """
    Service for encrypting and decrypting messages at rest
    
    Features:
    - Fernet symmetric encryption (AES-128-CBC + HMAC)
    - Master key from environment
    - Key derivation for room-specific keys
    - Version support for key rotation
    """
    
    def __init__(self):
        self._master_key = None
        self._fernet = None
        self._initialized = False
    
    def _initialize(self):
        """Initialize encryption with master key from settings"""
        if self._initialized:
            return
        
        if not settings.MESSAGE_ENCRYPTION_AT_REST:
            logger.info("📭 Message encryption at rest is DISABLED")
            self._initialized = True
            return
        
        master_key_str = settings.MESSAGE_ENCRYPTION_MASTER_KEY
        
        if not master_key_str:
            logger.warning("⚠️ MESSAGE_ENCRYPTION_MASTER_KEY not set! Generating temporary key...")
            logger.warning("⚠️ This key will NOT persist across restarts!")
            # Generate a temporary key for development
            self._master_key = Fernet.generate_key()
            logger.warning(f"⚠️ Generated key: {self._master_key.decode()}")
            logger.warning("⚠️ Set MESSAGE_ENCRYPTION_MASTER_KEY in docker-compose.yml for production!")
        else:
            try:
                # Try to use the provided key
                self._master_key = master_key_str.encode()
                # Test that it's valid Fernet key
                Fernet(self._master_key)
            except Exception as e:
                logger.error(f"❌ Invalid master key format: {e}")
                logger.info("🔧 Generating new Fernet key...")
                # If invalid, generate a new key
                self._master_key = Fernet.generate_key()
                logger.warning(f"⚠️ Generated new key: {self._master_key.decode()}")
                logger.warning("⚠️ Update MESSAGE_ENCRYPTION_MASTER_KEY with this value!")
        
        self._fernet = Fernet(self._master_key)
        self._initialized = True
        logger.info("🔐 BULLY! Encryption service initialized and ready!")
    
    @staticmethod
    def generate_fernet_key() -> str:
        """
        Generate a new Fernet key for use as master key
        
        Returns:
            Base64-encoded Fernet key as string
        """
        key = Fernet.generate_key()
        return key.decode()
    
    def encrypt_message(self, plaintext: str, encryption_version: int = 1) -> Optional[str]:
        """
        Encrypt a message for at-rest storage
        
        Args:
            plaintext: The message to encrypt
            encryption_version: Version number for key rotation support
        
        Returns:
            Base64-encoded encrypted message, or None if encryption disabled
        """
        self._initialize()
        
        if not settings.MESSAGE_ENCRYPTION_AT_REST:
            # Return plaintext if encryption disabled
            return plaintext
        
        try:
            encrypted_bytes = self._fernet.encrypt(plaintext.encode('utf-8'))
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Failed to encrypt message: {e}")
            # In production, you might want to fail hard here
            # For now, return plaintext with warning
            logger.warning("⚠️ Falling back to plaintext storage!")
            return plaintext
    
    def decrypt_message(self, encrypted_text: str, encryption_version: int = 1) -> Optional[str]:
        """
        Decrypt a message from at-rest storage
        
        Args:
            encrypted_text: Base64-encoded encrypted message
            encryption_version: Version number for key rotation support
        
        Returns:
            Decrypted plaintext message, or None if decryption fails
        """
        self._initialize()
        
        if not settings.MESSAGE_ENCRYPTION_AT_REST:
            # Return as-is if encryption disabled
            return encrypted_text
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
            decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except (InvalidToken, ValueError) as e:
            logger.error(f"❌ Failed to decrypt message: {e}")
            logger.error("⚠️ Message may be stored in plaintext or encrypted with different key")
            # Try returning as plaintext (in case it was stored before encryption was enabled)
            return encrypted_text
        except Exception as e:
            logger.error(f"❌ Unexpected decryption error: {e}")
            return None
    
    def derive_room_key(self, room_id: str) -> Optional[str]:
        """
        Derive a room-specific encryption key from master key
        
        This is for future E2EE support where each room has its own key
        
        Args:
            room_id: UUID of the room
        
        Returns:
            Base64-encoded Fernet key for the room, or None if encryption is disabled
        """
        self._initialize()
        
        # If encryption is disabled, return None
        if not settings.MESSAGE_ENCRYPTION_AT_REST or self._master_key is None:
            return None
        
        # Use PBKDF2 to derive a room-specific key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=room_id.encode('utf-8'),
            iterations=100000,
            backend=default_backend()
        )
        
        derived_key = base64.urlsafe_b64encode(kdf.derive(self._master_key))
        return derived_key.decode('utf-8')
    
    def encrypt_room_key(self, room_key: str) -> Optional[str]:
        """
        Encrypt a room key with the master key for storage
        
        This is for future E2EE support
        
        Args:
            room_key: Fernet key for the room
        
        Returns:
            Encrypted room key, or None if encryption is disabled or room_key is None
        """
        self._initialize()
        
        # If encryption is disabled or no room key provided, return None
        if not settings.MESSAGE_ENCRYPTION_AT_REST or room_key is None:
            return None
        
        try:
            encrypted_bytes = self._fernet.encrypt(room_key.encode('utf-8'))
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Failed to encrypt room key: {e}")
            return None
    
    def decrypt_room_key(self, encrypted_room_key: str) -> Optional[str]:
        """
        Decrypt a room key using the master key
        
        This is for future E2EE support
        
        Args:
            encrypted_room_key: Encrypted room key from database
        
        Returns:
            Decrypted Fernet key for the room, or None if encryption is disabled or key is None
        """
        self._initialize()
        
        # If encryption is disabled or no encrypted key provided, return None
        if not settings.MESSAGE_ENCRYPTION_AT_REST or encrypted_room_key is None:
            return None
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_room_key.encode('utf-8'))
            decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Failed to decrypt room key: {e}")
            return None
    
    def is_encryption_enabled(self) -> bool:
        """Check if encryption at rest is enabled"""
        return settings.MESSAGE_ENCRYPTION_AT_REST


# Global encryption service instance
encryption_service = EncryptionService()


# Utility functions for external use
def encrypt_message_content(plaintext: str) -> str:
    """Encrypt a message (convenience function)"""
    return encryption_service.encrypt_message(plaintext) or plaintext


def decrypt_message_content(encrypted_text: str) -> str:
    """Decrypt a message (convenience function)"""
    return encryption_service.decrypt_message(encrypted_text) or encrypted_text


def generate_master_key() -> str:
    """
    Generate a new master key for MESSAGE_ENCRYPTION_MASTER_KEY
    
    Run this once and add the output to your docker-compose.yml:
    
    ```yaml
    environment:
      - MESSAGE_ENCRYPTION_MASTER_KEY=<generated_key_here>
    ```
    """
    return EncryptionService.generate_fernet_key()


if __name__ == "__main__":
    # Helper for generating a new master key
    print("=" * 60)
    print("ROOSEVELT'S ENCRYPTION KEY GENERATOR")
    print("=" * 60)
    print("\nGenerated Fernet key for MESSAGE_ENCRYPTION_MASTER_KEY:\n")
    print(generate_master_key())
    print("\nAdd this to your docker-compose.yml environment:")
    print("  - MESSAGE_ENCRYPTION_MASTER_KEY=<key_above>")
    print("=" * 60)

