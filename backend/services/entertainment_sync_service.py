"""
Entertainment Sync Service
Service for managing Sonarr/Radarr sync configurations and state
"""

import logging
import base64
import secrets
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from models.entertainment_sync_models import (
    SyncConfigCreate, SyncConfigUpdate, SyncConfig, SyncItem, ItemFilters
)
from services.database_manager.database_helpers import (
    fetch_all, fetch_one, execute, fetch_value
)
from config import settings

logger = logging.getLogger(__name__)


class DuplicateConfigError(Exception):
    """Raised when attempting to create a duplicate sync configuration"""
    pass


class EntertainmentSyncService:
    """
    Service for managing entertainment sync configurations
    
    Handles CRUD operations for Radarr/Sonarr configurations and sync state tracking
    """
    
    def __init__(self):
        self._fernet = None
        self._initialized = False
    
    def _initialize_encryption(self):
        """Initialize encryption with master key from settings"""
        if self._initialized:
            return
        
        master_key_str = settings.SECRET_KEY
        
        if not master_key_str:
            logger.warning("SECRET_KEY not set! Generating temporary key...")
            self._master_key = Fernet.generate_key()
        else:
            try:
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'entertainment_sync_salt',
                    iterations=100000,
                    backend=default_backend()
                )
                derived_key = base64.urlsafe_b64encode(kdf.derive(master_key_str.encode()))
                self._master_key = derived_key
                self._fernet = Fernet(self._master_key)
            except Exception as e:
                logger.error(f"Failed to initialize encryption: {e}")
                self._master_key = Fernet.generate_key()
                self._fernet = Fernet(self._master_key)
        
        self._initialized = True
    
    def _encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key using Fernet"""
        self._initialize_encryption()
        encrypted_bytes = self._fernet.encrypt(api_key.encode('utf-8'))
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    
    def _decrypt_api_key(self, encrypted_api_key: str) -> str:
        """Decrypt API key using Fernet"""
        self._initialize_encryption()
        encrypted_bytes = base64.b64decode(encrypted_api_key.encode('utf-8'))
        decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
        return decrypted_bytes.decode('utf-8')
    
    async def create_sync_config(
        self, user_id: str, config: SyncConfigCreate
    ) -> UUID:
        """
        Create a new sync configuration
        
        Args:
            user_id: User ID who owns this configuration
            config: Configuration data
            
        Returns:
            UUID of created configuration
        """
        try:
            # Encrypt API key before storing
            encrypted_api_key = self._encrypt_api_key(config.api_key)
            
            config_id = await fetch_value(
                """
                INSERT INTO entertainment_sync_config (
                    user_id, source_type, api_url, api_key, enabled, sync_frequency_minutes,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
                RETURNING config_id
                """,
                user_id,
                config.source_type,
                config.api_url,
                encrypted_api_key,
                config.enabled,
                config.sync_frequency_minutes
            )
            
            logger.info(f"Created sync config {config_id} for user {user_id}")
            return UUID(str(config_id))
        except Exception as e:
            error_str = str(e)
            if "duplicate key" in error_str and "entertainment_sync_config_user_id_source_type_api_url_key" in error_str:
                logger.warning(f"Duplicate config attempted for user {user_id}, source {config.source_type}, url {config.api_url}")
                raise DuplicateConfigError(
                    f"A {config.source_type.title()} configuration for this URL already exists. "
                    f"Please update the existing configuration or use a different URL."
                )
            logger.error(f"Failed to create sync config: {e}")
            raise
    
    async def get_user_configs(self, user_id: str) -> List[SyncConfig]:
        """
        Get all sync configurations for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of sync configurations (API keys not decrypted in response)
        """
        try:
            rows = await fetch_all(
                """
                SELECT 
                    config_id, user_id, source_type, api_url, enabled,
                    sync_frequency_minutes, last_sync_at, last_sync_status,
                    items_synced, sync_error, created_at, updated_at
                FROM entertainment_sync_config
                WHERE user_id = $1
                ORDER BY created_at DESC
                """,
                user_id
            )
            
            configs = []
            for row in rows:
                configs.append(SyncConfig(**dict(row)))
            
            return configs
        except Exception as e:
            logger.error(f"Failed to get user configs: {e}")
            return []
    
    async def get_config(self, config_id: UUID, user_id: str) -> Optional[SyncConfig]:
        """
        Get a specific sync configuration
        
        Args:
            config_id: Configuration ID
            user_id: User ID (for authorization check)
            
        Returns:
            Sync configuration or None
        """
        try:
            row = await fetch_one(
                """
                SELECT 
                    config_id, user_id, source_type, api_url, enabled,
                    sync_frequency_minutes, last_sync_at, last_sync_status,
                    items_synced, sync_error, created_at, updated_at
                FROM entertainment_sync_config
                WHERE config_id = $1 AND user_id = $2
                """,
                str(config_id),
                user_id
            )
            
            if row:
                return SyncConfig(**dict(row))
            return None
        except Exception as e:
            logger.error(f"Failed to get config {config_id}: {e}")
            return None
    
    async def get_config_with_api_key(
        self, config_id: UUID, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get configuration with decrypted API key (for internal use only)
        
        Args:
            config_id: Configuration ID
            user_id: Optional user ID (for authorization check, None for internal use)
            
        Returns:
            Dict with config data including decrypted API key
        """
        try:
            if user_id:
                row = await fetch_one(
                    """
                    SELECT * FROM entertainment_sync_config
                    WHERE config_id = $1 AND user_id = $2
                    """,
                    str(config_id),
                    user_id
                )
            else:
                # Internal use - no user check
                row = await fetch_one(
                    """
                    SELECT * FROM entertainment_sync_config
                    WHERE config_id = $1
                    """,
                    str(config_id)
                )
            
            if row:
                config_dict = dict(row)
                # Decrypt API key
                encrypted_key = config_dict.get("api_key")
                if encrypted_key:
                    config_dict["api_key"] = self._decrypt_api_key(encrypted_key)
                return config_dict
            return None
        except Exception as e:
            logger.error(f"Failed to get config with API key {config_id}: {e}")
            return None
    
    async def update_sync_config(
        self, config_id: UUID, user_id: str, updates: SyncConfigUpdate
    ) -> bool:
        """
        Update sync configuration
        
        Args:
            config_id: Configuration ID
            user_id: User ID (for authorization check)
            updates: Update data
            
        Returns:
            True if successful
        """
        try:
            update_fields = []
            params = []
            param_index = 1
            
            if updates.api_url is not None:
                update_fields.append(f"api_url = ${param_index}")
                params.append(updates.api_url)
                param_index += 1
            
            if updates.api_key is not None:
                encrypted_key = self._encrypt_api_key(updates.api_key)
                update_fields.append(f"api_key = ${param_index}")
                params.append(encrypted_key)
                param_index += 1
            
            if updates.enabled is not None:
                update_fields.append(f"enabled = ${param_index}")
                params.append(updates.enabled)
                param_index += 1
            
            if updates.sync_frequency_minutes is not None:
                update_fields.append(f"sync_frequency_minutes = ${param_index}")
                params.append(updates.sync_frequency_minutes)
                param_index += 1
            
            if not update_fields:
                return True  # No updates to apply
            
            update_fields.append(f"updated_at = NOW()")
            params.append(str(config_id))
            params.append(user_id)
            
            query = f"""
                UPDATE entertainment_sync_config
                SET {', '.join(update_fields)}
                WHERE config_id = ${param_index} AND user_id = ${param_index + 1}
            """
            
            await execute(query, *params)
            logger.info(f"Updated sync config {config_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update sync config {config_id}: {e}")
            return False
    
    async def delete_sync_config(self, config_id: UUID, user_id: str) -> bool:
        """
        Delete sync configuration and all associated items
        
        This method:
        1. Fetches all sync items for this config
        2. Deletes vector embeddings from Qdrant
        3. Deletes knowledge graph entities from Neo4j
        4. Deletes sync items from database (via CASCADE when config is deleted)
        5. Deletes the config itself
        
        Args:
            config_id: Configuration ID
            user_id: User ID (for authorization check)
            
        Returns:
            True if successful
        """
        try:
            # First, get all sync items for this config to clean up vectors and KG
            sync_items = await self.get_sync_items(config_id)
            
            if sync_items:
                logger.info(f"Cleaning up {len(sync_items)} items for config {config_id}")
                
                # Get services for cleanup
                from services.service_container import get_service_container
                service_container = await get_service_container()
                
                # Clean up vector embeddings and KG entities for each item
                for item in sync_items:
                    try:
                        vector_doc_id = item.vector_document_id
                        if vector_doc_id:
                            # Delete from Qdrant vector store
                            if hasattr(service_container, 'embedding_manager') and service_container.embedding_manager:
                                await service_container.embedding_manager.delete_document_chunks(
                                    vector_doc_id,
                                    user_id
                                )
                                logger.debug(f"Deleted vector embeddings for {vector_doc_id}")
                            
                            # Delete from Neo4j knowledge graph
                            if hasattr(service_container, 'knowledge_graph_service') and service_container.knowledge_graph_service:
                                await service_container.knowledge_graph_service.delete_document_entities(vector_doc_id)
                                logger.debug(f"Deleted KG entities for {vector_doc_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up item {item.item_id} ({item.external_id}): {e}")
                        # Continue with other items even if one fails
            
            # Delete the config (this will CASCADE delete sync_items from database)
            await execute(
                """
                DELETE FROM entertainment_sync_config
                WHERE config_id = $1 AND user_id = $2
                """,
                str(config_id),
                user_id
            )
            logger.info(f"Deleted sync config {config_id} and cleaned up {len(sync_items) if sync_items else 0} associated items")
            return True
        except Exception as e:
            logger.error(f"Failed to delete sync config {config_id}: {e}")
            return False
    
    async def get_enabled_configs(self) -> List[Dict[str, Any]]:
        """
        Get all enabled sync configurations (for background tasks)
        
        Returns:
            List of config dicts with decrypted API keys
        """
        try:
            rows = await fetch_all(
                """
                SELECT * FROM entertainment_sync_config
                WHERE enabled = true
                ORDER BY last_sync_at NULLS FIRST
                """
            )
            
            configs = []
            for row in rows:
                config_dict = dict(row)
                # Decrypt API key
                encrypted_key = config_dict.get("api_key")
                if encrypted_key:
                    config_dict["api_key"] = self._decrypt_api_key(encrypted_key)
                configs.append(config_dict)
            
            return configs
        except Exception as e:
            logger.error(f"Failed to get enabled configs: {e}")
            return []
    
    async def update_sync_status(
        self,
        config_id: UUID,
        status: str,
        items_synced: Optional[int] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Update sync status after a sync operation
        
        Args:
            config_id: Configuration ID
            status: Status string ('success', 'failed', 'running')
            items_synced: Number of items synced
            error: Error message if failed
            
        Returns:
            True if successful
        """
        try:
            update_fields = [
                "last_sync_status = $1",
                "last_sync_at = NOW()",
                "updated_at = NOW()"
            ]
            params = [status]
            param_index = 2
            
            if items_synced is not None:
                update_fields.append(f"items_synced = ${param_index}")
                params.append(items_synced)
                param_index += 1
            
            if error is not None:
                update_fields.append(f"sync_error = ${param_index}")
                params.append(error)
                param_index += 1
            else:
                update_fields.append("sync_error = NULL")
            
            params.append(str(config_id))
            
            query = f"""
                UPDATE entertainment_sync_config
                SET {', '.join(update_fields)}
                WHERE config_id = ${param_index}
            """
            
            await execute(query, *params)
            return True
        except Exception as e:
            logger.error(f"Failed to update sync status for {config_id}: {e}")
            return False
    
    async def get_sync_items(
        self, config_id: UUID, filters: Optional[ItemFilters] = None
    ) -> List[SyncItem]:
        """
        Get synced items for a configuration
        
        Args:
            config_id: Configuration ID
            filters: Optional filters
            
        Returns:
            List of sync items
        """
        try:
            query = """
                SELECT 
                    item_id, config_id, external_id, external_type, title,
                    tmdb_id, tvdb_id, season_number, episode_number,
                    parent_series_id, metadata_hash, last_synced_at, vector_document_id
                FROM entertainment_sync_items
                WHERE config_id = $1
            """
            params = [str(config_id)]
            param_index = 2
            
            if filters and filters.external_type:
                query += f" AND external_type = ${param_index}"
                params.append(filters.external_type)
                param_index += 1
            
            query += " ORDER BY title"
            
            if filters:
                query += f" LIMIT ${param_index} OFFSET ${param_index + 1}"
                params.append(filters.limit)
                params.append(filters.skip)
            
            rows = await fetch_all(query, *params)
            
            items = []
            for row in rows:
                items.append(SyncItem(**dict(row)))
            
            return items
        except Exception as e:
            logger.error(f"Failed to get sync items for {config_id}: {e}")
            return []
    
    async def upsert_sync_item(
        self,
        config_id: UUID,
        external_id: str,
        external_type: str,
        title: str,
        metadata_hash: str,
        vector_document_id: Optional[str] = None,
        tmdb_id: Optional[int] = None,
        tvdb_id: Optional[int] = None,
        season_number: Optional[int] = None,
        episode_number: Optional[int] = None,
        parent_series_id: Optional[str] = None
    ) -> UUID:
        """
        Insert or update a sync item
        
        Args:
            config_id: Configuration ID
            external_id: External item ID
            external_type: Item type ('movie', 'series', 'episode')
            title: Item title
            metadata_hash: Hash of metadata for change detection
            vector_document_id: Vector document ID
            tmdb_id: Optional TMDB ID
            tvdb_id: Optional TVDB ID
            season_number: Optional season number
            episode_number: Optional episode number
            parent_series_id: Optional parent series ID
            
        Returns:
            Item ID
        """
        try:
            item_id = await fetch_value(
                """
                INSERT INTO entertainment_sync_items (
                    config_id, external_id, external_type, title, metadata_hash,
                    vector_document_id, tmdb_id, tvdb_id, season_number,
                    episode_number, parent_series_id, last_synced_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                ON CONFLICT (config_id, external_id, external_type)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    metadata_hash = EXCLUDED.metadata_hash,
                    vector_document_id = EXCLUDED.vector_document_id,
                    tmdb_id = EXCLUDED.tmdb_id,
                    tvdb_id = EXCLUDED.tvdb_id,
                    season_number = EXCLUDED.season_number,
                    episode_number = EXCLUDED.episode_number,
                    parent_series_id = EXCLUDED.parent_series_id,
                    last_synced_at = NOW()
                RETURNING item_id
                """,
                str(config_id),
                external_id,
                external_type,
                title,
                metadata_hash,
                vector_document_id,
                tmdb_id,
                tvdb_id,
                season_number,
                episode_number,
                parent_series_id
            )
            
            return UUID(str(item_id))
        except Exception as e:
            logger.error(f"Failed to upsert sync item: {e}")
            raise
    
    async def delete_sync_item(self, item_id: UUID) -> bool:
        """
        Delete a sync item
        
        Args:
            item_id: Item ID
            
        Returns:
            True if successful
        """
        try:
            await execute(
                "DELETE FROM entertainment_sync_items WHERE item_id = $1",
                str(item_id)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete sync item {item_id}: {e}")
            return False
    
    async def get_sync_items_by_external_ids(
        self, config_id: UUID, external_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get sync items by external IDs (for change detection)
        
        Args:
            config_id: Configuration ID
            external_ids: List of external IDs
            
        Returns:
            Dict mapping external_id to item data
        """
        try:
            if not external_ids:
                return {}
            
            placeholders = ','.join([f'${i+1}' for i in range(len(external_ids))])
            rows = await fetch_all(
                f"""
                SELECT * FROM entertainment_sync_items
                WHERE config_id = ${len(external_ids) + 1}
                AND external_id IN ({placeholders})
                """,
                *external_ids,
                str(config_id)
            )
            
            result = {}
            for row in rows:
                row_dict = dict(row)
                external_id = row_dict['external_id']
                external_type = row_dict['external_type']
                key = f"{external_id}_{external_type}"
                result[key] = row_dict
            
            return result
        except Exception as e:
            logger.error(f"Failed to get sync items by external IDs: {e}")
            return {}

