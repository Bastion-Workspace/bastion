"""
Org-Mode Settings Service
Manages user-specific org-mode configuration and preferences
"""

import logging
import json
from typing import Optional
from datetime import datetime

from config import settings
from models.org_settings_models import (
    OrgModeSettings,
    OrgModeSettingsUpdate,
    TodoStateSequence,
    AgendaPreferences,
    DisplayPreferences,
    JournalPreferences
)

logger = logging.getLogger(__name__)


class OrgSettingsService:
    """
    Service for managing org-mode settings
    
    **BULLY!** Handles all org-mode configuration like a well-organized cavalry!
    """
    
    def __init__(self):
        pass
    
    async def _get_pool(self):
        """Get database pool from DatabaseManager"""
        from services.database_manager.database_manager_service import get_database_manager
        db_manager = await get_database_manager()
        # Access private _pool attribute (it's the asyncpg pool)
        return db_manager._pool
    
    def _get_default_settings(self, user_id: str) -> OrgModeSettings:
        """
        Get default org-mode settings

        **By George!** Sensible defaults for a fresh start!
        """
        return OrgModeSettings(
            user_id=user_id,
            inbox_file=None,  # Auto-discover
            refile_max_level=2,  # Show * and ** headings
            todo_sequences=[
                TodoStateSequence(
                    name="Default",
                    active_states=["TODO", "NEXT", "WAITING"],
                    done_states=["DONE", "CANCELED"],
                    is_default=True
                )
            ],
            tags=[],
            agenda_preferences=AgendaPreferences(),
            display_preferences=DisplayPreferences(),
            journal_preferences=None,  # Will use default from model
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    async def get_settings(self, user_id: str) -> OrgModeSettings:
        """
        Get org-mode settings for a user
        
        If no settings exist, returns default settings
        """
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # Try to fetch existing settings
                row = await conn.fetchrow(
                    "SELECT settings_json, created_at, updated_at FROM org_settings WHERE user_id = $1",
                    user_id
                )
                
                if row:
                    # Parse settings from JSON
                    logger.info(f"🔍 ROOSEVELT: Raw settings_json type: {type(row['settings_json'])}")
                    logger.info(f"🔍 ROOSEVELT: Raw settings_json content: {row['settings_json']}")
                    
                    # Handle both dict and string JSON
                    if isinstance(row['settings_json'], dict):
                        settings_dict = row['settings_json'].copy()
                    else:
                        # Parse JSON string
                        settings_dict = json.loads(row['settings_json'])
                    
                    settings_dict['user_id'] = user_id
                    settings_dict['created_at'] = row['created_at']
                    settings_dict['updated_at'] = row['updated_at']
                    
                    logger.info(f"✅ ROOSEVELT: Loaded settings for user {user_id}")
                    return OrgModeSettings(**settings_dict)
                else:
                    # Return default settings
                    logger.info(f"📝 ROOSEVELT: No settings found for user {user_id}, using defaults")
                    return self._get_default_settings(user_id)
        
        except Exception as e:
            logger.error(f"❌ Failed to get org settings for user {user_id}: {e}")
            # Return defaults on error
            return self._get_default_settings(user_id)
    
    async def create_or_update_settings(
        self,
        user_id: str,
        settings_update: OrgModeSettingsUpdate
    ) -> OrgModeSettings:
        """
        Create or update org-mode settings for a user
        
        **BULLY!** Persistent configuration like a well-drilled cavalry!
        """
        try:
            # Get existing settings or defaults
            current_settings = await self.get_settings(user_id)
            
            # Update with new values (only update provided fields)
            if settings_update.inbox_file is not None:
                current_settings.inbox_file = settings_update.inbox_file
            
            if settings_update.refile_max_level is not None:
                current_settings.refile_max_level = settings_update.refile_max_level
            
            if settings_update.todo_sequences is not None:
                current_settings.todo_sequences = settings_update.todo_sequences
            
            if settings_update.tags is not None:
                current_settings.tags = settings_update.tags
            
            if settings_update.agenda_preferences is not None:
                current_settings.agenda_preferences = settings_update.agenda_preferences
            
            if settings_update.display_preferences is not None:
                current_settings.display_preferences = settings_update.display_preferences
            
            if settings_update.journal_preferences is not None:
                journal_prefs = settings_update.journal_preferences
                
                # If journal is being enabled, validate location
                if journal_prefs.enabled and journal_prefs.journal_location:
                    # Get username for path resolution
                    from services.database_manager.database_helpers import fetch_one
                    row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
                    username = row['username'] if row else user_id
                    
                    # Validate folder exists
                    from services.org_journal_service import get_org_journal_service
                    journal_service = await get_org_journal_service()
                    valid, error_msg = await journal_service._validate_journal_location(
                        user_id, username, journal_prefs.journal_location
                    )
                    
                    if not valid:
                        raise ValueError(f"Journal location validation failed: {error_msg}")
                
                current_settings.journal_preferences = journal_prefs

            # Update timestamp
            current_settings.updated_at = datetime.now()
            
            # Convert to JSON for storage
            settings_dict = current_settings.dict(exclude={'user_id', 'created_at', 'updated_at'})
            settings_json = json.dumps(settings_dict)
            
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # Upsert settings
                await conn.execute("""
                    INSERT INTO org_settings (user_id, settings_json, created_at, updated_at)
                    VALUES ($1, $2::jsonb, $3, $4)
                    ON CONFLICT (user_id)
                    DO UPDATE SET
                        settings_json = $2::jsonb,
                        updated_at = $4
                """, user_id, settings_json, current_settings.created_at, current_settings.updated_at)
            
            logger.info(f"✅ ROOSEVELT: Saved org settings for user {user_id}")
            return current_settings
        
        except Exception as e:
            logger.error(f"❌ Failed to save org settings for user {user_id}: {e}")
            raise
    
    async def delete_settings(self, user_id: str) -> bool:
        """
        Delete org-mode settings for a user (reset to defaults)
        """
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM org_settings WHERE user_id = $1",
                    user_id
                )
            
            logger.info(f"✅ ROOSEVELT: Deleted org settings for user {user_id}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to delete org settings for user {user_id}: {e}")
            return False
    
    async def get_todo_states(self, user_id: str) -> dict:
        """
        Get all TODO states for a user (active and done)
        
        Returns:
            {
                "active": ["TODO", "NEXT", "WAITING"],
                "done": ["DONE", "CANCELED"],
                "all": ["TODO", "NEXT", "WAITING", "DONE", "CANCELED"]
            }
        """
        settings = await self.get_settings(user_id)
        
        # Collect all states from all sequences
        active_states = set()
        done_states = set()
        
        for sequence in settings.todo_sequences:
            active_states.update(sequence.active_states)
            done_states.update(sequence.done_states)
        
        all_states = list(active_states | done_states)
        
        return {
            "active": sorted(list(active_states)),
            "done": sorted(list(done_states)),
            "all": sorted(all_states)
        }
    
    async def get_tags(self, user_id: str) -> list:
        """
        Get all predefined tags for a user
        """
        settings = await self.get_settings(user_id)
        return [tag.dict() for tag in settings.tags]


# Singleton instance
_org_settings_service = None


async def get_org_settings_service():
    """
    Get or create the org settings service singleton
    
    **BULLY!** One service to rule them all!
    """
    global _org_settings_service
    
    if _org_settings_service is None:
        _org_settings_service = OrgSettingsService()
        logger.info("✅ ROOSEVELT: Org Settings Service initialized!")
    
    return _org_settings_service

