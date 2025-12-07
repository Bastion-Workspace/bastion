"""
Settings Service - Handles persistent application configuration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from services.database_manager.database_helpers import fetch_all, fetch_one, execute
from sqlalchemy import text  # kept only for type hints in comments; remove if unused

from config import settings

logger = logging.getLogger(__name__)


class SettingsService:
    """Service for managing persistent application settings"""
    
    def __init__(self):
        # Use shared database manager helpers to avoid extra connection pools
        self._settings_cache = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the settings service"""
        try:
            logger.info("🔧 Initializing Settings Service...")
            
            # Test connection
            await self._test_connection()
            
            # Load settings into cache
            await self._load_settings_cache()
            
            self._initialized = True
            logger.info("✅ Settings Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Settings Service: {e}")
            raise
    
    async def _test_connection(self):
        """Test database connection"""
        try:
            _ = await fetch_one("SELECT 1")
            logger.info("✅ Database connection successful")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    async def _load_settings_cache(self):
        """Load all settings into memory cache"""
        try:
            rows = await fetch_all("SELECT key, value, data_type FROM settings")
            for row in rows:
                key, value, value_type = row["key"], row["value"], row["data_type"]
                self._settings_cache[key] = self._convert_value(value, value_type)
            logger.info(f"📚 Loaded {len(self._settings_cache)} settings into cache")
        except Exception as e:
            logger.error(f"❌ Failed to load settings cache: {e}")
            # Continue with empty cache if database is not ready
            self._settings_cache = {}
    
    def _convert_value(self, value: str, value_type: str) -> Any:
        """Convert string value to appropriate type"""
        if value is None:
            return None
        
        try:
            if value_type == "integer":
                return int(value)
            elif value_type == "float":
                return float(value)
            elif value_type == "boolean":
                return value.lower() in ("true", "1", "yes", "on")
            elif value_type == "json":
                return json.loads(value)
            else:  # string
                return value
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"⚠️ Failed to convert setting value '{value}' to {value_type}: {e}")
            return value
    
    def _convert_to_string(self, value: Any, value_type: str) -> str:
        """Convert value to string for database storage"""
        if value is None:
            return None
        
        if value_type == "json":
            return json.dumps(value)
        elif value_type == "boolean":
            return "true" if value else "false"
        else:
            return str(value)
    
    async def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        if not self._initialized:
            logger.warning("⚠️ Settings service not initialized, using default")
            return default
        
        # Try cache first
        if key in self._settings_cache:
            return self._settings_cache[key]
        
        # Fallback to database
        try:
            row = await fetch_one("SELECT value, data_type FROM settings WHERE key = $1", key)
            if row:
                value, value_type = row["value"], row["data_type"]
                converted_value = self._convert_value(value, value_type)
                self._settings_cache[key] = converted_value
                return converted_value
        except Exception as e:
            logger.error(f"❌ Failed to get setting '{key}': {e}")
        
        return default
    
    async def set_setting(self, key: str, value: Any, value_type: str = "string", 
                         description: str = None, category: str = "general") -> bool:
        """Set a setting value"""
        if not self._initialized:
            logger.warning("⚠️ Settings service not initialized")
            return False
        
        try:
            string_value = self._convert_to_string(value, value_type)
            await execute(
                """
                INSERT INTO settings (key, value, data_type, description, category, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    data_type = EXCLUDED.data_type,
                    description = COALESCE(EXCLUDED.description, settings.description),
                    category = COALESCE(EXCLUDED.category, settings.category),
                    updated_at = EXCLUDED.updated_at
                """,
                key, string_value, value_type, description, category
            )
            self._settings_cache[key] = value
            logger.info(f"✅ Setting '{key}' updated to '{value}'")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to set setting '{key}': {e}")
            return False
    
    async def get_settings_by_category(self, category: str) -> Dict[str, Any]:
        """Get all settings in a category"""
        if not self._initialized:
            return {}
        
        try:
            rows = await fetch_all(
                "SELECT key, value, data_type FROM settings WHERE category = $1",
                category,
            )
            settings_dict = {}
            for row in rows:
                key, value, value_type = row["key"], row["value"], row["data_type"]
                converted_value = self._convert_value(value, value_type)
                display_key = key
                prefix = f"{category}."
                if key.startswith(prefix):
                    display_key = key[len(prefix):]
                settings_dict[display_key] = converted_value
                self._settings_cache[key] = converted_value
            return settings_dict
        except Exception as e:
            logger.error(f"❌ Failed to get settings for category '{category}': {e}")
            return {}
    
    async def get_all_settings(self) -> Dict[str, Dict[str, Any]]:
        """Get all settings grouped by category"""
        if not self._initialized:
            return {}
        
        try:
            rows = await fetch_all(
                """
                SELECT key, value, data_type, description, category
                FROM settings
                ORDER BY category, key
                """
            )
            settings_by_category = {}
            for row in rows:
                key = row["key"]
                value = row["value"]
                value_type = row["data_type"]
                description = row["description"]
                category = row["category"]
                if category not in settings_by_category:
                    settings_by_category[category] = {}
                display_value = self._convert_value(value, value_type)
                display_key = key
                prefix = f"{category}."
                if key.startswith(prefix):
                    display_key = key[len(prefix):]
                settings_by_category[category][display_key] = {
                    "value": display_value,
                    "type": value_type,
                    "description": description
                }
                self._settings_cache[key] = display_value
            return settings_by_category
        except Exception as e:
            logger.error(f"❌ Failed to get all settings: {e}")
            return {}
    
    async def delete_setting(self, key: str) -> bool:
        """Delete a setting"""
        if not self._initialized:
            return False
        
        try:
            result = await execute("DELETE FROM settings WHERE key = $1", key)
            if key in self._settings_cache:
                del self._settings_cache[key]
            # execute returns e.g., "DELETE 1"; treat presence of "DELETE" as success indicator
            return isinstance(result, str) and result.startswith("DELETE")
        except Exception as e:
            logger.error(f"❌ Failed to delete setting '{key}': {e}")
            return False
    
    async def bulk_update_settings(self, settings_dict: Dict[str, Any]) -> Dict[str, bool]:
        """Update multiple settings at once"""
        results = {}
        
        for key, value in settings_dict.items():
            # Determine value type
            if isinstance(value, bool):
                value_type = "boolean"
            elif isinstance(value, int):
                value_type = "integer"
            elif isinstance(value, float):
                value_type = "float"
            elif isinstance(value, (dict, list)):
                value_type = "json"
            else:
                value_type = "string"
            
            results[key] = await self.set_setting(key, value, value_type)
        
        return results
    
    # Convenience methods for common settings
    async def get_llm_model(self) -> str:
        """Get current LLM model"""
        return await self.get_setting("llm_model", settings.DEFAULT_MODEL or "")
    
    async def set_llm_model(self, model: str) -> bool:
        """Set LLM model"""
        return await self.set_setting(
            "llm_model", 
            model, 
            "string", 
            "Default LLM model for chat and queries", 
            "llm"
        )
    
    async def get_llm_temperature(self) -> float:
        """Get LLM temperature"""
        return await self.get_setting("llm_temperature", 0.7)
    
    async def set_llm_temperature(self, temperature: float) -> bool:
        """Set LLM temperature"""
        return await self.set_setting(
            "llm_temperature", 
            temperature, 
            "float", 
            "Temperature setting for LLM responses (0.0-1.0)", 
            "llm"
        )
    
    async def get_rag_settings(self) -> Dict[str, Any]:
        """Get all RAG-related settings"""
        return await self.get_settings_by_category("rag")
    
    async def get_ui_settings(self) -> Dict[str, Any]:
        """Get all UI-related settings"""
        return await self.get_settings_by_category("ui")
    
    async def get_enabled_models(self) -> List[str]:
        """Get list of enabled model IDs"""
        enabled_models = await self.get_setting("enabled_models", [])
        return enabled_models if isinstance(enabled_models, list) else []
    
    async def set_enabled_models(self, model_ids: List[str]) -> bool:
        """Set list of enabled model IDs"""
        return await self.set_setting(
            "enabled_models",
            model_ids,
            "json",
            "List of enabled OpenRouter model IDs",
            "llm"
        )
    
    async def get_classification_model(self) -> str:
        """Get current classification model (fast model for intent classification)"""
        classification_model = await self.get_setting("classification_model", None)
        if classification_model:
            return classification_model
        
        # Fallback to main LLM model if no classification model is set
        return await self.get_llm_model()
    
    async def set_classification_model(self, model: str) -> bool:
        """Set classification model (fast model for intent classification)"""
        return await self.set_setting(
            "classification_model", 
            model, 
            "string", 
            "Fast LLM model for intent classification (separate from main chat model)", 
            "llm"
        )

    async def get_text_completion_model(self) -> Optional[str]:
        """Get preferred fast text-completion model (separate from chat model)."""
        return await self.get_setting("text_completion_model", None)

    async def set_text_completion_model(self, model: str) -> bool:
        """Set preferred fast text-completion model (separate from chat model)."""
        return await self.set_setting(
            "text_completion_model",
            model,
            "string",
            "Fast text-completion model for editor/proofreading tasks",
            "llm"
        )

    async def get_image_generation_model(self) -> str:
        """Get current image generation model (for OpenRouter image models)."""
        return await self.get_setting("image_generation_model", "")

    async def set_image_generation_model(self, model: str) -> bool:
        """Set image generation model used for creating images via OpenRouter."""
        return await self.set_setting(
            "image_generation_model",
            model,
            "string",
            "OpenRouter model used for image generation",
            "llm"
        )
    
    async def get_user_timezone(self, user_id: str) -> str:
        """Get user's timezone preference"""
        try:
            row = await fetch_one("SELECT preferences FROM users WHERE user_id = $1", user_id)
            if row and (row.get("preferences") is not None or (len(row) > 0 and row[0] is not None)):
                # Access by key if dict-like, else by index
                prefs = row.get("preferences") if isinstance(row, dict) else row[0]
                if isinstance(prefs, str):
                    try:
                        prefs = json.loads(prefs)
                    except Exception:
                        prefs = {}
                if isinstance(prefs, dict):
                    return prefs.get("timezone", "UTC")
            return "UTC"
        except Exception as e:
            logger.warning(f"Failed to get timezone for user {user_id}: {e}")
            return "UTC"
    
    async def set_user_timezone(self, user_id: str, timezone: str) -> bool:
        """Set user's timezone preference"""
        try:
            row = await fetch_one("SELECT preferences FROM users WHERE user_id = $1", user_id)
            if not row:
                logger.warning(f"User {user_id} not found")
                return False
            prefs = row.get("preferences") if isinstance(row, dict) else row[0]
            if isinstance(prefs, str):
                try:
                    prefs = json.loads(prefs)
                except Exception:
                    prefs = {}
            if not isinstance(prefs, dict):
                prefs = {}
            prefs["timezone"] = timezone
            await execute("UPDATE users SET preferences = $1, updated_at = NOW() WHERE user_id = $2", json.dumps(prefs), user_id)
            logger.info(f"Updated timezone for user {user_id} to {timezone}")
            return True
        except Exception as e:
            logger.error(f"Failed to set timezone for user {user_id}: {e}")
            return False
    
    async def get_user_prompt_settings(self, user_id: str):
        """Get prompt settings for a specific user"""
        try:
            from services.prompt_service import UserPromptSettings, PoliticalBias, PersonaStyle
            rows = await fetch_all(
                "SELECT key, value FROM user_settings WHERE user_id = $1 AND key LIKE 'prompt_%'",
                user_id,
            )
            if not rows:
                return UserPromptSettings()
            settings_dict = {}
            for row in rows:
                key = row["key"] if isinstance(row, dict) else row[0]
                value = row["value"] if isinstance(row, dict) else row[1]
                clean_key = key.replace('prompt_', '')
                settings_dict[clean_key] = value
            return UserPromptSettings(
                ai_name=settings_dict.get('ai_name', 'Alex'),
                political_bias=PoliticalBias(settings_dict.get('political_bias', 'neutral')),
                persona_style=PersonaStyle(settings_dict.get('persona_style', 'professional'))
            )
        except Exception as e:
            logger.error(f"❌ Failed to get prompt settings for user {user_id}: {str(e)}")
            from services.prompt_service import UserPromptSettings
            return UserPromptSettings()

    async def save_user_prompt_settings(self, user_id: str, user_settings) -> bool:
        """Save prompt settings for a specific user"""
        try:
            settings_to_save = {
                'prompt_ai_name': (user_settings.ai_name, 'string'),
                'prompt_political_bias': (user_settings.political_bias.value, 'string'),
                'prompt_persona_style': (user_settings.persona_style.value, 'string')
            }
            for key, (value, data_type) in settings_to_save.items():
                # Try update first
                update_result = await execute(
                    "UPDATE user_settings SET value = $3, data_type = $4, updated_at = NOW() WHERE user_id = $1 AND key = $2",
                    user_id, key, value, data_type
                )
                updated = isinstance(update_result, str) and update_result.startswith("UPDATE") and update_result.endswith(" 1")
                if not updated:
                    # Insert new record (use ON CONFLICT if unique constraint exists)
                    try:
                        await execute(
                            """
                            INSERT INTO user_settings (user_id, key, value, data_type, created_at, updated_at)
                            VALUES ($1, $2, $3, $4, NOW(), NOW())
                            ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value, data_type = EXCLUDED.data_type, updated_at = NOW()
                            """,
                            user_id, key, value, data_type
                        )
                    except Exception:
                        # Fallback without ON CONFLICT
                        await execute(
                            "INSERT INTO user_settings (user_id, key, value, data_type, created_at, updated_at) VALUES ($1, $2, $3, $4, NOW(), NOW())",
                            user_id, key, value, data_type
                        )
            logger.info(f"✅ Saved prompt settings for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save prompt settings for user {user_id}: {str(e)}")
            return False

    async def close(self):
        """Clean up resources"""
        # No dedicated engine to close when using shared database helpers
        logger.info("🔄 Settings Service closed")


# Global settings service instance
settings_service = SettingsService()
