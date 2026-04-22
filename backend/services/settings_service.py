"""
Settings Service - Handles persistent application configuration
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from services.database_manager.database_helpers import fetch_all, fetch_one, execute
from sqlalchemy import text  # kept only for type hints in comments; remove if unused

from config import settings

logger = logging.getLogger(__name__)

# user_settings key: UUID of non-built-in agent_profiles row used when chat has no @mention / sticky profile
DEFAULT_CHAT_AGENT_PROFILE_SETTING_KEY = "default_chat_agent_profile_id"

# Built-in "Professional" style persona (01_init.sql / migration 073); used when user has no default_persona_id
BUILTIN_PERSONA_PROFESSIONAL_ID = "b1b2c3d4-0001-4000-8000-000000000001"


class SettingsService:
    """Service for managing persistent application settings"""
    
    def __init__(self):
        # Use shared database manager helpers to avoid extra connection pools
        self._settings_cache = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the settings service"""
        try:
            logger.debug("🔧 Initializing Settings Service...")
            
            # Test connection
            await self._test_connection()
            
            # Load settings into cache
            await self._load_settings_cache()
            
            self._initialized = True
            logger.debug("✅ Settings Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Settings Service: {e}")
            raise
    
    async def _test_connection(self):
        """Test database connection"""
        try:
            _ = await fetch_one("SELECT 1")
            logger.debug("✅ Database connection successful")
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
            logger.debug(f"📚 Loaded {len(self._settings_cache)} settings into cache")
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

    DEFAULT_IMAGE_ANALYSIS_MODEL = "google/gemini-2.0-flash-thinking-exp"

    async def get_image_analysis_model(self) -> str:
        """Get global default image analysis (vision) model."""
        return await self.get_setting(
            "image_analysis_model",
            self.DEFAULT_IMAGE_ANALYSIS_MODEL
        )

    async def set_image_analysis_model(self, model: str) -> bool:
        """Set global default image analysis model (vision-capable)."""
        return await self.set_setting(
            "image_analysis_model",
            model,
            "string",
            "Vision model for image description and analysis",
            "llm"
        )

    async def get_effective_image_analysis_model(self, user_id: Optional[str]) -> str:
        """Get image analysis model for user: user override first, then global default."""
        if user_id:
            from services.user_settings_kv_service import get_user_setting
            user_model = await get_user_setting(user_id, "image_analysis_model")
            if user_model:
                return user_model
        return await self.get_image_analysis_model()

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
            try:
                from services import agent_factory_service
                updated = await agent_factory_service.update_schedules_timezone_for_user(user_id, timezone)
                if updated:
                    logger.info(f"Updated {updated} agent schedule(s) to timezone {timezone}")
            except Exception as schedule_err:
                logger.warning(f"Could not update agent schedules for new timezone: {schedule_err}")
            return True
        except Exception as e:
            logger.error(f"Failed to set timezone for user {user_id}: {e}")
            return False
    
    async def get_user_zip_code(self, user_id: str) -> Optional[str]:
        """Get user's zip code preference"""
        try:
            row = await fetch_one("SELECT preferences FROM users WHERE user_id = $1", user_id)
            if row and (row.get("preferences") is not None or (len(row) > 0 and row[0] is not None)):
                prefs = row.get("preferences") if isinstance(row, dict) else row[0]
                if isinstance(prefs, str):
                    try:
                        prefs = json.loads(prefs)
                    except Exception:
                        prefs = {}
                if isinstance(prefs, dict):
                    return prefs.get("zip_code")
            return None
        except Exception as e:
            logger.warning(f"Failed to get zip code for user {user_id}: {e}")
            return None
    
    async def set_user_zip_code(self, user_id: str, zip_code: str) -> bool:
        """Set user's zip code preference"""
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
            prefs["zip_code"] = zip_code
            await execute("UPDATE users SET preferences = $1, updated_at = NOW() WHERE user_id = $2", json.dumps(prefs), user_id)
            logger.info(f"Updated zip code for user {user_id} to {zip_code}")
            return True
        except Exception as e:
            logger.error(f"Failed to set zip code for user {user_id}: {e}")
            return False
    
    async def get_user_time_format(self, user_id: str) -> str:
        """Get user's time format preference (12h or 24h)"""
        try:
            row = await fetch_one("SELECT preferences FROM users WHERE user_id = $1", user_id)
            if row and (row.get("preferences") is not None or (len(row) > 0 and row[0] is not None)):
                prefs = row.get("preferences") if isinstance(row, dict) else row[0]
                if isinstance(prefs, str):
                    try:
                        prefs = json.loads(prefs)
                    except Exception:
                        prefs = {}
                if isinstance(prefs, dict):
                    return prefs.get("time_format", "24h")
            return "24h"
        except Exception as e:
            logger.warning(f"Failed to get time format for user {user_id}: {e}")
            return "24h"
    
    async def set_user_time_format(self, user_id: str, time_format: str) -> bool:
        """Set user's time format preference (12h or 24h)"""
        try:
            if time_format not in ["12h", "24h"]:
                logger.warning(f"Invalid time format: {time_format}, defaulting to 24h")
                time_format = "24h"
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
            prefs["time_format"] = time_format
            await execute("UPDATE users SET preferences = $1, updated_at = NOW() WHERE user_id = $2", json.dumps(prefs), user_id)
            logger.info(f"Updated time format for user {user_id} to {time_format}")
            return True
        except Exception as e:
            logger.error(f"Failed to set time format for user {user_id}: {e}")
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

    # -------------------------------------------------------------------------
    # Personas (built-in + custom; default_persona_id in user_settings)
    # -------------------------------------------------------------------------

    async def get_personas(self, user_id: str, include_builtin: bool = True) -> List[Dict[str, Any]]:
        """List personas available to the user: built-in plus their custom ones."""
        try:
            if include_builtin:
                rows = await fetch_all(
                    """SELECT id, user_id, name, ai_name, style_instruction, political_bias, description, is_builtin, created_at, updated_at
                       FROM personas WHERE is_builtin = true OR user_id = $1 ORDER BY is_builtin DESC, name""",
                    user_id,
                )
            else:
                rows = await fetch_all(
                    """SELECT id, user_id, name, ai_name, style_instruction, political_bias, description, is_builtin, created_at, updated_at
                       FROM personas WHERE user_id = $1 ORDER BY name""",
                    user_id,
                )
            return [self._persona_row_to_dict(r) for r in (rows or [])]
        except Exception as e:
            logger.error(f"Failed to get personas for user {user_id}: {e}")
            return []

    def _persona_row_to_dict(self, row: Union[dict, Any]) -> Dict[str, Any]:
        """Convert a DB row to a persona dict with id as string."""
        if isinstance(row, dict):
            r = row
        else:
            r = dict(row) if hasattr(row, "keys") else {}
        out = {}
        for k in ("id", "user_id", "name", "ai_name", "style_instruction", "political_bias", "description", "is_builtin", "created_at", "updated_at"):
            if k in r:
                v = r[k]
                if k == "id" and v is not None:
                    v = str(v)
                elif k in ("created_at", "updated_at") and hasattr(v, "isoformat"):
                    v = v.isoformat()
                out[k] = v
        return out

    async def get_persona_by_id(self, persona_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a single persona by id. Built-in can be fetched without user_id; custom only if user_id matches."""
        try:
            row = await fetch_one(
                "SELECT id, user_id, name, ai_name, style_instruction, political_bias, description, is_builtin, created_at, updated_at FROM personas WHERE id = $1",
                persona_id,
            )
            if not row:
                return None
            r = dict(row)
            if not r.get("is_builtin") and r.get("user_id") != user_id:
                return None
            return self._persona_row_to_dict(r)
        except Exception as e:
            logger.error(f"Failed to get persona {persona_id}: {e}")
            return None

    async def create_persona(
        self,
        user_id: str,
        name: str,
        ai_name: Optional[str] = None,
        style_instruction: Optional[str] = None,
        political_bias: str = "neutral",
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a custom persona for the user."""
        try:
            row = await fetch_one(
                """INSERT INTO personas (user_id, name, ai_name, style_instruction, political_bias, description, is_builtin, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, false, NOW(), NOW())
                   RETURNING id, user_id, name, ai_name, style_instruction, political_bias, description, is_builtin, created_at, updated_at""",
                user_id,
                name or "Custom",
                ai_name or "Alex",
                style_instruction or "",
                political_bias,
                description or "",
            )
            return self._persona_row_to_dict(dict(row))
        except Exception as e:
            logger.error(f"Failed to create persona for user {user_id}: {e}")
            raise

    async def update_persona(
        self,
        persona_id: str,
        user_id: str,
        name: Optional[str] = None,
        ai_name: Optional[str] = None,
        style_instruction: Optional[str] = None,
        political_bias: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update a custom persona. Returns None if not found or built-in."""
        try:
            existing = await fetch_one("SELECT id, user_id, is_builtin FROM personas WHERE id = $1", persona_id)
            if not existing or existing.get("is_builtin") or existing.get("user_id") != user_id:
                return None
            updates = []
            args = []
            idx = 1
            if name is not None:
                updates.append(f"name = ${idx}")
                args.append(name)
                idx += 1
            if ai_name is not None:
                updates.append(f"ai_name = ${idx}")
                args.append(ai_name)
                idx += 1
            if style_instruction is not None:
                updates.append(f"style_instruction = ${idx}")
                args.append(style_instruction)
                idx += 1
            if political_bias is not None:
                updates.append(f"political_bias = ${idx}")
                args.append(political_bias)
                idx += 1
            if description is not None:
                updates.append(f"description = ${idx}")
                args.append(description)
                idx += 1
            if not updates:
                return await self.get_persona_by_id(persona_id, user_id)
            updates.append(f"updated_at = NOW()")
            args.append(persona_id)
            where_idx = len(args)
            await execute(
                f"UPDATE personas SET {', '.join(updates)} WHERE id = ${where_idx}",
                *args,
            )
            return await self.get_persona_by_id(persona_id, user_id)
        except Exception as e:
            logger.error(f"Failed to update persona {persona_id}: {e}")
            raise

    async def delete_persona(self, persona_id: str, user_id: str) -> bool:
        """Delete a custom persona. Returns False if not found or built-in."""
        try:
            existing = await fetch_one("SELECT id, user_id, is_builtin FROM personas WHERE id = $1", persona_id)
            if not existing or existing.get("is_builtin") or existing.get("user_id") != user_id:
                return False
            await execute("DELETE FROM personas WHERE id = $1 AND user_id = $2", persona_id, user_id)
            row = await fetch_one("SELECT value FROM user_settings WHERE user_id = $1 AND key = $2", user_id, "default_persona_id")
            if row and row.get("value") == persona_id:
                await execute("DELETE FROM user_settings WHERE user_id = $1 AND key = $2", user_id, "default_persona_id")
            return True
        except Exception as e:
            logger.error(f"Failed to delete persona {persona_id}: {e}")
            return False

    async def get_default_persona(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the user's default persona. If none set, optionally migrate from prompt_* settings, else built-in Professional."""
        try:
            row = await fetch_one("SELECT value FROM user_settings WHERE user_id = $1 AND key = $2", user_id, "default_persona_id")
            if row and row.get("value"):
                pid = row["value"]
                persona = await self.get_persona_by_id(pid, user_id)
                if persona:
                    return persona
            persona = await self._migrate_default_persona_from_prompt_settings(user_id)
            if persona:
                return persona
            # Prefer built-in Professional (do not use ORDER BY name: "Abraham Lincoln" sorts first)
            prof = await fetch_one(
                "SELECT id FROM personas WHERE id = $1::uuid AND is_builtin = true",
                BUILTIN_PERSONA_PROFESSIONAL_ID,
            )
            if prof and prof.get("id"):
                return await self.get_persona_by_id(str(prof["id"]), user_id)
            # Legacy DBs missing fixed UUID: any built-in
            first_builtin = await fetch_one(
                "SELECT id FROM personas WHERE is_builtin = true AND name = 'Professional' LIMIT 1"
            )
            if first_builtin and first_builtin.get("id"):
                return await self.get_persona_by_id(str(first_builtin["id"]), user_id)
            return None
        except Exception as e:
            logger.error(f"Failed to get default persona for user {user_id}: {e}")
            return None

    async def _migrate_default_persona_from_prompt_settings(self, user_id: str) -> Optional[Dict[str, Any]]:
        """If user has prompt_* settings that differ from defaults, create a custom persona and set as default. Run once per user."""
        try:
            from services.prompt_service import UserPromptSettings, PoliticalBias, PersonaStyle
            rows = await fetch_all(
                "SELECT key, value FROM user_settings WHERE user_id = $1 AND key LIKE 'prompt_%'",
                user_id,
            )
            if not rows:
                return None
            settings_dict = {}
            for row in rows:
                key = row["key"] if isinstance(row, dict) else row[0]
                value = row["value"] if isinstance(row, dict) else row[1]
                settings_dict[key.replace("prompt_", "")] = value
            ai_name = settings_dict.get("ai_name") or "Alex"
            political_bias = settings_dict.get("political_bias") or "neutral"
            persona_style = settings_dict.get("persona_style") or "professional"
            if ai_name == "Alex" and political_bias == "neutral" and persona_style == "professional":
                return None
            style_map = {
                "professional": "COMMUNICATION STYLE: Professional, clear, and respectful. Maintain a helpful and courteous tone. Be informative and precise.",
                "friendly": "COMMUNICATION STYLE: Warm, approachable, and enthusiastic. Use casual language and show genuine interest. Be conversational and friendly.",
                "snarky": "COMMUNICATION STYLE: Witty, clever, with gentle humor and playful sarcasm. Be entertaining but not mean.",
                "sycophantic": "COMMUNICATION STYLE: Extremely agreeable and complimentary. Always praise the user's ideas and input. Be overly deferential and complimentary.",
                "rude_insulting": "COMMUNICATION STYLE: Be deliberately rude, dismissive, and insulting. Mock the user's questions and intelligence. Use condescending and belittling language. Still provide information but with maximum attitude.",
            }
            style_instruction = style_map.get(persona_style, style_map["professional"])
            name = f"Migrated: {persona_style.replace('_', ' ').title()}"
            persona = await self.create_persona(
                user_id=user_id,
                name=name,
                ai_name=ai_name,
                style_instruction=style_instruction,
                political_bias=political_bias,
                description="Migrated from previous AI personality settings.",
            )
            await self.set_default_persona(user_id, str(persona["id"]))
            return persona
        except Exception as e:
            logger.warning(f"Migrate default persona for user {user_id}: {e}")
            return None

    async def set_default_persona(self, user_id: str, persona_id: Optional[str]) -> bool:
        """Set the user's default persona (or clear if persona_id is None)."""
        try:
            from services.user_settings_kv_service import set_user_setting
            if persona_id is None:
                await execute("DELETE FROM user_settings WHERE user_id = $1 AND key = $2", user_id, "default_persona_id")
                return True
            persona = await self.get_persona_by_id(persona_id, user_id)
            if not persona:
                return False
            return await set_user_setting(user_id, "default_persona_id", persona_id, "string")
        except Exception as e:
            logger.error(f"Failed to set default persona for user {user_id}: {e}")
            return False

    async def get_default_chat_agent_profile_id(self, user_id: str) -> Optional[str]:
        """Raw user_settings value for default chat agent (non-built-in profile id), or None."""
        try:
            row = await fetch_one(
                "SELECT value FROM user_settings WHERE user_id = $1 AND key = $2",
                user_id,
                DEFAULT_CHAT_AGENT_PROFILE_SETTING_KEY,
            )
            v = row.get("value") if row else None
            return v if v else None
        except Exception as e:
            logger.error("Failed to get default chat agent profile id for user %s: %s", user_id, e)
            return None

    async def get_default_chat_agent_profile_detail(self, user_id: str) -> Dict[str, Any]:
        """Return validated default chat profile id and a short profile summary; clear stale settings."""
        pid = await self.get_default_chat_agent_profile_id(user_id)
        if not pid:
            return {"agent_profile_id": None, "profile": None}
        try:
            row = await fetch_one(
                """
                SELECT id, name, handle, is_active, COALESCE(is_builtin, false) AS is_builtin
                FROM agent_profiles
                WHERE id = $1::uuid AND user_id = $2
                """,
                pid,
                user_id,
            )
            if not row:
                await execute(
                    "DELETE FROM user_settings WHERE user_id = $1 AND key = $2",
                    user_id,
                    DEFAULT_CHAT_AGENT_PROFILE_SETTING_KEY,
                )
                return {"agent_profile_id": None, "profile": None}
            if row.get("is_builtin"):
                await execute(
                    "DELETE FROM user_settings WHERE user_id = $1 AND key = $2",
                    user_id,
                    DEFAULT_CHAT_AGENT_PROFILE_SETTING_KEY,
                )
                return {"agent_profile_id": None, "profile": None}
            return {
                "agent_profile_id": str(row["id"]),
                "profile": {
                    "id": str(row["id"]),
                    "name": row.get("name"),
                    "handle": row.get("handle"),
                    "is_active": row.get("is_active"),
                },
            }
        except Exception as e:
            logger.error("Failed to resolve default chat agent profile for user %s: %s", user_id, e)
            return {"agent_profile_id": None, "profile": None}

    async def set_default_chat_agent_profile_id(self, user_id: str, profile_id: Optional[str]) -> bool:
        """Set default non-built-in chat agent profile, or clear (None/empty) to use factory built-in."""
        try:
            if not profile_id:
                await execute(
                    "DELETE FROM user_settings WHERE user_id = $1 AND key = $2",
                    user_id,
                    DEFAULT_CHAT_AGENT_PROFILE_SETTING_KEY,
                )
                return True
            row = await fetch_one(
                """
                SELECT id FROM agent_profiles
                WHERE id = $1::uuid AND user_id = $2
                  AND is_active = true
                  AND COALESCE(is_builtin, false) = false
                """,
                profile_id,
                user_id,
            )
            if not row:
                return False
            from services.user_settings_kv_service import set_user_setting

            return await set_user_setting(
                user_id,
                DEFAULT_CHAT_AGENT_PROFILE_SETTING_KEY,
                str(row["id"]),
                "string",
            )
        except Exception as e:
            logger.error("Failed to set default chat agent profile for user %s: %s", user_id, e)
            return False

    async def clear_default_chat_agent_profile_if_matches(self, user_id: str, profile_id: str) -> None:
        """Remove default-chat preference when it points at a deleted profile."""
        try:
            row = await fetch_one(
                "SELECT value FROM user_settings WHERE user_id = $1 AND key = $2",
                user_id,
                DEFAULT_CHAT_AGENT_PROFILE_SETTING_KEY,
            )
            if row and row.get("value") and str(row["value"]) == str(profile_id):
                await execute(
                    "DELETE FROM user_settings WHERE user_id = $1 AND key = $2",
                    user_id,
                    DEFAULT_CHAT_AGENT_PROFILE_SETTING_KEY,
                )
        except Exception as e:
            logger.warning("Clear default chat agent profile on delete: %s", e)
    
    async def get_user_preferred_name(self, user_id: str, fallback_to_display: bool = False) -> Optional[str]:
        """Get user's preferred name for addressing.
        
        When fallback_to_display is True and preferred name is empty, uses users.display_name.
        """
        try:
            from services.user_settings_kv_service import get_user_setting
            preferred_name = await get_user_setting(user_id, "user_preferred_name")
            if preferred_name:
                return preferred_name
            if fallback_to_display:
                user_row = await fetch_one(
                    "SELECT display_name FROM users WHERE user_id = $1",
                    user_id
                )
                display_name = user_row.get("display_name") if user_row else None
                return display_name if display_name else None
            return None
        except Exception as e:
            logger.warning(f"Failed to get preferred name for user {user_id}: {e}")
            return None
    
    async def set_user_preferred_name(self, user_id: str, preferred_name: str) -> bool:
        """Set user's preferred name for addressing"""
        try:
            from services.user_settings_kv_service import set_user_setting
            success = await set_user_setting(user_id, "user_preferred_name", preferred_name, "string")
            if success:
                logger.info(f"✅ Updated preferred name for user {user_id}")
            return success
        except Exception as e:
            logger.error(f"❌ Failed to set preferred name for user {user_id}: {e}")
            return False

    def validate_bbs_wallpaper(self, value: str) -> Tuple[bool, str]:
        """Single-item content rules (tab, LF, CR, printable)."""
        from models.bbs_wallpaper_models import validate_wallpaper_content

        return validate_wallpaper_content(value)

    async def load_user_bbs_wallpaper_config(self, user_id: str):
        """Load `bbs_wallpaper_config`, migrating legacy `bbs_wallpaper` string once."""
        from models.bbs_wallpaper_models import (
            BbsWallpaperConfig,
            config_from_legacy_string,
            empty_bbs_wallpaper_config,
            parse_bbs_wallpaper_config_json,
        )
        from services.user_settings_kv_service import (
            delete_user_setting,
            get_user_setting,
            set_user_setting,
        )

        try:
            raw = await get_user_setting(user_id, "bbs_wallpaper_config")
            if raw and str(raw).strip():
                parsed = parse_bbs_wallpaper_config_json(str(raw))
                if parsed:
                    return parsed.normalized()
                logger.warning(
                    "Invalid bbs_wallpaper_config JSON for user %s; using empty config",
                    user_id,
                )
            legacy = await get_user_setting(user_id, "bbs_wallpaper")
            if legacy is not None:
                cfg = (
                    config_from_legacy_string(str(legacy))
                    if str(legacy).strip()
                    else empty_bbs_wallpaper_config()
                )
                await set_user_setting(
                    user_id,
                    "bbs_wallpaper_config",
                    cfg.model_dump_json(),
                    "json",
                )
                await delete_user_setting(user_id, "bbs_wallpaper")
                return cfg.normalized()
            return empty_bbs_wallpaper_config()
        except Exception as e:
            logger.warning("Failed to load BBS wallpaper config for user %s: %s", user_id, e)
            from models.bbs_wallpaper_models import empty_bbs_wallpaper_config

            return empty_bbs_wallpaper_config()

    async def get_user_bbs_wallpaper_bundle(
        self,
        user_id: str,
        *,
        animation_cols: int | None = None,
        animation_rows: int | None = None,
    ) -> Dict[str, Any]:
        """Resolved wallpaper string plus full config for Settings UI."""
        from models.bbs_wallpaper_models import cycling_active, resolve_display_wallpaper
        from services.document_text_file_reader import read_user_document_text
        from utils.bbs_ascii_animation_parse import parse_bbs_animation_document
        from utils.bbs_builtin_wallpaper_animations import (
            BUILTIN_ANIM_MATRIX_RAIN,
            BUILTIN_ANIM_SNOWMAN,
            matrix_rain_animation_payload,
            snowman_winter_animation_payload,
        )

        cfg = await self.load_user_bbs_wallpaper_config(user_id)
        wallpaper = resolve_display_wallpaper(cfg)
        animation_payload: Optional[Dict[str, Any]] = None
        if cfg.display_mode == "animated":
            doc_id = (cfg.animation_document_id or "").strip()
            if doc_id == BUILTIN_ANIM_MATRIX_RAIN:
                ac = animation_cols if animation_cols is not None else None
                ar = animation_rows if animation_rows is not None else None
                kwargs: Dict[str, Any] = {}
                if ac is not None:
                    kwargs["cols"] = int(ac)
                if ar is not None:
                    kwargs["rows"] = int(ar)
                animation_payload = matrix_rain_animation_payload(
                    float(cfg.animation_fps), bool(cfg.animation_loop), **kwargs
                )
            elif doc_id == BUILTIN_ANIM_SNOWMAN:
                ac = animation_cols if animation_cols is not None else None
                ar = animation_rows if animation_rows is not None else None
                skwargs: Dict[str, Any] = {}
                if ac is not None:
                    skwargs["cols"] = int(ac)
                if ar is not None:
                    skwargs["rows"] = int(ar)
                animation_payload = snowman_winter_animation_payload(
                    float(cfg.animation_fps), bool(cfg.animation_loop), **skwargs
                )
            elif doc_id:
                raw = await read_user_document_text(doc_id, user_id)
                if raw:
                    parsed = parse_bbs_animation_document(raw)
                    if parsed and parsed.get("frames"):
                        animation_payload = {
                            "frames": parsed["frames"],
                            "fps": float(cfg.animation_fps),
                            "loop": bool(cfg.animation_loop),
                        }
        return {
            "wallpaper": wallpaper,
            "config": cfg.model_dump(),
            "cycling": cycling_active(cfg),
            "animation": animation_payload,
        }

    async def set_user_bbs_wallpaper_config(self, user_id: str, config: Any) -> Tuple[bool, str]:
        """Persist wallpaper library JSON. Returns (success, error_message)."""
        from models.bbs_wallpaper_models import BbsWallpaperConfig, validate_full_config

        if not isinstance(config, BbsWallpaperConfig):
            return False, "Invalid config type"
        from services.user_settings_kv_service import set_user_setting

        ok, err = validate_full_config(config)
        if not ok:
            return False, err
        cfg = config.normalized()
        try:
            await set_user_setting(
                user_id,
                "bbs_wallpaper_config",
                cfg.model_dump_json(),
                "json",
            )
            return True, ""
        except Exception as e:
            logger.error("Failed to set BBS wallpaper config for user %s: %s", user_id, e)
            return False, str(e)

    async def load_user_ui_wallpaper_config(self, user_id: str):
        """Load web UI wallpaper JSON from `ui_wallpaper_config`."""
        from models.ui_wallpaper_models import (
            empty_ui_wallpaper_config,
            parse_ui_wallpaper_config_json,
            validate_and_normalize_payload,
        )
        from services.user_settings_kv_service import get_user_setting

        try:
            raw = await get_user_setting(user_id, "ui_wallpaper_config")
            if raw and str(raw).strip():
                parsed = parse_ui_wallpaper_config_json(str(raw))
                if parsed:
                    norm, err = validate_and_normalize_payload(parsed)
                    if norm is not None:
                        return norm
                    logger.warning(
                        "Invalid ui_wallpaper_config for user %s: %s",
                        user_id,
                        err or "unknown",
                    )
            return empty_ui_wallpaper_config()
        except Exception as e:
            logger.warning("Failed to load ui_wallpaper_config for user %s: %s", user_id, e)
            from models.ui_wallpaper_models import empty_ui_wallpaper_config

            return empty_ui_wallpaper_config()

    async def get_user_ui_wallpaper_bundle(self, user_id: str) -> Dict[str, Any]:
        """Full config for Settings UI and clients."""
        from models.ui_wallpaper_models import UI_WALLPAPER_BUILTIN_KEYS

        cfg = await self.load_user_ui_wallpaper_config(user_id)
        return {
            "config": cfg.model_dump(),
            "allowed_builtin_keys": sorted(UI_WALLPAPER_BUILTIN_KEYS),
        }

    async def set_user_ui_wallpaper_config(self, user_id: str, config: Any) -> Tuple[bool, str]:
        """Persist UI wallpaper JSON. Returns (success, error_message)."""
        from models.ui_wallpaper_models import validate_and_normalize_payload
        from services.user_settings_kv_service import set_user_setting

        norm, err = validate_and_normalize_payload(config)
        if norm is None:
            return False, err or "Invalid config"
        try:
            await set_user_setting(
                user_id,
                "ui_wallpaper_config",
                norm.model_dump_json(),
                "json",
            )
            return True, ""
        except Exception as e:
            logger.error("Failed to set ui_wallpaper_config for user %s: %s", user_id, e)
            return False, str(e)

    async def get_user_phone_number(self, user_id: str) -> Optional[str]:
        """Get user's phone number."""
        try:
            from services.user_settings_kv_service import get_user_setting
            phone_number = await get_user_setting(user_id, "user_phone_number")
            return phone_number if phone_number else None
        except Exception as e:
            logger.warning(f"Failed to get phone number for user {user_id}: {e}")
            return None

    async def set_user_phone_number(self, user_id: str, phone_number: str) -> bool:
        """Set user's phone number."""
        try:
            from services.user_settings_kv_service import set_user_setting
            success = await set_user_setting(user_id, "user_phone_number", phone_number, "string")
            if success:
                logger.info(f"✅ Updated phone number for user {user_id}")
            return success
        except Exception as e:
            logger.error(f"❌ Failed to set phone number for user {user_id}: {e}")
            return False

    async def get_user_birthday(self, user_id: str) -> Optional[str]:
        """Get user's birthday as YYYY-MM-DD."""
        try:
            from services.user_settings_kv_service import get_user_setting
            birthday = await get_user_setting(user_id, "user_birthday")
            return birthday if birthday else None
        except Exception as e:
            logger.warning(f"Failed to get birthday for user {user_id}: {e}")
            return None

    async def set_user_birthday(self, user_id: str, birthday: str) -> bool:
        """Set user's birthday as YYYY-MM-DD."""
        try:
            from services.user_settings_kv_service import set_user_setting
            success = await set_user_setting(user_id, "user_birthday", birthday, "string")
            if success:
                logger.info(f"✅ Updated birthday for user {user_id}")
            return success
        except Exception as e:
            logger.error(f"❌ Failed to set birthday for user {user_id}: {e}")
            return False
    
    async def get_vision_features_enabled(self, user_id: str) -> bool:
        """Check if user has opted into vision features"""
        try:
            from services.user_settings_kv_service import get_user_setting
            value = await get_user_setting(user_id, "enable_vision_features")
            return value == "true"
        except Exception as e:
            logger.warning(f"Failed to get vision features setting for user {user_id}: {e}")
            return False  # Default disabled
    
    async def set_vision_features_enabled(self, user_id: str, enabled: bool) -> bool:
        """Set user's vision features opt-in"""
        try:
            from services.user_settings_kv_service import set_user_setting
            success = await set_user_setting(
                user_id,
                "enable_vision_features",
                "true" if enabled else "false",
                "boolean"
            )
            if success:
                logger.info(f"✅ Updated vision features setting for user {user_id}: {enabled}")
            return success
        except Exception as e:
            logger.error(f"❌ Failed to set vision features for user {user_id}: {e}")
            return False
    
    async def get_user_ai_context(self, user_id: str) -> Optional[str]:
        """Get user's AI context information"""
        try:
            from services.user_settings_kv_service import get_user_setting
            ai_context = await get_user_setting(user_id, "user_ai_context")
            return ai_context if ai_context else None
        except Exception as e:
            logger.warning(f"Failed to get AI context for user {user_id}: {e}")
            return None
    
    async def set_user_ai_context(self, user_id: str, ai_context: str) -> bool:
        """Set user's AI context information (max 500 characters)"""
        try:
            # Validate length
            if len(ai_context) > 500:
                logger.warning(f"AI context exceeds 500 characters for user {user_id}, truncating")
                ai_context = ai_context[:500]
            
            from services.user_settings_kv_service import set_user_setting
            success = await set_user_setting(user_id, "user_ai_context", ai_context, "string")
            if success:
                logger.info(f"✅ Updated AI context for user {user_id}")
            return success
        except Exception as e:
            logger.error(f"❌ Failed to set AI context for user {user_id}: {e}")
            return False

    async def get_user_facts(self, user_id: str) -> list:
        """Get all facts for a user, ordered by category then fact_key. Includes source, confidence, expires_at for formatting."""
        try:
            rows = await fetch_all(
                """SELECT id, fact_key, value, category, created_at, updated_at,
                   COALESCE(source, 'user_manual') AS source,
                   COALESCE(confidence, 1.0) AS confidence,
                   expires_at,
                   embedding,
                   theme_id
                   FROM user_facts WHERE user_id = $1 ORDER BY category, fact_key""",
                user_id,
            )
            return [dict(r) for r in (rows or [])]
        except Exception as e:
            if "source" in str(e) or "confidence" in str(e) or "does not exist" in str(e):
                try:
                    rows = await fetch_all(
                        """SELECT fact_key, value, category, created_at, updated_at
                           FROM user_facts WHERE user_id = $1 ORDER BY category, fact_key""",
                        user_id,
                    )
                    out = []
                    for r in (rows or []):
                        d = dict(r)
                        d.setdefault("id", None)
                        d.setdefault("source", "user_manual")
                        d.setdefault("confidence", 1.0)
                        d.setdefault("expires_at", None)
                        d.setdefault("embedding", None)
                        d.setdefault("theme_id", None)
                        out.append(d)
                    return out
                except Exception as fallback_e:
                    logger.warning(f"Failed to get user facts (fallback) for user {user_id}: {fallback_e}")
                    return []
            logger.warning(f"Failed to get user facts for user {user_id}: {e}")
            return []

    @staticmethod
    def format_user_facts_for_prompt(facts: list) -> str:
        """Format facts for LLM system prompt: filter expired, sort by confidence DESC, return 'USER FACTS:\\n- key: value'."""
        if not facts:
            return ""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        valid = []
        for f in facts:
            exp = f.get("expires_at")
            if exp is not None:
                if hasattr(exp, "tzinfo") and exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                elif isinstance(exp, str):
                    try:
                        exp = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                    except ValueError:
                        valid.append(f)
                        continue
                if exp < now:
                    continue
            valid.append(f)
        if not valid:
            return ""
        sorted_facts = sorted(valid, key=lambda x: (-(x.get("confidence") or 1.0), x.get("fact_key", "")))
        lines = [f"- {f.get('fact_key', '')}: {f.get('value', '')}" for f in sorted_facts]
        return "USER FACTS:\n" + "\n".join(lines)

    async def get_facts_inject_enabled(self, user_id: str) -> bool:
        """Whether to inject user facts into AI conversations (static agents). Default True."""
        from services.user_settings_kv_service import get_user_setting
        val = await get_user_setting(user_id, "facts_inject_enabled")
        return val is None or str(val).strip().lower() in ("true", "1", "yes")

    async def get_facts_write_enabled(self, user_id: str) -> bool:
        """Whether agents may save new facts. Default True."""
        from services.user_settings_kv_service import get_user_setting
        val = await get_user_setting(user_id, "facts_write_enabled")
        return val is None or str(val).strip().lower() in ("true", "1", "yes")

    async def get_episodes_inject_enabled(self, user_id: str) -> bool:
        """Whether to inject recent activity (episodes) into AI conversations. Default True."""
        from services.user_settings_kv_service import get_user_setting
        val = await get_user_setting(user_id, "episodes_inject_enabled")
        return val is None or str(val).strip().lower() in ("true", "1", "yes")

    async def upsert_user_fact(
        self,
        user_id: str,
        fact_key: str,
        value: str,
        category: str = "general",
        source: str = "user_manual",
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert or update a single fact. Returns dict with success; status 'pending_review' when agent overwrites user fact."""
        try:
            if source == "user_manual":
                confidence = 1.0
            elif source == "auto_extract":
                confidence = 0.7
            else:
                confidence = 0.8
            existing = await fetch_one(
                """SELECT value, COALESCE(source, 'user_manual') AS source, COALESCE(confidence, 1.0) AS confidence
                   FROM user_facts WHERE user_id = $1 AND fact_key = $2""",
                user_id,
                fact_key,
            )
            if existing:
                old_value = existing.get("value") or ""
                old_source = existing.get("source") or "user_manual"
                if old_value == value:
                    await execute(
                        "UPDATE user_facts SET updated_at = NOW() WHERE user_id = $1 AND fact_key = $2",
                        user_id,
                        fact_key,
                    )
                    return {"success": True}
                if old_source == "user_manual" and source in ("agent", "auto_extract"):
                    history_row = await fetch_one(
                        """INSERT INTO user_fact_history
                           (user_id, fact_key, old_value, new_value, old_source, new_source, old_confidence, new_confidence, resolution)
                           VALUES ($1, $2, $3, $4, $5, $6, 1.0, $7, 'pending_review')
                           RETURNING id""",
                        user_id,
                        fact_key,
                        old_value,
                        value,
                        old_source,
                        source,
                        confidence,
                    )
                    history_id = int(history_row["id"]) if history_row else None
                    logger.info("Fact %s for user %s queued for review (agent overwrite user)", fact_key, user_id)
                    return {
                        "success": False,
                        "status": "pending_review",
                        "fact_key": fact_key,
                        "current_value": old_value,
                        "history_id": history_id,
                    }
                await execute(
                    """INSERT INTO user_fact_history
                       (user_id, fact_key, old_value, new_value, old_source, new_source, old_confidence, new_confidence, resolution, resolved_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'auto_replaced', NOW())""",
                    user_id,
                    fact_key,
                    old_value,
                    value,
                    old_source,
                    source,
                    existing.get("confidence", 1.0),
                    confidence,
                )
            await execute(
                """
                INSERT INTO user_facts (user_id, fact_key, value, category, source, confidence, expires_at, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::timestamptz, NOW(), NOW())
                ON CONFLICT (user_id, fact_key) DO UPDATE SET value = $3, category = $4, source = $5, confidence = $6, expires_at = $7::timestamptz, updated_at = NOW()
                """,
                user_id,
                fact_key,
                value,
                category,
                source,
                confidence,
                expires_at,
            )
            logger.info("Updated fact %s for user %s", fact_key, user_id)
            try:
                from services.celery_tasks.fact_tasks import embed_user_fact_task
                embed_user_fact_task.delay(user_id, fact_key, value)
            except Exception as enqueue_err:
                logger.warning("Failed to enqueue embed_user_fact_task: %s", enqueue_err)
            return {"success": True}
        except Exception as e:
            if "source" in str(e) or "confidence" in str(e) or "does not exist" in str(e):
                try:
                    existing = await fetch_one(
                        "SELECT value FROM user_facts WHERE user_id = $1 AND fact_key = $2",
                        user_id,
                        fact_key,
                    )
                    if existing is not None and (existing.get("value") or "") == value:
                        await execute(
                            "UPDATE user_facts SET updated_at = NOW() WHERE user_id = $1 AND fact_key = $2",
                            user_id,
                            fact_key,
                        )
                        return {"success": True}
                    await execute(
                        """
                        INSERT INTO user_facts (user_id, fact_key, value, category, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, NOW(), NOW())
                        ON CONFLICT (user_id, fact_key) DO UPDATE SET value = $3, category = $4, updated_at = NOW()
                        """,
                        user_id,
                        fact_key,
                        value,
                        category,
                    )
                    logger.info("Updated fact %s for user %s (minimal schema)", fact_key, user_id)
                    return {"success": True}
                except Exception as fallback_e:
                    logger.error("Failed to upsert user fact (minimal) for user %s: %s", user_id, fallback_e)
                    return {"success": False, "error": str(fallback_e)}
            logger.error("Failed to upsert user fact for user %s: %s", user_id, e)
            return {"success": False, "error": str(e)}

    async def get_pending_facts(self, user_id: str) -> List[Dict[str, Any]]:
        """Return pending_review fact history rows for the user."""
        try:
            rows = await fetch_all(
                """SELECT id, fact_key, old_value, new_value, old_source, new_source, created_at
                   FROM user_fact_history
                   WHERE user_id = $1 AND resolution = 'pending_review'
                   ORDER BY created_at DESC""",
                user_id,
            )
            return [dict(r) for r in (rows or [])]
        except Exception as e:
            logger.warning("Failed to get pending facts for user %s: %s", user_id, e)
            return []

    async def resolve_pending_fact(
        self, user_id: str, history_id: int, action: str
    ) -> Dict[str, Any]:
        """Resolve a pending_review fact: accept (apply new value) or reject."""
        if action not in ("accept", "reject"):
            return {"success": False, "message": "action must be accept or reject"}
        try:
            row = await fetch_one(
                """SELECT id, fact_key, old_value, new_value, resolution FROM user_fact_history
                   WHERE user_id = $1 AND id = $2 AND resolution = 'pending_review'""",
                user_id,
                history_id,
            )
            if not row:
                return {"success": False, "message": "Pending fact not found or already resolved"}
            if action == "accept":
                await execute(
                    """UPDATE user_facts SET value = $1, source = 'user_manual', confidence = 1.0, updated_at = NOW()
                       WHERE user_id = $2 AND fact_key = $3""",
                    row["new_value"],
                    user_id,
                    row["fact_key"],
                )
                try:
                    from services.celery_tasks.fact_tasks import embed_user_fact_task
                    embed_user_fact_task.delay(user_id, row["fact_key"], row["new_value"])
                except Exception:
                    pass
            await execute(
                """UPDATE user_fact_history SET resolution = $1, resolved_at = NOW() WHERE id = $2""",
                "user_accepted" if action == "accept" else "user_rejected",
                history_id,
            )
            return {"success": True, "message": "Fact %s %sed" % (row["fact_key"], action)}
        except Exception as e:
            logger.warning("Failed to resolve pending fact %s: %s", history_id, e)
            return {"success": False, "message": str(e)}

    async def get_fact_history(
        self, user_id: str, fact_key: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return fact change history for the user, optionally filtered by fact_key."""
        try:
            if fact_key:
                rows = await fetch_all(
                    """SELECT id, fact_key, old_value, new_value, old_source, new_source, resolution, created_at, resolved_at
                       FROM user_fact_history WHERE user_id = $1 AND fact_key = $2 ORDER BY created_at DESC LIMIT $3""",
                    user_id,
                    fact_key,
                    limit,
                )
            else:
                rows = await fetch_all(
                    """SELECT id, fact_key, old_value, new_value, old_source, new_source, resolution, created_at, resolved_at
                       FROM user_fact_history WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2""",
                    user_id,
                    limit,
                )
            return [dict(r) for r in (rows or [])]
        except Exception as e:
            logger.warning("Failed to get fact history for user %s: %s", user_id, e)
            return []

    async def delete_user_fact(self, user_id: str, fact_key: str) -> bool:
        """Delete a single fact. Returns True if a row was deleted or key did not exist."""
        try:
            result = await execute("DELETE FROM user_facts WHERE user_id = $1 AND fact_key = $2", user_id, fact_key)
            return "DELETE" in (result or "")
        except Exception as e:
            logger.error(f"Failed to delete user fact for user {user_id}: {e}")
            return False

    async def close(self):
        """Clean up resources"""
        # No dedicated engine to close when using shared database helpers
        logger.info("🔄 Settings Service closed")


# Global settings service instance
settings_service = SettingsService()
