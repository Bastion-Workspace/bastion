"""
Capabilities Service - Roosevelt's Feature Flags and Permissions

Provides per-user capability flags with role-aware defaults.
Admins implicitly have all capabilities enabled.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Any, Optional

from services.user_settings_kv_service import get_user_setting, set_user_setting

logger = logging.getLogger(__name__)


FEATURE_KEYS = [
    # News
    "feature.news.view",
    "feature.news.agent",
    "feature.news.notifications",
    # Extend with more features as needed
]


class CapabilitiesService:
    def __init__(self) -> None:
        self._initialized = True

    async def get_effective_capabilities(self, user: Dict[str, Any]) -> Dict[str, bool]:
        """Compute effective capabilities. Admins have all features enabled."""
        try:
            role = (user.get("role") or "").lower()
            if role == "admin":
                return {key: True for key in FEATURE_KEYS}

            user_id = user.get("user_id") or user.get("id")
            raw = await get_user_setting(user_id, "capabilities") if user_id else None
            caps: Dict[str, bool] = {}
            if raw:
                try:
                    caps = json.loads(raw)
                except Exception:
                    caps = {}
            # Ensure unknown features default to False
            return {key: bool(caps.get(key, False)) for key in FEATURE_KEYS}
        except Exception as e:
            logger.error(f"❌ Capabilities resolution failed: {e}")
            return {key: False for key in FEATURE_KEYS}

    async def user_has_feature(self, user: Dict[str, Any], feature_key: str) -> bool:
        if (user.get("role") or "").lower() == "admin":
            return True
        caps = await self.get_effective_capabilities(user)
        return bool(caps.get(feature_key, False))

    async def set_user_capabilities(self, user_id: str, capabilities: Dict[str, bool]) -> bool:
        try:
            # Persist full map as JSON
            value = json.dumps({k: bool(v) for k, v in capabilities.items() if k in FEATURE_KEYS})
            return await set_user_setting(user_id, "capabilities", value, data_type="json")
        except Exception as e:
            logger.error(f"❌ Failed to set capabilities for {user_id}: {e}")
            return False


capabilities_service = CapabilitiesService()


