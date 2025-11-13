"""
Template Service - Roosevelt's "Template Command Center"
Manages user-defined report templates with full CRUD operations
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import text

from services.settings_service import settings_service
from models.report_template_models import (
    ReportTemplate, ReportTemplateSection, ReportTemplateField, 
    TemplateScope, generate_template_id, create_poi_template
)

logger = logging.getLogger(__name__)


class TemplateService:
    """Service for managing report templates"""
    
    def __init__(self):
        self.settings_service = settings_service
        self._template_cache: Dict[str, ReportTemplate] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize template service and load built-in templates"""
        try:
            logger.info("📋 Initializing Template Service...")
            
            # Ensure settings service is initialized
            if not self.settings_service._initialized:
                await self.settings_service.initialize()
            
            # Load built-in system templates
            await self._ensure_system_templates()
            
            # Load user templates into cache
            await self._load_template_cache()
            
            self._initialized = True
            logger.info(f"✅ Template Service initialized with {len(self._template_cache)} templates")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Template Service: {e}")
            raise
    
    async def _ensure_system_templates(self):
        """Ensure built-in system templates exist"""
        try:
            # Create Person of Interest template if it doesn't exist
            poi_template = create_poi_template()
            existing = await self.get_template(poi_template.template_id)
            
            if not existing:
                await self._store_template_in_settings(poi_template)
                logger.info(f"✅ Created system template: {poi_template.template_name}")
            else:
                logger.info(f"📋 System template already exists: {poi_template.template_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to ensure system templates: {e}")
    
    async def _load_template_cache(self):
        """Load all templates into memory cache"""
        try:
            # Get all template settings
            template_settings = await self.settings_service.get_settings_by_category("report_templates")
            
            for key, template_data in template_settings.items():
                if key.startswith("template_"):
                    try:
                        template = ReportTemplate.parse_obj(template_data)
                        self._template_cache[template.template_id] = template
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load template {key}: {e}")
            
            logger.info(f"📚 Loaded {len(self._template_cache)} templates into cache")
            
        except Exception as e:
            logger.error(f"❌ Failed to load template cache: {e}")
            self._template_cache = {}
    
    async def create_template(self, template_data: Dict[str, Any], user_id: str) -> ReportTemplate:
        """Create a new report template"""
        try:
            # Generate template ID if not provided
            if "template_id" not in template_data:
                template_data["template_id"] = generate_template_id(template_data["template_name"])
            
            # Set creation metadata
            template_data["created_by"] = user_id
            template_data["created_at"] = datetime.now()
            template_data["updated_at"] = datetime.now()
            
            # Create and validate template
            template = ReportTemplate.parse_obj(template_data)
            
            # Check for duplicate template ID
            existing = await self.get_template(template.template_id)
            if existing:
                raise ValueError(f"Template ID already exists: {template.template_id}")
            
            # Store template
            await self._store_template_in_settings(template)
            
            # Update cache
            self._template_cache[template.template_id] = template
            
            logger.info(f"✅ Created template: {template.template_name} by user {user_id}")
            return template
            
        except Exception as e:
            logger.error(f"❌ Failed to create template: {e}")
            raise
    
    async def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """Get a specific template by ID"""
        try:
            # Check cache first
            if template_id in self._template_cache:
                return self._template_cache[template_id]
            
            # Try to load from settings
            setting_key = f"template_{template_id}"
            template_data = await self.settings_service.get_setting(setting_key)
            
            if template_data:
                template = ReportTemplate.parse_obj(template_data)
                self._template_cache[template_id] = template
                return template
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get template {template_id}: {e}")
            return None
    
    async def get_user_templates(self, user_id: str) -> List[ReportTemplate]:
        """Get all templates created by a specific user"""
        try:
            user_templates = []
            
            for template in self._template_cache.values():
                if (template.created_by == user_id or 
                    template.scope in [TemplateScope.PUBLIC, TemplateScope.SYSTEM]):
                    user_templates.append(template)
            
            # Sort by creation date, newest first
            user_templates.sort(key=lambda t: t.created_at, reverse=True)
            
            logger.info(f"📋 Found {len(user_templates)} templates for user {user_id}")
            return user_templates
            
        except Exception as e:
            logger.error(f"❌ Failed to get user templates for {user_id}: {e}")
            return []
    
    async def get_public_templates(self) -> List[ReportTemplate]:
        """Get all public and system templates"""
        try:
            public_templates = []
            
            for template in self._template_cache.values():
                if template.scope in [TemplateScope.PUBLIC, TemplateScope.SYSTEM]:
                    public_templates.append(template)
            
            # Sort by category, then name
            public_templates.sort(key=lambda t: (t.category, t.template_name))
            
            logger.info(f"📋 Found {len(public_templates)} public templates")
            return public_templates
            
        except Exception as e:
            logger.error(f"❌ Failed to get public templates: {e}")
            return []
    
    async def get_templates_by_keywords(self, keywords: List[str]) -> List[ReportTemplate]:
        """Get templates that match any of the given keywords"""
        try:
            matching_templates = []
            keywords_lower = [k.lower() for k in keywords]
            
            for template in self._template_cache.values():
                template_keywords = [k.lower() for k in template.keywords]
                if any(keyword in template_keywords for keyword in keywords_lower):
                    matching_templates.append(template)
            
            # Sort by number of matching keywords (most matches first)
            def match_score(template):
                template_keywords = [k.lower() for k in template.keywords]
                return sum(1 for k in keywords_lower if k in template_keywords)
            
            matching_templates.sort(key=match_score, reverse=True)
            
            logger.info(f"🔍 Found {len(matching_templates)} templates matching keywords: {keywords}")
            return matching_templates
            
        except Exception as e:
            logger.error(f"❌ Failed to search templates by keywords: {e}")
            return []
    
    async def update_template(self, template_id: str, updates: Dict[str, Any], user_id: str) -> bool:
        """Update an existing template"""
        try:
            # Get existing template
            existing = await self.get_template(template_id)
            if not existing:
                raise ValueError(f"Template not found: {template_id}")
            
            # Check permissions
            if existing.created_by != user_id and existing.scope == TemplateScope.PRIVATE:
                raise PermissionError(f"User {user_id} cannot modify template {template_id}")
            
            if existing.scope == TemplateScope.SYSTEM:
                raise PermissionError("System templates cannot be modified")
            
            # Apply updates
            template_data = existing.dict()
            template_data.update(updates)
            template_data["updated_at"] = datetime.now()
            template_data["version"] = existing.version + 1
            
            # Validate updated template
            updated_template = ReportTemplate.parse_obj(template_data)
            
            # Store updated template
            await self._store_template_in_settings(updated_template)
            
            # Update cache
            self._template_cache[template_id] = updated_template
            
            logger.info(f"✅ Updated template: {template_id} by user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update template {template_id}: {e}")
            return False
    
    async def delete_template(self, template_id: str, user_id: str) -> bool:
        """Delete a template"""
        try:
            # Get existing template
            existing = await self.get_template(template_id)
            if not existing:
                raise ValueError(f"Template not found: {template_id}")
            
            # Check permissions
            if existing.created_by != user_id and existing.scope == TemplateScope.PRIVATE:
                raise PermissionError(f"User {user_id} cannot delete template {template_id}")
            
            if existing.scope == TemplateScope.SYSTEM:
                raise PermissionError("System templates cannot be deleted")
            
            # Delete from settings
            setting_key = f"template_{template_id}"
            success = await self.settings_service.delete_setting(setting_key)
            
            if success:
                # Remove from cache
                if template_id in self._template_cache:
                    del self._template_cache[template_id]
                
                logger.info(f"✅ Deleted template: {template_id} by user {user_id}")
                return True
            else:
                logger.error(f"❌ Failed to delete template setting: {template_id}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to delete template {template_id}: {e}")
            return False
    
    async def duplicate_template(self, template_id: str, new_name: str, user_id: str) -> Optional[ReportTemplate]:
        """Create a copy of an existing template"""
        try:
            # Get existing template
            existing = await self.get_template(template_id)
            if not existing:
                raise ValueError(f"Template not found: {template_id}")
            
            # Create new template data
            new_template_data = existing.dict()
            new_template_data["template_id"] = generate_template_id(new_name)
            new_template_data["template_name"] = new_name
            new_template_data["created_by"] = user_id
            new_template_data["created_at"] = datetime.now()
            new_template_data["updated_at"] = datetime.now()
            new_template_data["version"] = 1
            new_template_data["scope"] = TemplateScope.PRIVATE  # New templates are private by default
            
            # Create the duplicate
            return await self.create_template(new_template_data, user_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to duplicate template {template_id}: {e}")
            return None
    
    async def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about templates"""
        try:
            stats = {
                "total_templates": len(self._template_cache),
                "by_scope": {},
                "by_category": {},
                "by_user": {},
                "most_popular_keywords": {}
            }
            
            # Count by scope
            for template in self._template_cache.values():
                scope = template.scope.value
                stats["by_scope"][scope] = stats["by_scope"].get(scope, 0) + 1
                
                # Count by category
                category = template.category
                stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
                
                # Count by user
                user = template.created_by
                stats["by_user"][user] = stats["by_user"].get(user, 0) + 1
                
                # Count keywords
                for keyword in template.keywords:
                    stats["most_popular_keywords"][keyword] = stats["most_popular_keywords"].get(keyword, 0) + 1
            
            # Sort keywords by popularity
            stats["most_popular_keywords"] = dict(sorted(
                stats["most_popular_keywords"].items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get template stats: {e}")
            return {}
    
    async def _store_template_in_settings(self, template: ReportTemplate):
        """Store template in settings database"""
        try:
            setting_key = f"template_{template.template_id}"
            template_data = template.dict()
            
            success = await self.settings_service.set_setting(
                key=setting_key,
                value=template_data,
                value_type="json",
                description=f"Report template: {template.template_name}",
                category="report_templates"
            )
            
            if not success:
                raise RuntimeError(f"Failed to store template in settings: {template.template_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store template in settings: {e}")
            raise


# Global template service instance
template_service = TemplateService()

