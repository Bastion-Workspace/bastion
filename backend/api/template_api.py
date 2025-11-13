"""
Template API - Roosevelt's "Template Command Interface"
REST endpoints for report template management
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from services.template_service import template_service
from utils.auth_middleware import get_current_user
from models.api_models import AuthenticatedUserResponse
from models.report_template_models import (
    ReportTemplate, ReportTemplateSection, ReportTemplateField,
    TemplateScope, FieldType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/templates", tags=["Templates"])


# ===== REQUEST/RESPONSE MODELS =====

class TemplateCreateRequest(BaseModel):
    """Request model for creating a new template"""
    template_name: str = Field(description="Name of the template")
    description: str = Field(description="What this template is used for")
    category: str = Field(default="general", description="Template category")
    keywords: List[str] = Field(default_factory=list, description="Keywords for auto-detection")
    scope: TemplateScope = Field(default=TemplateScope.PRIVATE, description="Template visibility")
    sections: List[Dict[str, Any]] = Field(description="Template sections definition")


class TemplateUpdateRequest(BaseModel):
    """Request model for updating a template"""
    template_name: Optional[str] = Field(default=None, description="Updated template name")
    description: Optional[str] = Field(default=None, description="Updated description")
    category: Optional[str] = Field(default=None, description="Updated category")
    keywords: Optional[List[str]] = Field(default=None, description="Updated keywords")
    scope: Optional[TemplateScope] = Field(default=None, description="Updated visibility")
    sections: Optional[List[Dict[str, Any]]] = Field(default=None, description="Updated sections")


class TemplateDuplicateRequest(BaseModel):
    """Request model for duplicating a template"""
    new_name: str = Field(description="Name for the duplicated template")


class TemplateResponse(BaseModel):
    """Response model for template operations"""
    success: bool
    message: str
    template: Optional[ReportTemplate] = None
    templates: Optional[List[ReportTemplate]] = None


class TemplateStatsResponse(BaseModel):
    """Response model for template statistics"""
    success: bool
    stats: Dict[str, Any]


# ===== TEMPLATE CRUD ENDPOINTS =====

@router.post("/", response_model=TemplateResponse)
async def create_template(
    request: TemplateCreateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a new report template"""
    try:
        logger.info(f"📋 Creating template: {request.template_name} by user {current_user.username}")
        
        # Prepare template data
        template_data = request.dict()
        
        # Create template
        template = await template_service.create_template(template_data, current_user.user_id)
        
        return TemplateResponse(
            success=True,
            message=f"Template '{template.template_name}' created successfully",
            template=template
        )
        
    except ValueError as e:
        logger.error(f"❌ Template creation validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Template creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")


@router.get("/", response_model=TemplateResponse)
async def get_user_templates(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get all templates accessible to the current user"""
    try:
        logger.info(f"📋 Getting templates for user: {current_user.username}")
        
        templates = await template_service.get_user_templates(current_user.user_id)
        
        return TemplateResponse(
            success=True,
            message=f"Found {len(templates)} templates",
            templates=templates
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get user templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")


@router.get("/public", response_model=TemplateResponse)
async def get_public_templates():
    """Get all public and system templates"""
    try:
        logger.info("📋 Getting public templates")
        
        templates = await template_service.get_public_templates()
        
        return TemplateResponse(
            success=True,
            message=f"Found {len(templates)} public templates",
            templates=templates
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get public templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get public templates: {str(e)}")


@router.get("/search")
async def search_templates_by_keywords(
    keywords: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Search templates by keywords"""
    try:
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
        logger.info(f"🔍 Searching templates by keywords: {keyword_list}")
        
        templates = await template_service.get_templates_by_keywords(keyword_list)
        
        return TemplateResponse(
            success=True,
            message=f"Found {len(templates)} matching templates",
            templates=templates
        )
        
    except Exception as e:
        logger.error(f"❌ Template search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Template search failed: {str(e)}")


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get a specific template by ID"""
    try:
        logger.info(f"📋 Getting template: {template_id}")
        
        template = await template_service.get_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template not found: {template_id}")
        
        # Check if user has access to this template
        if (template.scope == TemplateScope.PRIVATE and 
            template.created_by != current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to this template")
        
        return TemplateResponse(
            success=True,
            message="Template retrieved successfully",
            template=template
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")


@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str,
    request: TemplateUpdateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update an existing template"""
    try:
        logger.info(f"📋 Updating template: {template_id} by user {current_user.username}")
        
        # Prepare updates (only include non-None values)
        updates = {k: v for k, v in request.dict().items() if v is not None}
        
        success = await template_service.update_template(template_id, updates, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update template")
        
        # Get updated template
        updated_template = await template_service.get_template(template_id)
        
        return TemplateResponse(
            success=True,
            message=f"Template '{template_id}' updated successfully",
            template=updated_template
        )
        
    except PermissionError as e:
        logger.error(f"❌ Permission denied updating template {template_id}: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        logger.error(f"❌ Template update validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Template update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update template: {str(e)}")


@router.delete("/{template_id}")
async def delete_template(
    template_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a template"""
    try:
        logger.info(f"📋 Deleting template: {template_id} by user {current_user.username}")
        
        success = await template_service.delete_template(template_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to delete template")
        
        return {
            "success": True,
            "message": f"Template '{template_id}' deleted successfully"
        }
        
    except PermissionError as e:
        logger.error(f"❌ Permission denied deleting template {template_id}: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        logger.error(f"❌ Template not found: {template_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Template deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete template: {str(e)}")


@router.post("/{template_id}/duplicate", response_model=TemplateResponse)
async def duplicate_template(
    template_id: str,
    request: TemplateDuplicateRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a copy of an existing template"""
    try:
        logger.info(f"📋 Duplicating template: {template_id} as '{request.new_name}' by user {current_user.username}")
        
        new_template = await template_service.duplicate_template(
            template_id, request.new_name, current_user.user_id
        )
        
        if not new_template:
            raise HTTPException(status_code=400, detail="Failed to duplicate template")
        
        return TemplateResponse(
            success=True,
            message=f"Template duplicated as '{new_template.template_name}'",
            template=new_template
        )
        
    except ValueError as e:
        logger.error(f"❌ Template duplication validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Template duplication failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to duplicate template: {str(e)}")


# ===== TEMPLATE UTILITY ENDPOINTS =====

@router.get("/stats/overview", response_model=TemplateStatsResponse)
async def get_template_stats(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get template usage statistics (admin only)"""
    try:
        if current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        logger.info(f"📊 Getting template stats for admin user: {current_user.username}")
        
        stats = await template_service.get_template_stats()
        
        return TemplateStatsResponse(
            success=True,
            stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get template stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get template stats: {str(e)}")


@router.get("/field-types/available")
async def get_available_field_types():
    """Get available field types for template creation"""
    try:
        field_types = [
            {
                "value": field_type.value,
                "label": field_type.value.replace("_", " ").title(),
                "description": _get_field_type_description(field_type)
            }
            for field_type in FieldType
        ]
        
        return {
            "success": True,
            "field_types": field_types
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get field types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get field types: {str(e)}")


def _get_field_type_description(field_type: FieldType) -> str:
    """Get description for field type"""
    descriptions = {
        FieldType.TEXT: "Single-line text input",
        FieldType.LONG_TEXT: "Multi-line text area",
        FieldType.LIST: "Bulleted or numbered list",
        FieldType.DATE: "Date field",
        FieldType.NUMBER: "Numeric value",
        FieldType.URL: "Web URL",
        FieldType.EMAIL: "Email address",
        FieldType.PHONE: "Phone number",
        FieldType.ADDRESS: "Physical address",
        FieldType.IMAGE: "Image or photo placeholder",
        FieldType.STRUCTURED_DATA: "Key-value pairs or structured information"
    }
    return descriptions.get(field_type, "Custom field type")


@router.post("/validate")
async def validate_template_structure(
    template_data: Dict[str, Any],
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Validate template structure without saving"""
    try:
        logger.info(f"🔍 Validating template structure for user: {current_user.username}")
        
        # Try to parse as ReportTemplate to validate structure
        template = ReportTemplate.parse_obj(template_data)
        
        return {
            "success": True,
            "message": "Template structure is valid",
            "validation_details": {
                "sections_count": len(template.sections),
                "total_fields": sum(len(section.fields) for section in template.sections),
                "required_sections": sum(1 for section in template.sections if section.required),
                "keywords": template.keywords
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Template validation failed: {e}")
        return {
            "success": False,
            "message": f"Template validation failed: {str(e)}",
            "validation_details": None
        }

