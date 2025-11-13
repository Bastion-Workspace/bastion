"""
Cyber Catalog API Endpoints
FastAPI endpoints for cyber data cataloging

**BULLY!** JSON-first cataloging API endpoints!
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import ValidationError

from models.cyber_catalog_models import (
    CyberCatalogConfig,
    CyberCatalogJSON,
    CyberCatalogValidationResult
)
from models.api_models import AuthenticatedUserResponse
from services.cyber_catalog_service import get_cyber_catalog_service
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cyber-catalog", tags=["Cyber Catalog"])


@router.post("/catalog")
async def catalog_folder(
    folder_path: str,
    config: Optional[CyberCatalogConfig] = None,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Catalog a folder of cyber/breach data files
    
    **BULLY!** JSON-first cataloging with validation!
    
    Args:
        folder_path: Path to folder to catalog
        config: Optional catalog configuration
        
    Returns:
        Catalog results with JSON path and validation info
    """
    try:
        logger.info(f"🔍 Cyber Catalog API: Cataloging folder {folder_path} for user {current_user.user_id}")
        
        # Use default config if not provided
        if config is None:
            config = CyberCatalogConfig(validate_only=True)
        
        # Ensure output path is set
        if not config.output_json_path:
            import tempfile
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.output_json_path = f"/tmp/cyber_catalog_{current_user.user_id}_{timestamp}.json"
        
        service = await get_cyber_catalog_service()
        result = await service.catalog_folder(folder_path, config)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Cataloging failed"))
        
        logger.info(f"✅ Cyber Catalog API: Cataloged {result.get('entry_count', 0)} files")
        return result
        
    except ValidationError as e:
        logger.error(f"❌ Cyber Catalog API ERROR: Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Cyber Catalog API ERROR: Failed to catalog folder: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to catalog folder: {str(e)}")


@router.post("/validate-json")
async def validate_catalog_json(
    json_path: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> CyberCatalogValidationResult:
    """
    Validate a catalog JSON file
    
    **BULLY!** Validate before committing to database!
    
    Args:
        json_path: Path to JSON catalog file
        
    Returns:
        Validation result with errors and warnings
    """
    try:
        logger.info(f"🔍 Cyber Catalog API: Validating JSON {json_path} for user {current_user.user_id}")
        
        # Load and validate catalog
        catalog = CyberCatalogJSON.from_json_file(json_path)
        validation = catalog.validate_catalog()
        
        logger.info(f"✅ Cyber Catalog API: Validation complete - Valid: {validation.is_valid}")
        return validation
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"JSON file not found: {json_path}")
    except Exception as e:
        logger.error(f"❌ Cyber Catalog API ERROR: Failed to validate JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate JSON: {str(e)}")


@router.get("/json/{json_path:path}")
async def get_catalog_json(
    json_path: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> CyberCatalogJSON:
    """
    Get catalog JSON file
    
    **BULLY!** Retrieve catalog for review!
    
    Args:
        json_path: Path to JSON catalog file
        
    Returns:
        Catalog JSON structure
    """
    try:
        logger.info(f"🔍 Cyber Catalog API: Loading JSON {json_path} for user {current_user.user_id}")
        
        catalog = CyberCatalogJSON.from_json_file(json_path)
        
        logger.info(f"✅ Cyber Catalog API: Loaded catalog with {len(catalog.entries)} entries")
        return catalog
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"JSON file not found: {json_path}")
    except Exception as e:
        logger.error(f"❌ Cyber Catalog API ERROR: Failed to load JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load JSON: {str(e)}")


@router.get("/config")
async def get_default_config(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> CyberCatalogConfig:
    """
    Get default catalog configuration
    
    **BULLY!** Get default config for cataloging!
    
    Returns:
        Default catalog configuration
    """
    return CyberCatalogConfig()


@router.post("/config")
async def save_config(
    config: CyberCatalogConfig,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Save catalog configuration (for future use with database)
    
    **BULLY!** Save configuration for later!
    
    Args:
        config: Catalog configuration to save
        
    Returns:
        Success confirmation
    """
    try:
        # TODO: Save config to database when DB integration is added
        logger.info(f"💾 Cyber Catalog API: Saving config for user {current_user.user_id}")
        
        return {
            "success": True,
            "message": "Configuration saved (JSON-only mode - DB integration coming soon)",
            "config": config.dict()
        }
        
    except Exception as e:
        logger.error(f"❌ Cyber Catalog API ERROR: Failed to save config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")

