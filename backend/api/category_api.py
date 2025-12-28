"""
Category Management API endpoints
Extracted from main.py for better modularity
"""

import logging
from fastapi import APIRouter, HTTPException

from services.service_container import get_service_container

logger = logging.getLogger(__name__)

router = APIRouter(tags=["categories"])


async def _get_category_service():
    """Get category service from service container"""
    container = await get_service_container()
    return container.category_service


@router.get("/api/categories")
async def get_all_categories():
    """Get all categories used across documents and notes"""
    category_service = await _get_category_service()
    try:
        logger.info("🏷️ Getting all universal categories")
        
        result = await category_service.get_all_categories()
        
        logger.info(f"✅ Retrieved {len(result['categories'])} categories")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/categories/suggestions")
async def get_category_suggestions(content: str = "", content_type: str = "note"):
    """Get category suggestions for content"""
    category_service = await _get_category_service()
    try:
        logger.info(f"🏷️ Getting category suggestions for {content_type}")
        
        suggestions = await category_service.suggest_categories(content, content_type)
        
        logger.info(f"✅ Generated {len(suggestions)} category suggestions")
        return {"suggestions": suggestions}
        
    except Exception as e:
        logger.error(f"❌ Failed to get category suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/categories")
async def create_category(request: dict):
    """Create a new category"""
    category_service = await _get_category_service()
    try:
        category_name = request.get("name", "").strip()
        description = request.get("description", "")
        category_type = request.get("type", "custom")
        
        if not category_name:
            raise HTTPException(status_code=400, detail="Category name is required")
        
        logger.info(f"🏷️ Creating category: {category_name}")
        
        category = await category_service.create_category(category_name, description, category_type)
        
        logger.info(f"✅ Category created: {category_name}")
        return category.dict()
        
    except Exception as e:
        logger.error(f"❌ Failed to create category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/categories/{category_name}")
async def update_category(category_name: str, request: dict):
    """Update a category"""
    category_service = await _get_category_service()
    try:
        description = request.get("description", "")
        
        logger.info(f"🏷️ Updating category: {category_name}")
        
        success = await category_service.update_category(category_name, description)
        if not success:
            raise HTTPException(status_code=404, detail="Category not found")
        
        logger.info(f"✅ Category updated: {category_name}")
        return {"status": "success", "message": f"Category '{category_name}' updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/categories/{category_name}")
async def delete_category(category_name: str):
    """Delete a category (only custom categories can be deleted)"""
    category_service = await _get_category_service()
    try:
        logger.info(f"🏷️ Deleting category: {category_name}")
        
        success = await category_service.delete_category(category_name)
        if not success:
            raise HTTPException(status_code=404, detail="Category not found or cannot be deleted")
        
        logger.info(f"✅ Category deleted: {category_name}")
        return {"status": "success", "message": f"Category '{category_name}' deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/categories/statistics")
async def get_category_statistics():
    """Get statistics about category usage"""
    category_service = await _get_category_service()
    try:
        logger.info("🏷️ Getting category statistics")
        
        stats = await category_service.get_category_statistics()
        
        logger.info("✅ Category statistics retrieved")
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get category statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
