"""
Search API endpoints
Extracted from main.py for better modularity
"""

import logging
from fastapi import APIRouter, HTTPException, Response

from models.api_models import DirectSearchRequest, DirectSearchResponse
from services.direct_search_service import DirectSearchService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post("/api/search/direct")
async def direct_search_documents(request: dict):
    """Perform direct semantic search without LLM processing"""
    try:
        logger.info(f"🔍 Direct search query: {request.get('query', '')[:100]}...")
        
        from models.api_models import DirectSearchRequest
        from services.direct_search_service import DirectSearchService
        
        search_request = DirectSearchRequest(**request)
        
        direct_search_service = await _get_search_service()
        
        result = await direct_search_service.search_documents(
            query=search_request.query,
            limit=search_request.limit,
            similarity_threshold=search_request.similarity_threshold,
            document_types=search_request.document_types,
            categories=search_request.categories,
            tags=search_request.tags,
            date_from=search_request.date_from,
            date_to=search_request.date_to,
            include_metadata=search_request.include_metadata
        )
        
        from models.api_models import DirectSearchResponse
        return DirectSearchResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Direct search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/search/filters")
async def get_search_filters():
    """Get available filter options for direct search"""
    try:
        logger.info("🔍 Getting search filter options")
        
        from services.direct_search_service import DirectSearchService
        
        direct_search_service = await _get_search_service()
        filters = await direct_search_service.get_search_filters()
        
        return filters
        
    except Exception as e:
        logger.error(f"❌ Failed to get search filters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/search/suggestions")
async def get_search_suggestions(query: str, limit: int = 10):
    """Get search suggestions based on partial query"""
    try:
        logger.info(f"🔍 Getting search suggestions for: {query}")
        
        from services.direct_search_service import DirectSearchService
        
        direct_search_service = await _get_search_service()
        suggestions = await direct_search_service.get_search_suggestions(query, limit)
        
        return {"suggestions": suggestions}
        
    except Exception as e:
        logger.error(f"❌ Failed to get search suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/search/export")
async def export_search_results(request: dict):
    """Export search results in various formats"""
    try:
        logger.info("📤 Exporting search results")
        
        from services.direct_search_service import DirectSearchService
        
        results = request.get("results", [])
        format_type = request.get("format", "json")
        
        direct_search_service = await _get_search_service()
        export_result = await direct_search_service.export_search_results(results, format_type)
        
        if format_type.lower() == "csv":
            from fastapi.responses import Response
            return Response(
                content=export_result["data"],
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={export_result['filename']}"
                }
            )
        else:
            return export_result
        
    except Exception as e:
        logger.error(f"❌ Failed to export search results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

