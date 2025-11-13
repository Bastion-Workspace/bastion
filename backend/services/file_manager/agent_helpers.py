"""
FileManager Agent Helpers - Easy-to-use functions for agents to place files
"""

import logging
from typing import Optional, Dict, Any, List

from . import get_file_manager
from .models.file_placement_models import FilePlacementRequest, SourceType

logger = logging.getLogger(__name__)


async def place_chat_response(
    content: str,
    title: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None
) -> str:
    """Place a chat response in the appropriate folder"""
    try:
        file_manager = await get_file_manager()
        
        request = FilePlacementRequest(
            content=content,
            title=title,
            source_type=SourceType.CHAT,
            source_metadata={
                "conversation_id": conversation_id,
                "response_type": "chat"
            },
            tags=tags or [],
            description=description,
            user_id=user_id,
            process_immediately=True
        )
        
        response = await file_manager.place_file(request)
        logger.info(f"✅ Chat response placed: {response.document_id}")
        return response.document_id
        
    except Exception as e:
        logger.error(f"❌ Failed to place chat response: {e}")
        raise


async def place_code_file(
    content: str,
    title: str,
    language: str,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None
) -> str:
    """Place a code file in the appropriate folder"""
    try:
        file_manager = await get_file_manager()
        
        request = FilePlacementRequest(
            content=content,
            title=title,
            source_type=SourceType.CODING,
            source_metadata={
                "language": language,
                "code_type": "generated"
            },
            tags=tags or [],
            description=description,
            user_id=user_id,
            process_immediately=True
        )
        
        response = await file_manager.place_file(request)
        logger.info(f"✅ Code file placed: {response.document_id}")
        return response.document_id
        
    except Exception as e:
        logger.error(f"❌ Failed to place code file: {e}")
        raise


async def place_web_content(
    content: str,
    title: str,
    url: str,
    domain: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None
) -> str:
    """Place web-scraped content in the appropriate folder"""
    try:
        file_manager = await get_file_manager()
        
        request = FilePlacementRequest(
            content=content,
            title=title,
            source_type=SourceType.WEB_SCRAPING,
            source_metadata={
                "url": url,
                "domain": domain,
                "scraped_at": "now"
            },
            tags=tags or [],
            description=description,
            user_id=user_id,
            process_immediately=True
        )
        
        response = await file_manager.place_file(request)
        logger.info(f"✅ Web content placed: {response.document_id}")
        return response.document_id
        
    except Exception as e:
        logger.error(f"❌ Failed to place web content: {e}")
        raise


async def place_manual_file(
    content: str,
    title: str,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None,
    folder_path: Optional[List[str]] = None
) -> str:
    """Place a manually created file in the specified folder"""
    try:
        file_manager = await get_file_manager()
        
        request = FilePlacementRequest(
            content=content,
            title=title,
            source_type=SourceType.MANUAL,
            tags=tags or [],
            description=description,
            user_id=user_id,
            folder_path=folder_path,
            process_immediately=True
        )
        
        response = await file_manager.place_file(request)
        logger.info(f"✅ Manual file placed: {response.document_id}")
        return response.document_id
        
    except Exception as e:
        logger.error(f"❌ Failed to place manual file: {e}")
        raise


async def place_calibre_book(
    content: str,
    title: str,
    author: str,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None
) -> str:
    """Place a Calibre book in the appropriate folder"""
    try:
        file_manager = await get_file_manager()
        
        request = FilePlacementRequest(
            content=content,
            title=title,
            source_type=SourceType.CALIBRE,
            source_metadata={
                "imported_from": "calibre",
                "book_type": "imported"
            },
            author=author,
            tags=tags or [],
            description=description,
            user_id=user_id,
            process_immediately=True
        )
        
        response = await file_manager.place_file(request)
        logger.info(f"✅ Calibre book placed: {response.document_id}")
        return response.document_id
        
    except Exception as e:
        logger.error(f"❌ Failed to place Calibre book: {e}")
        raise
