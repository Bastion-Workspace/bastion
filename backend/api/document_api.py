"""
Document Management API endpoints
Extracted from main.py for better modularity
"""

import asyncio
import logging
import os
import glob
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from io import BytesIO

import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import FileResponse

from models.api_models import (
    URLImportRequest, ImportImageRequest, QueryRequest, DocumentListResponse,
    DocumentUploadResponse, DocumentStatus, DocumentFilterRequest, DocumentUpdateRequest,
    BulkCategorizeRequest, DocumentCategoriesResponse, BulkOperationResponse,
    ProcessingStatus, DocumentType, DocumentInfo, DocumentCategory, AuthenticatedUserResponse
)
from services.service_container import get_service_container
from services.user_document_service import UserDocumentService
from services.auth_service import auth_service
from utils.auth_middleware import get_current_user, require_admin
from utils.websocket_manager import get_websocket_manager
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])


# Helper function to get services from container
async def _get_document_service():
    """Get document service from service container"""
    container = await get_service_container()
    return container.document_service


async def _get_folder_service():
    """Get folder service from service container"""
    container = await get_service_container()
    return container.folder_service


async def check_document_access(
    doc_id: str,
    current_user: AuthenticatedUserResponse,
    required_permission: str = "read"
) -> DocumentInfo:
    """
    Check if user has access to a document
    
    Args:
        doc_id: Document ID
        current_user: Current authenticated user
        required_permission: "read", "write", or "delete"
        
    Returns:
        DocumentInfo if access granted
        
    Raises:
        HTTPException: 403 if access denied, 404 if not found
    """
    document_service = await _get_document_service()
    doc_info = await document_service.get_document(doc_id)
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Admin has full access
    if current_user.role == "admin":
        return doc_info
    
    collection_type = getattr(doc_info, 'collection_type', 'user')
    doc_user_id = getattr(doc_info, 'user_id', None)
    doc_team_id = getattr(doc_info, 'team_id', None)
    
    # Global collection: read-only for users, write for admins
    if collection_type == 'global':
        if required_permission in ['write', 'delete']:
            raise HTTPException(status_code=403, detail="Only admins can modify global documents")
        return doc_info  # Anyone can read global docs
    
    # Team collection: check team membership
    if doc_team_id:
        from api.teams_api import team_service
        role = await team_service.check_team_access(doc_team_id, current_user.user_id)
        if not role:
            raise HTTPException(status_code=403, detail="Not a team member")
        
        # Check permission based on team role
        if required_permission == 'delete':
            if role != 'admin':
                raise HTTPException(status_code=403, detail="Only team admins can delete team documents")
        
        return doc_info
    
    # User collection: only owner has access
    if doc_user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return doc_info


# ===== DOCUMENT MANAGEMENT ENDPOINTS =====

# ===== DOCUMENT MANAGEMENT ENDPOINTS =====

@router.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(None),
    folder_id: str = Form(None),
    category: str = Form(None),
    tags: str = Form(None),  # Comma-separated tags
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Upload and process a document to global collection (admin only) - **BULLY!** Now with category and tags!"""
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()
    try:
        logger.info(f"📄 Admin {current_user.username} uploading document to global collection: {file.filename} to folder: {folder_id}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Validate folder access if folder_id is provided
        if folder_id:
            # Check if folder exists and is a global folder
            folder = await folder_service.get_folder(folder_id, None)  # Global folders have no user_id
            if not folder or folder.collection_type != "global":
                raise HTTPException(status_code=404, detail="Global folder not found or access denied")
        
        # Process document (no user_id = global collection)
        # **ROOSEVELT FIX**: Pass folder_id to ensure files land in correct subfolder on disk
        result = await document_service.upload_and_process(file, doc_type, folder_id=folder_id)
        
        # **ROOSEVELT METADATA UPDATE**: Update category and tags if provided
        if result.document_id and (category or tags):
            from models.api_models import DocumentUpdateRequest, DocumentCategory
            
            # Parse tags from comma-separated string
            tags_list = [tag.strip() for tag in tags.split(',')] if tags else []
            
            # Parse category enum
            doc_category = None
            if category:
                try:
                    doc_category = DocumentCategory(category)
                except ValueError:
                    logger.warning(f"⚠️ Invalid category '{category}', ignoring")
            
            # Update document metadata
            update_request = DocumentUpdateRequest(
                category=doc_category,
                tags=tags_list if tags_list else None
            )
            await document_service.update_document_metadata(result.document_id, update_request)
            logger.info(f"📋 Updated document metadata: category={category}, tags={tags_list}")
        
        # Assign document to folder if specified (immediate assignment with retry)
        if folder_id and result.document_id:
            # Try immediate folder assignment first
            try:
                success = await document_service.document_repository.update_document_folder(result.document_id, folder_id, None)  # None for admin/global
                if success:
                    logger.info(f"✅ Global document {result.document_id} assigned to folder {folder_id} immediately")
                    
                    # Small delay to ensure transaction is committed before frontend queries
                    await asyncio.sleep(0.1)
                    
                    # Send WebSocket notification for optimistic UI update
                    try:
                        from utils.websocket_manager import get_websocket_manager
                        websocket_manager = get_websocket_manager()
                        if websocket_manager:
                            await websocket_manager.send_to_session({
                                "type": "folder_event",
                                "action": "file_created",
                                "folder_id": folder_id,
                                "document_id": result.document_id,
                                "filename": file.filename,
                                "user_id": current_user.user_id,
                                "timestamp": datetime.now().isoformat()
                            }, current_user.user_id)
                            logger.info(f"📡 Sent file creation notification for user {current_user.user_id}")
                        else:
                            logger.warning("⚠️ WebSocket manager not available for file creation notification")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to send WebSocket notification: {e}")
                else:
                    # If immediate assignment fails, start background retry task
                    logger.warning(f"⚠️ Immediate folder assignment failed, starting background retry for document {result.document_id}")
                    
                    async def assign_document_to_folder():
                        document_service = await _get_document_service()
                        try:
                            # Wait a bit longer for the document to be fully committed
                            await asyncio.sleep(2.0)
                            
                            # Try folder assignment with retry logic
                            max_retries = 10
                            for attempt in range(max_retries):
                                try:
                                    success = await document_service.document_repository.update_document_folder(result.document_id, folder_id, None)  # None for admin/global
                                    if success:
                                        logger.info(f"✅ Global document {result.document_id} assigned to folder {folder_id} (background attempt {attempt + 1})")
                                        
                                        # Send WebSocket notification for optimistic UI update
                                        try:
                                            from utils.websocket_manager import get_websocket_manager
                                            websocket_manager = get_websocket_manager()
                                            if websocket_manager:
                                                await websocket_manager.send_to_session({
                                                    "type": "folder_event",
                                                    "action": "file_created",
                                                    "folder_id": folder_id,
                                                    "document_id": result.document_id,
                                                    "filename": file.filename,
                                                    "user_id": current_user.user_id,
                                                    "timestamp": datetime.now().isoformat()
                                                }, current_user.user_id)
                                                logger.info(f"📡 Sent file creation notification for user {current_user.user_id}")
                                            else:
                                                logger.warning("⚠️ WebSocket manager not available for file creation notification")
                                        except Exception as e:
                                            logger.warning(f"⚠️ Failed to send WebSocket notification: {e}")
                                        return
                                    else:
                                        logger.warning(f"⚠️ Failed to assign global document {result.document_id} to folder {folder_id} (background attempt {attempt + 1})")
                                        if attempt < max_retries - 1:
                                            await asyncio.sleep(2.0 * (attempt + 1))  # Longer exponential backoff
                                except Exception as e:
                                    logger.error(f"❌ Error assigning global document to folder (background attempt {attempt + 1}): {e}")
                                    if attempt < max_retries - 1:
                                        await asyncio.sleep(2.0 * (attempt + 1))
                            else:
                                logger.error(f"❌ Failed to assign global document {result.document_id} to folder {folder_id} after {max_retries} background attempts")
                        except Exception as e:
                            logger.error(f"❌ Background folder assignment failed: {e}")
                    
                    # Start the background task without waiting for it
                    asyncio.create_task(assign_document_to_folder())
                    
            except Exception as e:
                logger.error(f"❌ Immediate folder assignment failed: {e}")
                # Start background retry task
                asyncio.create_task(assign_document_to_folder())
        
        logger.info(f"✅ Global document uploaded successfully: {result.document_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Global upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/documents/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Upload and process multiple documents to global collection (admin only)"""
    document_service = await _get_document_service()
    try:
        logger.info(f"📄 Uploading {len(files)} documents")
        
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail=f"No filename provided for one of the files")
        
        # Use parallel document service for bulk upload
        if hasattr(document_service, 'upload_multiple_documents'):
            result = await document_service.upload_multiple_documents(files, enable_parallel=True)
        else:
            # Fallback to sequential processing
            logger.warning("⚠️ Parallel upload not available, falling back to sequential processing")
            upload_results = []
            successful_uploads = 0
            failed_uploads = 0
            
            for file in files:
                try:
                    single_result = await document_service.upload_and_process(file)
                    upload_results.append(single_result)
                    if single_result.status != ProcessingStatus.FAILED:
                        successful_uploads += 1
                    else:
                        failed_uploads += 1
                except Exception as e:
                    failed_uploads += 1
                    upload_results.append(DocumentUploadResponse(
                        document_id="",
                        filename=file.filename,
                        status=ProcessingStatus.FAILED,
                        message=f"Upload failed: {str(e)}"
                    ))
            
            from models.api_models import BulkUploadResponse
            result = BulkUploadResponse(
                total_files=len(files),
                successful_uploads=successful_uploads,
                failed_uploads=failed_uploads,
                upload_results=upload_results,
                processing_time=0,
                message=f"Sequential upload completed: {successful_uploads}/{len(files)} files successful"
            )
        
        logger.info(f"✅ Bulk upload completed: {result.successful_uploads}/{result.total_files} successful")
        return result
        
    except Exception as e:
        logger.error(f"❌ Bulk upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/documents/import-url", response_model=DocumentUploadResponse)
async def import_from_url(
    request: URLImportRequest,
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Import content from URL to global collection (admin only)"""
    document_service = await _get_document_service()
    try:
        logger.info(f"🔗 Admin {current_user.username} importing from URL: {request.url}")
        
        result = await document_service.import_from_url(request.url, request.content_type)
        
        logger.info(f"✅ URL imported successfully: {result.document_id}")
        return result
        
    except Exception as e:
        logger.error(f"❌ URL import failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/documents/import-image", response_model=DocumentUploadResponse)
async def import_image(
    request: ImportImageRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Import a generated image into the user's document library"""
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()
    try:
        import aiofiles
        from io import BytesIO
        
        logger.info(f"🖼️ User {current_user.username} importing image: {request.image_url}")
        
        # Extract filename from URL (e.g., /api/images/filename.png or /static/images/filename.png -> filename.png)
        image_url = request.image_url
        if image_url.startswith('/api/images/'):
            filename_from_url = image_url.replace('/api/images/', '').split('/')[-1]
        elif image_url.startswith('/static/images/'):
            filename_from_url = image_url.replace('/static/images/', '').split('/')[-1]
        else:
            # Fallback: try to extract filename from any path
            filename_from_url = image_url.split('/')[-1]
        
        # Use provided filename or fall back to extracted filename
        filename = request.filename or filename_from_url
        if not filename:
            raise HTTPException(status_code=400, detail="Could not determine filename from image URL")
        
        # Validate folder access if folder_id is provided
        folder_id = request.folder_id
        if folder_id:
            folder = await folder_service.get_folder(folder_id, current_user.user_id)
            if not folder:
                raise HTTPException(status_code=404, detail="Folder not found or access denied")
            
            # For admin users, allow importing to global folders
            if folder.collection_type == "global" and current_user.role != "admin":
                raise HTTPException(status_code=403, detail="Only admins can import to global folders")
        
        # Read image file from static images directory
        images_path = Path(f"{settings.UPLOAD_DIR}/web_sources/images")
        image_file_path = images_path / filename_from_url
        
        # Also check subdirectories (some images may be in document_id subdirectories)
        if not image_file_path.exists():
            # Search in subdirectories
            found = False
            for subdir in images_path.iterdir():
                if subdir.is_dir():
                    potential_path = subdir / filename_from_url
                    if potential_path.exists():
                        image_file_path = potential_path
                        found = True
                        break
            
            if not found:
                raise HTTPException(status_code=404, detail=f"Image file not found: {filename_from_url}")
        
        # Read the image file
        async with aiofiles.open(image_file_path, 'rb') as f:
            image_content = await f.read()
        
        # Create an UploadFile-like object from the image content
        image_buffer = BytesIO(image_content)
        image_buffer.seek(0)  # Ensure file pointer is at the start
        image_file = UploadFile(
            filename=filename,
            file=image_buffer
        )
        
        # Import the image using the document service
        result = await document_service.upload_and_process(
            image_file, 
            doc_type='image', 
            user_id=current_user.user_id, 
            folder_id=folder_id
        )
        
        logger.info(f"✅ Image imported successfully: {result.document_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Image import failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== USER DOCUMENT MANAGEMENT ENDPOINTS =====

@router.post("/api/user/documents/upload", response_model=DocumentUploadResponse)
async def upload_user_document(
    file: UploadFile = File(...),
    doc_type: str = Form(None),
    folder_id: str = Form(None),
    category: str = Form(None),
    tags: str = Form(None),  # Comma-separated tags
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Upload a document to user's private collection - Roosevelt Architecture - **BULLY!** Now with category and tags!"""
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()
    try:
        logger.info(f"📄 User {current_user.username} uploading document: {file.filename} to folder: {folder_id}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # **BULLY!** INBOX.ORG SAFEGUARD - Prevent duplicate inbox files
        if file.filename.lower() == "inbox.org":
            from pathlib import Path
            from services.database_manager.database_helpers import fetch_one, fetch_all
            
            # First check database for existing inbox.org (most reliable)
            existing_docs = await fetch_all(
                """
                SELECT document_id, filename, folder_id
                FROM document_metadata
                WHERE user_id = $1 AND LOWER(filename) = 'inbox.org'
                """,
                current_user.user_id
            )
            
            if existing_docs:
                # Found existing inbox in database - reject upload
                error_msg = (
                    f"⚠️ INBOX.ORG ALREADY EXISTS in your documents. "
                    f"Only one inbox.org per user is allowed. "
                    f"Use quick capture (Ctrl+Shift+C) to add content, or delete the existing inbox before uploading a new one."
                )
                if len(existing_docs) > 1:
                    error_msg += f" (Note: {len(existing_docs)} inbox files found in database - consider cleaning up duplicates)"
                
                logger.warning(f"🚫 BLOCKED: User {current_user.username} tried to upload inbox.org when one already exists (document_id: {existing_docs[0]['document_id']})")
                raise HTTPException(status_code=409, detail=error_msg)
            
            # Also check filesystem as backup (in case DB is out of sync)
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", current_user.user_id)
            username = row['username'] if row else current_user.user_id
            
            upload_dir = Path(settings.UPLOAD_DIR)
            user_base_dir = upload_dir / "Users" / username
            
            if user_base_dir.exists():
                existing_inboxes = list(user_base_dir.rglob("inbox.org"))
                if existing_inboxes:
                    # Found existing inbox on disk but not in DB - warn and allow upload (will recover file)
                    existing_paths = [str(f.relative_to(user_base_dir)) for f in existing_inboxes]
                    logger.warning(f"⚠️  Found orphaned inbox.org on disk at {existing_paths[0]} - allowing upload to recover file")
                    # Continue with upload - this will overwrite the orphaned file
        
        # Validate folder access if folder_id is provided
        if folder_id:
            # Check if folder exists and user has access
            folder = await folder_service.get_folder(folder_id, current_user.user_id)
            if not folder:
                raise HTTPException(status_code=404, detail="Folder not found or access denied")
            
            # For admin users, allow uploading to global folders
            if folder.collection_type == "global" and current_user.role != "admin":
                raise HTTPException(
                    status_code=403,
                    detail="Uploading files to Global folders requires Admin privileges"
                )
        
        # ROOSEVELT FIX: Pass folder_id to upload_and_process for transaction-level folder assignment
        # This ensures folder assignment happens within the same transaction as document creation
        
        # Process document with user_id and folder_id for immediate folder assignment
        result = await document_service.upload_and_process(file, doc_type, user_id=current_user.user_id, folder_id=folder_id)
        
        # **ROOSEVELT METADATA UPDATE**: Update category and tags if provided
        if result.document_id and (category or tags):
            from models.api_models import DocumentUpdateRequest, DocumentCategory
            
            # Parse tags from comma-separated string
            tags_list = [tag.strip() for tag in tags.split(',')] if tags else []
            
            # Parse category enum
            doc_category = None
            if category:
                try:
                    doc_category = DocumentCategory(category)
                except ValueError:
                    logger.warning(f"⚠️ Invalid category '{category}', ignoring")
            
            # Update document metadata
            update_request = DocumentUpdateRequest(
                category=doc_category,
                tags=tags_list if tags_list else None
            )
            await document_service.update_document_metadata(result.document_id, update_request)
            logger.info(f"📋 Updated user document metadata: category={category}, tags={tags_list}")
        
        # ROOSEVELT FIX: Folder assignment is now handled within the upload_and_process transaction
        # Send WebSocket notification for optimistic UI update if folder assignment was successful
        if folder_id and result.document_id:
            try:
                from utils.websocket_manager import get_websocket_manager
                websocket_manager = get_websocket_manager()
                if websocket_manager:
                    await websocket_manager.send_to_session({
                        "type": "folder_event",
                        "action": "file_created",
                        "folder_id": folder_id,
                        "document_id": result.document_id,
                        "filename": file.filename,
                        "user_id": current_user.user_id,
                        "timestamp": datetime.now().isoformat()
                    }, current_user.user_id)
                    logger.info(f"📡 Sent file creation notification for user {current_user.user_id}")
                else:
                    logger.warning("⚠️ WebSocket manager not available for file creation notification")
            except Exception as e:
                logger.warning(f"⚠️ Failed to send WebSocket notification: {e}")
        
        logger.info(f"✅ User document uploaded successfully: {result.document_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ User upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/user/documents/search")
async def search_user_and_global_documents(
    request: QueryRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Search both user's private documents and global shared documents"""
    document_service = await _get_document_service()
    try:
        logger.info(f"🔍 User {current_user.username} searching: {request.query[:50]}...")
        
        # Use embedding manager's hybrid search (searches both collections)
        results = await document_service.embedding_manager.search_similar(
            query_text=request.query,
            limit=getattr(request, 'limit', 10),
            score_threshold=getattr(request, 'similarity_threshold', 0.7),
            user_id=current_user.user_id  # This triggers hybrid search
        )
        
        logger.info(f"✅ Found {len(results)} results for user {current_user.username}")
        return {"results": results, "total_results": len(results)}
        
    except Exception as e:
        logger.error(f"❌ User search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/user/documents/has-org")
async def user_has_org_documents(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Lightweight check: does the user have any .org documents? Avoids loading document list."""
    document_service = await _get_document_service()
    try:
        has_org = await document_service.document_repository.user_has_org_documents(current_user.user_id)
        return {"has_org": has_org}
    except Exception as e:
        logger.debug(f"has-org check failed: {e}")
        return {"has_org": False}


@router.get("/api/user/documents", response_model=DocumentListResponse)
async def list_user_documents(
    skip: int = 0, 
    limit: int = 100,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List user's private documents"""
    document_service = await _get_document_service()
    try:
        logger.info(f"📄 Listing documents for user {current_user.username}")
        
        # Get user-specific documents
        documents = await document_service.document_repository.list_user_documents(
            current_user.user_id, skip, limit
        )
        
        logger.info(f"📄 Found {len(documents)} documents for user {current_user.username}")
        logger.debug(f"📄 User ID: {current_user.user_id}")
        logger.debug(f"📄 Documents: {[doc.document_id for doc in documents]}")
        return DocumentListResponse(documents=documents, total=len(documents))
        
    except Exception as e:
        logger.error(f"❌ Failed to list user documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/user/collection/stats")
async def get_user_collection_stats(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get user's collection statistics"""
    document_service = await _get_document_service()
    try:
        logger.info(f"📊 Getting collection stats for user {current_user.username}")
        
        # Get stats from embedding manager
        stats = await document_service.embedding_manager.get_user_collection_stats(current_user.user_id)
        
        return {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "collection_stats": stats
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get user collection stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/user/documents/debug")
async def debug_user_documents(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Debug endpoint to check user documents"""
    document_service = await _get_document_service()
    try:
        logger.info(f"🔍 Debug: Checking documents for user {current_user.username}")
        
        # Get user documents directly from repository
        documents = await document_service.document_repository.list_user_documents(
            current_user.user_id, 0, 100
        )
        
        # Get all documents to compare
        all_documents = await document_service.document_repository.list_documents(0, 100)
        
        return {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "user_documents_count": len(documents),
            "user_documents": [
                {
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "user_id": getattr(doc, 'user_id', None),
                    "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                    "status": doc.status.value if doc.status else None
                }
                for doc in documents
            ],
            "all_documents_count": len(all_documents),
            "all_documents_sample": [
                {
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "user_id": getattr(doc, 'user_id', None),
                    "upload_date": doc.upload_date.isoformat() if doc.upload_date else None
                }
                for doc in all_documents[:5]  # Show first 5
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/folders/debug")
async def debug_folders(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Debug endpoint to check folders"""
    folder_service = await _get_folder_service()
    try:
        logger.info(f"🔍 Debug: Checking folders for user {current_user.username}")
        
        # Get user folders
        user_folders = await folder_service.document_repository.get_folders_by_user(current_user.user_id, "user")
        
        # Get global folders
        global_folders = await folder_service.document_repository.get_folders_by_user(None, "global")
        
        # Get folder tree
        folder_tree = await folder_service.get_folder_tree(current_user.user_id, "user")
        
        return {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "role": current_user.role,
            "user_folders_count": len(user_folders),
            "global_folders_count": len(global_folders),
            "folder_tree_count": len(folder_tree),
            "user_folders": [
                {
                    "folder_id": folder["folder_id"],
                    "name": folder["name"],
                    "collection_type": folder["collection_type"],
                    "user_id": folder["user_id"],
                    "parent_folder_id": folder["parent_folder_id"]
                }
                for folder in user_folders
            ],
            "global_folders": [
                {
                    "folder_id": folder["folder_id"],
                    "name": folder["name"],
                    "collection_type": folder["collection_type"],
                    "user_id": folder["user_id"],
                    "parent_folder_id": folder["parent_folder_id"]
                }
                for folder in global_folders
            ],
            "folder_tree": [
                {
                    "folder_id": folder.folder_id,
                    "name": folder.name,
                    "collection_type": folder.collection_type,
                    "user_id": folder.user_id,
                    "parent_folder_id": folder.parent_folder_id
                }
                for folder in folder_tree
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Folder debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/api/admin/clear-documents")
async def clear_all_documents(
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Clear all documents from all user folders, vector DB collections, and knowledge graph (admin only)"""
    document_service = await _get_document_service()
    try:
        logger.info(f"🗑️ Admin {current_user.username} starting complete document clearance")
        
        # Initialize counters
        deleted_documents = 0
        deleted_collections = 0
        errors = []
        
        # Step 1: Get all documents across the system
        logger.info("📋 Getting all documents from database...")
        all_documents = await document_service.list_documents(skip=0, limit=10000)
        logger.info(f"📋 Found {len(all_documents)} documents to delete")
        
        # Step 2: Delete all documents (this will also clean up vector embeddings and knowledge graph entities)
        logger.info("🗑️ Deleting all documents...")
        for doc in all_documents:
            try:
                success = await document_service.delete_document(doc.document_id)
                if success:
                    deleted_documents += 1
                    logger.info(f"🗑️ Deleted document: {doc.filename} ({doc.document_id})")
                else:
                    errors.append(f"Failed to delete document {doc.filename}")
            except Exception as e:
                error_msg = f"Error deleting document {doc.filename}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Step 3: Get all users and clear their vector collections
        logger.info("👥 Getting all users to clear their collections...")
        users_response = await auth_service.get_users(skip=0, limit=1000)
        users = users_response.users
        
        logger.info(f"👥 Found {len(users)} users, clearing their vector collections...")
        user_doc_service = UserDocumentService()
        await user_doc_service.initialize()
        
        for user in users:
            try:
                success = await user_doc_service.delete_user_collection(user.user_id)
                if success:
                    deleted_collections += 1
                    logger.info(f"🗑️ Cleared collection for user: {user.username}")
                else:
                    errors.append(f"Failed to clear collection for user {user.username}")
            except Exception as e:
                error_msg = f"Error clearing collection for user {user.username}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Step 4: Clear the global vector collection completely via Vector Service
        logger.info("🗑️ Clearing global vector collection...")
        try:
            from clients.vector_service_client import get_vector_service_client
            from config import settings
            
            vector_client = await get_vector_service_client(required=False)
            
            # Check if global collection exists
            collections_result = await vector_client.list_collections()
            if not collections_result.get("success"):
                raise Exception(f"Failed to list collections: {collections_result.get('error')}")
            
            collection_names = [col["name"] for col in collections_result.get("collections", [])]
            
            if settings.VECTOR_COLLECTION_NAME in collection_names:
                # Delete and recreate the global collection via Vector Service
                delete_result = await vector_client.delete_collection(settings.VECTOR_COLLECTION_NAME)
                if not delete_result.get("success"):
                    raise Exception(f"Failed to delete collection: {delete_result.get('error')}")
                
                # Recreate empty collection
                create_result = await vector_client.create_collection(
                    collection_name=settings.VECTOR_COLLECTION_NAME,
                    vector_size=settings.EMBEDDING_DIMENSIONS,
                    distance="COSINE"
                )
                if not create_result.get("success"):
                    raise Exception(f"Failed to create collection: {create_result.get('error')}")
                
                logger.info("🗑️ Cleared and recreated global vector collection")
            else:
                logger.info("ℹ️ Global vector collection didn't exist")
        except Exception as e:
            error_msg = f"Error clearing global vector collection: {str(e)}"
            errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        # Step 5: Clear knowledge graph completely
        logger.info("🗑️ Clearing knowledge graph...")
        try:
            if knowledge_graph_service:
                await knowledge_graph_service.clear_all_data()
                logger.info("🗑️ Cleared all knowledge graph data")
            else:
                logger.warning("⚠️ Knowledge graph service not available")
        except Exception as e:
            error_msg = f"Error clearing knowledge graph: {str(e)}"
            errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        # Step 6: Clean up orphaned files
        logger.info("🧹 Cleaning up orphaned files...")
        try:
            from pathlib import Path
            upload_dir = Path(settings.UPLOAD_DIR)
            processed_dir = Path(settings.PROCESSED_DIR) if hasattr(settings, 'PROCESSED_DIR') else None
            
            deleted_files = 0
            
            # Clean upload directory
            if upload_dir.exists():
                for file_path in upload_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        deleted_files += 1
                logger.info(f"🧹 Cleaned {deleted_files} files from upload directory")
            
            # Clean processed directory if it exists
            if processed_dir and processed_dir.exists():
                processed_files = 0
                for file_path in processed_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        processed_files += 1
                logger.info(f"🧹 Cleaned {processed_files} files from processed directory")
        except Exception as e:
            error_msg = f"Error cleaning up files: {str(e)}"
            errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        # Prepare response
        success_message = f"✅ Clearance completed: {deleted_documents} documents deleted, {deleted_collections} user collections cleared"
        
        if errors:
            success_message += f". {len(errors)} errors encountered."
            logger.warning(f"⚠️ Clearance completed with {len(errors)} errors")
        else:
            logger.info("✅ Complete document clearance successful")
        
        return {
            "success": True,
            "message": success_message,
            "deleted_documents": deleted_documents,
            "deleted_collections": deleted_collections,
            "errors": errors[:10] if errors else [],  # Limit errors shown
            "total_errors": len(errors)
        }
        
    except Exception as e:
        logger.error(f"❌ Admin clearance failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


@router.post("/api/admin/clear-documents-database-only")
async def clear_documents_database_only(
    current_user: AuthenticatedUserResponse = Depends(require_admin()),
    rescan: bool = True,
):
    """Clear document records and indexes only; leave files on disk so they can be re-read.
    Use when the database is out of sync with disk. If rescan=true (default), trigger file
    watcher rescan so documents are re-added without restart."""
    document_service = await _get_document_service()
    container = await get_service_container()
    kg_service = getattr(container, "knowledge_graph_service", None)
    try:
        logger.info(f"Admin {current_user.username} starting database-only document clearance")

        deleted_documents = 0
        deleted_collections = 0
        errors = []

        all_documents = await document_service.list_documents(skip=0, limit=10000)
        logger.info(f"Found {len(all_documents)} document records to remove (files will remain on disk)")

        for doc in all_documents:
            try:
                success = await document_service.delete_document_database_only(doc.document_id)
                if success:
                    deleted_documents += 1
                else:
                    errors.append(f"Failed to remove record for {doc.filename}")
            except Exception as e:
                errors.append(f"Error removing {doc.filename}: {str(e)}")
                logger.error(f"Error removing document record: {e}")

        users_response = await auth_service.get_users(skip=0, limit=1000)
        user_doc_service = UserDocumentService()
        await user_doc_service.initialize()
        for user in users_response.users:
            try:
                if await user_doc_service.delete_user_collection(user.user_id):
                    deleted_collections += 1
            except Exception as e:
                errors.append(f"Error clearing collection for {user.username}: {str(e)}")

        try:
            from clients.vector_service_client import get_vector_service_client
            vector_client = await get_vector_service_client(required=False)
            collections_result = await vector_client.list_collections()
            if collections_result.get("success"):
                names = [c["name"] for c in collections_result.get("collections", [])]
                if settings.VECTOR_COLLECTION_NAME in names:
                    await vector_client.delete_collection(settings.VECTOR_COLLECTION_NAME)
                    await vector_client.create_collection(
                        collection_name=settings.VECTOR_COLLECTION_NAME,
                        vector_size=settings.EMBEDDING_DIMENSIONS,
                        distance="COSINE",
                    )
        except Exception as e:
            errors.append(f"Error clearing global vector collection: {str(e)}")

        if kg_service:
            try:
                await kg_service.clear_all_data()
            except Exception as e:
                errors.append(f"Error clearing knowledge graph: {str(e)}")

        rescan_result = None
        if rescan and deleted_documents >= 0:
            try:
                from services.file_watcher_service import get_file_watcher
                file_watcher = await get_file_watcher()
                rescan_result = await file_watcher.run_rescan()
            except Exception as e:
                errors.append(f"Rescan after clear: {str(e)}")
                logger.warning(f"Rescan after clear failed: {e}")

        message = (
            f"Database-only clearance completed: {deleted_documents} document records removed, "
            f"{deleted_collections} user collections cleared. Files were left on disk."
        )
        if rescan_result and rescan_result.get("success"):
            message += " File watcher rescan was run; documents on disk should be re-added."
        elif rescan and (not rescan_result or not rescan_result.get("success")):
            message += " Restart the app or trigger a rescan to re-add documents from disk."

        return {
            "success": True,
            "message": message,
            "deleted_documents": deleted_documents,
            "deleted_collections": deleted_collections,
            "rescan_run": rescan_result.get("success") if rescan_result else False,
            "errors": errors[:10] if errors else [],
            "total_errors": len(errors),
        }
    except Exception as e:
        logger.error(f"Database-only clearance failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/admin/clear-neo4j")
async def clear_neo4j(
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Clear all data from Neo4j knowledge graph (admin only)"""
    try:
        logger.info(f"🗑️ Admin {current_user.username} clearing Neo4j knowledge graph")
        
        if not knowledge_graph_service:
            raise HTTPException(status_code=503, detail="Knowledge graph service not available")
        
        # Clear all knowledge graph data
        await knowledge_graph_service.clear_all_data()
        
        # Get stats to confirm clearing
        try:
            stats = await knowledge_graph_service.get_graph_stats()
            logger.info(f"📊 Neo4j stats after clear: {stats}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get stats after clearing: {e}")
            stats = {"total_entities": 0, "total_documents": 0, "total_relationships": 0}
        
        logger.info("✅ Neo4j knowledge graph cleared successfully")
        
        return {
            "success": True,
            "message": "✅ Neo4j knowledge graph cleared successfully",
            "stats_after_clear": stats
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear Neo4j: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear Neo4j: {str(e)}")


@router.post("/api/admin/clear-qdrant")
async def clear_qdrant(
    current_user: AuthenticatedUserResponse = Depends(require_admin())
):
    """Clear all data from Qdrant vector database (admin only)"""
    try:
        logger.info(f"🗑️ Admin {current_user.username} clearing Qdrant vector database")
        
        from clients.vector_service_client import get_vector_service_client
        from config import settings
        
        # Initialize counters
        cleared_collections = 0
        cleared_global = False
        errors = []
        
        # Initialize Vector Service client
        vector_client = await get_vector_service_client(required=False)
        
        # Get all existing collections
        try:
            collections_result = await vector_client.list_collections()
            if not collections_result.get("success"):
                raise Exception(f"Failed to list collections: {collections_result.get('error')}")
            
            collection_names = [col["name"] for col in collections_result.get("collections", [])]
            logger.info(f"📋 Found {len(collection_names)} collections in Qdrant")
        except Exception as e:
            logger.error(f"❌ Failed to list Qdrant collections: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to connect to Vector Service: {str(e)}")
        
        # Clear global collection (named "documents" per VECTOR_COLLECTION_NAME)
        logger.info(f"🔍 Looking for global collection: {settings.VECTOR_COLLECTION_NAME}")
        if settings.VECTOR_COLLECTION_NAME in collection_names:
            try:
                # Delete and recreate global collection via Vector Service
                logger.info(f"🗑️ Deleting global collection: {settings.VECTOR_COLLECTION_NAME}")
                delete_result = await vector_client.delete_collection(settings.VECTOR_COLLECTION_NAME)
                if not delete_result.get("success"):
                    raise Exception(f"Delete failed: {delete_result.get('error')}")
                
                # Recreate empty global collection
                logger.info(f"🆕 Recreating global collection: {settings.VECTOR_COLLECTION_NAME}")
                create_result = await vector_client.create_collection(
                    collection_name=settings.VECTOR_COLLECTION_NAME,
                    vector_size=settings.EMBEDDING_DIMENSIONS,
                    distance="COSINE"
                )
                if not create_result.get("success"):
                    raise Exception(f"Create failed: {create_result.get('error')}")
                
                cleared_global = True
                cleared_collections += 1
                logger.info(f"✅ Cleared and recreated global collection: {settings.VECTOR_COLLECTION_NAME}")
            except Exception as e:
                error_msg = f"Failed to clear global collection {settings.VECTOR_COLLECTION_NAME}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        else:
            logger.info(f"ℹ️ Global collection '{settings.VECTOR_COLLECTION_NAME}' not found in Qdrant (may not exist)")
        
        # Clear user collections (collections that start with 'user_')
        user_collections = [name for name in collection_names if name.startswith('user_')]
        logger.info(f"👥 Found {len(user_collections)} user collections to clear")
        
        for collection_name in user_collections:
            try:
                delete_result = await vector_client.delete_collection(collection_name)
                if delete_result.get("success"):
                    cleared_collections += 1
                    logger.info(f"🗑️ Deleted user collection: {collection_name}")
                else:
                    raise Exception(delete_result.get("error", "Unknown error"))
            except Exception as e:
                error_msg = f"Failed to delete collection {collection_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Clear team collections (collections that start with 'team_')
        team_collections = [name for name in collection_names if name.startswith('team_')]
        logger.info(f"👥 Found {len(team_collections)} team collections to clear")
        
        for collection_name in team_collections:
            try:
                delete_result = await vector_client.delete_collection(collection_name)
                if delete_result.get("success"):
                    cleared_collections += 1
                    logger.info(f"🗑️ Deleted team collection: {collection_name}")
                else:
                    raise Exception(delete_result.get("error", "Unknown error"))
            except Exception as e:
                error_msg = f"Failed to delete collection {collection_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Get final collection count
        try:
            final_result = await vector_client.list_collections()
            if final_result.get("success"):
                remaining_count = len(final_result.get("collections", []))
                logger.info(f"📊 Collections remaining after clear: {remaining_count}")
            else:
                remaining_count = "unknown"
        except Exception as e:
            logger.warning(f"⚠️ Could not get final collection count: {e}")
            remaining_count = "unknown"
        
        # Prepare response
        success_message = f"✅ Qdrant cleared: {cleared_collections} collections processed"
        if cleared_global:
            success_message += " (including global collection)"
        
        if errors:
            success_message += f", {len(errors)} errors encountered"
            logger.warning(f"⚠️ Qdrant clearing completed with {len(errors)} errors")
        else:
            logger.info("✅ Qdrant vector database cleared successfully")
        
        return {
            "success": True,
            "message": success_message,
            "cleared_collections": cleared_collections,
            "cleared_global": cleared_global,
            "remaining_collections": remaining_count,
            "errors": errors[:5] if errors else [],  # Limit errors shown
            "total_errors": len(errors)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear Qdrant: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear Qdrant: {str(e)}")


@router.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(skip: int = 0, limit: int = 100):
    """List global/admin documents only"""
    document_service = await _get_document_service()
    try:
        # Get only global documents (admin uploads or approved submissions)
        documents = await document_service.document_repository.list_global_documents(skip, limit)
        return DocumentListResponse(documents=documents, total=len(documents))
        
    except Exception as e:
        logger.error(f"❌ Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/{doc_id}/status", response_model=DocumentStatus)
async def get_processing_status(doc_id: str):
    """Get document processing status and quality metrics"""
    document_service = await _get_document_service()
    try:
        status = await document_service.get_document_status(doc_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get document status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/documents/{doc_id}/reprocess")
async def reprocess_document(doc_id: str, current_user: AuthenticatedUserResponse = Depends(require_admin())):
    """Re-process a failed or completed document"""
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()
    try:
        logger.info(f"🔄 Re-processing document: {doc_id}")
        
        # Get document info
        doc_info = await document_service.get_document(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reset status to processing
        await document_service.document_repository.update_status(doc_id, ProcessingStatus.PROCESSING)
        
        # Clear existing embeddings for this document
        try:
            await document_service.embedding_manager.delete_document_chunks(doc_id)
            logger.info(f"🗑️  Cleared existing embeddings for document {doc_id}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to clear embeddings for {doc_id}: {e}")
        
        # Determine file path using new folder structure
        file_path = None
        
        # Try new folder structure first
        try:
            from services.service_container import get_service_container
            container = await get_service_container()
            folder_service = container.folder_service
            
            logger.info(f"🔍 Looking for file: {doc_info.filename}")
            logger.info(f"🔍 Document folder_id: {doc_info.folder_id if hasattr(doc_info, 'folder_id') else 'None'}")
            logger.info(f"🔍 Document user_id: {doc_info.user_id if hasattr(doc_info, 'user_id') else 'None'}")
            
            # Try without doc_id prefix first (new style)
            folder_path = await folder_service.get_document_file_path(
                filename=doc_info.filename,
                folder_id=doc_info.folder_id if hasattr(doc_info, 'folder_id') else None,
                user_id=doc_info.user_id if hasattr(doc_info, 'user_id') else None,
                collection_type="global" if not doc_info.user_id else "user"
            )
            
            logger.info(f"🔍 Checking path (no prefix): {folder_path}")
            
            if folder_path and Path(folder_path).exists():
                file_path = Path(folder_path)
                logger.info(f"📂 Found file in new structure (no prefix): {file_path}")
            else:
                # Try with doc_id prefix (legacy style)
                filename_with_id = f"{doc_id}_{doc_info.filename}"
                folder_path = await folder_service.get_document_file_path(
                    filename=filename_with_id,
                    folder_id=doc_info.folder_id if hasattr(doc_info, 'folder_id') else None,
                    user_id=doc_info.user_id if hasattr(doc_info, 'user_id') else None,
                    collection_type="global" if not doc_info.user_id else "user"
                )
                
                logger.info(f"🔍 Checking path (with prefix): {folder_path}")
                
                if folder_path and Path(folder_path).exists():
                    file_path = Path(folder_path)
                    logger.info(f"📂 Found file in new structure (with prefix): {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Error checking new folder structure: {e}")
        
        # Fallback to legacy structure
        if not file_path:
            upload_dir = Path(settings.UPLOAD_DIR)
            for potential_file in upload_dir.glob(f"{doc_id}_*"):
                file_path = potential_file
                logger.info(f"📂 Found file in legacy structure: {file_path}")
                break
        
        if not file_path or not file_path.exists():
            # If original file not found, check if it's a URL import
            if doc_info.doc_type == DocumentType.URL:
                # Re-import from URL (global documents don't have user_id)
                asyncio.create_task(document_service._process_url_async(doc_id, doc_info.filename, "html"))
                logger.info(f"🔗 Re-importing URL document: {doc_info.filename}")
            else:
                raise HTTPException(status_code=404, detail="Original file not found - cannot reprocess")
        else:
            # Re-process the existing file, preserving original user_id/collection type
            doc_type = document_service._detect_document_type(doc_info.filename)
            asyncio.create_task(document_service._process_document_async(
                doc_id, 
                file_path, 
                doc_type, 
                user_id=doc_info.user_id  # Preserve original collection type
            ))
            logger.info(f"📄 Re-processing file: {file_path} (user_id={doc_info.user_id}, collection={'global' if not doc_info.user_id else 'user'})")
        
        logger.info(f"✅ Document {doc_id} queued for re-processing")
        return {
            "status": "success", 
            "message": f"Document {doc_id} queued for re-processing",
            "document_id": doc_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to re-process document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/user/documents/rescan")
async def rescan_user_files(
    dry_run: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Scan user's filesystem and recover orphaned files
    
    **BULLY!** Bring those lost files back into the fold!
    
    Args:
        dry_run: If true, only report what would be recovered without making changes
    
    Returns:
        Recovery results with statistics
    """
    try:
        from services.file_recovery_service import get_file_recovery_service
        
        logger.info(f"🔍 ROOSEVELT: Rescanning files for user {current_user.user_id}")
        
        recovery_service = await get_file_recovery_service()
        result = await recovery_service.scan_and_recover_user_files(
            user_id=current_user.user_id,
            dry_run=dry_run
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ File rescan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/user/documents/{doc_id}/reprocess")
async def reprocess_user_document(doc_id: str, current_user: AuthenticatedUserResponse = Depends(get_current_user)):
    """Re-process a user's failed or completed document"""
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()
    try:
        logger.info(f"🔄 Re-processing user document: {doc_id} for user: {current_user.user_id}")
        
        # Get document info and verify ownership
        doc_info = await document_service.get_document(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user owns this document (this would need to be implemented in the document service)
        # For now, we'll assume user documents are stored with user_id in the database
        # This is a simplified check - in a real implementation, you'd verify ownership
        
        # Reset status to processing
        await document_service.document_repository.update_status(doc_id, ProcessingStatus.PROCESSING)
        
        # Clear existing embeddings for this document
        try:
            await document_service.embedding_manager.delete_document_chunks(doc_id)
            logger.info(f"🗑️  Cleared existing embeddings for document {doc_id}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to clear embeddings for {doc_id}: {e}")
        
        # Determine file path using new folder structure
        file_path = None
        
        # Try new folder structure first
        try:
            from services.service_container import get_service_container
            container = await get_service_container()
            folder_service = container.folder_service
            
            logger.info(f"🔍 Looking for file: {doc_info.filename}")
            logger.info(f"🔍 Document folder_id: {doc_info.folder_id if hasattr(doc_info, 'folder_id') else 'None'}")
            logger.info(f"🔍 Document user_id: {doc_info.user_id if hasattr(doc_info, 'user_id') else 'None'}")
            
            # Try without doc_id prefix first (new style)
            folder_path = await folder_service.get_document_file_path(
                filename=doc_info.filename,
                folder_id=doc_info.folder_id if hasattr(doc_info, 'folder_id') else None,
                user_id=current_user.user_id,
                collection_type="user"
            )
            
            logger.info(f"🔍 Checking path (no prefix): {folder_path}")
            
            if folder_path and Path(folder_path).exists():
                file_path = Path(folder_path)
                logger.info(f"📂 Found file in new structure (no prefix): {file_path}")
            else:
                # Try with doc_id prefix (legacy style)
                filename_with_id = f"{doc_id}_{doc_info.filename}"
                folder_path = await folder_service.get_document_file_path(
                    filename=filename_with_id,
                    folder_id=doc_info.folder_id if hasattr(doc_info, 'folder_id') else None,
                    user_id=current_user.user_id,
                    collection_type="user"
                )
                
                logger.info(f"🔍 Checking path (with prefix): {folder_path}")
                
                if folder_path and Path(folder_path).exists():
                    file_path = Path(folder_path)
                    logger.info(f"📂 Found file in new structure (with prefix): {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Error checking new folder structure: {e}")
        
        # Fallback to legacy structure
        if not file_path:
            upload_dir = Path(settings.UPLOAD_DIR)
            for potential_file in upload_dir.glob(f"{doc_id}_*"):
                file_path = potential_file
                logger.info(f"📂 Found file in legacy structure: {file_path}")
                break
        
        if not file_path or not file_path.exists():
            # If original file not found, check if it's a URL import
            if doc_info.doc_type == DocumentType.URL:
                # Re-import from URL with user_id
                asyncio.create_task(document_service._process_url_async(doc_id, doc_info.filename, "html", current_user.user_id))
                logger.info(f"🔗 Re-importing URL document: {doc_info.filename}")
            else:
                raise HTTPException(status_code=404, detail="Original file not found - cannot reprocess")
        else:
            # Re-process the existing file with user_id
            doc_type = document_service._detect_document_type(doc_info.filename)
            asyncio.create_task(document_service._process_document_async(doc_id, file_path, doc_type, current_user.user_id))
            logger.info(f"📄 Re-processing file: {file_path}")
        
        logger.info(f"✅ User document {doc_id} queued for re-processing")
        return {
            "status": "success", 
            "message": f"Document {doc_id} queued for re-processing",
            "document_id": doc_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to re-process user document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/{doc_id}/pdf")
async def get_document_pdf(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Serve the original PDF file for a document"""
    folder_service = await _get_folder_service()
    try:
        logger.info(f"📄 Serving PDF file for document: {doc_id}")
        
        # SECURITY: Check access authorization
        doc_info = await check_document_access(doc_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if it's a PDF document
        if not doc_info.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Document is not a PDF")
        
        # **ROOSEVELT FIX**: Use folder service to get correct file path
        from pathlib import Path
        import os
        
        filename = getattr(doc_info, 'filename', None)
        user_id = getattr(doc_info, 'user_id', None)
        folder_id = getattr(doc_info, 'folder_id', None)
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        # SECURITY: Sanitize filename to prevent path traversal
        if filename:
            safe_filename = os.path.basename(filename)
            if not safe_filename or safe_filename in ('.', '..'):
                logger.error(f"Invalid filename in document metadata: {filename}")
                raise HTTPException(status_code=500, detail="Invalid file metadata")
            filename = safe_filename
        
        file_path = None
        
        if filename:
            # Try new folder structure first
            try:
                file_path_str = await folder_service.get_document_file_path(
                    filename=filename,
                    folder_id=folder_id,
                    user_id=user_id,
                    collection_type=collection_type
                )
                file_path = Path(file_path_str)
                
                # SECURITY: Verify resolved path is within uploads directory
                try:
                    uploads_base = Path(settings.UPLOAD_DIR).resolve()
                    file_path_resolved = file_path.resolve()
                    
                    if not str(file_path_resolved).startswith(str(uploads_base)):
                        logger.error(f"Path traversal attempt detected in document: {doc_id} -> {file_path_resolved}")
                        raise HTTPException(status_code=403, detail="Access denied")
                except Exception as e:
                    logger.error(f"Path validation error for document {doc_id}: {e}")
                    raise HTTPException(status_code=403, detail="Access denied")
                
                if not file_path.exists():
                    logger.warning(f"⚠️ PDF not found at computed path: {file_path}")
                    file_path = None
            except Exception as e:
                logger.warning(f"⚠️ Failed to compute file path with folder service: {e}")
        
        # Fall back to legacy flat structure if not found in new structure
        if file_path is None or not file_path.exists():
            upload_dir = Path(settings.UPLOAD_DIR)
            legacy_paths = [
                upload_dir / f"{doc_id}_{doc_info.filename}",
                upload_dir / doc_info.filename
            ]
            
            for legacy_path in legacy_paths:
                if legacy_path.exists():
                    file_path = legacy_path
                    logger.info(f"📄 Found PDF in legacy location: {file_path}")
                    break
        
        if file_path is None or not file_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found on disk")
        
        logger.info(f"✅ Serving PDF file: {file_path}")
        return FileResponse(
            path=str(file_path),
            filename=doc_info.filename,
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to serve PDF file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/{doc_id}/file")
async def get_document_file(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Serve the original file for a document (audio, images, etc.)"""
    folder_service = await _get_folder_service()
    try:
        logger.info(f"📄 Serving file for document: {doc_id}")
        
        # SECURITY: Check access authorization
        doc_info = await check_document_access(doc_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get file path using folder service
        from pathlib import Path
        from fastapi.responses import FileResponse
        import os
        import mimetypes
        
        filename = getattr(doc_info, 'filename', None)
        user_id = getattr(doc_info, 'user_id', None)
        folder_id = getattr(doc_info, 'folder_id', None)
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        # SECURITY: Sanitize filename to prevent path traversal
        if filename:
            safe_filename = os.path.basename(filename)
            if not safe_filename or safe_filename in ('.', '..'):
                logger.error(f"Invalid filename in document metadata: {filename}")
                raise HTTPException(status_code=500, detail="Invalid file metadata")
            filename = safe_filename
        
        file_path = None
        
        if filename:
            # Try new folder structure first
            try:
                file_path_str = await folder_service.get_document_file_path(
                    filename=filename,
                    folder_id=folder_id,
                    user_id=user_id,
                    collection_type=collection_type
                )
                file_path = Path(file_path_str)
                
                # SECURITY: Verify resolved path is within uploads directory
                try:
                    uploads_base = Path(settings.UPLOAD_DIR).resolve()
                    file_path_resolved = file_path.resolve()
                    
                    if not str(file_path_resolved).startswith(str(uploads_base)):
                        logger.error(f"Path traversal attempt detected in document: {doc_id} -> {file_path_resolved}")
                        raise HTTPException(status_code=403, detail="Access denied")
                except Exception as e:
                    logger.error(f"Path validation error for document {doc_id}: {e}")
                    raise HTTPException(status_code=403, detail="Access denied")
                
                if not file_path.exists():
                    logger.warning(f"⚠️ File not found at computed path: {file_path}")
                    file_path = None
            except Exception as e:
                logger.warning(f"⚠️ Failed to compute file path with folder service: {e}")
        
        # Fall back to legacy flat structure if not found in new structure
        if file_path is None or not file_path.exists():
            upload_dir = Path(settings.UPLOAD_DIR)
            legacy_paths = [
                upload_dir / f"{doc_id}_{doc_info.filename}",
                upload_dir / doc_info.filename
            ]
            
            for legacy_path in legacy_paths:
                if legacy_path.exists():
                    file_path = legacy_path
                    logger.info(f"📄 Found file in legacy location: {file_path}")
                    break
        
        if file_path is None or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        # Determine media type from file extension
        media_type, _ = mimetypes.guess_type(str(file_path))
        if not media_type:
            media_type = "application/octet-stream"
        
        logger.info(f"✅ Serving file: {file_path} (type: {media_type})")
        return FileResponse(
            path=str(file_path),
            filename=doc_info.filename,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to serve file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a document and all its embeddings"""
    document_service = await _get_document_service()
    try:
        logger.info(f"🗑️  Deleting document: {doc_id}")
        
        # SECURITY: Check delete authorization
        await check_document_access(doc_id, current_user, "delete")
        
        success = await document_service.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"✅ Document {doc_id} deleted successfully by user {current_user.user_id}")
        return {"status": "success", "message": f"Document {doc_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/stats")
async def get_documents_stats():
    """Get statistics about stored documents and embeddings"""
    document_service = await _get_document_service()
    try:
        stats = await document_service.get_documents_stats()
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get documents stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/documents/cleanup")
async def cleanup_orphaned_embeddings():
    """Clean up embeddings for documents that no longer exist"""
    document_service = await _get_document_service()
    try:
        logger.info("🧹 Starting cleanup of orphaned embeddings...")
        
        cleaned_count = await document_service.cleanup_orphaned_embeddings()
        
        return {
            "status": "success", 
            "message": f"Cleaned up {cleaned_count} orphaned document embeddings"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup orphaned embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/duplicates")
async def get_duplicate_documents():
    """Get all duplicate documents grouped by file hash"""
    document_service = await _get_document_service()
    try:
        logger.info("🔍 Getting duplicate documents")
        
        duplicates = await document_service.get_duplicate_documents()
        
        # Convert to a more API-friendly format
        duplicate_groups = []
        for file_hash, docs in duplicates.items():
            duplicate_groups.append({
                "file_hash": file_hash,
                "document_count": len(docs),
                "total_size": sum(doc.file_size for doc in docs),
                "documents": [
                    {
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "upload_date": doc.upload_date.isoformat(),
                        "file_size": doc.file_size,
                        "status": doc.status.value
                    }
                    for doc in docs
                ]
            })
        
        logger.info(f"✅ Found {len(duplicate_groups)} groups of duplicate documents")
        return {
            "duplicate_groups": duplicate_groups,
            "total_groups": len(duplicate_groups),
            "total_duplicates": sum(group["document_count"] for group in duplicate_groups),
            "wasted_storage": sum(group["total_size"] - (group["total_size"] // group["document_count"]) for group in duplicate_groups)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get duplicate documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== DOCUMENT CATEGORIZATION ENDPOINTS =====

@router.post("/api/documents/filter", response_model=DocumentListResponse)
async def filter_documents(filter_request: DocumentFilterRequest):
    """Filter and search documents with advanced criteria"""
    document_service = await _get_document_service()
    try:
        logger.info(f"🔍 Filtering documents with criteria")
        
        result = await document_service.filter_documents(filter_request)
        
        logger.info(f"✅ Found {result.total} documents matching criteria")
        return result
        
    except Exception as e:
        logger.error(f"❌ Document filtering failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/documents/{doc_id}/metadata")
async def update_document_metadata(doc_id: str, update_request: DocumentUpdateRequest):
    """Update document metadata (title, category, tags, etc.)"""
    document_service = await _get_document_service()
    try:
        logger.info(f"📝 Updating metadata for document: {doc_id}")
        
        success = await document_service.update_document_metadata(doc_id, update_request)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"✅ Document metadata updated: {doc_id}")
        return {"status": "success", "message": f"Document {doc_id} metadata updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update document metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/documents/bulk-categorize", response_model=BulkOperationResponse)
async def bulk_categorize_documents(bulk_request: BulkCategorizeRequest):
    """Bulk categorize multiple documents"""
    document_service = await _get_document_service()
    try:
        logger.info(f"📋 Bulk categorizing {len(bulk_request.document_ids)} documents")
        
        result = await document_service.bulk_categorize_documents(bulk_request)
        
        logger.info(f"✅ Bulk categorization completed: {result.success_count} successful")
        return result
        
    except Exception as e:
        logger.error(f"❌ Bulk categorization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/categories", response_model=DocumentCategoriesResponse)
async def get_document_categories_overview():
    """Get overview of document categories and tags"""
    document_service = await _get_document_service()
    try:
        logger.info("📊 Getting document categories overview")
        
        overview = await document_service.get_document_categories_overview()
        
        logger.info(f"✅ Categories overview retrieved: {len(overview.categories)} categories, {len(overview.tags)} tags")
        return overview
        
    except Exception as e:
        logger.error(f"❌ Failed to get categories overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/{doc_id}/content")
async def get_document_content(
    doc_id: str,
    request: Request,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get document content by ID"""
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()
    try:
        logger.info(f"📄 API: Getting content for document {doc_id}")
        
        # SECURITY: Check read authorization
        document = await check_document_access(doc_id, current_user, "read")
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # ALWAYS get content from the actual file on disk
        full_content = None  # None = file not found, "" = empty file, "content" = file with content
        content_source = "file"
        
        try:
            from pathlib import Path
            
            # Use folder service to find the file in the new structure
            filename = getattr(document, 'filename', None)
            user_id = getattr(document, 'user_id', None)
            folder_id = getattr(document, 'folder_id', None)
            collection_type = getattr(document, 'collection_type', 'user')
            
            # Skip content loading for PDFs - they're served via the /pdf endpoint
            # But still compute the file path for canonical_path
            if filename and filename.lower().endswith('.pdf'):
                logger.info(f"📄 API: Skipping content load for PDF: {filename} (use /pdf endpoint instead)")
                full_content = ""  # Empty content for PDFs
                content_source = "pdf_binary"
                # Still compute file path for metadata
                try:
                    file_path_str = await folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=folder_id,
                        user_id=user_id,
                        collection_type=collection_type
                    )
                    file_path = Path(file_path_str)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to compute file path for PDF: {e}")
            # Skip content loading for image files - they're binary and served via /api/images/ endpoint
            elif filename and any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg']):
                logger.info(f"🖼️ API: Skipping content load for image: {filename} (use /api/images/ endpoint instead)")
                full_content = ""  # Empty content for images
                content_source = "image_binary"
                # Still compute file path for metadata
                try:
                    file_path_str = await folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=folder_id,
                        user_id=user_id,
                        collection_type=collection_type
                    )
                    file_path = Path(file_path_str)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to compute file path for image: {e}")
            # Skip content loading for DocX files - they're binary and served via /file endpoint
            elif filename and filename.lower().endswith('.docx'):
                logger.info(f"📄 API: Skipping content load for DocX: {filename} (use /file endpoint instead)")
                full_content = ""  # Empty content for DocX files
                content_source = "docx_binary"
                # Still compute file path for metadata
                try:
                    file_path_str = await folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=folder_id,
                        user_id=user_id,
                        collection_type=collection_type
                    )
                    file_path = Path(file_path_str)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to compute file path for DocX: {e}")
            elif filename:
                # Try new folder structure first
                try:
                    file_path_str = await folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=folder_id,
                        user_id=user_id,
                        collection_type=collection_type
                    )
                    file_path = Path(file_path_str)
                    
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            full_content = f.read()
                        logger.info(f"✅ API: Loaded content from file: {file_path}")
                    else:
                        logger.warning(f"⚠️ File not found at computed path: {file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to compute file path with folder service: {e}")
                
                # Fall back to searching user directory for org files
                if full_content is None and filename.endswith('.org'):
                    # For org files, search user directory tree
                    username = await folder_service._get_username(user_id) if user_id else None
                    if username:
                        user_dir = Path(settings.UPLOAD_DIR) / "Users" / username
                        if user_dir.exists():
                            # Search recursively for the org file
                            matching_files = list(user_dir.rglob(filename))
                            if matching_files:
                                file_path = matching_files[0]  # Use first match
                                if file_path.exists():
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        full_content = f.read()
                                    logger.info(f"✅ API: Found org file in subdirectory: {file_path}")
                
                # Fall back to old paths if still not found
                if full_content is None:
                    upload_dir = Path(settings.UPLOAD_DIR)
                    legacy_paths = [
                        upload_dir / f"{doc_id}_{filename}",
                        upload_dir / filename
                    ]
                    
                    # For markdown, also check web_sources
                    if filename.endswith('.md'):
                        import glob
                        legacy_paths.extend([
                            upload_dir / "web_sources" / "rss_articles" / "*" / filename,
                            upload_dir / "web_sources" / "scraped_content" / "*" / filename
                        ])
                    
                    import glob
                    for path_pattern in legacy_paths:
                        matches = glob.glob(str(path_pattern)) if '*' in str(path_pattern) else [str(path_pattern)]
                        if matches:
                            file_path = Path(matches[0])
                            if file_path.exists():
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    full_content = f.read()
                                logger.info(f"✅ API: Loaded content from legacy path: {file_path}")
                                break
            
            # If file not found, this is an error - vectors are NOT for viewing
            # **BULLY!** Allow empty files - distinguish between "file not found" vs "empty file"
            if full_content is None:
                logger.error(f"❌ API: File not found for document {doc_id} (filename: {getattr(document, 'filename', 'unknown')})")
                raise HTTPException(status_code=404, detail=f"Document file not found on disk for {doc_id}")
            elif full_content == "":
                logger.info(f"📝 API: Document {doc_id} has empty content (file exists but is empty)")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ API: Failed to load file for document {doc_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load document file: {str(e)}")
        
        # **BULLY!** For editing, we need the full content WITH frontmatter
        # Only strip frontmatter for read-only display, not for editing
        display_content = full_content
        
        # **ROOSEVELT'S CANONICAL PATH FIX**: Ensure we always have canonical_path and folder info
        # Get folder information for frontend use
        folder_name = None
        try:
            if folder_id:
                folder = await folder_service.get_folder(folder_id, user_id)
                if folder:
                    # folder is a DocumentFolder Pydantic model, access attribute directly
                    folder_name = getattr(folder, 'name', None)
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch folder name: {e}")
        
        # Ensure canonical_path is set - this is critical for relative reference resolution!
        canonical_path_str = None
        if 'file_path' in locals() and file_path:
            canonical_path_str = str(file_path)
        else:
            # If we got here, we found the content but file_path wasn't preserved
            # Reconstruct it from the folder service
            try:
                if filename:
                    file_path_str = await folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=folder_id,
                        user_id=user_id,
                        collection_type=collection_type
                    )
                    canonical_path_str = file_path_str
            except Exception as e:
                logger.warning(f"⚠️ Could not construct canonical_path: {e}")
        
        # Get updated_at from database directly (not in DocumentInfo model)
        updated_at = None
        try:
            from services.database_manager.database_helpers import fetch_one
            row = await fetch_one(
                "SELECT updated_at FROM document_metadata WHERE document_id = $1",
                doc_id,
                rls_context={'user_id': '', 'user_role': 'admin'}
            )
            if row and row.get('updated_at'):
                updated_at = row['updated_at'].isoformat() if hasattr(row['updated_at'], 'isoformat') else str(row['updated_at'])
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch updated_at: {e}")
        
        # Get metadata from document
        metadata = {
            "document_id": document.document_id,
            "title": document.title,
            "filename": document.filename,
            "author": document.author,
            "description": document.description,
            "category": document.category.value if document.category else None,
            "tags": document.tags,
            "created_at": document.upload_date.isoformat() if document.upload_date else None,
            "updated_at": updated_at,
            "status": document.status.value if document.status else None,
            "file_size": document.file_size,
            "language": document.language,
            "user_id": getattr(document, 'user_id', None),
            "collection_type": getattr(document, 'collection_type', None),
            "folder_id": folder_id,  # **ROOSEVELT: Add folder_id**
            "folder_name": folder_name,  # **ROOSEVELT: Add folder_name for display**
            "canonical_path": canonical_path_str  # **ROOSEVELT: Reliable canonical path!**
        }
        
        response_data = {
            "content": display_content,
            "metadata": metadata,
            "total_length": len(display_content),
            "content_source": content_source,
            "chunk_count": 0  # For PDFs and other files, chunk count is not relevant for viewing
        }
        
        logger.info(f"✅ API: Returning content for {doc_id} from {content_source}: {len(full_content)} characters")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ API: Failed to get document content for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document content: {str(e)}")


@router.post("/api/documents/{doc_id}/exempt")
async def exempt_document_from_vectorization(
    doc_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Exempt a document from vectorization and knowledge graph processing"""
    document_service = await _get_document_service()
    try:
        success = await document_service.exempt_document_from_vectorization(
            doc_id, 
            current_user.user_id
        )
        if success:
            return {"status": "success", "message": "Document exempted from vectorization"}
        else:
            raise HTTPException(status_code=500, detail="Failed to exempt document")
    except Exception as e:
        logger.error(f"❌ Failed to exempt document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/documents/{doc_id}/exempt")
async def remove_document_exemption(
    doc_id: str,
    inherit: bool = False,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Remove exemption from document.
    
    Args:
        doc_id: Document ID
        inherit: If True, set to inherit from folder. If False, set to explicit vectorize.
    """
    document_service = await _get_document_service()
    try:
        success = await document_service.remove_document_exemption(
            doc_id,
            current_user.user_id,
            inherit=inherit
        )
        if success:
            if inherit:
                return {"status": "success", "message": "Document now inherits from folder"}
            else:
                return {"status": "success", "message": "Document exemption removed and re-processed"}
        else:
            raise HTTPException(status_code=500, detail="Failed to remove exemption")
    except Exception as e:
        logger.error(f"❌ Failed to remove exemption for document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/folders/{folder_id}/exempt")
async def exempt_folder_from_vectorization(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Exempt a folder and all descendants from vectorization"""
    logger.info(f"🚫 API: Exempting folder {folder_id} for user {current_user.user_id}")
    folder_service = await _get_folder_service()
    try:
        from services.service_container import get_service_container
        container = await get_service_container()
        folder_service = container.folder_service

        success = await folder_service.exempt_folder_from_vectorization(
            folder_id,
            current_user.user_id
        )
        if success:
            logger.info(f"✅ API: Folder {folder_id} exempted successfully")
            return {"status": "success", "message": "Folder and descendants exempted from vectorization"}
        else:
            logger.error(f"❌ API: Failed to exempt folder {folder_id} - method returned false")
            raise HTTPException(status_code=500, detail="Failed to exempt folder")
    except Exception as e:
        logger.error(f"❌ Failed to exempt folder {folder_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/folders/{folder_id}/exempt")
async def remove_folder_exemption(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Remove exemption from a folder (set to inherit from parent), re-process all documents"""
    folder_service = await _get_folder_service()
    try:
        from services.service_container import get_service_container
        container = await get_service_container()
        folder_service = container.folder_service
        
        success = await folder_service.remove_folder_exemption(
            folder_id,
            current_user.user_id
        )
        if success:
            return {"status": "success", "message": "Folder exemption removed - now inherits from parent"}
        else:
            raise HTTPException(status_code=500, detail="Failed to remove exemption")
    except Exception as e:
        logger.error(f"❌ Failed to remove exemption for folder {folder_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/folders/{folder_id}/exempt/override")
async def override_folder_exemption(
    folder_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Set folder to explicitly NOT exempt (override parent exemption)"""
    folder_service = await _get_folder_service()
    try:
        from services.service_container import get_service_container
        container = await get_service_container()
        folder_service = container.folder_service
        
        success = await folder_service.override_folder_exemption(
            folder_id,
            current_user.user_id
        )
        if success:
            return {"status": "success", "message": "Folder set to override parent exemption - not exempt"}
        else:
            raise HTTPException(status_code=500, detail="Failed to set override")
    except Exception as e:
        logger.error(f"❌ Failed to set override for folder {folder_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/documents/{doc_id}/content")
async def update_document_content(
    doc_id: str,
    request: Request,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update a text-based document's content on disk and re-embed chunks.
    Supports .txt, .md, .org. Non-text or binary docs are rejected.
    """
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()
    try:
        # SECURITY: Check write authorization
        await check_document_access(doc_id, current_user, "write")
        
        body = await request.json()
        new_content = body.get("content")
        if new_content is None:
            raise HTTPException(status_code=400, detail="Missing 'content' in request body")
        if not isinstance(new_content, str):
            raise HTTPException(status_code=400, detail="'content' must be a string")

        # Fetch document metadata
        doc_info = await document_service.get_document(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")

        filename = getattr(doc_info, 'filename', None) or ""
        if not filename:
            raise HTTPException(status_code=400, detail="Document filename missing")

        # Only allow text-editable types
        editable_exts = (".txt", ".md", ".org")
        if not str(filename).lower().endswith(editable_exts):
            raise HTTPException(status_code=400, detail="Only .txt, .md, and .org documents can be edited")

        # Locate file path on disk using folder service
        from pathlib import Path
        user_id = getattr(doc_info, 'user_id', None)
        folder_id = getattr(doc_info, 'folder_id', None)
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        file_path = None
        try:
            file_path_str = await folder_service.get_document_file_path(
                filename=filename,
                folder_id=folder_id,
                user_id=user_id,
                collection_type=collection_type
            )
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                logger.warning(f"⚠️ File not found at computed path: {file_path}")
                file_path = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to compute file path with folder service: {e}")
        
        # Fall back to legacy paths if not found
        if file_path is None or not file_path.exists():
            upload_dir = Path(settings.UPLOAD_DIR)
            legacy_paths = [
                upload_dir / f"{doc_id}_{filename}",
                upload_dir / filename
            ]
            
            # For markdown, also check web_sources
            if filename.lower().endswith('.md'):
                import glob
                legacy_paths.extend([
                    upload_dir / "web_sources" / "rss_articles" / "*" / filename,
                    upload_dir / "web_sources" / "scraped_content" / "*" / filename
                ])
            
            import glob
            for path_pattern in legacy_paths:
                matches = glob.glob(str(path_pattern)) if '*' in str(path_pattern) else [str(path_pattern)]
                if matches:
                    candidate = Path(matches[0])
                    if candidate.exists():
                        file_path = candidate
                        break
            
            if file_path is None or not file_path.exists():
                raise HTTPException(status_code=404, detail="Original file not found on disk")

        # If content is unchanged, skip disk write and background re-indexing.
        # This avoids unnecessary chunk deletion + re-embedding for no-op saves.
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_content = f.read()
            if existing_content == new_content:
                logger.info(f"📝 No-op save detected for {doc_id}; skipping re-indexing")
                return {
                    "status": "success",
                    "message": "No changes detected. Skipping re-indexing.",
                    "document_id": doc_id,
                }
        except UnicodeDecodeError:
            # If the file isn't valid UTF-8, fall back to the existing behavior
            # (write + reprocess) rather than rejecting or guessing.
            logger.warning(f"⚠️ Unable to read existing content as UTF-8 for {doc_id}; proceeding with save")
        except Exception as e:
            logger.warning(f"⚠️ Failed to compare existing content for {doc_id}; proceeding with save: {e}")

        # Write content to disk
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logger.info(f"📝 Updated file on disk: {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to write updated content: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save content: {str(e)}")

        # Update file size
        try:
            await document_service.document_repository.update_file_size(doc_id, len(new_content.encode('utf-8')))
        except Exception as e:
            logger.warning(f"⚠️ Failed to update file size metadata: {e}")

        # Extract and store document links for file relation graph (text types only)
        if str(filename).lower().endswith(('.org', '.md', '.txt')):
            try:
                from services.link_extraction_service import get_link_extraction_service
                link_service = await get_link_extraction_service()
                rls_context = {'user_id': current_user.user_id, 'user_role': current_user.role}
                await link_service.extract_and_store_links(doc_id, new_content, rls_context)
            except Exception as link_err:
                logger.warning("Link extraction failed for %s: %s", doc_id, link_err)

        # Check if document is exempt from vectorization BEFORE processing
        is_exempt = await document_service.document_repository.is_document_exempt(doc_id, current_user.user_id)
        if is_exempt:
            logger.info(f"🚫 Document {doc_id} is exempt from vectorization - skipping embedding and entity extraction")
            await document_service.document_repository.update_status(doc_id, ProcessingStatus.COMPLETED)
            return {"status": "success", "message": "Content updated (exempt from vectorization)", "document_id": doc_id}

        # Queue re-embedding and entity extraction in background so save returns immediately
        await document_service.document_repository.update_status(doc_id, ProcessingStatus.EMBEDDING)
        await document_service._emit_document_status_update(doc_id, ProcessingStatus.EMBEDDING.value, current_user.user_id)
        try:
            await document_service.embedding_manager.delete_document_chunks(doc_id)
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete old chunks for {doc_id}: {e}")
        if document_service.kg_service:
            try:
                await document_service.kg_service.delete_document_entities(doc_id)
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete old KG entities for {doc_id}: {e}")

        from services.celery_tasks.document_tasks import reprocess_document_after_save_task
        reprocess_document_after_save_task.apply_async(args=[doc_id, current_user.user_id], queue="default")
        logger.info(f"Content saved for {doc_id}; re-indexing queued in background")
        return {
            "status": "success",
            "message": "Content saved. Re-indexing in background.",
            "document_id": doc_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update document content for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update document content: {str(e)}")

