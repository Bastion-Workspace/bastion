"""
Document Repository - Database operations for document metadata
Handles PostgreSQL operations for document storage and retrieval
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4

import asyncpg
from asyncpg import Pool

from config import settings
from models.api_models import (
    DocumentInfo, ProcessingStatus, DocumentType, DocumentCategory,
    QualityMetrics, DocumentFilterRequest
)
from repositories.document_repository_extensions import DocumentRepositoryZipExtensions

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Repository for document metadata database operations"""
    
    def __init__(self):
        self.pool: Optional[Pool] = None
        self.zip_extensions: Optional[DocumentRepositoryZipExtensions] = None
        self._database_manager = None
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            logger.info("🔧 Initializing Document Repository...")
            
            # Use DatabaseManager for connection management
            from services.database_manager.database_manager_service import get_database_manager
            self._database_manager = await get_database_manager()
            
            logger.info("✅ Document Repository initialized with DatabaseManager")
            
            # Initialize ZIP extensions
            self.zip_extensions = DocumentRepositoryZipExtensions(self)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Document Repository: {e}")
            raise
    
    async def initialize_with_pool(self, shared_pool):
        """Initialize with shared database connection pool"""
        try:
            logger.info("🔧 Initializing Document Repository with shared pool...")
            
            # Use provided shared pool
            self.pool = shared_pool
            
            logger.info("✅ Document Repository initialized with shared pool")
            
            # Initialize ZIP extensions
            self.zip_extensions = DocumentRepositoryZipExtensions(self)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Document Repository with shared pool: {e}")
            raise
    
    async def create(self, doc_info: DocumentInfo) -> bool:
        """Create a new document record"""
        try:
            from services.database_manager.database_helpers import execute
            
            logger.info(f"🔧 DEBUG: Starting document creation for {doc_info.document_id}")
            
            # Convert quality metrics to JSON if present
            quality_json = None
            if doc_info.quality_metrics:
                quality_json = json.dumps(doc_info.quality_metrics.dict())
            
            # Set RLS context manually
            user_id = getattr(doc_info, 'user_id', None)
            collection_type = getattr(doc_info, 'collection_type', 'user')
            
            logger.info(f"🔧 DEBUG: Setting RLS context - user_id: {user_id}, collection_type: {collection_type}")
            
            if user_id:
                # Set user context for user documents
                await execute(
                    "SELECT set_config('app.current_user_id', $1, true)",
                    user_id
                )
                await execute(
                    "SELECT set_config('app.current_user_role', 'user', true)"
                )
                logger.info(f"🔧 DEBUG: Set user context for document creation")
            elif collection_type == "global":
                # Set admin context for global documents
                await execute(
                    "SELECT set_config('app.current_user_id', '', true)"
                )
                await execute(
                    "SELECT set_config('app.current_user_role', 'admin', true)"
                )
                logger.info(f"🔧 DEBUG: Set admin context for document creation")
                
            logger.info(f"🔧 DEBUG: Executing INSERT statement for document {doc_info.document_id}")
            
            # Documents default to NULL (inherit) so they follow folder settings dynamically
            # Only set explicit exemption if provided in doc_info
            folder_id = getattr(doc_info, 'folder_id', None)
            exempt_from_vectorization = getattr(doc_info, 'exempt_from_vectorization', None)  # Default to NULL (inherit)

            # If inheriting and folder is exempt, set explicit TRUE to prevent processing
            if exempt_from_vectorization is None and folder_id:
                try:
                    folder_exempt = await self.is_folder_exempt(folder_id)
                    if folder_exempt:
                        exempt_from_vectorization = True
                        logger.info(f"🚫 Document {doc_info.document_id} inherits exemption from folder {folder_id} → setting TRUE at creation")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to check folder exemption for {folder_id}: {e}")
            
            # If user explicitly set exemption, use that; otherwise inherit from folder
            if exempt_from_vectorization is not None:
                logger.info(f"📝 Document {doc_info.document_id} created with explicit exemption: {exempt_from_vectorization}")
            else:
                logger.info(f"📝 Document {doc_info.document_id} will inherit vectorization setting from folder")
            
            await execute("""
                INSERT INTO document_metadata (
                    document_id, filename, title, category, tags, description,
                    author, language, publication_date, doc_type, file_size, file_hash, processing_status,
                    upload_date, quality_score, page_count, chunk_count, entity_count, metadata_json, user_id,
                    collection_type, folder_id, exempt_from_vectorization
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23)
            """, 
                doc_info.document_id,
                doc_info.filename,
                doc_info.title,
                doc_info.category.value if doc_info.category else None,
                doc_info.tags,
                doc_info.description,
                doc_info.author,
                doc_info.language,
                doc_info.publication_date,
                doc_info.doc_type.value,
                doc_info.file_size,
                doc_info.file_hash,
                doc_info.status.value,
                doc_info.upload_date,
                doc_info.quality_metrics.overall_score if doc_info.quality_metrics else None,
                getattr(doc_info, 'page_count', 0),
                getattr(doc_info, 'chunk_count', 0),
                getattr(doc_info, 'entity_count', 0),
                quality_json,
                user_id,
                getattr(doc_info, 'collection_type', 'user'),  # Default to 'user' if not specified
                folder_id,  # Allow NULL for folder_id
                exempt_from_vectorization  # Include exemption status
            )
            
            logger.info(f"📝 Created document record: {doc_info.document_id} for user: {getattr(doc_info, 'user_id', None)} with collection_type: {getattr(doc_info, 'collection_type', 'user')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create document {doc_info.document_id}: {e}")
            logger.error(f"❌ Exception details: {type(e).__name__}: {str(e)}")
            return False

    async def create_with_folder(self, doc_info: DocumentInfo, folder_id: str = None) -> bool:
        """Create document and assign to folder in a single transaction - Roosevelt Architecture"""
        try:
            from services.database_manager.database_helpers import execute
            
            # ROOSEVELT FIX: RLS is disabled on document_metadata (like document_folders)
            # No need for RLS context - consistent architecture across all tables
            user_id = getattr(doc_info, 'user_id', None)
            logger.info(f"🔧 DEBUG: Creating document with RLS disabled - user will be: {user_id}")
            
            # Extract title from filename (remove extension)
            from pathlib import Path
            title = Path(doc_info.filename).stem
            
            # Documents default to NULL (inherit) so they follow folder settings dynamically
            # Only set explicit exemption if provided in doc_info
            exempt_from_vectorization = getattr(doc_info, 'exempt_from_vectorization', None)  # Default to NULL (inherit)

            # If inheriting and folder is exempt, set explicit TRUE to prevent processing
            if exempt_from_vectorization is None and folder_id:
                try:
                    folder_exempt = await self.is_folder_exempt(folder_id)
                    if folder_exempt:
                        exempt_from_vectorization = True
                        logger.info(f"🚫 Document {doc_info.document_id} inherits exemption from folder {folder_id} → setting TRUE at creation")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to check folder exemption for {folder_id}: {e}")
            
            # If user explicitly set exemption, use that; otherwise inherit from folder
            if exempt_from_vectorization is not None:
                logger.info(f"📝 Document {doc_info.document_id} created with explicit exemption: {exempt_from_vectorization}")
            else:
                logger.info(f"📝 Document {doc_info.document_id} will inherit vectorization setting from folder")
            
            await execute(
                """
                INSERT INTO document_metadata (
                    document_id, filename, title, doc_type, upload_date, file_size,
                    file_hash, processing_status, quality_score, page_count, chunk_count, entity_count,
                    user_id, collection_type, folder_id, exempt_from_vectorization
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (document_id) DO NOTHING
                """,
                doc_info.document_id,
                doc_info.filename,
                title,  # Add title
                doc_info.doc_type.value,
                doc_info.upload_date,
                doc_info.file_size,
                doc_info.file_hash,
                doc_info.status.value,
                doc_info.quality_metrics.overall_score if doc_info.quality_metrics else None,
                getattr(doc_info, 'page_count', 0),
                getattr(doc_info, 'chunk_count', 0),
                getattr(doc_info, 'entity_count', 0),
                user_id,
                getattr(doc_info, 'collection_type', 'user'),  # Default to 'user' if not specified
                folder_id,  # Include folder_id in the initial insert for atomic operation
                exempt_from_vectorization  # Include exemption status
            )
            
            logger.info(f"📝 Created document record: {doc_info.document_id} for user: {user_id} with collection_type: {getattr(doc_info, 'collection_type', 'user')}")
            if folder_id:
                logger.info(f"📁 Assigned document {doc_info.document_id} to folder {folder_id} in creation transaction")
            if exempt_from_vectorization:
                logger.info(f"🚫 Document {doc_info.document_id} created with vectorization exemption")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create document with folder {doc_info.document_id}: {e}")
            logger.error(f"❌ Exception details: {type(e).__name__}: {str(e)}")
            return False
    
    async def get_by_id(self, document_id: str) -> Optional[DocumentInfo]:
        """Get a document by ID"""
        try:
            from services.database_manager.database_helpers import execute, fetch_one
            
            # Set admin context to access all documents
            await execute(
                "SELECT set_config('app.current_user_id', '', true)"
            )
            await execute(
                "SELECT set_config('app.current_user_role', 'admin', true)"
            )
            
            logger.info(f"🔍 DEBUG: Looking for document {document_id} with admin context")
            
            row = await fetch_one("""
                SELECT * FROM document_metadata WHERE document_id = $1
            """, document_id)
            
            if row:
                logger.info(f"🔍 DEBUG: Found document {document_id} - user_id: {row.get('user_id')}, collection_type: {row.get('collection_type')}")
                return self._row_to_document_info(row)
            else:
                logger.warning(f"🔍 DEBUG: Document {document_id} not found with admin context")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get document {document_id}: {e}")
            return None
    
    async def update(self, document_id: str, **updates) -> bool:
        """Update document fields"""
        try:
            if not updates:
                return True
            
            # Build dynamic update query
            set_clauses = []
            values = []
            param_count = 1
            
            for field, value in updates.items():
                if field == 'status' and hasattr(value, 'value'):
                    # Map status to processing_status column
                    set_clauses.append(f"processing_status = ${param_count}")
                    values.append(value.value)
                    param_count += 1
                    continue
                elif field == 'status':
                    # Handle string status values
                    set_clauses.append(f"processing_status = ${param_count}")
                    values.append(value)
                    param_count += 1
                    continue
                elif field == 'category' and hasattr(value, 'value'):
                    value = value.value
                elif field == 'doc_type' and hasattr(value, 'value'):
                    value = value.value
                elif field == 'quality_metrics' and value:
                    # Convert quality metrics to JSON and extract score
                    if hasattr(value, 'dict'):
                        quality_json = json.dumps(value.dict())
                        set_clauses.append(f"metadata_json = ${param_count}")
                        values.append(quality_json)
                        param_count += 1
                        
                        set_clauses.append(f"quality_score = ${param_count}")
                        values.append(value.overall_score)
                        param_count += 1
                        continue
                
                set_clauses.append(f"{field} = ${param_count}")
                values.append(value)
                param_count += 1
            
            if not set_clauses:
                return True
            
            # Add document_id as last parameter
            values.append(document_id)
            
            query = f"""
                UPDATE document_metadata 
                SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ${param_count}
            """
            
            from services.database_manager.database_helpers import execute, fetch_one
            
            # ROOSEVELT FIX: Check document ownership first to set correct RLS context
            try:
                # First, check document ownership with admin context
                await execute("SELECT set_config('app.current_user_id', '', true)")
                await execute("SELECT set_config('app.current_user_role', 'admin', true)")
                
                doc_check = await fetch_one("SELECT user_id FROM document_metadata WHERE document_id = $1", document_id)
                
                if doc_check and doc_check['user_id']:
                    # User document - set user context for the update
                    await execute("SELECT set_config('app.current_user_id', $1, true)", doc_check['user_id'])
                    await execute("SELECT set_config('app.current_user_role', 'user', true)")
                    logger.debug(f"🔍 Using user context for document update: {doc_check['user_id']}")
                else:
                    # Global document - keep admin context
                    logger.debug(f"🔍 Using admin context for global document update")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not determine document ownership for {document_id}, using admin context: {e}")
                # Fallback to admin context
                await execute("SELECT set_config('app.current_user_id', '', true)")
                await execute("SELECT set_config('app.current_user_role', 'admin', true)")
            
            result = await execute(query, *values)
            
            # Check if any rows were updated
            rows_updated = int(result.split()[-1])
            if rows_updated > 0:
                logger.debug(f"📝 Updated document: {document_id}")
                return True
            else:
                logger.warning(f"⚠️ No document found to update: {document_id}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to update document {document_id}: {e}")
            return False
    
    async def delete(self, document_id: str) -> bool:
        """Delete a document record"""
        try:
            # Use shared DB helper to ensure proper connection context (avoids None pool)
            from services.database_manager.database_helpers import execute
            result = await execute(
                """
                DELETE FROM document_metadata WHERE document_id = $1
                """,
                document_id
            )
            rows_deleted = int(result.split()[-1])
            if rows_deleted > 0:
                logger.debug(f"🗑️ Deleted document record: {document_id}")
                return True
            else:
                logger.warning(f"⚠️ No document found to delete: {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete document {document_id}: {e}")
            return False
    
    async def list_documents(self, skip: int = 0, limit: int = 100) -> List[DocumentInfo]:
        """List documents with pagination"""
        try:
            from services.database_manager.database_helpers import fetch_all
            
            rows = await fetch_all("""
                SELECT * FROM document_metadata 
                ORDER BY upload_date DESC 
                LIMIT $1 OFFSET $2
            """, limit, skip)
            
            return [self._row_to_document_info(row) for row in rows]
            
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {e}")
            return []

    async def list_user_documents(self, user_id: str, skip: int = 0, limit: int = 100) -> List[DocumentInfo]:
        """List documents for a specific user"""
        try:
            logger.debug(f"📄 Listing user documents for user_id: {user_id}, skip: {skip}, limit: {limit}")
            from services.database_manager.database_helpers import fetch_all
            
            rows = await fetch_all("""
                SELECT * FROM document_metadata 
                WHERE user_id = $1
                ORDER BY upload_date DESC 
                LIMIT $2 OFFSET $3
            """, user_id, limit, skip)
            
            logger.debug(f"📄 Query returned {len(rows)} rows for user {user_id}")
            documents = [self._row_to_document_info(row) for row in rows]
            logger.debug(f"📄 Converted to {len(documents)} DocumentInfo objects")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Failed to list user documents: {e}")
            return []

    async def list_global_documents(self, skip: int = 0, limit: int = 100) -> List[DocumentInfo]:
        """List global/admin documents (user_id is NULL or collection_type is 'global')"""
        try:
            from services.database_manager.database_helpers import fetch_all
            
            rows = await fetch_all("""
                SELECT * FROM document_metadata 
                WHERE user_id IS NULL OR collection_type = 'global'
                ORDER BY upload_date DESC 
                LIMIT $1 OFFSET $2
            """, limit, skip)
            
            return [self._row_to_document_info(row) for row in rows]
            
        except Exception as e:
            logger.error(f"❌ Failed to list global documents: {e}")
            return []
    
    async def filter_documents(self, filter_request: DocumentFilterRequest) -> Tuple[List[DocumentInfo], int]:
        """Filter documents with advanced criteria"""
        try:
            # Build dynamic WHERE clause
            where_clauses = []
            values = []
            param_count = 1

            # Text search
            if filter_request.search_query:
                search_term = f"%{filter_request.search_query.lower()}%"
                where_clauses.append(f"""
                    (LOWER(filename) LIKE ${param_count} OR
                     LOWER(title) LIKE ${param_count} OR
                     LOWER(description) LIKE ${param_count} OR
                     LOWER(author) LIKE ${param_count})
                """)
                values.append(search_term)
                param_count += 1

            # Category filter
            if filter_request.category:
                where_clauses.append(f"category = ${param_count}")
                values.append(filter_request.category.value)
                param_count += 1

            # Tags filter (document must have all specified tags)
            if filter_request.tags:
                where_clauses.append(f"tags @> ${param_count}")
                values.append(filter_request.tags)
                param_count += 1

            # Document type filter
            if filter_request.doc_type:
                where_clauses.append(f"doc_type = ${param_count}")
                values.append(filter_request.doc_type.value)
                param_count += 1

            # Status filter
            if filter_request.status:
                where_clauses.append(f"processing_status = ${param_count}")
                values.append(filter_request.status.value)
                param_count += 1

            # Author filter
            if filter_request.author:
                where_clauses.append(f"LOWER(author) LIKE ${param_count}")
                values.append(f"%{filter_request.author.lower()}%")
                param_count += 1

            # Language filter
            if filter_request.language:
                where_clauses.append(f"language = ${param_count}")
                values.append(filter_request.language)
                param_count += 1

            # Date range filters
            if filter_request.date_from:
                where_clauses.append(f"upload_date >= ${param_count}")
                values.append(filter_request.date_from)
                param_count += 1

            if filter_request.date_to:
                where_clauses.append(f"upload_date <= ${param_count}")
                values.append(filter_request.date_to)
                param_count += 1

            # Publication date range filters
            if filter_request.publication_date_from:
                where_clauses.append(f"publication_date >= ${param_count}")
                values.append(filter_request.publication_date_from)
                param_count += 1

            if filter_request.publication_date_to:
                where_clauses.append(f"publication_date <= ${param_count}")
                values.append(filter_request.publication_date_to)
                param_count += 1

            # Quality filter
            if filter_request.min_quality:
                where_clauses.append(f"quality_score >= ${param_count}")
                values.append(filter_request.min_quality)
                param_count += 1

            # Build WHERE clause
            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            # Build ORDER BY clause
            order_by = "upload_date"
            if filter_request.sort_by == "filename":
                order_by = "filename"
            elif filter_request.sort_by == "title":
                order_by = "title"
            elif filter_request.sort_by == "size":
                order_by = "file_size"
            elif filter_request.sort_by == "quality":
                order_by = "quality_score"
            elif filter_request.sort_by == "publication_date":
                order_by = "publication_date"

            order_direction = "DESC" if filter_request.sort_order.lower() == "desc" else "ASC"

            # Use proper database helpers instead of direct pool access
            from services.database_manager.database_helpers import fetch_all, fetch_value

            # Get total count
            count_query = f"SELECT COUNT(*) FROM document_metadata {where_sql}"
            total_count = await fetch_value(count_query, *values)

            # Get filtered documents
            query = f"""
                SELECT * FROM document_metadata {where_sql}
                ORDER BY {order_by} {order_direction}
                LIMIT ${param_count} OFFSET ${param_count + 1}
            """
            values.extend([filter_request.limit, filter_request.skip])

            rows = await fetch_all(query, *values)
            documents = [self._row_to_document_info(row) for row in rows]

            return documents, total_count

        except Exception as e:
            logger.error(f"❌ Failed to filter documents: {e}")
            return [], 0
    
    async def find_by_hash(self, file_hash: str) -> Optional[DocumentInfo]:
        """Find existing document with the same file hash"""
        try:
            from services.database_manager.database_helpers import fetch_one
            
            # Set admin context to access all documents
            rls_context = {'user_id': '', 'user_role': 'admin'}
            
            row = await fetch_one("""
                SELECT * FROM document_metadata WHERE file_hash = $1 LIMIT 1
            """, file_hash, rls_context=rls_context)
            
            if row:
                return self._row_to_document_info(row)
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to find duplicate by hash: {e}")
            return None
    
    async def find_by_filename_and_context(
        self, 
        filename: str, 
        user_id: Optional[str], 
        collection_type: str,
        folder_id: Optional[str] = None
    ) -> Optional[DocumentInfo]:
        """Find document by filename and user/folder context
        
        **ROOSEVELT DUPLICATE DETECTIVE - DATABASE LAYER!** 🔍
        """
        try:
            from services.database_manager.database_helpers import fetch_one
            
            logger.info(f"🔍 DB QUERY: Searching for filename='{filename}', user_id={user_id}, collection={collection_type}, folder_id={folder_id}")
            
            # Build query based on context
            if user_id:
                # User document
                if folder_id:
                    query = """
                        SELECT * FROM document_metadata 
                        WHERE filename = $1 AND user_id = $2 AND collection_type = $3 AND folder_id = $4
                        LIMIT 1
                    """
                    logger.info(f"🔍 DB QUERY: Using user+folder query with 4 parameters")
                    rls_context = {'user_id': user_id, 'user_role': 'user'}
                    row = await fetch_one(query, filename, user_id, collection_type, folder_id, rls_context=rls_context)
                else:
                    query = """
                        SELECT * FROM document_metadata 
                        WHERE filename = $1 AND user_id = $2 AND collection_type = $3 AND folder_id IS NULL
                        LIMIT 1
                    """
                    logger.info(f"🔍 DB QUERY: Using user (no folder) query with 3 parameters")
                    rls_context = {'user_id': user_id, 'user_role': 'user'}
                    row = await fetch_one(query, filename, user_id, collection_type, rls_context=rls_context)
            else:
                # Global document
                if folder_id:
                    query = """
                        SELECT * FROM document_metadata 
                        WHERE filename = $1 AND collection_type = $2 AND folder_id = $3 AND user_id IS NULL
                        LIMIT 1
                    """
                    logger.info(f"🔍 DB QUERY: Using global+folder query with 3 parameters")
                    rls_context = {'user_id': '', 'user_role': 'admin'}
                    row = await fetch_one(query, filename, collection_type, folder_id, rls_context=rls_context)
                else:
                    query = """
                        SELECT * FROM document_metadata 
                        WHERE filename = $1 AND collection_type = $2 AND user_id IS NULL AND folder_id IS NULL
                        LIMIT 1
                    """
                    logger.info(f"🔍 DB QUERY: Using global (no folder) query with 2 parameters")
                    rls_context = {'user_id': '', 'user_role': 'admin'}
                    row = await fetch_one(query, filename, collection_type, rls_context=rls_context)
            
            if row:
                doc_info = self._row_to_document_info(row)
                logger.info(f"✅ DB QUERY: FOUND document_id={doc_info.document_id}")
                return doc_info
            else:
                logger.info(f"❌ DB QUERY: NO MATCH FOUND in database")
                return None
            
        except Exception as e:
            logger.error(f"❌ Failed to find document by filename and context: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def update_filename(self, document_id: str, new_filename: str) -> bool:
        """Update document filename"""
        try:
            from services.database_manager.database_helpers import execute
            
            # Set admin context to allow update
            rls_context = {'user_id': '', 'user_role': 'admin'}
            
            result = await execute("""
                UPDATE document_metadata 
                SET filename = $1, updated_at = NOW()
                WHERE document_id = $2
            """, new_filename, document_id, rls_context=rls_context)
            
            logger.info(f"✅ Updated filename for document {document_id}: {new_filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update filename: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get document statistics"""
        try:
            from services.database_manager.database_helpers import fetch_one
            
            stats = await fetch_one("""
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(*) FILTER (WHERE processing_status = 'completed') as completed_documents,
                    COUNT(*) FILTER (WHERE processing_status = 'failed') as failed_documents,
                    COUNT(*) FILTER (WHERE processing_status IN ('processing', 'embedding')) as processing_documents,
                    SUM(file_size) as total_size,
                    AVG(quality_score) as avg_quality
                FROM document_metadata
            """)
            
            return {
                "total_documents": stats["total_documents"],
                "completed_documents": stats["completed_documents"],
                "failed_documents": stats["failed_documents"],
                "processing_documents": stats["processing_documents"],
                "total_size": stats["total_size"] or 0,
                "avg_quality": float(stats["avg_quality"]) if stats["avg_quality"] else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get documents stats: {e}")
            return {}
    
    async def get_categories_overview(self):
        """Get categories and tags overview"""
        try:
            from models.api_models import DocumentCategoriesResponse, CategorySummary, TagSummary
            from services.database_manager.database_helpers import fetch_all, fetch_value

            # Get category statistics
            category_stats = await fetch_all("""
                SELECT
                    category,
                    COUNT(*) as count,
                    SUM(file_size) as total_size,
                    AVG(quality_score) as avg_quality
                FROM document_metadata
                WHERE category IS NOT NULL
                GROUP BY category
                ORDER BY count DESC
            """)

            # Get uncategorized count
            uncategorized = await fetch_value("""
                SELECT COUNT(*) FROM document_metadata WHERE category IS NULL
            """)

            # Get total documents count
            total_docs = await fetch_value("""
                SELECT COUNT(*) FROM document_metadata
            """)

            # Get tag statistics (this is more complex with PostgreSQL arrays)
            tag_stats = await fetch_all("""
                SELECT
                    unnest(tags) as tag,
                    COUNT(*) as count
                FROM document_metadata
                WHERE tags IS NOT NULL AND array_length(tags, 1) > 0
                GROUP BY tag
                ORDER BY count DESC
            """)

            # Build category summaries
            categories = []
            for row in category_stats:
                categories.append(CategorySummary(
                    category=DocumentCategory(row["category"]),
                    count=row["count"],
                    total_size=row["total_size"] or 0,
                    avg_quality=float(row["avg_quality"]) if row["avg_quality"] else None
                ))

            # Build tag summaries
            tags = []
            for row in tag_stats:
                # Get categories for this tag
                tag_categories = await fetch_all("""
                    SELECT DISTINCT category
                    FROM document_metadata
                    WHERE $1 = ANY(tags) AND category IS NOT NULL
                """, row["tag"])

                tag_category_list = [cat["category"] for cat in tag_categories]

                tags.append(TagSummary(
                    tag=row["tag"],
                    count=row["count"],
                    categories=tag_category_list
                ))

            return DocumentCategoriesResponse(
                categories=categories,
                tags=tags,
                total_documents=total_docs,
                uncategorized_count=uncategorized
            )

        except Exception as e:
            logger.error(f"❌ Failed to get categories overview: {e}")
            from models.api_models import DocumentCategoriesResponse
            return DocumentCategoriesResponse(
                categories=[],
                tags=[],
                total_documents=0,
                uncategorized_count=0
            )
    
    async def get_duplicate_documents(self) -> Dict[str, List[str]]:
        """Get duplicate documents grouped by hash"""
        try:
            from services.database_manager.database_helpers import fetch_all

            rows = await fetch_all("""
                SELECT file_hash, array_agg(document_id) as document_ids
                FROM document_metadata
                WHERE file_hash IS NOT NULL
                GROUP BY file_hash
                HAVING COUNT(*) > 1
            """)

            return {row["file_hash"]: row["document_ids"] for row in rows}

        except Exception as e:
            logger.error(f"❌ Failed to get duplicate documents: {e}")
            return {}
    
    async def get_duplicates(self) -> Dict[str, List[DocumentInfo]]:
        """Get duplicate documents grouped by hash, returning full DocumentInfo objects"""
        try:
            from services.database_manager.database_helpers import fetch_all

            # Get all documents that have duplicates
            rows = await fetch_all("""
                SELECT dm.* FROM document_metadata dm
                WHERE dm.file_hash IN (
                    SELECT file_hash FROM document_metadata
                    WHERE file_hash IS NOT NULL
                    GROUP BY file_hash
                    HAVING COUNT(*) > 1
                )
                ORDER BY dm.file_hash, dm.upload_date
            """)

            # Group by hash
            duplicates = {}
            for row in rows:
                doc_info = self._row_to_document_info(row)
                file_hash = doc_info.file_hash
                if file_hash not in duplicates:
                    duplicates[file_hash] = []
                duplicates[file_hash].append(doc_info)

            return duplicates

        except Exception as e:
            logger.error(f"❌ Failed to get duplicate documents: {e}")
            return {}
    
    async def update_status(self, document_id: str, status: ProcessingStatus) -> bool:
        """Update document status"""
        return await self.update(document_id, status=status)
    
    async def update_file_size(self, document_id: str, file_size: int) -> bool:
        """Update document file size"""
        return await self.update(document_id, file_size=file_size)
    
    async def update_quality_metrics(self, document_id: str, quality_metrics: QualityMetrics) -> bool:
        """Update document quality metrics"""
        return await self.update(document_id, quality_metrics=quality_metrics)
    
    async def update_metadata(self, document_id: str, update_request) -> bool:
        """Update document metadata from update request"""
        updates = {}
        
        if update_request.title is not None:
            updates['title'] = update_request.title
        
        if update_request.category is not None:
            updates['category'] = update_request.category
        
        if update_request.tags is not None:
            updates['tags'] = update_request.tags
        
        if update_request.description is not None:
            updates['description'] = update_request.description
        
        if update_request.author is not None:
            updates['author'] = update_request.author
        
        if update_request.publication_date is not None:
            updates['publication_date'] = update_request.publication_date
        
        if updates:
            return await self.update(document_id, **updates)
        
        return True
    
    async def bulk_categorize(self, bulk_request) -> Tuple[int, List[str]]:
        """Bulk categorize multiple documents"""
        try:
            success_count = 0
            failed_documents = []

            for doc_id in bulk_request.document_ids:
                try:
                    # Check if document exists and update
                    updates = {'category': bulk_request.category}

                    if bulk_request.tags:
                        # For bulk operations, we'll replace tags rather than merge
                        # to keep the behavior simple and predictable
                        updates['tags'] = bulk_request.tags

                    success = await self.update(doc_id, **updates)
                    if success:
                        success_count += 1
                    else:
                        failed_documents.append(doc_id)

                except Exception as e:
                    logger.error(f"❌ Failed to update document {doc_id}: {e}")
                    failed_documents.append(doc_id)

            return success_count, failed_documents

        except Exception as e:
            logger.error(f"❌ Bulk categorization failed: {e}")
            return 0, bulk_request.document_ids
    
    async def get_all_document_ids(self) -> List[str]:
        """Get all document IDs"""
        try:
            from services.database_manager.database_helpers import fetch_all

            rows = await fetch_all("""
                SELECT document_id FROM document_metadata
            """)

            return [row["document_id"] for row in rows]

        except Exception as e:
            logger.error(f"❌ Failed to get all document IDs: {e}")
            return []
    
    async def get_documents_by_status(self, status: ProcessingStatus) -> List[DocumentInfo]:
        """Get all documents with a specific processing status"""
        try:
            from services.database_manager.database_helpers import fetch_all
            
            rows = await fetch_all("""
                SELECT * FROM document_metadata 
                WHERE processing_status = $1
                ORDER BY upload_date ASC
            """, status.value)
            
            return [self._row_to_document_info(row) for row in rows]
            
        except Exception as e:
            logger.error(f"❌ Failed to get documents by status {status.value}: {e}")
            return []
    
    def _row_to_document_info(self, row) -> DocumentInfo:
        """Convert database row to DocumentInfo object"""
        # Parse quality metrics from JSON if present
        quality_metrics = None
        if row["metadata_json"]:
            try:
                quality_data = json.loads(row["metadata_json"])
                quality_metrics = QualityMetrics(**quality_data)
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse quality metrics: {e}")
        
        return DocumentInfo(
            document_id=row["document_id"],
            filename=row["filename"],
            title=row["title"],
            category=DocumentCategory(row["category"]) if row["category"] else None,
            tags=row["tags"] or [],
            description=row["description"],
            author=row["author"],
            language=row["language"],
            publication_date=row["publication_date"],
            doc_type=DocumentType(row["doc_type"]),
            file_size=row["file_size"],
            file_hash=row["file_hash"],
            status=ProcessingStatus(row["processing_status"]),
            upload_date=row["upload_date"],
            page_count=row.get("page_count", 0),
            chunk_count=row.get("chunk_count", 0),
            entity_count=row.get("entity_count", 0),
            quality_metrics=quality_metrics,
            user_id=row.get("user_id", None),
            folder_id=row.get("folder_id", None),
            exempt_from_vectorization=row.get("exempt_from_vectorization", None)
        )
    
    async def execute_query(self, query: str, *params, rls_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results"""
        try:
            from services.database_manager.database_helpers import fetch_all
            
            rows = await fetch_all(query, *params, rls_context=rls_context)
            return rows
                
        except Exception as e:
            logger.error(f"❌ Failed to execute query: {e}")
            raise
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document as dictionary (for compatibility with PDF segmentation service)"""
        try:
            from services.database_manager.database_helpers import fetch_one
            
            row = await fetch_one("""
                SELECT * FROM document_metadata WHERE document_id = $1
            """, document_id)
            
            return row
                
        except Exception as e:
            logger.error(f"❌ Failed to get document {document_id}: {e}")
            return None
    
    async def store_document_metadata(self, doc_info: DocumentInfo, user_id: str = None) -> bool:
        """Store document metadata directly (for text documents and web content)"""
        try:
            from services.database_manager.database_helpers import execute
            
            # Convert quality metrics to JSON if present
            quality_json = None
            if doc_info.quality_metrics:
                quality_json = json.dumps(doc_info.quality_metrics.dict())
            
            # Convert additional metadata to JSON
            additional_metadata = None
            if hasattr(doc_info, 'metadata') and doc_info.metadata:
                additional_metadata = json.dumps(doc_info.metadata)
            
            # Set RLS context manually if user_id is provided
            if user_id:
                await execute(
                    "SELECT set_config('app.current_user_id', $1, true)",
                    user_id
                )
                await execute(
                    "SELECT set_config('app.current_user_role', 'user', true)"
                )
            
            await execute("""
                    INSERT INTO document_metadata (
                        document_id, filename, title, category, tags, description,
                        author, language, publication_date, doc_type, file_size, file_hash, processing_status,
                        upload_date, quality_score, page_count, chunk_count, entity_count, metadata_json, user_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                    ON CONFLICT (document_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        title = EXCLUDED.title,
                        category = EXCLUDED.category,
                        tags = EXCLUDED.tags,
                        description = EXCLUDED.description,
                        author = EXCLUDED.author,
                        language = EXCLUDED.language,
                        publication_date = EXCLUDED.publication_date,
                        doc_type = EXCLUDED.doc_type,
                        file_size = EXCLUDED.file_size,
                        file_hash = EXCLUDED.file_hash,
                        processing_status = EXCLUDED.processing_status,
                        upload_date = EXCLUDED.upload_date,
                        quality_score = EXCLUDED.quality_score,
                        page_count = EXCLUDED.page_count,
                        chunk_count = EXCLUDED.chunk_count,
                        entity_count = EXCLUDED.entity_count,
                        metadata_json = EXCLUDED.metadata_json,
                        user_id = EXCLUDED.user_id,
                        updated_at = CURRENT_TIMESTAMP
                """, 
                    doc_info.document_id,
                    doc_info.filename,
                    doc_info.title,
                    doc_info.category.value if doc_info.category else None,
                    doc_info.tags,
                    doc_info.description,
                    doc_info.author,
                    doc_info.language,
                    doc_info.publication_date,
                    doc_info.doc_type.value,
                    doc_info.file_size,
                    doc_info.file_hash or "",  # Provide empty string if None
                    doc_info.status.value if hasattr(doc_info, 'status') else ProcessingStatus.COMPLETED.value,
                    doc_info.upload_date,
                    doc_info.quality_metrics.overall_score if doc_info.quality_metrics else None,
                    getattr(doc_info, 'page_count', 0),
                    getattr(doc_info, 'chunk_count', 0),
                    getattr(doc_info, 'entity_count', 0),
                    additional_metadata,
                    user_id or getattr(doc_info, 'user_id', None)
            )
            
            logger.debug(f"📝 Stored document metadata: {doc_info.document_id} for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store document metadata {doc_info.document_id}: {e}")
            return False
    
    async def update_chunk_count(self, document_id: str, chunk_count: int) -> bool:
        """Update the chunk count for a document"""
        try:
            from services.database_manager.database_helpers import execute
            
            await execute("""
                UPDATE document_metadata 
                SET chunk_count = $1, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = $2
            """, chunk_count, document_id)
            
            logger.debug(f"📊 Updated chunk count for {document_id}: {chunk_count}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update chunk count for {document_id}: {e}")
            return False
    
    async def update_entity_count(self, document_id: str, entity_count: int) -> bool:
        """Update the entity count for a document"""
        try:
            from services.database_manager.database_helpers import execute

            await execute("""
                UPDATE document_metadata
                SET entity_count = $1, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = $2
            """, entity_count, document_id)

            logger.debug(f"🔗 Updated entity count for {document_id}: {entity_count}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update entity count for {document_id}: {e}")
            return False
    
    # ZIP Hierarchy Methods (delegated to extensions)
    async def mark_as_zip_container(self, document_id: str) -> bool:
        """Mark a document as a ZIP container"""
        return await self.zip_extensions.mark_as_zip_container(document_id)
    
    async def set_parent_relationship(self, child_id: str, parent_id: str, original_zip_path: str = None) -> bool:
        """Set parent-child relationship for ZIP extracted files"""
        return await self.zip_extensions.set_parent_relationship(child_id, parent_id, original_zip_path)
    
    async def get_zip_children(self, parent_document_id: str) -> List[Dict[str, Any]]:
        """Get all files extracted from a ZIP document"""
        return await self.zip_extensions.get_zip_children(parent_document_id)
    
    async def get_zip_containers(self) -> List[Dict[str, Any]]:
        """Get all ZIP container documents"""
        return await self.zip_extensions.get_zip_containers()
    
    async def toggle_metadata_inheritance(self, document_id: str, inherit: bool) -> bool:
        """Toggle metadata inheritance for a ZIP extracted file"""
        return await self.zip_extensions.toggle_metadata_inheritance(document_id, inherit)
    
    async def get_parent_document(self, child_document_id: str) -> Optional[Dict[str, Any]]:
        """Get parent document for a ZIP extracted file"""
        return await self.zip_extensions.get_parent_document(child_document_id)
    
    async def get_documents_with_hierarchy(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get documents with their hierarchy information"""
        return await self.zip_extensions.get_documents_with_hierarchy(limit, offset)
    
    async def update_child_metadata_inheritance(self, parent_document_id: str, metadata_updates: Dict[str, Any]) -> int:
        """Update metadata for all child documents that inherit from parent"""
        return await self.zip_extensions.update_child_metadata_inheritance(parent_document_id, metadata_updates)
    
    async def delete_zip_hierarchy(self, parent_document_id: str) -> Dict[str, int]:
        """Delete a ZIP file and all its extracted children"""
        return await self.zip_extensions.delete_zip_hierarchy(parent_document_id)
    
    async def get_filter_options(self) -> Dict[str, Any]:
        """Get available filter options for search"""
        try:
            from services.database_manager.database_helpers import fetch_all, fetch_one

            # Get unique document types
            doc_types = await fetch_all("""
                SELECT DISTINCT doc_type FROM document_metadata
                WHERE doc_type IS NOT NULL
                ORDER BY doc_type
            """)

            # Get unique categories
            categories = await fetch_all("""
                SELECT DISTINCT category FROM document_metadata
                WHERE category IS NOT NULL
                ORDER BY category
            """)

            # Get unique tags (flattened from JSON arrays)
            tags = await fetch_all("""
                SELECT DISTINCT jsonb_array_elements_text(tags) as tag
                FROM document_metadata
                WHERE tags IS NOT NULL AND jsonb_array_length(tags) > 0
                ORDER BY tag
            """)

            # Get date range
            date_range = await fetch_one("""
                SELECT
                    MIN(upload_date) as earliest_date,
                    MAX(upload_date) as latest_date
                FROM document_metadata
            """)

            return {
                "doc_types": [row["doc_type"] for row in doc_types],
                "categories": [row["category"] for row in categories],
                "tags": [row["tag"] for row in tags],
                "earliest_date": date_range["earliest_date"] if date_range else None,
                "latest_date": date_range["latest_date"] if date_range else None
            }

        except Exception as e:
            logger.error(f"❌ Failed to get filter options: {e}")
            return {
                "doc_types": [],
                "categories": [],
                "tags": [],
                "earliest_date": None,
                "latest_date": None
            }

    # === GLOBAL SUBMISSION WORKFLOW METHODS ===
    
    async def update_submission_status(
        self,
        document_id: str,
        submission_status: str,
        submitted_by: str,
        submitted_at: datetime,
        submission_reason: str = None
    ):
        """Update document submission status and metadata"""
        try:
            from services.database_manager.database_helpers import execute

            await execute("""
                UPDATE documents
                SET
                    submission_status = $1,
                    submitted_by = $2,
                    submitted_at = $3,
                    submission_reason = $4,
                    updated_at = $5
                WHERE document_id = $6
            """, submission_status, submitted_by, submitted_at, submission_reason, datetime.utcnow(), document_id)

            logger.info(f"📝 Updated submission status for document {document_id}: {submission_status}")

        except Exception as e:
            logger.error(f"❌ Failed to update submission status for {document_id}: {e}")
            raise

    async def get_pending_submissions(self, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Get documents pending global approval"""
        try:
            from services.database_manager.database_helpers import fetch_all

            rows = await fetch_all("""
                SELECT
                    document_id, filename, title, doc_type, category, tags, description,
                    author, language, publication_date, upload_date, file_size, file_hash,
                    status, chunk_count, entity_count, quality_ocr_confidence,
                    quality_language_confidence, quality_vocabulary_score,
                    quality_pattern_score, quality_overall_score,
                    submission_status, submitted_by, submitted_at, submission_reason,
                    reviewed_by, reviewed_at, review_comment, collection_type
                FROM documents
                WHERE submission_status = 'pending_approval'
                ORDER BY submitted_at ASC
                OFFSET $1 LIMIT $2
            """, skip, limit)

            return rows

        except Exception as e:
            logger.error(f"❌ Failed to get pending submissions: {e}")
            raise

    async def count_pending_submissions(self) -> int:
        """Count documents pending global approval"""
        try:
            from services.database_manager.database_helpers import fetch_value

            count = await fetch_value("""
                SELECT COUNT(*) FROM documents
                WHERE submission_status = 'pending_approval'
            """)
            return count or 0

        except Exception as e:
            logger.error(f"❌ Failed to count pending submissions: {e}")
            raise

    async def update_review_status(
        self,
        document_id: str,
        submission_status: str,
        reviewed_by: str,
        reviewed_at: datetime,
        review_comment: str = None,
        collection_type: str = "user"
    ):
        """Update document review status after admin approval/rejection"""
        try:
            from services.database_manager.database_helpers import execute

            await execute("""
                UPDATE documents
                SET
                    submission_status = $1,
                    reviewed_by = $2,
                    reviewed_at = $3,
                    review_comment = $4,
                    collection_type = $5,
                    updated_at = $6
                WHERE document_id = $7
            """, submission_status, reviewed_by, reviewed_at, review_comment, collection_type, datetime.utcnow(), document_id)

            logger.info(f"📝 Updated review status for document {document_id}: {submission_status}")

        except Exception as e:
            logger.error(f"❌ Failed to update review status for {document_id}: {e}")
            raise

    # ===== FOLDER OPERATIONS =====
    
    async def create_or_get_folder(self, folder_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new folder or return existing one if already present
        
        **ROOSEVELT'S UPSERT CAVALRY!** 🏇
        Uses PostgreSQL ON CONFLICT to handle race conditions at database level
        
        Returns:
            Dict with folder data (either newly created or existing)
            None if operation failed
        """
        try:
            from services.database_manager.database_helpers import fetch_one
            
            user_id = folder_data.get("user_id")
            team_id = folder_data.get("team_id")
            collection_type = folder_data.get("collection_type", "user")
            folder_name = folder_data.get("name")
            parent_id = folder_data.get("parent_folder_id")
            
            logger.info(f"📁 Repository: Create or get folder '{folder_name}' (parent: {parent_id}, user: {user_id}, team: {team_id}, collection: {collection_type})")
            
            # **ROOSEVELT'S NULL-SAFE UPSERT!**
            # PostgreSQL's ON CONFLICT with partial indexes requires different syntax
            # for root folders (NULL parent) vs. non-root folders
            
            if parent_id is None:
                # Root folder - need different UPSERT for user vs global vs team folders
                if collection_type == "team" and team_id is not None:
                    # TEAM root folder - includes team_id in conflict constraint
                    row = await fetch_one("""
                        INSERT INTO document_folders (
                            folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (team_id, name, collection_type) 
                        WHERE parent_folder_id IS NULL AND team_id IS NOT NULL
                        DO UPDATE SET updated_at = EXCLUDED.updated_at
                        RETURNING folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                    """, 
                        folder_data["folder_id"],
                        folder_name,
                        parent_id,
                        user_id,
                        team_id,
                        collection_type,
                        folder_data["created_at"],
                        folder_data["updated_at"]
                    )
                elif user_id is not None:
                    # USER root folder - includes user_id in conflict constraint
                    row = await fetch_one("""
                        INSERT INTO document_folders (
                            folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (user_id, name, collection_type) 
                        WHERE parent_folder_id IS NULL AND user_id IS NOT NULL
                        DO UPDATE SET updated_at = EXCLUDED.updated_at
                        RETURNING folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                    """, 
                        folder_data["folder_id"],
                        folder_name,
                        parent_id,
                        user_id,
                        team_id,
                        collection_type,
                        folder_data["created_at"],
                        folder_data["updated_at"]
                    )
                else:
                    # GLOBAL root folder - user_id is NULL, constraint doesn't include user_id
                    row = await fetch_one("""
                        INSERT INTO document_folders (
                            folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (name, collection_type) 
                        WHERE parent_folder_id IS NULL AND user_id IS NULL
                        DO UPDATE SET updated_at = EXCLUDED.updated_at
                        RETURNING folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                    """, 
                        folder_data["folder_id"],
                        folder_name,
                        parent_id,
                        user_id,
                        team_id,
                        collection_type,
                        folder_data["created_at"],
                        folder_data["updated_at"]
                    )
            else:
                # Non-root folder - different UPSERT for user vs global vs team folders
                if collection_type == "team" and team_id is not None:
                    # TEAM non-root folder - includes team_id in conflict constraint
                    row = await fetch_one("""
                        INSERT INTO document_folders (
                            folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (team_id, name, parent_folder_id, collection_type)
                        WHERE parent_folder_id IS NOT NULL AND team_id IS NOT NULL
                        DO UPDATE SET updated_at = EXCLUDED.updated_at
                        RETURNING folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                    """, 
                        folder_data["folder_id"],
                        folder_name,
                        parent_id,
                        user_id,
                        team_id,
                        collection_type,
                        folder_data["created_at"],
                        folder_data["updated_at"]
                    )
                elif user_id is not None:
                    # USER non-root folder - includes user_id in conflict constraint
                    row = await fetch_one("""
                        INSERT INTO document_folders (
                            folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (user_id, name, parent_folder_id, collection_type)
                        WHERE parent_folder_id IS NOT NULL AND user_id IS NOT NULL
                        DO UPDATE SET updated_at = EXCLUDED.updated_at
                        RETURNING folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                    """, 
                        folder_data["folder_id"],
                        folder_name,
                        parent_id,
                        user_id,
                        team_id,
                        collection_type,
                        folder_data["created_at"],
                        folder_data["updated_at"]
                    )
                else:
                    # GLOBAL non-root folder - user_id is NULL, constraint doesn't include user_id
                    row = await fetch_one("""
                        INSERT INTO document_folders (
                            folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (name, parent_folder_id, collection_type)
                        WHERE parent_folder_id IS NOT NULL AND user_id IS NULL
                        DO UPDATE SET updated_at = EXCLUDED.updated_at
                        RETURNING folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                    """, 
                        folder_data["folder_id"],
                        folder_name,
                        parent_id,
                        user_id,
                        team_id,
                        collection_type,
                        folder_data["created_at"],
                        folder_data["updated_at"]
                    )
            
            if not row:
                logger.error(f"❌ Repository: UPSERT returned no row for folder '{folder_name}'")
                return None
            
            # Check if we created new or found existing
            if row['folder_id'] == folder_data["folder_id"]:
                logger.info(f"✅ Repository: Created NEW folder '{folder_name}' → {row['folder_id']}")
            else:
                logger.info(f"📁 Repository: Found EXISTING folder '{folder_name}' → {row['folder_id']} (requested {folder_data['folder_id']})")
            
            # Convert row to dict and ensure UUID fields are strings
            result = dict(row)
            # Convert UUID to string for team_id (asyncpg returns UUID objects)
            if result.get('team_id') and not isinstance(result['team_id'], str):
                result['team_id'] = str(result['team_id'])
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Repository: Failed to create or get folder '{folder_data.get('name')}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def create_folder(self, folder_data: Dict[str, Any]) -> bool:
        """
        Create a new folder (DEPRECATED - use create_or_get_folder instead)
        
        **DEPRECATED**: This method doesn't handle race conditions properly.
        Use create_or_get_folder() which uses UPSERT pattern.
        
        Kept for backwards compatibility only.
        """
        try:
            logger.warning(f"⚠️ DEPRECATED: create_folder called for '{folder_data.get('name')}' - consider using create_or_get_folder")
            
            from services.database_manager.database_helpers import execute
            
            user_id = folder_data.get("user_id")
            collection_type = folder_data.get("collection_type", "user")
            
            await execute("""
                INSERT INTO document_folders (
                    folder_id, name, parent_folder_id, user_id, collection_type, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
                folder_data["folder_id"],
                folder_data["name"],
                folder_data["parent_folder_id"],
                user_id,
                folder_data["collection_type"],
                folder_data["created_at"],
                folder_data["updated_at"]
            )
            
            logger.info(f"✅ Repository: Folder created successfully: {folder_data['name']} for user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Repository: Failed to create folder: {e}")
            logger.error(f"❌ Repository: Folder data was: {folder_data}")
            return False
    
    async def get_folder(self, folder_id: str) -> Optional[Dict[str, Any]]:
        """Get folder by ID"""
        try:
            from services.database_manager.database_helpers import fetch_one
            
            row = await fetch_one("""
                SELECT folder_id, name, parent_folder_id, user_id, team_id, collection_type, exempt_from_vectorization, created_at, updated_at
                FROM document_folders WHERE folder_id = $1
            """, folder_id)
            
            if row:
                # Convert UUID to string for team_id (asyncpg returns UUID objects)
                result = dict(row)
                if result.get('team_id') and not isinstance(result['team_id'], str):
                    result['team_id'] = str(result['team_id'])
                return result
            
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get folder {folder_id}: {e}")
            return None
    
    async def get_folders_by_user(self, user_id: str = None, collection_type: str = "user") -> List[Dict[str, Any]]:
        """Get all folders for a user"""
        try:
            logger.debug(f"📁 Getting folders for user_id: {user_id}, collection_type: {collection_type}")
            from services.database_manager.database_helpers import fetch_all
            
            if user_id:
                rows = await fetch_all("""
                    SELECT folder_id, name, parent_folder_id, user_id, team_id, collection_type, exempt_from_vectorization, created_at, updated_at
                    FROM document_folders 
                    WHERE user_id = $1 AND collection_type = $2
                    ORDER BY name
                """, user_id, collection_type)
            else:
                rows = await fetch_all("""
                    SELECT folder_id, name, parent_folder_id, user_id, team_id, collection_type, exempt_from_vectorization, created_at, updated_at
                    FROM document_folders 
                    WHERE collection_type = $1
                    ORDER BY name
                """, collection_type)
            
            # Convert UUID to string for team_id (asyncpg returns UUID objects)
            result = []
            for row in rows:
                row_dict = dict(row)
                if row_dict.get('team_id') and not isinstance(row_dict['team_id'], str):
                    row_dict['team_id'] = str(row_dict['team_id'])
                result.append(row_dict)
            
            return result
        except Exception as e:
            logger.error(f"❌ Failed to get folders for user {user_id}: {e}")
            return []
    
    async def get_folders_by_teams(self, team_ids: List[str]) -> List[Dict[str, Any]]:
        """Get all folders for a list of teams"""
        try:
            if not team_ids:
                return []
            
            logger.debug(f"📁 Getting folders for teams: {team_ids}")
            from services.database_manager.database_helpers import fetch_all
            
            rows = await fetch_all("""
                SELECT folder_id, name, parent_folder_id, user_id, team_id, collection_type, created_at, updated_at
                FROM document_folders 
                WHERE team_id = ANY($1) AND collection_type = 'team'
                ORDER BY name
            """, team_ids)
            
            # Convert UUID to string for team_id (asyncpg returns UUID objects)
            result = []
            for row in rows:
                row_dict = dict(row)
                if row_dict.get('team_id') and not isinstance(row_dict['team_id'], str):
                    row_dict['team_id'] = str(row_dict['team_id'])
                result.append(row_dict)
            
            return result
        except Exception as e:
            logger.error(f"❌ Failed to get folders for teams: {e}")
            return []
    
    async def get_subfolders(self, parent_folder_id: str) -> List[Dict[str, Any]]:
        """Get subfolders of a folder"""
        try:
            from services.database_manager.database_helpers import fetch_all
            
            rows = await fetch_all("""
                SELECT folder_id, name, parent_folder_id, user_id, collection_type, created_at, updated_at
                FROM document_folders 
                WHERE parent_folder_id = $1
                ORDER BY name
            """, parent_folder_id)
            
            return rows
        except Exception as e:
            logger.error(f"❌ Failed to get subfolders for {parent_folder_id}: {e}")
            return []
    
    async def get_documents_by_folder(self, folder_id: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get documents in a folder"""
        try:
            logger.info(f"🔍 Repository: Getting documents for folder {folder_id}")
            
            from services.database_manager.database_helpers import fetch_all, fetch_one
            
            # Set RLS context based on user_id
            if user_id:
                rls_context = {'user_id': user_id, 'user_role': 'user'}
                logger.info(f"🔍 Repository: Using user context for folder query - user_id: {user_id}")
            else:
                rls_context = {'user_id': '', 'user_role': 'admin'}
                logger.info(f"🔍 Repository: Using admin context for folder query")
            
            # Log RLS context being used
            logger.info(f"🔍 DEBUG: Using RLS context - user_id: {rls_context.get('user_id', '')}, role: {rls_context.get('user_role', '')}")
            
            # **ROOSEVELT FIX:** Handle NULL folder_id (root-level documents)
            # In SQL, folder_id = NULL doesn't work - we need IS NULL
            if folder_id is None:
                logger.info(f"🔍 Repository: Querying ROOT-LEVEL documents (folder_id IS NULL)")
                rows = await fetch_all("""
                    SELECT document_id, filename, title, category, tags, description, author, language,
                           publication_date, doc_type, file_size, file_hash, processing_status, upload_date,
                           quality_score, page_count, chunk_count, entity_count, metadata_json, user_id,
                           submission_status, submitted_by, submitted_at, submission_reason, reviewed_by,
                           reviewed_at, review_comment, collection_type, folder_id, exempt_from_vectorization
                    FROM document_metadata 
                    WHERE folder_id IS NULL
                    ORDER BY filename
                """, rls_context=rls_context)
            else:
                # Query documents in specific folder with RLS context
                rows = await fetch_all("""
                    SELECT document_id, filename, title, category, tags, description, author, language,
                           publication_date, doc_type, file_size, file_hash, processing_status, upload_date,
                           quality_score, page_count, chunk_count, entity_count, metadata_json, user_id,
                           submission_status, submitted_by, submitted_at, submission_reason, reviewed_by,
                           reviewed_at, review_comment, collection_type, folder_id, exempt_from_vectorization
                    FROM document_metadata 
                    WHERE folder_id = $1
                    ORDER BY filename
                """, folder_id, rls_context=rls_context)
            
            logger.info(f"🔍 DEBUG: Raw query found {len(rows)} documents in folder {folder_id}")
            for row in rows:
                logger.info(f"🔍 DEBUG: Document {row['document_id']}: title='{row['title']}', user_id={row['user_id']}, collection_type='{row['collection_type']}', status='{row['processing_status']}'")
            
            documents = [self._row_to_document_info(row) for row in rows]
            logger.info(f"✅ Repository: Found {len(documents)} documents in folder {folder_id}")
            return documents
        except Exception as e:
            logger.error(f"❌ Failed to get documents in folder {folder_id}: {e}")
            return []
    
    async def get_root_documents_by_collection(self, collection_type: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Get documents at root level (folder_id IS NULL) for a specific collection type
        
        **ROOSEVELT FIX:** Root-level documents need collection_type filter!
        """
        try:
            logger.info(f"🔍 Repository: Getting root-level documents for collection_type: {collection_type}")
            
            from services.database_manager.database_helpers import fetch_all
            
            # Set RLS context based on user_id
            if user_id:
                rls_context = {'user_id': user_id, 'user_role': 'user'}
                logger.info(f"🔍 Repository: Using user context - user_id: {user_id}")
            else:
                rls_context = {'user_id': '', 'user_role': 'admin'}
                logger.info(f"🔍 Repository: Using admin context")
            
            # Query root-level documents for this collection type
            rows = await fetch_all("""
                SELECT document_id, filename, title, category, tags, description, author, language,
                       publication_date, doc_type, file_size, file_hash, processing_status, upload_date,
                       quality_score, page_count, chunk_count, entity_count, metadata_json, user_id,
                       submission_status, submitted_by, submitted_at, submission_reason, reviewed_by,
                       reviewed_at, review_comment, collection_type, folder_id, exempt_from_vectorization
                FROM document_metadata 
                WHERE folder_id IS NULL AND collection_type = $1
                ORDER BY filename
            """, collection_type, rls_context=rls_context)
            
            logger.info(f"🔍 DEBUG: Raw query found {len(rows)} root-level {collection_type} documents")
            for row in rows:
                logger.info(f"🔍 DEBUG: Document {row['document_id']}: title='{row['title']}', user_id={row['user_id']}, collection_type='{row['collection_type']}', status='{row['processing_status']}'")
            
            documents = [self._row_to_document_info(row) for row in rows]
            logger.info(f"✅ Repository: Found {len(documents)} root-level {collection_type} documents")
            return documents
        except Exception as e:
            logger.error(f"❌ Failed to get root-level documents for {collection_type}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def get_document_count_in_folder(self, folder_id: str) -> int:
        """Get count of documents in a folder"""
        try:
            from services.database_manager.database_helpers import fetch_one
            
            result = await fetch_one("""
                SELECT COUNT(*) FROM document_metadata WHERE folder_id = $1
            """, folder_id)
            return result.get('count', 0) if result else 0
        except Exception as e:
            logger.error(f"❌ Failed to get document count for folder {folder_id}: {e}")
            return 0
    
    async def get_subfolder_count(self, folder_id: str) -> int:
        """Get count of subfolders in a folder"""
        try:
            from services.database_manager.database_helpers import fetch_one
            
            result = await fetch_one("""
                SELECT COUNT(*) FROM document_folders WHERE parent_folder_id = $1
            """, folder_id)
            return result.get('count', 0) if result else 0
        except Exception as e:
            logger.error(f"❌ Failed to get subfolder count for folder {folder_id}: {e}")
            return 0
    
    async def update_folder(self, folder_id: str, updates: Dict[str, Any]) -> bool:
        """Update folder information"""
        try:
            from services.database_manager.database_helpers import execute
            
            # Build dynamic update query
            set_clauses = []
            params = []
            param_count = 1
            
            for key, value in updates.items():
                if key in ['name', 'parent_folder_id', 'category', 'tags', 'inherit_tags', 'updated_at']:
                    set_clauses.append(f"{key} = ${param_count}")
                    params.append(value)
                    param_count += 1
            
            if not set_clauses:
                return True
            
            query = f"UPDATE document_folders SET {', '.join(set_clauses)} WHERE folder_id = ${param_count}"
            params.append(folder_id)
            
            await execute(query, *params)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update folder {folder_id}: {e}")
            return False
    
    async def update_folder_metadata(self, folder_id: str, category: str = None, tags: List[str] = None, inherit_tags: bool = None) -> bool:
        """
        Update folder metadata (category, tags, inherit_tags)
        
        **ROOSEVELT FOLDER TAGGING**: Store metadata for automatic inheritance!
        """
        try:
            from services.database_manager.database_helpers import execute
            
            updates = {}
            if category is not None:
                updates['category'] = category
            if tags is not None:
                updates['tags'] = tags
            if inherit_tags is not None:
                updates['inherit_tags'] = inherit_tags
            
            if not updates:
                return True
            
            updates['updated_at'] = datetime.utcnow()
            
            return await self.update_folder(folder_id, updates)
            
        except Exception as e:
            logger.error(f"❌ Failed to update folder metadata {folder_id}: {e}")
            return False
    
    async def get_folder_metadata(self, folder_id: str) -> Dict[str, Any]:
        """Get folder metadata (category, tags, inherit_tags)"""
        try:
            from services.database_manager.database_helpers import fetch_one
            
            query = """
                SELECT category, tags, inherit_tags 
                FROM document_folders 
                WHERE folder_id = $1
            """
            
            result = await fetch_one(query, folder_id)
            if result:
                return {
                    'category': result.get('category'),
                    'tags': result.get('tags', []),
                    'inherit_tags': result.get('inherit_tags', True)
                }
            return {'category': None, 'tags': [], 'inherit_tags': True}
            
        except Exception as e:
            logger.error(f"❌ Failed to get folder metadata {folder_id}: {e}")
            return {'category': None, 'tags': [], 'inherit_tags': True}
    
    async def get_all_tags(self) -> List[str]:
        """
        Get all unique tags from documents and folders
        
        **ROOSEVELT TAG DETECTION**: Used for fuzzy matching user queries!
        """
        try:
            from services.database_manager.database_helpers import fetch_all
            
            # Get tags from both documents and folders
            # **ROOSEVELT SQL FIX**: Wrap unnest in subquery to filter on alias
            query = """
                SELECT tag
                FROM (
                    SELECT DISTINCT unnest(tags) as tag
                    FROM (
                        SELECT tags FROM document_metadata WHERE tags IS NOT NULL AND array_length(tags, 1) > 0
                        UNION ALL
                        SELECT tags FROM document_folders WHERE tags IS NOT NULL AND array_length(tags, 1) > 0
                    ) combined_tags
                ) unnested_tags
                WHERE tag IS NOT NULL AND tag != ''
                ORDER BY tag
            """
            
            rows = await fetch_all(query)
            tags = [row['tag'] for row in rows if row.get('tag')]
            logger.info(f"📋 Found {len(tags)} unique tags in system")
            return tags
            
        except Exception as e:
            logger.error(f"❌ Failed to get all tags: {e}")
            return []
    
    async def get_all_categories(self) -> List[str]:
        """
        Get all unique categories from documents and folders
        
        **ROOSEVELT TAG DETECTION**: Used for fuzzy matching user queries!
        """
        try:
            from services.database_manager.database_helpers import fetch_all
            
            # Get categories from both documents and folders
            query = """
                SELECT DISTINCT category
                FROM (
                    SELECT category FROM document_metadata WHERE category IS NOT NULL AND category != ''
                    UNION ALL
                    SELECT category FROM document_folders WHERE category IS NOT NULL AND category != ''
                ) combined_categories
                WHERE category IS NOT NULL AND category != ''
                ORDER BY category
            """
            
            rows = await fetch_all(query)
            categories = [row['category'] for row in rows if row.get('category')]
            logger.info(f"📋 Found {len(categories)} unique categories in system")
            return categories
            
        except Exception as e:
            logger.error(f"❌ Failed to get all categories: {e}")
            return []
    
    async def delete_folder(self, folder_id: str) -> bool:
        """Delete a folder"""
        try:
            from services.database_manager.database_helpers import execute
            
            await execute("DELETE FROM document_folders WHERE folder_id = $1", folder_id)
            logger.info(f"✅ Repository: Deleted folder {folder_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete folder {folder_id}: {e}")
            return False
    
    async def update_document_exemption_status(self, document_id: str, exempt_status: bool = None) -> bool:
        """
        Update document exemption from vectorization status.
        
        Args:
            document_id: Document ID
            exempt_status: True=exempt, False=not exempt (override), None=inherit from folder
        
        Returns:
            True if update successful
        """
        try:
            from services.database_manager.database_helpers import execute
            await execute("""
                UPDATE document_metadata 
                SET exempt_from_vectorization = $1, updated_at = CURRENT_TIMESTAMP
                WHERE document_id = $2
            """, exempt_status, document_id)
            status_str = "inherit from folder" if exempt_status is None else ("exempt" if exempt_status else "not exempt (override)")
            logger.info(f"Updated exemption status for document {document_id}: {status_str}")
            return True
        except Exception as e:
            logger.error(f"Failed to update exemption status for document {document_id}: {e}")
            return False
    
    async def update_folder_exemption_status(self, folder_id: str, exempt_status: bool = None) -> bool:
        """
        Update folder exemption from vectorization status.
        
        Args:
            folder_id: Folder ID
            exempt_status: True=exempt, False=not exempt (override parent), None=inherit from parent
        
        Returns:
            True if update successful
        """
        try:
            from services.database_manager.database_helpers import execute
            await execute("""
                UPDATE document_folders 
                SET exempt_from_vectorization = $1, updated_at = CURRENT_TIMESTAMP
                WHERE folder_id = $2
            """, exempt_status, folder_id)
            status_str = "inherit from parent" if exempt_status is None else ("exempt" if exempt_status else "not exempt (override)")
            logger.info(f"Updated exemption status for folder {folder_id}: {status_str}")
            return True
        except Exception as e:
            logger.error(f"Failed to update exemption status for folder {folder_id}: {e}")
            return False
    
    async def get_folder_descendants(self, folder_id: str) -> Tuple[List[str], List[str]]:
        """Get all descendant folder IDs and document IDs for a folder (recursive)"""
        try:
            from services.database_manager.database_helpers import fetch_all
            
            # Get all descendant folders recursively
            folder_rows = await fetch_all("""
                WITH RECURSIVE folder_tree AS (
                    SELECT folder_id FROM document_folders WHERE folder_id = $1
                    UNION ALL
                    SELECT f.folder_id 
                    FROM document_folders f
                    INNER JOIN folder_tree ft ON f.parent_folder_id = ft.folder_id
                )
                SELECT folder_id FROM folder_tree WHERE folder_id != $1
            """, folder_id)
            
            descendant_folder_ids = [row['folder_id'] for row in folder_rows]
            
            # Get all documents in this folder and all descendant folders
            all_folder_ids = [folder_id] + descendant_folder_ids
            doc_rows = await fetch_all("""
                SELECT document_id 
                FROM document_metadata 
                WHERE folder_id = ANY($1)
            """, all_folder_ids)
            
            descendant_document_ids = [row['document_id'] for row in doc_rows]
            
            return descendant_folder_ids, descendant_document_ids
        except Exception as e:
            logger.error(f"Failed to get folder descendants for {folder_id}: {e}")
            return [], []
    
    async def is_folder_exempt(self, folder_id: str) -> bool:
        """
        Check if folder is exempt from vectorization.
        
        Three-state system:
        - TRUE: Folder is exempt
        - FALSE: Folder is NOT exempt (explicit override of parent)
        - NULL: Inherit from parent folder
        
        Returns True if folder or any ancestor is explicitly exempt (TRUE).
        Returns False if folder is explicitly not exempt (FALSE) or no exemption found.
        """
        try:
            if not folder_id:
                return False
            
            from services.database_manager.database_helpers import fetch_one
            
            # Recursively check folder hierarchy
            # Stop at first explicit exemption (TRUE) or explicit override (FALSE)
            folder_row = await fetch_one("""
                WITH RECURSIVE folder_path AS (
                    -- Start with the target folder
                    SELECT folder_id, parent_folder_id, exempt_from_vectorization, 0 as depth
                    FROM document_folders WHERE folder_id = $1
                    UNION ALL
                    -- Walk up to parent folders
                    SELECT f.folder_id, f.parent_folder_id, f.exempt_from_vectorization, fp.depth + 1
                    FROM document_folders f
                    INNER JOIN folder_path fp ON f.folder_id = fp.parent_folder_id
                )
                SELECT exempt_from_vectorization, depth
                FROM folder_path 
                WHERE exempt_from_vectorization IS NOT NULL  -- Stop at first explicit setting (TRUE or FALSE)
                ORDER BY depth ASC  -- Check from target folder up to root
                LIMIT 1
            """, folder_id)
            
            if folder_row is None:
                # No explicit exemption found in hierarchy - not exempt
                logger.debug(f"✅ Folder {folder_id} has no explicit exemption - not exempt")
                return False
            
            # If we found an explicit setting, use it
            is_exempt = folder_row['exempt_from_vectorization'] is True
            if is_exempt:
                logger.debug(f"🚫 Folder {folder_id} (or ancestor) is exempt from vectorization")
            else:
                logger.debug(f"✅ Folder {folder_id} (or ancestor) explicitly overrides exemption - not exempt")
            
            return is_exempt
        except Exception as e:
            logger.error(f"❌ Failed to check folder exemption status for {folder_id}: {e}")
            return False
    
    async def is_document_exempt(self, document_id: str) -> bool:
        """
        Check if document is exempt from vectorization.
        
        Three-state system for documents:
        - TRUE: Document is exempt
        - FALSE: Document is NOT exempt (explicit override of folder)
        - NULL: Inherit from folder
        
        Returns True if document is explicitly exempt (TRUE) or inherits exemption from folder.
        Returns False if document is explicitly not exempt (FALSE) or folder is not exempt.
        """
        try:
            from services.database_manager.database_helpers import fetch_one
            
            # First check the document itself
            doc_row = await fetch_one("""
                SELECT exempt_from_vectorization, folder_id
                FROM document_metadata 
                WHERE document_id = $1
            """, document_id)
            
            if not doc_row:
                logger.warning(f"⚠️ Document {document_id} not found in database for exemption check")
                return False
            
            # Check if document has explicit exemption setting
            doc_exempt = doc_row['exempt_from_vectorization']
            
            if doc_exempt is True:
                # Document is explicitly exempt
                logger.info(f"🚫 Document {document_id} is directly exempt from vectorization")
                return True
            elif doc_exempt is False:
                # Document explicitly overrides folder exemption - not exempt
                logger.info(f"✅ Document {document_id} explicitly overrides folder exemption - not exempt")
                return False
            else:
                # Document inherits from folder (NULL)
                folder_id = doc_row['folder_id']
                if not folder_id:
                    logger.debug(f"📁 Document {document_id} has no folder_id - not exempt")
                    return False
                
                # Use helper method to check folder exemption
                is_exempt = await self.is_folder_exempt(folder_id)
                if is_exempt:
                    logger.info(f"🚫 Document {document_id} inherits exemption from folder {folder_id} (or ancestor)")
                else:
                    logger.debug(f"✅ Document {document_id} inherits non-exemption from folder {folder_id}")
                
                return is_exempt
        except Exception as e:
            logger.error(f"❌ Failed to check exemption status for document {document_id}: {e}")
            return False
    
    async def update_document_folder(self, document_id: str, folder_id: str = None, user_id: str = None) -> bool:
        """Update the folder assignment of a document - Roosevelt Architecture"""
        try:
            from services.database_manager.database_helpers import execute, fetch_one
            
            # Set RLS context based on user_id
            rls_context = None
            if user_id:
                rls_context = {'user_id': user_id, 'user_role': 'user'}
                logger.info(f"🔍 Repository: Using user context for document folder update - user_id: {user_id}")
            else:
                rls_context = {'user_id': '', 'user_role': 'admin'}
                logger.info(f"🔍 Repository: Using admin context for document folder update")
            
            # Debug: Check if document exists before update
            doc_check = await fetch_one("""
                SELECT document_id, user_id, collection_type FROM document_metadata WHERE document_id = $1
            """, document_id, rls_context=rls_context)
            
            if not doc_check:
                logger.warning(f"⚠️ DEBUG: Document {document_id} not found in database!")
                return False
            
            logger.info(f"🔍 DEBUG: Document {document_id} found for update. User_id: {doc_check['user_id']}, Collection_type: {doc_check['collection_type']}")
            
            # Check current exemption status and new folder exemption
            # Documents inherit exemption from their folder, so when moved, update accordingly
            current_doc = await fetch_one("""
                SELECT exempt_from_vectorization, folder_id as old_folder_id 
                FROM document_metadata WHERE document_id = $1
            """, document_id, rls_context=rls_context)
            
            current_exempt = current_doc.get('exempt_from_vectorization', False) if current_doc else False
            old_folder_id = current_doc.get('old_folder_id') if current_doc else None
            
            # Determine new exemption status based on new folder
            if folder_id:
                # Check if new folder is exempt
                folder_exempt = await self.is_folder_exempt(folder_id)
                exempt_from_vectorization = folder_exempt
                if folder_exempt:
                    logger.info(f"🚫 Document {document_id} inheriting exemption from new folder {folder_id}")
                else:
                    logger.debug(f"✅ Document {document_id} moved to non-exempt folder {folder_id}")
            else:
                # Moving to root - no folder exemption
                exempt_from_vectorization = False
                logger.debug(f"✅ Document {document_id} moved to root - removing folder-based exemption")
            
            # Note: If a document was directly exempted by user (not via folder), 
            # the user can re-exempt it after moving. The exemption status is now 
            # based on the folder, ensuring consistency.
            
            # Perform the update
            query = """
                UPDATE document_metadata
                SET folder_id = $1, exempt_from_vectorization = $3
                WHERE document_id = $2
            """
            await execute(query, folder_id, document_id, exempt_from_vectorization, rls_context=rls_context)
            
            # Verify the update was successful by checking the document again
            verify_check = await fetch_one("""
                SELECT document_id, folder_id FROM document_metadata WHERE document_id = $1
            """, document_id, rls_context=rls_context)
            
            if verify_check and verify_check['folder_id'] == folder_id:
                logger.info(f"✅ Document {document_id} folder updated to {folder_id} and verified")
                return True
            else:
                logger.warning(f"⚠️ DEBUG: Folder update verification failed for {document_id}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to update document folder for {document_id}: {e}")
            return False
    
    async def find_documents_by_tags(
        self,
        required_tags: List[str],
        user_id: Optional[str] = None,
        collection_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Find documents that contain ALL of the specified tags

        Args:
            required_tags: List of tags that ALL must be present on the document
            user_id: Optional user ID to filter results
            collection_type: Optional collection type filter ('user' or 'global')
            limit: Maximum number of results to return

        Returns:
            List of document dictionaries with metadata
        """
        try:
            # Import fetch_all directly like in the working manual test
            from services.database_manager.database_helpers import fetch_all

            query = """
                SELECT
                    document_id, filename, title, category, tags, description,
                    author, language, publication_date, doc_type, file_size,
                    file_hash, processing_status, upload_date, quality_score,
                    page_count, chunk_count, entity_count, user_id, collection_type
                FROM document_metadata
                WHERE tags @> $1
                ORDER BY upload_date DESC
                LIMIT $2
            """

            # Call exactly like the working manual test
            documents = await fetch_all(query, required_tags, limit)

            logger.info(f"📄 Found {len(documents)} documents with tags {required_tags}")
            return documents

        except Exception as e:
            logger.error(f"❌ Failed to find documents by tags {required_tags}: {e}")
            return []

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("🔄 Document Repository closed")

