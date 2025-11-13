"""
Document Repository Extensions for ZIP Hierarchy
Extends DocumentRepository with methods for parent-child relationships and ZIP handling
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentRepositoryZipExtensions:
    """Extension methods for ZIP hierarchy support in DocumentRepository"""
    
    def __init__(self, repository):
        """Initialize with reference to main DocumentRepository"""
        self.repository = repository
        self.pool = repository.pool
    
    async def mark_as_zip_container(self, document_id: str) -> bool:
        """Mark a document as a ZIP container"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE document_metadata 
                    SET is_zip_container = TRUE, updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = $1
                """, document_id)
                
                logger.debug(f"📦 Marked document as ZIP container: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to mark document as ZIP container: {e}")
            return False
    
    async def set_parent_relationship(self, child_id: str, parent_id: str, original_zip_path: str = None) -> bool:
        """Set parent-child relationship for ZIP extracted files"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE document_metadata 
                    SET parent_document_id = $1, original_zip_path = $2, updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = $3
                """, parent_id, original_zip_path, child_id)
                
                logger.debug(f"📁 Set parent relationship: {child_id} -> {parent_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to set parent relationship: {e}")
            return False
    
    async def get_zip_children(self, parent_document_id: str) -> List[Dict[str, Any]]:
        """Get all files extracted from a ZIP document"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        document_id, filename, title, doc_type, category, tags,
                        author, language, publication_date, file_size, processing_status,
                        upload_date, quality_score, original_zip_path, inherit_parent_metadata
                    FROM document_metadata 
                    WHERE parent_document_id = $1
                    ORDER BY original_zip_path, filename
                """, parent_document_id)
                
                result = []
                for row in rows:
                    result.append({
                        'document_id': row['document_id'],
                        'filename': row['filename'],
                        'title': row['title'],
                        'doc_type': row['doc_type'],
                        'category': row['category'],
                        'tags': row['tags'] or [],
                        'author': row['author'],
                        'language': row['language'],
                        'publication_date': row['publication_date'],
                        'file_size': row['file_size'],
                        'processing_status': row['processing_status'],
                        'upload_date': row['upload_date'],
                        'quality_score': row['quality_score'],
                        'original_zip_path': row['original_zip_path'],
                        'inherit_parent_metadata': row['inherit_parent_metadata']
                    })
                
                logger.debug(f"📁 Found {len(result)} children for ZIP {parent_document_id}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to get ZIP children: {e}")
            return []
    
    async def get_zip_containers(self) -> List[Dict[str, Any]]:
        """Get all ZIP container documents"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        document_id, filename, title, doc_type, category, tags,
                        author, upload_date, file_size, processing_status, quality_score
                    FROM document_metadata 
                    WHERE is_zip_container = TRUE
                    ORDER BY upload_date DESC
                """)
                
                result = []
                for row in rows:
                    result.append({
                        'document_id': row['document_id'],
                        'filename': row['filename'],
                        'title': row['title'],
                        'doc_type': row['doc_type'],
                        'category': row['category'],
                        'tags': row['tags'] or [],
                        'author': row['author'],
                        'upload_date': row['upload_date'],
                        'file_size': row['file_size'],
                        'processing_status': row['processing_status'],
                        'quality_score': row['quality_score']
                    })
                
                logger.debug(f"📦 Found {len(result)} ZIP containers")
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to get ZIP containers: {e}")
            return []
    
    async def toggle_metadata_inheritance(self, document_id: str, inherit: bool) -> bool:
        """Toggle metadata inheritance for a ZIP extracted file"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE document_metadata 
                    SET inherit_parent_metadata = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE document_id = $2
                """, inherit, document_id)
                
                logger.debug(f"🔄 Toggled metadata inheritance for {document_id}: {inherit}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to toggle metadata inheritance: {e}")
            return False
    
    async def get_parent_document(self, child_document_id: str) -> Optional[Dict[str, Any]]:
        """Get parent document for a ZIP extracted file"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT 
                        p.document_id, p.filename, p.title, p.doc_type, p.category, 
                        p.tags, p.author, p.upload_date, p.file_size
                    FROM document_metadata p
                    INNER JOIN document_metadata c ON c.parent_document_id = p.document_id
                    WHERE c.document_id = $1
                """, child_document_id)
                
                if row:
                    return {
                        'document_id': row['document_id'],
                        'filename': row['filename'],
                        'title': row['title'],
                        'doc_type': row['doc_type'],
                        'category': row['category'],
                        'tags': row['tags'] or [],
                        'author': row['author'],
                        'upload_date': row['upload_date'],
                        'file_size': row['file_size']
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get parent document: {e}")
            return None
    
    async def get_documents_with_hierarchy(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get documents with their hierarchy information"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        d.document_id, d.filename, d.title, d.doc_type, d.category, d.tags,
                        d.author, d.upload_date, d.file_size, d.processing_status, d.quality_score,
                        d.parent_document_id, d.is_zip_container, d.original_zip_path,
                        d.inherit_parent_metadata,
                        p.filename as parent_filename
                    FROM document_metadata d
                    LEFT JOIN document_metadata p ON d.parent_document_id = p.document_id
                    ORDER BY 
                        CASE WHEN d.parent_document_id IS NULL THEN d.upload_date 
                             ELSE p.upload_date END DESC,
                        d.parent_document_id NULLS FIRST,
                        d.original_zip_path,
                        d.filename
                    LIMIT $1 OFFSET $2
                """, limit, offset)
                
                result = []
                for row in rows:
                    result.append({
                        'document_id': row['document_id'],
                        'filename': row['filename'],
                        'title': row['title'],
                        'doc_type': row['doc_type'],
                        'category': row['category'],
                        'tags': row['tags'] or [],
                        'author': row['author'],
                        'upload_date': row['upload_date'],
                        'file_size': row['file_size'],
                        'processing_status': row['processing_status'],
                        'quality_score': row['quality_score'],
                        'parent_document_id': row['parent_document_id'],
                        'is_zip_container': row['is_zip_container'],
                        'original_zip_path': row['original_zip_path'],
                        'inherit_parent_metadata': row['inherit_parent_metadata'],
                        'parent_filename': row['parent_filename']
                    })
                
                logger.debug(f"📋 Retrieved {len(result)} documents with hierarchy")
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to get documents with hierarchy: {e}")
            return []
    
    async def update_child_metadata_inheritance(self, parent_document_id: str, metadata_updates: Dict[str, Any]) -> int:
        """Update metadata for all child documents that inherit from parent"""
        try:
            updated_count = 0
            
            async with self.pool.acquire() as conn:
                # Get all children that inherit metadata
                children = await conn.fetch("""
                    SELECT document_id FROM document_metadata 
                    WHERE parent_document_id = $1 AND inherit_parent_metadata = TRUE
                """, parent_document_id)
                
                # Update each child's metadata
                for child in children:
                    child_id = child['document_id']
                    
                    # Build update query dynamically based on provided metadata
                    update_fields = []
                    values = []
                    param_index = 1
                    
                    for field, value in metadata_updates.items():
                        if field in ['title', 'category', 'tags', 'description', 'author', 'publication_date']:
                            update_fields.append(f"{field} = ${param_index}")
                            values.append(value)
                            param_index += 1
                    
                    if update_fields:
                        update_fields.append(f"updated_at = ${param_index}")
                        values.append(datetime.utcnow())
                        values.append(child_id)  # For WHERE clause
                        
                        query = f"""
                            UPDATE document_metadata 
                            SET {', '.join(update_fields)}
                            WHERE document_id = ${param_index + 1}
                        """
                        
                        await conn.execute(query, *values)
                        updated_count += 1
                
                logger.info(f"📝 Updated metadata for {updated_count} child documents")
                return updated_count
                
        except Exception as e:
            logger.error(f"❌ Failed to update child metadata inheritance: {e}")
            return 0
    
    async def delete_zip_hierarchy(self, parent_document_id: str) -> Dict[str, int]:
        """Delete a ZIP file and all its extracted children"""
        try:
            deleted_children = 0
            deleted_parent = 0
            
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # First, get count of children that will be deleted
                    child_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM document_metadata 
                        WHERE parent_document_id = $1
                    """, parent_document_id)
                    
                    # Delete children first (foreign key constraint)
                    await conn.execute("""
                        DELETE FROM document_metadata 
                        WHERE parent_document_id = $1
                    """, parent_document_id)
                    deleted_children = child_count
                    
                    # Delete parent ZIP document
                    result = await conn.execute("""
                        DELETE FROM document_metadata 
                        WHERE document_id = $1
                    """, parent_document_id)
                    deleted_parent = 1 if result == "DELETE 1" else 0
                
                logger.info(f"🗑️ Deleted ZIP hierarchy: {deleted_children} children + {deleted_parent} parent")
                return {
                    'deleted_children': deleted_children,
                    'deleted_parent': deleted_parent,
                    'total_deleted': deleted_children + deleted_parent
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to delete ZIP hierarchy: {e}")
            return {'deleted_children': 0, 'deleted_parent': 0, 'total_deleted': 0} 