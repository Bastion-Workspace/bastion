"""
Free-form Notes Service - Handles user-created notes that integrate with search
"""

import asyncio
import json
import logging
import uuid
import zipfile
import io
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Tuple
from repositories.document_repository import DocumentRepository
from services.embedding_service_wrapper import get_embedding_service
from models.api_models import (
    FreeFormNoteRequest, FreeFormNoteFilterRequest, FreeFormNoteInfo,
    FreeFormNoteResponse, FreeFormNotesListResponse
)

logger = logging.getLogger(__name__)


class FreeFormNotesService:
    """Service for managing free-form notes"""
    
    def __init__(self):
        self.document_repository = None
        self.embedding_manager = None
        self.current_user_id = "default-user"  # Default for single-user mode
    
    async def initialize(self):
        """Initialize the notes service"""
        logger.info("🗒️ Initializing Free-form Notes Service...")
        
        # Initialize document repository
        self.document_repository = DocumentRepository()
        await self.document_repository.initialize()
        
        # Initialize embedding service wrapper
        self.embedding_manager = await get_embedding_service()
        
        logger.info("✅ Free-form Notes Service initialized")
    
    def set_current_user(self, user_id: str):
        """Set the current user for operations (for multi-user support)"""
        self.current_user_id = user_id
    
    async def create_note(self, request: FreeFormNoteRequest) -> FreeFormNoteInfo:
        """Create a new free-form note"""
        try:
            note_id = str(uuid.uuid4())
            logger.info(f"🗒️ Creating note: {request.title}")
            
            # Insert note into database (avoid RETURNING to sidestep RLS RETURNING visibility)
            insert_query = """
                INSERT INTO free_form_notes (
                    note_id, title, content, note_date, tags, category, folder_id, metadata_json, user_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            metadata = {
                "created_via": "api",
                "word_count": len(request.content.split()),
                "char_count": len(request.content)
            }
            
            from services.database_manager.database_helpers import execute, fetch_one
            rls = {'user_id': self.current_user_id, 'user_role': 'user'}

            exec_result = await execute(
                insert_query,
                note_id,
                request.title,
                request.content,
                request.note_date,
                request.tags or [],
                request.category,
                request.folder_id,
                json.dumps(metadata),
                self.current_user_id,
                rls_context=rls
            )
            if not exec_result or not exec_result.upper().startswith("INSERT"):
                raise Exception("Failed to create note in database")

            # Fetch timestamps after insert (visible via SELECT policy)
            ts_row = await fetch_one(
                "SELECT created_at, updated_at FROM free_form_notes WHERE note_id = $1 AND user_id = $2",
                note_id,
                self.current_user_id,
                rls_context=rls
            )
            created_at = ts_row['created_at'] if ts_row and 'created_at' in ts_row else datetime.utcnow()
            updated_at = ts_row['updated_at'] if ts_row and 'updated_at' in ts_row else created_at
            
            # Process note for embedding (async)
            asyncio.create_task(self._process_note_for_embedding(note_id, request.title, request.content))
            
            logger.info(f"✅ Created note: {note_id}")
            
            # Return full note information for immediate UI update
            return FreeFormNoteInfo(
                note_id=note_id,
                title=request.title,
                content=request.content,
                note_date=request.note_date,
                tags=request.tags or [],
                category=request.category,
                created_at=created_at,
                updated_at=updated_at,
                embedding_processed=False
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create note: {e}")
            raise Exception(f"Failed to create note: {str(e)}")
    
    async def get_note(self, note_id: str) -> Optional[FreeFormNoteInfo]:
        """Get a single note by ID"""
        try:
            query = """
                SELECT note_id, title, content, note_date, tags, category,
                       embedding_processed, created_at, updated_at
                FROM free_form_notes
                WHERE note_id = $1 AND user_id = $2
            """
            
            result = await self.document_repository.execute_query(query, note_id, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            
            if not result:
                return None
            
            note = result[0]
            
            return FreeFormNoteInfo(
                note_id=note['note_id'],
                title=note['title'],
                content=note['content'],
                note_date=note['note_date'],
                tags=note['tags'] or [],
                category=note['category'],
                created_at=note['created_at'],
                updated_at=note['updated_at'],
                embedding_processed=note['embedding_processed']
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get note {note_id}: {e}")
            return None
    
    async def update_note(self, note_id: str, request: FreeFormNoteRequest) -> FreeFormNoteInfo:
        """Update an existing note"""
        try:
            logger.info(f"🗒️ Updating note: {note_id}")
            
            # Update note in database (ensure user owns the note)
            query = """
                UPDATE free_form_notes
                SET title = $2, content = $3, note_date = $4, tags = $5, 
                    category = $6, updated_at = CURRENT_TIMESTAMP,
                    embedding_processed = FALSE
                WHERE note_id = $1 AND user_id = $7
                RETURNING title, created_at, updated_at
            """
            
            result = await self.document_repository.execute_query(
                query,
                note_id,
                request.title,
                request.content,
                request.note_date,
                request.tags or [],
                request.category,
                self.current_user_id,
                rls_context={'user_id': self.current_user_id, 'user_role': 'user'}
            )
            
            if not result:
                raise Exception(f"Note {note_id} not found")
            
            # Get the timestamps from the update result
            db_result = result[0]
            created_at = db_result['created_at']
            updated_at = db_result['updated_at']
            
            # Re-process note for embedding (async)
            asyncio.create_task(self._process_note_for_embedding(note_id, request.title, request.content))
            
            logger.info(f"✅ Updated note: {note_id}")
            
            # Return full note information for immediate UI update
            return FreeFormNoteInfo(
                note_id=note_id,
                title=request.title,
                content=request.content,
                note_date=request.note_date,
                tags=request.tags or [],
                category=request.category,
                created_at=created_at,
                updated_at=updated_at,
                embedding_processed=False
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to update note {note_id}: {e}")
            raise Exception(f"Failed to update note: {str(e)}")
    
    async def delete_note(self, note_id: str) -> FreeFormNoteResponse:
        """Delete a note"""
        try:
            logger.info(f"🗒️ Deleting note: {note_id}")
            
            # Get note title before deletion (ensure user owns the note)
            title_query = "SELECT title FROM free_form_notes WHERE note_id = $1 AND user_id = $2"
            title_result = await self.document_repository.execute_query(title_query, note_id, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            
            if not title_result:
                raise Exception(f"Note {note_id} not found")
            
            title = title_result[0]['title']
            
            # Delete from vector database first
            await self._remove_note_from_embedding(note_id)
            
            # Delete from database (ensure user owns the note)
            delete_query = "DELETE FROM free_form_notes WHERE note_id = $1 AND user_id = $2"
            await self.document_repository.execute_query(delete_query, note_id, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            
            logger.info(f"✅ Deleted note: {note_id}")
            
            return FreeFormNoteResponse(
                note_id=note_id,
                title=title,
                message="Note deleted successfully"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to delete note {note_id}: {e}")
            raise Exception(f"Failed to delete note: {str(e)}")
    
    async def list_notes(self, filter_request: FreeFormNoteFilterRequest) -> FreeFormNotesListResponse:
        """List notes with filtering and pagination"""
        try:
            logger.info(f"🗒️ Listing notes with filters")
            
            # Build WHERE clause
            where_conditions = ["user_id = $1"]  # Always filter by user
            params = [self.current_user_id]
            param_count = 1
            
            if filter_request.search_query:
                param_count += 1
                where_conditions.append(f"(title ILIKE ${param_count} OR content ILIKE ${param_count})")
                params.append(f"%{filter_request.search_query}%")
            
            if filter_request.category:
                param_count += 1
                where_conditions.append(f"category = ${param_count}")
                params.append(filter_request.category)
            
            if filter_request.tags:
                param_count += 1
                where_conditions.append(f"tags @> ${param_count}")
                params.append(filter_request.tags)
            
            if filter_request.date_from:
                param_count += 1
                where_conditions.append(f"note_date >= ${param_count}")
                params.append(filter_request.date_from)
            
            if filter_request.date_to:
                param_count += 1
                where_conditions.append(f"note_date <= ${param_count}")
                params.append(filter_request.date_to)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Build ORDER BY clause
            sort_fields = {
                "created_at": "created_at",
                "note_date": "note_date",
                "title": "title",
                "updated_at": "updated_at"
            }
            
            sort_field = sort_fields.get(filter_request.sort_by, "created_at")
            sort_order = "ASC" if filter_request.sort_order == "asc" else "DESC"
            
            # Get total count
            count_query = f"""
                SELECT COUNT(*) as total
                FROM free_form_notes
                WHERE {where_clause}
            """
            
            count_result = await self.document_repository.execute_query(count_query, *params, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            total = count_result[0]['total'] if count_result else 0
            
            # Get notes
            notes_query = f"""
                SELECT note_id, title, content, note_date, tags, category,
                       embedding_processed, created_at, updated_at
                FROM free_form_notes
                WHERE {where_clause}
                ORDER BY {sort_field} {sort_order}
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            
            params.extend([filter_request.limit, filter_request.skip])
            
            notes_result = await self.document_repository.execute_query(notes_query, *params, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            
            # Convert to response models
            notes = []
            for note in notes_result or []:
                notes.append(FreeFormNoteInfo(
                    note_id=note['note_id'],
                    title=note['title'],
                    content=note['content'],
                    note_date=note['note_date'],
                    tags=note['tags'] or [],
                    category=note['category'],
                    created_at=note['created_at'],
                    updated_at=note['updated_at'],
                    embedding_processed=note['embedding_processed']
                ))
            
            # Get category and tag counts
            categories = await self._get_category_counts()
            tags = await self._get_tag_counts()
            
            return FreeFormNotesListResponse(
                notes=notes,
                total=total,
                categories=categories,
                tags=tags,
                filters_applied={
                    "search_query": filter_request.search_query,
                    "category": filter_request.category,
                    "tags": filter_request.tags,
                    "date_from": filter_request.date_from,
                    "date_to": filter_request.date_to
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to list notes: {e}")
            raise Exception(f"Failed to list notes: {str(e)}")
    
    async def export_notes_to_zip(self, note_ids: Optional[List[str]] = None) -> bytes:
        """Export notes to a ZIP file"""
        try:
            logger.info(f"🗒️ Exporting notes to ZIP")
            
            # Get notes to export
            if note_ids:
                # Export specific notes (ensure they belong to current user)
                placeholders = ",".join([f"${i+2}" for i in range(len(note_ids))])
                query = f"""
                    SELECT note_id, title, content, note_date, tags, category, created_at
                    FROM free_form_notes
                    WHERE user_id = $1 AND note_id IN ({placeholders})
                    ORDER BY created_at
                """
                notes_result = await self.document_repository.execute_query(query, self.current_user_id, *note_ids, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            else:
                # Export all notes for current user
                query = """
                    SELECT note_id, title, content, note_date, tags, category, created_at
                    FROM free_form_notes
                    WHERE user_id = $1
                    ORDER BY created_at
                """
                notes_result = await self.document_repository.execute_query(query, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            
            if not notes_result:
                raise Exception("No notes found to export")
            
            # Create ZIP file in memory
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add metadata file
                metadata = {
                    "export_date": datetime.utcnow().isoformat(),
                    "export_type": "free_form_notes",
                    "total_notes": len(notes_result),
                    "exported_by": "Plato Knowledge Base"
                }
                
                zip_file.writestr("export_metadata.json", str(metadata))
                
                # Add each note as a text file
                for note in notes_result:
                    # Create filename (sanitize title)
                    safe_title = "".join(c for c in note['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_title = safe_title[:50]  # Limit length
                    filename = f"{safe_title}_{note['note_id'][:8]}.txt"
                    
                    # Create note content
                    note_content = f"""Title: {note['title']}
Date: {note['note_date'] or 'Not specified'}
Category: {note['category'] or 'Uncategorized'}
Tags: {', '.join(note['tags'] or [])}
Created: {note['created_at']}
Note ID: {note['note_id']}

{'-' * 50}

{note['content']}
"""
                    
                    zip_file.writestr(filename, note_content)
                
                # Add summary file
                summary_content = f"""Free-form Notes Export Summary
Export Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
Total Notes: {len(notes_result)}

Notes Included:
"""
                
                for note in notes_result:
                    summary_content += f"- {note['title']} (ID: {note['note_id'][:8]})\n"
                
                zip_file.writestr("export_summary.txt", summary_content)
            
            zip_buffer.seek(0)
            zip_data = zip_buffer.getvalue()
            
            logger.info(f"✅ Exported {len(notes_result)} notes to ZIP ({len(zip_data)} bytes)")
            
            return zip_data
            
        except Exception as e:
            logger.error(f"❌ Failed to export notes to ZIP: {e}")
            raise Exception(f"Failed to export notes: {str(e)}")
    
    async def search_notes_for_query(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search notes using vector similarity for integration with main search"""
        try:
            # Use embedding manager to search for notes
            # Notes are stored with document_id = "note:{note_id}"
            results = await self.embedding_manager.search_similar(
                query_text=query,
                limit=limit * 2,  # Get more to filter for notes
                score_threshold=0.3,
                use_query_expansion=False
            )
            
            # Filter for note results only
            note_results = []
            for result in results:
                if result['document_id'].startswith('note:'):
                    note_id = result['document_id'][5:]  # Remove 'note:' prefix
                    
                    # Get note details
                    note_info = await self.get_note(note_id)
                    if note_info:
                        note_results.append({
                            'chunk_id': result['chunk_id'],
                            'document_id': result['document_id'],
                            'content': result['content'],
                            'score': result['score'],
                            'note_info': note_info,
                            'source_type': 'free_form_note'
                        })
                
                if len(note_results) >= limit:
                    break
            
            return note_results
            
        except Exception as e:
            logger.error(f"❌ Failed to search notes: {e}")
            return []
    
    async def _process_note_for_embedding(self, note_id: str, title: str, content: str):
        """Process note for embedding integration (called asynchronously)"""
        try:
            logger.info(f"🗒️ Processing note for embedding: {note_id}")
            
            # Create combined content for embedding
            combined_content = f"Title: {title}\n\nContent: {content}"
            
            # Create chunks (notes are typically short, so one chunk is usually enough)
            chunks = []
            
            # If content is long, split it into chunks
            max_chunk_size = 1000
            if len(combined_content) <= max_chunk_size:
                chunks.append({
                    'chunk_id': f"note_{note_id}_chunk_0",
                    'content': combined_content,
                    'chunk_index': 0,
                    'metadata': {
                        'source_type': 'free_form_note',
                        'note_id': note_id,
                        'title': title
                    }
                })
            else:
                # Split into multiple chunks
                words = combined_content.split()
                chunk_words = []
                chunk_index = 0
                
                for word in words:
                    chunk_words.append(word)
                    current_chunk = ' '.join(chunk_words)
                    
                    if len(current_chunk) >= max_chunk_size:
                        chunks.append({
                            'chunk_id': f"note_{note_id}_chunk_{chunk_index}",
                            'content': current_chunk,
                            'chunk_index': chunk_index,
                            'metadata': {
                                'source_type': 'free_form_note',
                                'note_id': note_id,
                                'title': title
                            }
                        })
                        chunk_words = []
                        chunk_index += 1
                
                # Add remaining words as final chunk
                if chunk_words:
                    chunks.append({
                        'chunk_id': f"note_{note_id}_chunk_{chunk_index}",
                        'content': ' '.join(chunk_words),
                        'chunk_index': chunk_index,
                        'metadata': {
                            'source_type': 'free_form_note',
                            'note_id': note_id,
                            'title': title
                        }
                    })
            
            # Store chunks in vector database with note: prefix for document_id
            document_id = f"note:{note_id}"
            
            # Remove existing embeddings for this note first
            await self._remove_note_from_embedding(note_id)
            
            # Add new embeddings with enhanced metadata
            # Get note details for metadata enhancement
            note_details = await self.get_note(note_id)
            
            for chunk in chunks:
                # Enhance metadata with note-specific information
                enhanced_metadata = chunk['metadata'].copy()
                if note_details:
                    enhanced_metadata.update({
                        'category': note_details.category,
                        'tags': note_details.tags,
                        'created_at': note_details.created_at.isoformat() if note_details.created_at else None,
                        'note_date': note_details.note_date.isoformat() if note_details.note_date else None,
                        'document_type': 'note',
                        'title': note_details.title
                    })
                
                await self.embedding_manager.store_chunk_embedding(
                    chunk_id=chunk['chunk_id'],
                    document_id=document_id,
                    content=chunk['content'],
                    metadata=enhanced_metadata
                )
            
            # Mark note as processed
            update_query = """
                UPDATE free_form_notes
                SET embedding_processed = TRUE
                WHERE note_id = $1 AND user_id = $2
            """
            await self.document_repository.execute_query(update_query, note_id, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            
            logger.info(f"✅ Note {note_id} processed for embedding ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to process note {note_id} for embedding: {e}")
    
    async def _remove_note_from_embedding(self, note_id: str):
        """Remove note embeddings from vector database"""
        try:
            document_id = f"note:{note_id}"
            
            # Delete from vector database
            await self.embedding_manager.delete_document_embeddings(document_id)
            
            logger.info(f"🗑️ Removed embeddings for note: {note_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to remove embeddings for note {note_id}: {e}")
    
    async def _get_category_counts(self) -> Dict[str, int]:
        """Get count of notes by category"""
        try:
            query = """
                SELECT category, COUNT(*) as count
                FROM free_form_notes
                WHERE user_id = $1 AND category IS NOT NULL
                GROUP BY category
                ORDER BY count DESC
            """
            
            result = await self.document_repository.execute_query(query, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            
            return {row['category']: row['count'] for row in result or []}
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get category counts: {e}")
            return {}
    
    async def _get_tag_counts(self) -> Dict[str, int]:
        """Get count of notes by tag"""
        try:
            query = """
                SELECT unnest(tags) as tag, COUNT(*) as count
                FROM free_form_notes
                WHERE user_id = $1 AND tags IS NOT NULL AND array_length(tags, 1) > 0
                GROUP BY tag
                ORDER BY count DESC
                LIMIT 50
            """
            
            result = await self.document_repository.execute_query(query, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            
            return {row['tag']: row['count'] for row in result or []}
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get tag counts: {e}")
            return {}
    
    async def get_notes(self, filter_request: FreeFormNoteFilterRequest) -> FreeFormNotesListResponse:
        """Alias for list_notes to match main.py expectations"""
        return await self.list_notes(filter_request)
    
    async def get_note_tags(self) -> List[str]:
        """Get all unique note tags"""
        try:
            query = """
                SELECT DISTINCT unnest(tags) as tag
                FROM free_form_notes
                WHERE user_id = $1 AND tags IS NOT NULL AND array_length(tags, 1) > 0
                ORDER BY tag
            """
            
            result = await self.document_repository.execute_query(query, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            return [row['tag'] for row in result or []]
            
        except Exception as e:
            logger.error(f"❌ Failed to get note tags: {e}")
            raise Exception(f"Failed to get note tags: {str(e)}")
    
    async def get_note_categories(self) -> List[str]:
        """Get all unique note categories"""
        try:
            query = """
                SELECT DISTINCT category
                FROM free_form_notes
                WHERE user_id = $1 AND category IS NOT NULL
                ORDER BY category
            """
            
            result = await self.document_repository.execute_query(query, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            return [row['category'] for row in result or []]
            
        except Exception as e:
            logger.error(f"❌ Failed to get note categories: {e}")
            raise Exception(f"Failed to get note categories: {str(e)}")
    
    async def get_categories_and_tags(self) -> Dict[str, Any]:
        """Get categories and tags for the frontend"""
        try:
            categories = await self.get_note_categories()
            tags = await self.get_note_tags()
            
            return {
                "categories": categories,
                "tags": tags
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get categories and tags: {e}")
            raise Exception(f"Failed to get categories and tags: {str(e)}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get notes statistics"""
        try:
            # Get total count
            total_query = "SELECT COUNT(*) as total FROM free_form_notes WHERE user_id = $1"
            total_result = await self.document_repository.execute_query(total_query, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            total = total_result[0]['total'] if total_result else 0
            
            # Get category counts
            categories = await self._get_category_counts()
            
            # Get tag counts  
            tags = await self._get_tag_counts()
            
            # Get recent notes count (this month and week)
            recent_query = """
                SELECT 
                    SUM(CASE WHEN created_at >= date_trunc('month', CURRENT_DATE) THEN 1 ELSE 0 END) as this_month,
                    SUM(CASE WHEN created_at >= date_trunc('week', CURRENT_DATE) THEN 1 ELSE 0 END) as this_week
                FROM free_form_notes
                WHERE user_id = $1
            """
            recent_result = await self.document_repository.execute_query(recent_query, self.current_user_id, rls_context={'user_id': self.current_user_id, 'user_role': 'user'})
            recent = recent_result[0] if recent_result else {"this_month": 0, "this_week": 0}
            
            return {
                "total_notes": total,
                "notes_by_category": categories,
                "notes_by_tag": tags,
                "notes_this_month": recent.get("this_month", 0),
                "notes_this_week": recent.get("this_week", 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            raise Exception(f"Failed to get statistics: {str(e)}")
    
    async def search_notes(self, search_request: dict) -> Dict[str, Any]:
        """Search notes using vector similarity"""
        try:
            query = search_request.get("query", "")
            max_results = search_request.get("max_results", 10)
            similarity_threshold = search_request.get("similarity_threshold", 0.7)
            
            if not query:
                return {"notes": [], "total": 0}
            
            # Search using vector similarity
            note_results = await self.search_notes_for_query(query, max_results)
            
            # Convert to response format
            notes = []
            for result in note_results:
                note_info = result.get("note_info")
                if note_info:
                    notes.append({
                        "note_id": note_info.note_id,
                        "title": note_info.title,
                        "content": note_info.content,
                        "note_date": note_info.note_date,
                        "tags": note_info.tags,
                        "category": note_info.category,
                        "created_at": note_info.created_at,
                        "updated_at": note_info.updated_at,
                        "relevance_score": result.get("score", 0.0)
                    })
            
            return {
                "notes": notes,
                "total": len(notes)
            }
            
        except Exception as e:
            logger.error(f"❌ Note search failed: {e}")
            raise Exception(f"Note search failed: {str(e)}")

    async def close(self):
        """Clean up resources"""
        if self.document_repository:
            await self.document_repository.close()
        if self.embedding_manager:
            await self.embedding_manager.close()
        logger.info("🗒️ Free-form Notes Service closed")
