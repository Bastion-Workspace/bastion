"""
File System Watcher Service
Monitors uploads directory for changes and syncs with database and vector store
"""

import hashlib
import logging
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from ds_config import settings
from ds_processing.ocr_in_progress import contains as ocr_in_progress_contains

logger = logging.getLogger(__name__)

# Types that must use the binary pipeline (never UTF-8 text decode).
# Align with document_service_v2._detect_document_type: all extensions except
# txt, md, org, html, htm, eml, srt, vtt.
BINARY_DOC_TYPES = frozenset({
    "pdf",
    "docx",
    "epub",
    "zip",
    "pptx",
    "mp4",
    "mkv",
    "avi",
    "mov",
    "webm",
    "image",
})


def _sha256_file(path: Path) -> Optional[str]:
    """SHA-256 hex digest of file bytes. None if missing or unreadable."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class DocumentFileHandler(FileSystemEventHandler):
    """
    Handles file system events for document files
    
    **BULLY!** This runs in a watchdog thread, so we need to bridge to the main event loop!
    """
    
    def __init__(self, document_service, folder_service, event_loop, file_watcher_service=None):
        super().__init__()
        self.document_service = document_service
        self.folder_service = folder_service
        self.event_loop = event_loop  # Reference to main event loop
        self.file_watcher_service = file_watcher_service  # Reference to FileWatcherService for admin user access
        self.processing_files = set()  # Track files currently being written by us
        self.debounce_time = 2.0  # Wait 2 seconds after last modification
        self.pending_events = {}  # File path -> (event, timestamp)
        
    def _should_ignore_path(self, path: str, is_directory: bool = False) -> bool:
        """Check if we should ignore this path
        
        **ROOSEVELT FIX:** Directories are NEVER ignored - only files are filtered!
        """
        path_lower = path.lower()
        
        # Document version snapshots live under .versions/{document_id}/ and must not
        # be treated as normal documents (no DB records, no vectorization).
        if '/.versions/' in path_lower or '\\.versions\\' in path_lower:
            return True

        # **BULLY!** Ignore sensitive system and log directories!
        # This prevents "log leakage" into the vector database
        ignored_dirs = [
            '/logs/', '\\logs\\', 
            '/processed/', '\\processed\\',
            '/node_modules/', '\\node_modules\\',
            '/.git/', '\\.git\\',
            '/.cursor/', '\\.cursor\\'
        ]
        if any(ignored in path_lower for ignored in ignored_dirs):
            return True
        
        # Teams directories are managed by the application (team_service)
        # File watcher should not process Teams files - they have access control
        if '/teams/' in path_lower or '\\teams\\' in path_lower:
            return True
        
        # web_sources is for generated images and web scraping - not tracked in document system
        if '/web_sources/' in path_lower or '\\web_sources\\' in path_lower:
            return True

        # **BULLY!** NEVER ignore other directories - only filter files!
        if is_directory:
            # Ignore hidden directories
            if '/.~' in path or '\\.~' in path:
                return True
            return False  # All other directories are important!
        
        # For files: apply normal filtering
        # Ignore hidden files and temp files
        if '/.~' in path or '\\.~' in path:
            return True
        if path_lower.endswith('.tmp') or path_lower.endswith('.swp'):
            return True
        if '~$' in path:  # Office temp files
            return True
        
        # Ignore messaging attachments - these are not documents!
        # Messaging attachments are in: uploads/messaging/{room_id}/
        if '/messaging/' in path_lower:
            return True
        
        # NEVER ignore .metadata.json files - these are image metadata sidecars
        if path_lower.endswith('.metadata.json'):
            return False  # Always process metadata sidecars
        
        # For Global/Comics: process image files so they show in UI
        # (they won't be vectorized, but users can right-click to add metadata)
        if '/global/comics' in path_lower:
            # Allow image files in Comics folder
            image_extensions = [
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico',
                '.tiff', '.tif', '.heic', '.heif'
            ]
            if any(path_lower.endswith(ext) for ext in image_extensions):
                return False  # Process images (add to database, visible in UI)
        
        # For team posts: allow text/document files but ignore images
        # Team post attachments are in: uploads/Teams/{team_id}/posts/
        if '/teams/' in path_lower and '/posts/' in path_lower:
            # Allow text and document files to be vectorized
            text_doc_extensions = [
                '.md', '.org', '.txt', '.pdf', '.docx', '.pptx', '.html', '.htm', '.epub',
                '.csv', '.json', '.xml', '.rtf', '.odt'
            ]
            if any(path_lower.endswith(ext) for ext in text_doc_extensions):
                return False  # Process these files
            
            # Allow image files in team posts (so they show in UI)
            image_extensions = [
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico',
                '.tiff', '.tif', '.heic', '.heif'
            ]
            if any(path_lower.endswith(ext) for ext in image_extensions):
                return False  # Process images (add to database, visible in UI)
            
            # Ignore other non-text files in team posts
            return True
            
        # Ignore non-document files (for other locations)
        valid_extensions = [
            '.md', '.org', '.txt', '.pdf', '.docx', '.pptx', '.html', '.htm', '.epub',  # Documents
            '.srt', '.vtt',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'  # Images
        ]
        if not any(path_lower.endswith(ext) for ext in valid_extensions):
            return True
            
        # Ignore if it's in our processing set
        if path in self.processing_files:
            return True
            
        return False
    
    async def _get_document_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Find document record by matching file path
        Maps: uploads/Users/{username}/{folder}/{filename} -> document_id
        
        **ROOSEVELT DUPLICATE DETECTIVE!** 🔍
        """
        try:
            path = Path(file_path)
            filename = path.name
            
            # If this is a metadata sidecar file, resolve lookup filename
            if filename.lower().endswith('.metadata.json'):
                try:
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    schema_type = (metadata.get('schema_type') or '').strip().lower()
                    if schema_type == 'document':
                        doc_filename = metadata.get('document_filename', '')
                        if doc_filename:
                            filename = doc_filename
                            logger.debug(
                                "Document metadata sidecar; lookup by document filename: %s", filename
                            )
                        else:
                            logger.warning(
                                "No document_filename in document metadata sidecar: %s", file_path
                            )
                    else:
                        filename = path.name
                        logger.debug(
                            "Image metadata sidecar; lookup by sidecar filename: %s", filename
                        )
                except Exception as e:
                    logger.warning("Failed to read metadata sidecar for lookup: %s: %s", file_path, e)
            
            # Parse the path to extract user context
            parts = path.parts
            uploads_idx = -1
            for i, part in enumerate(parts):
                if part == 'uploads':
                    uploads_idx = i
                    break
            
            if uploads_idx == -1:
                logger.warning(f"⚠️ File path doesn't contain 'uploads': {file_path}")
                return None
            
            # Determine collection type and context
            if uploads_idx + 1 < len(parts):
                collection_dir = parts[uploads_idx + 1]
                
                if collection_dir == 'Users' and uploads_idx + 2 < len(parts):
                    # User file: uploads/Users/{username}/{folders...}/{filename}
                    username = parts[uploads_idx + 2]
                    collection_type = 'user'
                    
                    # Get user_id from username
                    user_id = await self._get_user_id_from_username(username)
                    if not user_id:
                        logger.warning(f"⚠️ Could not find user_id for username: {username}")
                        return None
                    
                    # **BULLY!** Get the DEEPEST folder_id from the FULL folder hierarchy!
                    folder_id = None
                    folder_start_idx = uploads_idx + 3
                    folder_end_idx = len(parts) - 1  # Exclude filename
                    
                    if folder_start_idx < folder_end_idx:
                        # We have folders in the path - resolve to the deepest one
                        folder_parts = parts[folder_start_idx:folder_end_idx]
                        folder_id = await self._resolve_deepest_folder_id(folder_parts, user_id, collection_type)
                    
                elif collection_dir == 'Global':
                    # Global file: uploads/Global/{folders...}/{filename}
                    collection_type = 'global'
                    user_id = None
                    
                    # **BULLY!** Get the DEEPEST folder_id from the FULL folder hierarchy!
                    folder_id = None
                    folder_start_idx = uploads_idx + 2
                    folder_end_idx = len(parts) - 1  # Exclude filename
                    
                    if folder_start_idx < folder_end_idx:
                        # We have folders in the path - resolve to the deepest one
                        folder_parts = parts[folder_start_idx:folder_end_idx]
                        folder_id = await self._resolve_deepest_folder_id(folder_parts, None, collection_type)
                elif collection_dir == 'Teams':
                    # Teams files are managed by the application (team_service)
                    # File watcher should not process them - they have access control
                    logger.debug(f"⏭️  Skipping Teams file (managed by application): {file_path}")
                    return None
                else:
                    logger.warning(f"⚠️ Unknown collection directory: {collection_dir}")
                    return None
            else:
                logger.warning(f"⚠️ Invalid path structure: {file_path}")
                return None
            
            # Find document by filename, user_id, and folder_id
            document = await self.document_service.document_repository.find_by_filename_and_context(
                filename=filename,
                user_id=user_id,
                collection_type=collection_type,
                folder_id=folder_id
            )
            
            if document:
                return {
                    'document_id': document.document_id,
                    'filename': document.filename,
                    'user_id': user_id,
                    'collection_type': collection_type,
                    'folder_id': folder_id
                }
            else:
                logger.info(f"❌ NOT FOUND: No existing record for {filename} (will create new)")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error finding document by path: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _get_user_id_from_username(self, username: str) -> Optional[str]:
        """Get user_id from username"""
        try:
            # Use a synchronous database query to get user_id
            import psycopg2
            from ds_config import settings
            
            conn = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB
            )
            
            try:
                cur = conn.cursor()
                cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
                row = cur.fetchone()
                
                if row:
                    return row[0]
                else:
                    return None
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"❌ Error getting user_id for username {username}: {e}")
            return None
    
    async def _get_folder_id(self, folder_name: str, user_id: Optional[str], collection_type: str) -> Optional[str]:
        """Get folder_id from folder name"""
        try:
            folders = await self.folder_service.get_folder_tree(user_id, collection_type)
            
            for folder in folders:
                if folder.name == folder_name:
                    return folder.folder_id
            
            return None
                
        except Exception as e:
            logger.error(f"❌ Error getting folder_id for {folder_name}: {e}")
            return None
    
    async def _resolve_deepest_folder_id(self, folder_parts: tuple, user_id: Optional[str], collection_type: str) -> Optional[str]:
        """
        Resolve folder hierarchy from path parts to find the deepest folder_id
        
        **BULLY!** This matches the logic in _ensure_folder_hierarchy!
        
        Args:
            folder_parts: Tuple of folder names from path (e.g., ('Test', 'SubFolder'))
            user_id: User ID or None for global
            collection_type: 'user' or 'global'
            
        Returns:
            folder_id of the deepest folder, or None if not found
        """
        try:
            if not folder_parts:
                return None
            
            # **TRUST BUST!** get_folder_tree() returns VIRTUAL roots with real folders as children!
            # We need the RAW folders from the database, not wrapped in virtual roots!
            from ds_db.document_repository import DocumentRepository
            doc_repo = DocumentRepository()
            folders_data = await doc_repo.get_folders_by_user(user_id, collection_type)
            
            logger.debug(f"🔍 RESOLVE: Got {len(folders_data)} RAW folders from database")
            logger.debug(f"🔍 RESOLVE: Looking for path: {folder_parts}")
            
            # Build a lookup map: (name, parent_folder_id) -> folder_id
            # Use .get() since folders_data is a list of dicts, not DocumentFolder objects
            folder_map = {(f.get('name'), f.get('parent_folder_id')): f.get('folder_id') for f in folders_data}
            
            logger.debug(f"🔍 RESOLVE: Built folder map with {len(folder_map)} entries")
            logger.debug(f"🔍 RESOLVE: Available folders: {[(name, parent) for (name, parent) in folder_map.keys()]}")
            
            # Walk the folder hierarchy from root to deepest
            parent_folder_id = None
            current_folder_id = None
            
            for folder_name in folder_parts:
                # Look for folder with this name and current parent
                key = (folder_name, parent_folder_id)
                logger.debug(f"🔍 RESOLVE: Looking for folder '{folder_name}' with parent {parent_folder_id}")
                
                if key in folder_map:
                    current_folder_id = folder_map[key]
                    parent_folder_id = current_folder_id  # Next level uses this as parent
                    logger.debug(f"✅ RESOLVE: Found folder '{folder_name}' → {current_folder_id}")
                else:
                    # Folder doesn't exist in hierarchy - return None
                    logger.debug(f"❌ RESOLVE: Folder '{folder_name}' NOT found under parent {parent_folder_id}")
                    logger.debug(f"❌ RESOLVE: This file's folder doesn't exist in DB - cannot find existing document!")
                    return None
            
            return current_folder_id
            
        except Exception as e:
            logger.error(f"❌ Error resolving folder hierarchy: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _resolve_deepest_folder_id_for_team(self, folder_parts: tuple, team_id: str, team_creator_id: str) -> Optional[str]:
        """
        Resolve folder hierarchy for team folders from path parts to find the deepest folder_id
        
        Args:
            folder_parts: Tuple of folder names from path (e.g., ('Team Documents', 'SubFolder'))
            team_id: Team ID
            team_creator_id: User ID of the team creator (for RLS context)
            
        Returns:
            folder_id of the deepest folder, or None if not found
        """
        try:
            if not folder_parts:
                return None
            
            # Get team folders from database using database helper
            from ds_db.database_manager.database_helpers import fetch_all
            
            # Use team creator's context for RLS (they have admin rights to see all team folders)
            folders_data = await fetch_all(
                "SELECT folder_id, name, parent_folder_id FROM document_folders WHERE team_id = $1",
                team_id,
                rls_context={'user_id': team_creator_id, 'user_role': 'admin'}
            )
            
            logger.debug(f"🔍 RESOLVE TEAM: Got {len(folders_data)} team folders from database for team {team_id}")
            logger.debug(f"🔍 RESOLVE TEAM: Looking for path: {folder_parts}")
            
            # Build a lookup map: (name, parent_folder_id) -> folder_id
            folder_map = {}
            for f in folders_data:
                name = f.get('name')
                parent_id = f.get('parent_folder_id')
                folder_id = f.get('folder_id')
                folder_map[(name, parent_id)] = folder_id
            
            logger.debug(f"🔍 RESOLVE TEAM: Built folder map with {len(folder_map)} entries")
            
            # Walk the folder hierarchy from root to deepest
            parent_folder_id = None
            current_folder_id = None
            
            for folder_name in folder_parts:
                # Look for folder with this name and current parent
                key = (folder_name, parent_folder_id)
                logger.debug(f"🔍 RESOLVE TEAM: Looking for folder '{folder_name}' with parent {parent_folder_id}")
                
                if key in folder_map:
                    current_folder_id = folder_map[key]
                    parent_folder_id = current_folder_id  # Next level uses this as parent
                    logger.debug(f"✅ RESOLVE TEAM: Found folder '{folder_name}' → {current_folder_id}")
                else:
                    # Folder doesn't exist in hierarchy - return None
                    logger.debug(f"❌ RESOLVE TEAM: Folder '{folder_name}' NOT found under parent {parent_folder_id}")
                    return None
            
            return current_folder_id
            
        except Exception as e:
            logger.error(f"❌ Error resolving team folder hierarchy: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification"""
        if event.is_directory or self._should_ignore_path(event.src_path, event.is_directory):
            return
        
        # Debounce: Wait for file to stop being modified
        self.pending_events[event.src_path] = (event, time.time())
        logger.debug(f"📝 File modified (debouncing): {event.src_path}")
    
    def on_created(self, event: FileSystemEvent):
        """Handle file or folder creation"""
        if self._should_ignore_path(event.src_path, event.is_directory):
            logger.debug(f"⏭️  Ignoring path (file watcher): {event.src_path} (is_directory={event.is_directory})")
            return
        
        if event.is_directory:
            # Handle folder creation immediately (no debounce needed)
            logger.info(f"📁 Folder created: {event.src_path}")
            asyncio.run_coroutine_threadsafe(
                self._handle_folder_created(event.src_path),
                self.event_loop
            )
        else:
            # File creation - debounce to wait for complete write
            self.pending_events[event.src_path] = (event, time.time())
            logger.debug(f"📝 File created (debouncing): {event.src_path}")
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file/folder deletion"""
        if self._should_ignore_path(event.src_path, event.is_directory):
            return
        
        if event.is_directory:
            logger.info(f"🗑️ Folder deleted: {event.src_path}")
            # Schedule coroutine in main event loop from watchdog thread
            asyncio.run_coroutine_threadsafe(
                self._handle_folder_deleted(event.src_path),
                self.event_loop
            )
        else:
            logger.info(f"🗑️ File deleted: {event.src_path}")
            # Schedule coroutine in main event loop from watchdog thread
            asyncio.run_coroutine_threadsafe(
                self._handle_file_deleted(event.src_path),
                self.event_loop
            )
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file or folder move/rename"""
        if self._should_ignore_path(event.dest_path, event.is_directory):
            return
        
        if event.is_directory:
            # **BULLY!** Handle folder move/rename
            logger.info(f"📦 Folder moved: {event.src_path} -> {event.dest_path}")
            asyncio.run_coroutine_threadsafe(
                self._handle_folder_moved(event.src_path, event.dest_path),
                self.event_loop
            )
        else:
            # File move/rename
            logger.info(f"📦 File moved: {event.src_path} -> {event.dest_path}")
            asyncio.run_coroutine_threadsafe(
                self._handle_file_moved(event.src_path, event.dest_path),
                self.event_loop
            )
    
    async def _handle_file_modified(self, file_path: str):
        """Process file modification"""
        try:
            logger.info(f"Processing modified file: {file_path}")

            if ocr_in_progress_contains(file_path):
                logger.debug(f"Skipping modified file (OCR in progress): {file_path}")
                return

            path = Path(file_path)
            if path.name.lower().endswith('.metadata.json'):
                logger.info(f"Modified file is image metadata sidecar - routing to ImageSidecarService")
                await self._handle_new_file(file_path)
                return

            doc_info = await self._get_document_by_path(file_path)

            if not doc_info:
                logger.info(f"Modified file has no document record, treating as new: {file_path}")
                await self._handle_new_file(file_path)
                return

            # Avoid overlapping in-place reprocessing with startup scan / _handle_new_file on same path
            if file_path in self.processing_files:
                logger.debug(
                    "Skipping modified reprocess; path already in flight: %s", file_path
                )
                return
            self.processing_files.add(file_path)
            try:
                document_id = doc_info['document_id']
                user_id = doc_info['user_id']
                filename = doc_info.get('filename') or path.name
                doc_type = self.document_service._detect_document_type(filename)

                # Skip full reprocess when only metadata changed (chmod/mtime/touch) and
                # DB still has a file_hash matching on-disk bytes (same size).
                full_meta = await self.document_service.get_document(document_id)
                try:
                    st_size = path.stat().st_size if path.is_file() else None
                except OSError:
                    st_size = None
                if (
                    full_meta
                    and full_meta.file_hash
                    and st_size is not None
                    and full_meta.file_size == st_size
                ):
                    disk_hash = _sha256_file(path)
                    if disk_hash and disk_hash == full_meta.file_hash:
                        logger.debug(
                            "Skipping reprocess; on-disk content matches stored file_hash: %s %s",
                            document_id,
                            file_path,
                        )
                        return

                if doc_type in BINARY_DOC_TYPES:
                    logger.info(f"Re-processing binary document {document_id} due to file change (full pipeline)")
                    from ds_models.api_models import ProcessingStatus
                    await self.document_service.document_repository.update_status(
                        document_id, ProcessingStatus.PROCESSING
                    )
                    await self.document_service._emit_document_status_update(
                        document_id, "processing", user_id
                    )
                    await self.document_service._process_document_async(
                        document_id, path, doc_type, user_id
                    )
                    return

                logger.info(f"Re-processing document {document_id} due to file change")

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Import ProcessingStatus
                from ds_models.api_models import ProcessingStatus
                
                # Update status to PROCESSING and notify UI
                await self.document_service.document_repository.update_status(
                    document_id, ProcessingStatus.PROCESSING
                )
                await self.document_service._emit_document_status_update(
                    document_id, "processing", user_id
                )
                
                # Re-process chunks and embeddings
                await self.document_service.document_repository.update_status(
                    document_id, ProcessingStatus.EMBEDDING
                )
                await self.document_service._emit_document_status_update(
                    document_id, "embedding", user_id
                )
                
                chunks = await self.document_service.document_processor.process_text_content(
                    content, document_id, {}
                )
                
                # **BULLY!** Skip vectorization for .org files - they use specialized org-mode search!
                if file_path.lower().endswith('.org'):
                    logger.info(f"📋 Skipping vectorization for org file: {file_path} (uses org-mode search)")
                    # Still process chunks for metadata, but don't embed
                    if chunks:
                        logger.info(f"✅ Re-processed {len(chunks)} chunks (no embedding) for org document {document_id}")
                # Check if document is exempt from vectorization
                elif await self.document_service.document_repository.is_document_exempt(document_id, user_id):
                    logger.info(f"🚫 Document {document_id} is exempt from vectorization - skipping embedding and KG extraction")
                elif chunks:
                    # **ROOSEVELT'S COMPLETE CLEANUP!** Delete old vectors AND knowledge graph entities
                    await self.document_service.embedding_manager.delete_document_chunks(document_id, user_id)
                    
                    # Delete old knowledge graph entities
                    if self.document_service.kg_service:
                        try:
                            await self.document_service.kg_service.delete_document_entities(document_id)
                            logger.info(f"🗑️  Deleted old knowledge graph entities for {document_id}")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to delete old KG entities for {document_id}: {e}")
                    
                    # Fetch document metadata for embedding storage
                    document_category = None
                    document_tags = None
                    document_title = None
                    document_author = None
                    document_filename = None
                    try:
                        doc_info = await self.document_service.get_document(document_id)
                        if doc_info:
                            document_category = doc_info.category.value if doc_info.category else None
                            document_tags = doc_info.tags if doc_info.tags else None
                            document_title = doc_info.title if doc_info.title else None
                            document_author = doc_info.author if doc_info.author else None
                            document_filename = doc_info.filename if doc_info.filename else None
                    except Exception as e:
                        logger.warning(f"⚠️  Could not fetch document metadata for {document_id}: {e}")
                    
                    # Store new embeddings with metadata
                    await self.document_service.embedding_manager.embed_and_store_chunks(
                        chunks,
                        user_id=user_id,
                        document_category=document_category,
                        document_tags=document_tags,
                        document_title=document_title,
                        document_author=document_author,
                        document_filename=document_filename
                    )
                    logger.info(f"✅ Re-processed {len(chunks)} chunks for document {document_id}")
                    
                    # **BY GEORGE!** Extract and store NEW entities using PROPER spaCy NER
                    if self.document_service.kg_service:
                        try:
                            # Use DocumentProcessor's sophisticated entity extraction (spaCy + patterns)
                            entities = await self.document_service.document_processor._extract_entities(content, chunks)
                            if entities:
                                await self.document_service.kg_service.store_entities(entities, document_id, chunks)
                                logger.info(f"🔗 Extracted and stored {len(entities)} NEW entities using spaCy NER for {document_id}")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to extract/store new entities for {document_id}: {e}")
                        
                        # **BULLY! ENTERTAINMENT KG EXTRACTION!** 🎬
                        # If this is an entertainment document, extract domain-specific relationships
                        try:
                            from shims.services.entertainment_kg_extractor import (
                                get_entertainment_kg_extractor,
                            )
                            ent_extractor = get_entertainment_kg_extractor()
                            
                            # Get document info for tag checking
                            doc_info = await self.document_service.get_document(document_id)
                            
                            if doc_info and ent_extractor.should_extract_from_document(doc_info):
                                logger.info(f"🎬 Extracting entertainment entities from {document_id}")
                                ent_entities, ent_relationships = ent_extractor.extract_entities_and_relationships(
                                    content, doc_info
                                )
                                
                                if ent_entities or ent_relationships:
                                    await self.document_service.kg_service.store_entertainment_entities_and_relationships(
                                        ent_entities, ent_relationships, document_id
                                    )
                                    logger.info(f"🎬 Stored entertainment graph: {len(ent_entities)} entities, {len(ent_relationships)} relationships")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to extract/store entertainment entities for {document_id}: {e}")
                
                # Update status to COMPLETED and notify UI
                await self.document_service.document_repository.update_status(
                    document_id, ProcessingStatus.COMPLETED
                )
                await self.document_service._emit_document_status_update(
                    document_id, "completed", user_id
                )
                
                logger.info(f"✅ File sync complete: {file_path}")
            finally:
                self.processing_files.discard(file_path)
            
        except Exception as e:
            logger.error(f"❌ Error handling file modification: {e}")
    
    async def _handle_file_deleted(self, file_path: str):
        """Process file deletion
        
        Handles file deletions detected by the file system watcher.
        This may be called when:
        1. File is deleted externally (outside the API) - need to clean up DB record
        2. File is deleted by API - document may already be deleted, so check first
        """
        try:
            logger.info(f"🗑️ Processing deleted file: {file_path}")
            
            # Check for and delete orphaned sidecar metadata file (stem: image.jpg -> image.metadata.json)
            from pathlib import Path
            file_path_obj = Path(file_path)
            sidecar_path = file_path_obj.parent / f"{file_path_obj.stem}.metadata.json"
            if sidecar_path.exists():
                try:
                    sidecar_path.unlink()
                    logger.info(f"🗑️ Deleted orphaned sidecar metadata: {sidecar_path}")
                except Exception as sidecar_e:
                    logger.warning(f"⚠️ Failed to delete orphaned sidecar {sidecar_path}: {sidecar_e}")
            
            # Find document record
            doc_info = await self._get_document_by_path(file_path)
            
            if not doc_info:
                logger.debug(f"📄 Deleted file has no document record: {file_path}")
                return
            
            document_id = doc_info['document_id']
            user_id = doc_info['user_id']
            folder_id = doc_info['folder_id']
            
            # Verify document still exists in database before attempting deletion
            # This handles the case where the API already deleted it
            existing_doc = await self.document_service.document_repository.get_by_id(document_id)
            if not existing_doc:
                logger.info(f"📄 Document {document_id} already deleted from database (likely by API) - skipping")
                return
            
            logger.info(f"🗑️ Deleting document record for {document_id} (file was deleted)")
            
            # Emit WebSocket notification BEFORE deletion so we can include folder_id
            try:
                from ds_services.websocket_notifier import WebSocketNotifier
                from utils.websocket_manager import get_websocket_manager
                
                ws_manager = get_websocket_manager()
                ws_notifier = WebSocketNotifier(ws_manager)
                
                # Send file_deleted event with folder context
                await ws_notifier.notify_file_deleted(
                    document_id=document_id,
                    folder_id=folder_id,
                    user_id=user_id
                )
                logger.info(f"📡 Sent deletion notification for document: {document_id} in folder: {folder_id}")
            except Exception as e:
                logger.error(f"❌ Failed to send deletion notification: {e}")
            
            # Full cleanup: metadata + image embeddings (face/object), DB rows, document record (file already gone)
            success = await self.document_service.delete_document_database_only(document_id)
            if success:
                logger.info(f"✅ Document deleted from system: {document_id}")
            else:
                logger.warning(f"⚠️ Document {document_id} may have already been deleted")
            
        except Exception as e:
            logger.error(f"❌ Error handling file deletion: {e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_folder_deleted(self, folder_path: str):
        """Process folder deletion
        
        **ROOSEVELT'S FOLDER DELETION CAVALRY!** 🗑️
        Delete folder and all its contents from the database
        """
        try:
            from pathlib import Path
            logger.info(f"🗑️ Processing deleted folder: {folder_path}")
            
            # Parse path to get folder context
            path = Path(folder_path)
            parts = path.parts
            
            uploads_idx = -1
            for i, part in enumerate(parts):
                if part == 'uploads':
                    uploads_idx = i
                    break
            
            if uploads_idx == -1:
                logger.warning(f"⚠️ Folder path doesn't contain 'uploads': {folder_path}")
                return
            
            # Determine collection type and context
            if uploads_idx + 1 >= len(parts):
                logger.warning(f"⚠️ Invalid folder path structure: {folder_path}")
                return
            
            collection_dir = parts[uploads_idx + 1]
            
            if collection_dir == 'Users' and uploads_idx + 2 < len(parts):
                # User folder: uploads/Users/{username}/{folders...}
                username = parts[uploads_idx + 2]
                collection_type = 'user'
                
                # Get user_id from username
                user_id = await self._get_user_id_from_username(username)
                if not user_id:
                    logger.warning(f"⚠️ Could not find user_id for username: {username}")
                    return
                
                # Get folder path parts (everything after username)
                folder_start_idx = uploads_idx + 3
                folder_parts = parts[folder_start_idx:]
                
                logger.info(f"📁 Deleted folder path: {' -> '.join(folder_parts)}")
                
                # Resolve to folder_id
                if folder_parts:
                    folder_id = await self._resolve_deepest_folder_id(tuple(folder_parts), user_id, collection_type)
                else:
                    # Root user folder deleted? This shouldn't happen
                    logger.warning(f"⚠️ Root user folder deleted - skipping")
                    return
                    
            elif collection_dir == 'Global':
                # Global folder: uploads/Global/{folders...}
                collection_type = 'global'
                user_id = None
                
                folder_start_idx = uploads_idx + 2
                folder_parts = parts[folder_start_idx:]
                
                logger.info(f"📁 Deleted global folder path: {' -> '.join(folder_parts)}")
                
                if folder_parts:
                    folder_id = await self._resolve_deepest_folder_id(tuple(folder_parts), None, collection_type)
                else:
                    logger.warning(f"⚠️ Root global folder deleted - skipping")
                    return
            
            elif collection_dir == 'Teams':
                # Teams folders are managed by the application (team_service)
                # File watcher should not process them - they have access control
                if uploads_idx + 2 < len(parts):
                    # Team folder: uploads/Teams/{team_id}/documents/{folders...}
                    collection_type = 'team'
                    team_id = parts[uploads_idx + 2]
                    
                    # Get the team creator directly from database for RLS context
                    try:
                        from ds_db.database_manager.database_helpers import fetch_one
                        team = await fetch_one(
                            "SELECT team_id, created_by FROM teams WHERE team_id = $1",
                            team_id,
                            rls_context={'user_id': '', 'user_role': 'admin'}  # Admin context to bypass RLS for team lookup
                        )
                        if not team:
                            logger.info(f"⏭️  Team not found for deletion: {team_id} (managed by application)")
                            return
                        # Use team creator as user_id for RLS context (they have admin rights)
                        user_id = team.get('created_by')
                        if not user_id:
                            logger.info(f"⏭️  Team {team_id} has no creator recorded (managed by application)")
                            return
                    except Exception as e:
                        logger.info(f"⏭️  Could not find team creator for {team_id}: {e} (managed by application)")
                        return
                    
                    # Skip if this is just the team_id directory or documents directory
                    if uploads_idx + 3 >= len(parts):
                        logger.info(f"⏭️  Team root directory - no folder record to delete (managed by application)")
                        return
                    
                    documents_dir = parts[uploads_idx + 3]
                    if documents_dir != 'documents':
                        logger.info(f"⏭️  Skipping non-documents team folder: {folder_path} (managed by application)")
                        return
                    
                    # Get folder path parts (everything after documents/)
                    folder_start_idx = uploads_idx + 4
                    folder_parts = parts[folder_start_idx:]
                    
                    if not folder_parts:
                        logger.info(f"⏭️  Team documents root - no folder record to delete (managed by application)")
                        return
                    
                    logger.info(f"📁 Deleted team folder path: {' -> '.join(folder_parts)}")
                    
                    # Resolve to folder_id using team_id and creator
                    folder_id = await self._resolve_deepest_folder_id_for_team(tuple(folder_parts), team_id, user_id)
                else:
                    # Teams root directory - skip
                    logger.info(f"⏭️  Teams root directory - skipping (managed by application)")
                    return
            
            else:
                logger.warning(f"⚠️ Unknown collection directory: {collection_dir}")
                return
            
            if not folder_id:
                logger.warning(f"⚠️ Folder not found in database: {folder_path}")
                logger.info(f"📡 Sending generic folder tree refresh to ensure UI consistency")
                
                # **ROOSEVELT FIX:** Even if folder not in DB, tell frontend to refresh!
                # The folder might have been in the tree but not properly synced
                try:
                    from ds_services.websocket_notifier import WebSocketNotifier
                    from utils.websocket_manager import get_websocket_manager
                    
                    ws_manager = get_websocket_manager()
                    ws_notifier = WebSocketNotifier(ws_manager)
                    
                    # Send generic refresh message
                    message = {
                        "type": "folder_tree_refresh",
                        "reason": "folder_deleted_from_disk",
                        "path": str(folder_path),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if user_id:
                        await ws_manager.send_to_session(message, user_id)
                        logger.info(f"📡 Sent folder tree refresh to user {user_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to send refresh notification: {e}")
                
                return
            
            logger.info(f"🗑️ Deleting folder from database: {folder_id}")
            
            # **BULLY!** Get folder metadata BEFORE deleting for notification
            folder_name = folder_parts[-1] if folder_parts else "Unknown"
            
            # **BULLY!** Send WebSocket notification BEFORE deletion!
            try:
                from ds_services.websocket_notifier import WebSocketNotifier
                from utils.websocket_manager import get_websocket_manager
                
                ws_manager = get_websocket_manager()
                ws_notifier = WebSocketNotifier(ws_manager)
                
                # Send folder deletion event with folder data
                await ws_notifier.notify_folder_event(
                    event_type='deleted',
                    folder_data={
                        'folder_id': folder_id,
                        'name': folder_name
                    },
                    user_id=user_id
                )
                logger.info(f"📡 Sent folder deletion notification: {folder_name} ({folder_id})")
            except Exception as e:
                logger.error(f"❌ Failed to send folder deletion notification: {e}")
            
            # Delete folder from database (cascade will delete documents)
            await self.folder_service.delete_folder(folder_id, user_id, recursive=True, current_user_role='admin')
            
            logger.info(f"✅ Folder deleted from system: {folder_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling folder deletion: {e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_folder_created(self, folder_path: str):
        """Process folder creation
        
        Create folder record in database when folder is created on disk.
        Handles folders created via backend disk manipulation, WebDAV, or any filesystem operation.
        """
        try:
            from pathlib import Path
            logger.info(f"📁 Processing created folder: {folder_path}")
            
            # Verify folder still exists (might have been deleted quickly)
            path = Path(folder_path)
            if not path.exists() or not path.is_dir():
                logger.warning(f"⚠️ Folder no longer exists or is not a directory: {folder_path}")
                return
            
            # Parse path to get folder context
            path = Path(folder_path)
            parts = path.parts
            
            uploads_idx = -1
            for i, part in enumerate(parts):
                if part == 'uploads':
                    uploads_idx = i
                    break
            
            if uploads_idx == -1:
                logger.warning(f"⚠️ Folder path doesn't contain 'uploads': {folder_path}")
                return
            
            # Determine collection type and context
            if uploads_idx + 1 >= len(parts):
                logger.warning(f"⚠️ Invalid folder path structure: {folder_path}")
                return
            
            collection_dir = parts[uploads_idx + 1]
            
            if collection_dir == 'Users' and uploads_idx + 2 < len(parts):
                # User folder: uploads/Users/{username}/{folders...}
                username = parts[uploads_idx + 2]
                collection_type = 'user'
                
                # Get user_id from username
                user_id = await self._get_user_id_from_username(username)
                if not user_id:
                    logger.warning(f"⚠️ Could not find user_id for username: {username}")
                    return
                
                # Get folder path parts (everything after username)
                folder_start_idx = uploads_idx + 3
                folder_parts = parts[folder_start_idx:]
                
            elif collection_dir == 'Global':
                # Global folder: uploads/Global/{folders...}
                collection_type = 'global'
                user_id = None
                
                folder_start_idx = uploads_idx + 2
                folder_parts = parts[folder_start_idx:]
            
            elif collection_dir == 'Teams':
                # Team folders are managed by the application (team_service, folder_service)
                # File watcher just observes them - no need to recreate in database
                logger.info(f"📁 Team folder created: {folder_path} (managed by application)")
                return
            
            else:
                logger.warning(f"⚠️ Unknown collection directory: {collection_dir}")
                return
            
            if not folder_parts:
                logger.info(f"📁 Root directory - no folder record needed")
                return
            
            logger.info(f"📁 Creating folder hierarchy: {' -> '.join(folder_parts)}")
            
            # Create folder hierarchy in database
            folder_id = await self._ensure_folder_hierarchy(tuple(folder_parts), user_id, collection_type)
            
            if folder_id:
                logger.info(f"✅ Folder created in database: {folder_id}")
                
                # Send WebSocket notification
                try:
                    from ds_services.websocket_notifier import WebSocketNotifier
                    from utils.websocket_manager import get_websocket_manager
                    
                    ws_manager = get_websocket_manager()
                    ws_notifier = WebSocketNotifier(ws_manager)
                    
                    await ws_notifier.notify_folder_event(
                        event_type='created',
                        folder_data={
                            'folder_id': folder_id,
                            'name': folder_parts[-1]
                        },
                        user_id=user_id
                    )
                    logger.info(f"📡 Sent folder creation notification: {folder_parts[-1]} ({folder_id})")
                except Exception as e:
                    logger.error(f"❌ Failed to send folder creation notification: {e}")
            else:
                logger.warning(f"⚠️ Failed to create folder hierarchy in database")
            
        except Exception as e:
            logger.error(f"❌ Error handling folder creation: {e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_folder_moved(self, old_path: str, new_path: str):
        """Process folder move/rename
        
        **ROOSEVELT'S FOLDER RELOCATION CAVALRY!** 📦
        Update folder record when folder is moved or renamed on disk
        
        When a folder is moved programmatically (via API), the database is already updated.
        We just need to ensure the folder record exists at the new location.
        File events are handled separately - no need to reprocess all files.
        """
        try:
            from pathlib import Path
            logger.info(f"📦 Processing moved folder: {old_path} -> {new_path}")
            
            # Parse NEW path to verify folder exists in database
            path = Path(new_path)
            parts = path.parts
            
            uploads_idx = -1
            for i, part in enumerate(parts):
                if part == 'uploads':
                    uploads_idx = i
                    break
            
            if uploads_idx == -1:
                logger.warning(f"⚠️ Folder path doesn't contain 'uploads': {new_path}")
                return
            
            # Determine collection type
            if uploads_idx + 1 >= len(parts):
                logger.warning(f"⚠️ Invalid folder path structure: {new_path}")
                return
            
            collection_dir = parts[uploads_idx + 1]
            
            if collection_dir == 'Users' and uploads_idx + 2 < len(parts):
                username = parts[uploads_idx + 2]
                collection_type = 'user'
                
                # Get user_id from username
                user_id = await self._get_user_id_from_username(username)
                if not user_id:
                    logger.warning(f"⚠️ Could not find user_id for username: {username}")
                    return
                
                # Get folder path parts (everything after username)
                folder_start_idx = uploads_idx + 3
                folder_parts = parts[folder_start_idx:]
                
                if not folder_parts:
                    logger.info(f"📁 Root directory move - no action needed")
                    return
                
                logger.info(f"📁 Verifying folder exists at new location: {' -> '.join(folder_parts)}")
                
                # Check if folder exists at new location in database
                folder_id = await self._resolve_deepest_folder_id(tuple(folder_parts), user_id, collection_type)
                
                if folder_id:
                    logger.info(f"✅ Folder already exists in database at new location: {folder_id}")
                    # Folder is already in the correct location in DB (from programmatic move)
                    # No need to process individual files - they keep their folder_id
                else:
                    logger.warning(f"⚠️ Folder not found at new location - may be external move")
                    # This might be an external move (not via API)
                    # Handle as delete + create for external moves
                    logger.info(f"📁 Handling external folder move as delete + create")
                    await self._handle_folder_deleted(old_path)
                    await self._handle_folder_created(new_path)
            
            elif collection_dir == 'Global':
                # Similar logic for global folders
                collection_type = 'global'
                user_id = None
                
                folder_start_idx = uploads_idx + 2
                folder_parts = parts[folder_start_idx:]
                
                if not folder_parts:
                    logger.info(f"📁 Root directory move - no action needed")
                    return
                
                folder_id = await self._resolve_deepest_folder_id(tuple(folder_parts), None, collection_type)
                
                if folder_id:
                    logger.info(f"✅ Folder already exists in database at new location: {folder_id}")
                else:
                    logger.info(f"📁 Handling external folder move as delete + create")
                    await self._handle_folder_deleted(old_path)
                    await self._handle_folder_created(new_path)
            
            elif collection_dir == 'Teams':
                # Team folders are managed by application - skip file watcher processing
                logger.info(f"📁 Team folder moved: {new_path} (managed by application)")
                return
            
            else:
                logger.warning(f"⚠️ Unknown collection directory: {collection_dir}")
                return
            
        except Exception as e:
            logger.error(f"❌ Error handling folder move: {e}")
            import traceback
            traceback.print_exc()
    
    async def _handle_file_moved(self, old_path: str, new_path: str):
        """Process file move/rename"""
        try:
            logger.info(f"📦 Processing moved file: {old_path} -> {new_path}")
            
            # Find document record by old path
            doc_info = await self._get_document_by_path(old_path)
            
            if not doc_info:
                logger.debug(f"📄 Moved file has no document record: {old_path}")
                return
            
            document_id = doc_info['document_id']
            
            # Extract new filename
            new_filename = Path(new_path).name
            
            logger.info(f"📝 Updating document {document_id} filename to: {new_filename}")
            
            # Update filename in database
            await self.document_service.document_repository.update_filename(
                document_id, new_filename
            )
            
            # Get actual document ownership from database to determine correct collection
            # This ensures we update the right collection (team > user > global)
            try:
                actual_doc = await self.document_service.document_repository.get_by_id(document_id)
                if actual_doc:
                    # Get actual ownership info from document record
                    actual_user_id = getattr(actual_doc, 'user_id', None)
                    actual_team_id = getattr(actual_doc, 'team_id', None)
                    
                    # Update embedding metadata to reflect new filename
                    # Use actual document ownership, not path-derived user_id
                    await self.document_service._update_qdrant_metadata(
                        document_id=document_id,
                        document_filename=new_filename,
                        user_id=actual_user_id,
                        team_id=actual_team_id
                    )
                    logger.info(f"✅ Updated embedding metadata for moved file: {document_id} (team={actual_team_id}, user={actual_user_id})")
                else:
                    logger.warning(f"⚠️ Could not fetch document record for {document_id} - skipping vector update")
            except Exception as vector_error:
                # Non-critical: database update succeeded, vector update is optional
                logger.warning(f"⚠️ Failed to update vector metadata for moved file (non-critical): {vector_error}")
            
            # Emit WebSocket update (use path-derived user_id for notification routing)
            await self.document_service._emit_document_status_update(
                document_id, "updated", doc_info.get('user_id')
            )
            
            logger.info(f"✅ Document filename updated: {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling file move: {e}")

    async def _resolve_upload_context_from_path(self, path: Path) -> Optional[Dict[str, Any]]:
        """Resolve collection_type, user_id, folder_id from uploads/Users/... or uploads/Global/..."""
        parts = path.parts
        uploads_idx = -1
        for i, part in enumerate(parts):
            if part == "uploads":
                uploads_idx = i
                break
        if uploads_idx == -1:
            logger.warning("File path does not contain 'uploads': %s", path)
            return None
        if uploads_idx + 1 >= len(parts):
            logger.warning("Invalid path structure: %s", path)
            return None
        collection_dir = parts[uploads_idx + 1]
        if collection_dir == "Users" and uploads_idx + 2 < len(parts):
            username = parts[uploads_idx + 2]
            collection_type = "user"
            user_id = await self._get_user_id_from_username(username)
            if not user_id:
                logger.warning("Could not find user_id for username: %s", username)
                return None
            folder_id = None
            folder_start_idx = uploads_idx + 3
            folder_end_idx = len(parts) - 1
            if folder_start_idx < folder_end_idx:
                folder_parts = tuple(parts[folder_start_idx:folder_end_idx])
                folder_id = await self._ensure_folder_hierarchy(
                    folder_parts, user_id, collection_type
                )
            else:
                folder_parts = tuple()
            return {
                "collection_type": collection_type,
                "user_id": user_id,
                "folder_id": folder_id,
                "folder_parts": folder_parts,
            }
        if collection_dir == "Global":
            collection_type = "global"
            user_id = None
            folder_id = None
            folder_start_idx = uploads_idx + 2
            folder_end_idx = len(parts) - 1
            if folder_start_idx < folder_end_idx:
                folder_parts = tuple(parts[folder_start_idx:folder_end_idx])
                folder_id = await self._ensure_folder_hierarchy(
                    folder_parts, None, collection_type
                )
            else:
                folder_parts = tuple()
            return {
                "collection_type": collection_type,
                "user_id": user_id,
                "folder_id": folder_id,
                "folder_parts": folder_parts,
            }
        if collection_dir == "Teams":
            logger.info("Skipping Teams path (managed by application): %s", path)
            return None
        logger.warning("Unknown collection directory: %s", collection_dir)
        return None

    async def _handle_new_file(self, file_path: str):
        """
        Process a new file discovered by the file watcher
        
        **BULLY!** Create document records for files added via WebDAV!
        Uses the SAME service methods as UI uploads for consistency!
        """
        if file_path in self.processing_files:
            logger.info(
                "Skipping _handle_new_file (already in flight): %s", file_path
            )
            return
        self.processing_files.add(file_path)
        try:
            from pathlib import Path
            from uuid import uuid4
            from ds_models.api_models import ProcessingStatus
            import mimetypes
            
            path = Path(file_path)
            filename = path.name
            
            # .metadata.json: document schema uses document_sidecar_service; image sidecar uses unified pipeline
            if filename.lower().endswith('.metadata.json'):
                try:
                    import json
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    schema_type = (metadata.get('schema_type') or '').strip().lower()
                except Exception as e:
                    logger.warning("Failed to read metadata sidecar: %s: %s", file_path, e)
                    return

                if schema_type == 'document':
                    # Document metadata sidecar: require existing document; sync to DB + re-embed
                    existing_doc = await self._get_document_by_path(file_path)
                    if not existing_doc:
                        logger.warning(f"No existing document found for document metadata sidecar: {file_path}")
                        return
                    try:
                        from ds_services.document_sidecar_service import get_document_sidecar_service
                        sidecar_service = await get_document_sidecar_service()
                        result = await sidecar_service.process_document_metadata(file_path, document_info=existing_doc)
                        if result.get("success"):
                            logger.info(f"Document metadata sidecar processed: {existing_doc.get('document_id')}")
                            try:
                                from ds_services.websocket_notifier import WebSocketNotifier
                                from utils.websocket_manager import get_websocket_manager
                                ws_manager = get_websocket_manager()
                                ws_notifier = WebSocketNotifier(ws_manager)
                                path_str = str(path).lower()
                                if '/global/' in path_str:
                                    await ws_notifier.notify_folder_updated(
                                        folder_id=None, user_id=None, collection_type='global'
                                    )
                                elif '/users/' in path_str:
                                    parts = path.parts
                                    uploads_idx = parts.index('uploads') if 'uploads' in parts else -1
                                    if uploads_idx >= 0 and uploads_idx + 2 < len(parts):
                                        username = parts[uploads_idx + 2]
                                        user_id = await self._get_user_id_from_username(username)
                                        if user_id:
                                            await ws_notifier.notify_folder_updated(
                                                folder_id=None, user_id=user_id, collection_type='user'
                                            )
                            except Exception as e:
                                logger.warning(f"Failed to send folder update notification: {e}")
                        else:
                            logger.error(f"Failed to process document metadata: {result.get('error')}")
                    except Exception as e:
                        logger.error(f"Error processing document metadata JSON: {e}")
                    return

                logger.info("Image metadata sidecar (unified pipeline): %s", file_path)
                ctx = await self._resolve_upload_context_from_path(path)
                if not ctx:
                    return
                from ds_services.image_sidecar_service import get_image_sidecar_service

                # Idempotent: ingest from image branch may have already run upsert; watcher re-entry
                # hits find_by_filename_and_context update path plus a safe re-embed kick.
                isvc = await get_image_sidecar_service()
                result = await isvc.upsert_and_process_image_sidecar(
                    file_path=file_path,
                    metadata_data=metadata,
                    folder_id=ctx["folder_id"],
                    user_id=ctx["user_id"],
                    collection_type=ctx["collection_type"],
                    document_service=self.document_service,
                )
                if not result.get("success"):
                    logger.error(
                        "Image sidecar ingest failed for %s: %s",
                        file_path,
                        result.get("error"),
                    )
                    return
                try:
                    from ds_services.websocket_notifier import WebSocketNotifier
                    from utils.websocket_manager import get_websocket_manager

                    ws_manager = get_websocket_manager()
                    ws_notifier = WebSocketNotifier(ws_manager)
                    path_str = str(path).lower()
                    if "/global/" in path_str:
                        await ws_notifier.notify_folder_updated(
                            folder_id=None, user_id=None, collection_type="global"
                        )
                    elif "/users/" in path_str:
                        parts = path.parts
                        uploads_idx = parts.index("uploads") if "uploads" in parts else -1
                        if uploads_idx >= 0 and uploads_idx + 2 < len(parts):
                            username = parts[uploads_idx + 2]
                            user_id_ws = await self._get_user_id_from_username(username)
                            if user_id_ws:
                                await ws_notifier.notify_folder_updated(
                                    folder_id=None, user_id=user_id_ws, collection_type="user"
                                )
                except Exception as e:
                    logger.warning("Folder update notification failed: %s", e)
                return

            # Check if this file already has a document record (prevents duplicates)
            existing_doc = await self._get_document_by_path(file_path)
            if existing_doc:
                logger.debug(f"✅ Document already exists for {file_path}, skipping creation (ID: {existing_doc['document_id']})")
                return
            
            logger.info(f"📄 Creating document record for new file: {file_path}")
            
            # Parse path to get user and folder context
            parts = path.parts
            logger.info(f"📁 Path parts: {parts}")
            
            uploads_idx = -1
            for i, part in enumerate(parts):
                if part == 'uploads':
                    uploads_idx = i
                    break
            
            logger.info(f"📁 Uploads index: {uploads_idx}, Total parts: {len(parts)}")
            
            if uploads_idx == -1:
                logger.warning(f"⚠️ File path doesn't contain 'uploads': {file_path}")
                return
            
            # Determine collection type and context
            if uploads_idx + 1 >= len(parts):
                logger.warning(f"⚠️ Invalid path structure: {file_path}")
                return
                
            collection_dir = parts[uploads_idx + 1]
            
            if collection_dir == 'Users' and uploads_idx + 2 < len(parts):
                # User file: uploads/Users/{username}/{folders...}/{filename}
                username = parts[uploads_idx + 2]
                collection_type = 'user'
                
                # Get user_id from username
                user_id = await self._get_user_id_from_username(username)
                if not user_id:
                    logger.warning(f"⚠️ Could not find user_id for username: {username}")
                    return
                
                # Process folder hierarchy and get/create folder_id
                folder_id = None
                # Check if there are folders between username and filename
                # uploads_idx + 1 = 'Users'
                # uploads_idx + 2 = username
                # uploads_idx + 3 = first folder (if exists)
                # len(parts) - 1 = filename
                folder_start_idx = uploads_idx + 3
                folder_end_idx = len(parts) - 1
                
                logger.info(f"📁 Folder range: [{folder_start_idx}:{folder_end_idx}] in parts of length {len(parts)}")
                
                if folder_start_idx < folder_end_idx:
                    # Extract folder path components
                    folder_parts = parts[folder_start_idx:folder_end_idx]
                    logger.info(f"📁 Folder path parts: {folder_parts}")
                    folder_id = await self._ensure_folder_hierarchy(folder_parts, user_id, collection_type)
                    logger.info(f"📁 Folder ID result: {folder_id}")
                else:
                    logger.info(f"📁 No folders in path (file directly under user directory)")
                    
            elif collection_dir == 'Global':
                # Global file: uploads/Global/{folders...}/{filename}
                collection_type = 'global'
                user_id = None
                
                # Process folder hierarchy
                folder_id = None
                folder_start_idx = uploads_idx + 2
                folder_end_idx = len(parts) - 1
                
                logger.info(f"📁 Global folder range: [{folder_start_idx}:{folder_end_idx}] in parts of length {len(parts)}")
                
                if folder_start_idx < folder_end_idx:
                    folder_parts = parts[folder_start_idx:folder_end_idx]
                    logger.info(f"📁 Global folder path parts: {folder_parts}")
                    folder_id = await self._ensure_folder_hierarchy(folder_parts, None, collection_type)
                    logger.info(f"📁 Global folder ID result: {folder_id}")
                else:
                    logger.info(f"📁 No folders in path (file directly under Global)")
            elif collection_dir == 'Teams':
                # Teams files are managed by the application (team_service)
                # File watcher should not process them - they have access control
                logger.info(f"⏭️  Skipping Teams file (managed by application): {file_path}")
                return
            else:
                logger.warning(f"⚠️ Unknown collection directory: {collection_dir}")
                return
            
            # Generate document_id
            document_id = str(uuid4())
            
            # Determine document type from extension
            # **BULLY!** Use EXACT DocumentType enum values!
            file_ext = path.suffix.lower()
            doc_type_map = {
                '.md': 'md',  # Must match DocumentType.MD
                '.org': 'org',  # Must match DocumentType.ORG
                '.txt': 'txt',  # Must match DocumentType.TXT
                '.pdf': 'pdf',  # Must match DocumentType.PDF
                '.docx': 'docx',  # Must match DocumentType.DOCX
                '.pptx': 'pptx',  # Must match DocumentType.PPTX
                '.html': 'html',  # Must match DocumentType.HTML
                '.htm': 'html',  # Must match DocumentType.HTML
                '.epub': 'epub',  # Must match DocumentType.EPUB
                '.eml': 'eml',  # Must match DocumentType.EML
                '.srt': 'srt',
                '.vtt': 'vtt',
                # **ROOSEVELT FIX:** Images stored but NOT vectorized!
                '.jpg': 'image',
                '.jpeg': 'image',
                '.png': 'image',
                '.gif': 'image',
                '.bmp': 'image',
                '.webp': 'image',
                # Audio files - stored but NOT vectorized
                '.mp3': 'mp3',
                '.aac': 'aac',
                '.wav': 'wav',
                '.flac': 'flac',
                '.ogg': 'ogg',
                '.m4a': 'm4a',
                '.wma': 'wma',
                '.opus': 'opus',
            }
            doc_type = doc_type_map.get(file_ext, 'txt')  # Default to txt
            
            # Get mime type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            # Get file size
            file_size = path.stat().st_size
            
            # Create title from filename (remove extension)
            title = path.stem
            
            logger.info(f"📄 Creating document: {filename} (type: {doc_type}, folder: {folder_id}, user: {user_id})")
            
            # **BULLY!** Use the SAME method as UI uploads for consistency!
            from ds_models.api_models import DocumentInfo, DocumentType, ProcessingStatus
            from datetime import datetime
            
            # Create DocumentInfo object (same as UI upload)
            doc_info = DocumentInfo(
                document_id=document_id,
                filename=filename,
                doc_type=DocumentType(doc_type),
                upload_date=datetime.utcnow(),
                file_size=file_size,
                file_hash=None,  # WebDAV files don't need hash checks (already on disk)
                status=ProcessingStatus.PROCESSING,
                user_id=user_id,
                collection_type=collection_type
            )
            
            # Use DocumentRepository's transactional create method (same as UI!)
            logger.info(f"🔧 DEBUG: About to create document - document_id: {document_id}, filename: {filename}, folder_id: {folder_id}, user_id: {user_id}, collection_type: {collection_type}")
            
            creation_success = await self.document_service.document_repository.create_with_folder(
                doc_info, folder_id
            )
            
            if not creation_success:
                logger.error(f"❌ Failed to create document record: {document_id}")
                return
                
            logger.info(f"✅ Document record created via repository: {document_id}")
            logger.info(f"✅ Document should be in folder: {folder_id} (collection: {collection_type}, user: {user_id})")
            
            # Emit WebSocket notifications
            # 1. Document status update
            await self.document_service._emit_document_status_update(
                document_id, "created", user_id
            )
            
            # 2. **BULLY!** Folder update notification for UI refresh!
            if folder_id:
                try:
                    from ds_services.websocket_notifier import WebSocketNotifier
                    from utils.websocket_manager import get_websocket_manager
                    
                    ws_manager = get_websocket_manager()
                    ws_notifier = WebSocketNotifier(ws_manager)
                    await ws_notifier.notify_file_created(
                        document_id=document_id,
                        folder_id=folder_id,
                        user_id=user_id,
                        metadata={"filename": filename, "doc_type": doc_type}
                    )
                    logger.info(f"📡 Sent folder update notification for document: {document_id} in folder: {folder_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to send folder update notification: {e}")
            
            # Agent folder watches: trigger custom agents when a new file appears in a watched folder
            if folder_id:
                try:
                    from ds_db.database_manager.database_helpers import fetch_all
                    from shims.services.celery_tasks.agent_tasks import dispatch_folder_file_reaction
                    watches = await fetch_all(
                        "SELECT agent_profile_id, user_id, file_type_filter FROM agent_folder_watches "
                        "WHERE folder_id = $1 AND is_active = true",
                        folder_id,
                    )
                    folder_path_display = " / ".join(folder_parts) if folder_parts else ""
                    file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                    for w in watches:
                        ft_filter = (w.get("file_type_filter") or "").strip()
                        if ft_filter:
                            allowed = [x.strip().lower() for x in ft_filter.split(",") if x.strip()]
                            if allowed and file_ext not in allowed:
                                continue
                        dispatch_folder_file_reaction.delay(
                            agent_profile_id=str(w["agent_profile_id"]),
                            user_id=str(w["user_id"]),
                            document_id=document_id,
                            filename=filename,
                            folder_id=folder_id,
                            folder_path=folder_path_display,
                            file_type=doc_type or file_ext or "",
                        )
                except Exception as e:
                    logger.warning("Agent folder watch dispatch failed: %s", e)
            
            # **BULLY!** Process ALL files asynchronously - don't block the file watcher!
            logger.info(f"🔄 Starting async processing for document: {document_id} (type: {doc_type})")
            
            # Create task with exception handling callback (same for ALL file types)
            task = asyncio.create_task(
                self.document_service._process_document_async(
                    document_id, file_path, doc_type, user_id
                )
            )
            
            # Add callback to catch and log any exceptions
            def handle_processing_error(future):
                try:
                    future.result()  # This will raise if the task failed
                    logger.info(f"✅ Async processing completed for: {document_id} ({doc_type})")
                except Exception as e:
                    logger.error(f"❌ PROCESSING FAILED for {document_id} ({filename}, {doc_type}): {e}")
                    import traceback
                    traceback.print_exc()
            
            task.add_done_callback(handle_processing_error)
            
            logger.info(f"✅ New file record created: {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Error handling new file: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.processing_files.discard(file_path)
    
    async def _ensure_folder_hierarchy(self, folder_parts: tuple, user_id: Optional[str], collection_type: str) -> Optional[str]:
        """
        Ensure all folders in the hierarchy exist, creating them if needed
        Returns the folder_id of the deepest folder
        
        Uses UPSERT pattern - no need for complex caching or race condition handling!
        The database handles everything!
        """
        try:
            logger.info(f"📁 Building folder hierarchy: {folder_parts} for user {user_id}, collection: {collection_type}")
            
            # For Global folders, we need an admin user's ID for RLS context
            admin_user_id = user_id  # For user folders, use the user's own ID
            if collection_type == 'global':
                # Get system admin user for RLS context from FileWatcherService
                if self.file_watcher_service:
                    admin_user_id = await self.file_watcher_service._get_system_admin_user_id()
                    if not admin_user_id:
                        logger.error(f"❌ Cannot create Global folders: No admin user available for RLS context")
                        return None
                    logger.info(f"📁 Using admin user {admin_user_id} for Global folder creation")
                else:
                    logger.error(f"❌ Cannot create Global folders: FileWatcherService reference not available")
                    return None
            
            parent_id = None
            current_folder_id = None
            
            # Walk through each level of the hierarchy
            for i, folder_name in enumerate(folder_parts):
                logger.info(f"📁 Level {i+1}/{len(folder_parts)}: Ensuring '{folder_name}' exists (parent: {parent_id})")
                
                # File watcher acts as admin creating folders
                # For Global folders: user_id=None (global), admin_user_id=system_admin (RLS context)
                # For User folders: user_id=user, admin_user_id=user (standard)
                folder = await self.folder_service.create_folder(
                    name=folder_name,
                    parent_folder_id=parent_id,
                    user_id=user_id,
                    collection_type=collection_type,
                    current_user_role="admin",
                    admin_user_id=admin_user_id
                )
                
                if not folder:
                    logger.error(f"❌ Failed to ensure folder '{folder_name}' at level {i+1}")
                    logger.error(f"❌ Hierarchy path so far: {' -> '.join(folder_parts[:i+1])}")
                    return None
                
                current_folder_id = folder.folder_id
                logger.info(f"✅ Folder ensured: {folder_name} → {current_folder_id}")
                
                # This folder becomes the parent for the next level
                parent_id = current_folder_id
            
            logger.info(f"✅ Folder hierarchy complete: {' -> '.join(folder_parts)}")
            logger.info(f"✅ Final folder_id: {current_folder_id}")
            return current_folder_id
            
        except Exception as e:
            logger.error(f"❌ Error ensuring folder hierarchy: {e}")
            logger.error(f"❌ Failed path: {' -> '.join(folder_parts) if folder_parts else 'empty'}")
            import traceback
            traceback.print_exc()
            return None
    
    async def process_pending_events(self):
        """Process pending events after debounce period
        
        **ROOSEVELT'S PARALLEL CAVALRY CHARGE!**
        Process multiple files in parallel with error isolation!
        """
        current_time = time.time()
        paths_to_process = []
        
        for path, (event, timestamp) in list(self.pending_events.items()):
            if current_time - timestamp >= self.debounce_time:
                paths_to_process.append(path)
                del self.pending_events[path]
        
        if not paths_to_process:
            return

        before = len(paths_to_process)
        paths_to_process = [p for p in paths_to_process if p not in self.processing_files]
        skipped_inflight = before - len(paths_to_process)
        if skipped_inflight:
            logger.debug(
                "Skipping %d debounced path(s) already in flight (startup scan / ingest)",
                skipped_inflight,
            )
        if not paths_to_process:
            return
        
        logger.info(f"🔄 Processing {len(paths_to_process)} debounced file events")
        
        # **BULLY!** Process files in parallel with error isolation!
        # Each file gets its own async task so one failure doesn't block others
        async def process_with_error_handling(path):
            try:
                await self._handle_file_modified(path)
            except Exception as e:
                logger.error(f"❌ Failed to process file {path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create tasks for all files and run them concurrently
        tasks = [process_with_error_handling(path) for path in paths_to_process]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"✅ Completed processing {len(paths_to_process)} file events")


class FileWatcherService:
    """
    Service that monitors the uploads directory for file changes
    """
    
    def __init__(self, document_service, folder_service):
        self.document_service = document_service
        self.folder_service = folder_service
        self.observer = None
        self.event_handler = None
        self.running = False
        self._system_admin_user_id = None  # Cached system admin user_id for RLS context
    
    async def _get_system_admin_user_id(self) -> Optional[str]:
        """
        Get a system admin user_id for file watcher operations.
        This is needed for RLS context when creating Global folders.
        
        Returns the user_id of any admin user, or None if no admin exists.
        """
        if self._system_admin_user_id:
            return self._system_admin_user_id
        
        try:
            import psycopg2
            from ds_config import settings
            
            conn = psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB
            )
            
            try:
                cur = conn.cursor()
                cur.execute("SELECT user_id FROM users WHERE role = 'admin' ORDER BY created_at ASC LIMIT 1")
                row = cur.fetchone()
                
                if row:
                    self._system_admin_user_id = row[0]
                    logger.info(f"✅ File watcher using admin user {self._system_admin_user_id} for Global operations")
                    return self._system_admin_user_id
                else:
                    logger.warning(f"⚠️ No admin user found - file watcher cannot create Global folders")
                    return None
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"❌ Error getting system admin user: {e}")
            return None
    
    async def _fix_org_file_folder_ids(self):
        """
        **DEPRECATED MIGRATION - DISABLED!**
        
        **TRUST BUST!** This legacy function was FORCIBLY MOVING all org files 
        into the OrgMode folder on EVERY startup - even files correctly placed 
        in other folders via WebDAV!
        
        Modern file creation (UI + WebDAV) properly assigns folder_id at creation time.
        No need for this aggressive startup "fix" that causes more problems than it solves!
        """
        logger.info("✅ Org file folder migration DISABLED - modern creation handles folders correctly")
        return  # Do nothing!
    
    async def _scan_and_import_folders(self, uploads_dir):
        """
        **ROOSEVELT'S FOLDER IMPORT CAVALRY!**
        
        Scan for folders on disk and create database records for any that don't exist.
        This ensures folders added via filesystem (WebDAV, etc.) appear in the UI.
        
        **BULLY!** Import those folders into the database!
        """
        try:
            from pathlib import Path
            logger.info("📁 Scanning for folders on disk that need to be imported...")
            
            imported_count = 0
            scanned_count = 0
            
            # Scan User folders
            users_dir = uploads_dir / 'Users'
            if users_dir.exists():
                for user_dir in users_dir.iterdir():
                    if not user_dir.is_dir():
                        continue
                    
                    username = user_dir.name
                    
                    # Get user_id from username
                    user_id = await self.event_handler._get_user_id_from_username(username)
                    if not user_id:
                        logger.warning(f"⚠️ Could not find user_id for username: {username}")
                        continue
                    
                    # Walk all subdirectories under this user
                    logger.debug(f"📁 Scanning user directory: {user_dir} for user {username}")
                    for dir_path in user_dir.rglob('*'):
                        if not dir_path.is_dir():
                            continue
                        
                        scanned_count += 1
                        
                        # Get folder path relative to user directory
                        rel_path = dir_path.relative_to(user_dir)
                        folder_parts = rel_path.parts
                        
                        # Skip the user root directory itself (empty parts)
                        if not folder_parts:
                            logger.debug(f"⏭️  Skipping user root directory for {username}")
                            continue
                        
                        # Check if this folder exists in database
                        folder_id = await self.event_handler._resolve_deepest_folder_id(
                            tuple(folder_parts), user_id, 'user'
                        )
                        
                        if not folder_id:
                            # Folder doesn't exist in database - create it!
                            logger.info(f"📁 Importing user folder: {rel_path} for user {username}")
                            try:
                                created_folder_id = await self.event_handler._ensure_folder_hierarchy(
                                    tuple(folder_parts), user_id, 'user'
                                )
                                if created_folder_id:
                                    imported_count += 1
                                    logger.info(f"✅ Imported folder: {rel_path} → {created_folder_id}")
                                else:
                                    logger.warning(f"⚠️ Failed to create folder hierarchy for: {rel_path}")
                            except Exception as e:
                                logger.error(f"❌ Failed to import folder {rel_path}: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            logger.debug(f"✅ User folder already exists in database: {rel_path} → {folder_id}")
            
            # Scan Global folders
            global_dir = uploads_dir / 'Global'
            if global_dir.exists():
                logger.info(f"📁 Scanning Global directory: {global_dir}")
                for dir_path in global_dir.rglob('*'):
                    if not dir_path.is_dir():
                        continue
                    
                    scanned_count += 1
                    
                    # Get folder path relative to Global directory
                    rel_path = dir_path.relative_to(global_dir)
                    folder_parts = rel_path.parts
                    
                    # Skip the Global root directory itself (empty parts)
                    if not folder_parts:
                        logger.debug(f"⏭️  Skipping Global root directory")
                        continue
                    
                    # Check if this folder exists in database
                    folder_id = await self.event_handler._resolve_deepest_folder_id(
                        tuple(folder_parts), None, 'global'
                    )
                    
                    if not folder_id:
                        # Folder doesn't exist in database - create it!
                        logger.info(f"📁 Importing global folder: {rel_path}")
                        try:
                            created_folder_id = await self.event_handler._ensure_folder_hierarchy(
                                tuple(folder_parts), None, 'global'
                            )
                            if created_folder_id:
                                imported_count += 1
                                logger.info(f"✅ Imported global folder: {rel_path} → {created_folder_id}")
                            else:
                                logger.warning(f"⚠️ Failed to create folder hierarchy for: {rel_path}")
                        except Exception as e:
                            logger.error(f"❌ Failed to import global folder {rel_path}: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        logger.debug(f"✅ Global folder already exists in database: {rel_path} → {folder_id}")
            
            logger.info(f"📁 FOLDER IMPORT COMPLETE!")
            logger.info(f"   📊 Folders scanned: {scanned_count}")
            logger.info(f"   ✅ Folders imported: {imported_count}")
            
            if imported_count > 0:
                logger.info(f"   ✅ {imported_count} folders imported into database!")
            
        except Exception as e:
            logger.error(f"❌ Folder import scan failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def _cleanup_missing_folders(self, uploads_dir):
        """
        **ROOSEVELT'S FOLDER CLEANUP CAVALRY!**
        
        Remove folders from database that no longer exist on disk.
        This ensures database stays in sync with filesystem during startup.
        
        **By George!** Clean up those orphaned folders!
        """
        try:
            from pathlib import Path
            logger.debug("🧹 Cleaning up missing folders from database...")
            
            cleaned_count = 0
            checked_count = 0
            
            # Get all user folders from database
            from ds_db.document_repository import DocumentRepository
            doc_repo = DocumentRepository()
            
            # Check user folders
            user_folders = await doc_repo.get_folders_by_user(None, 'user')  # Get all user folders
            logger.debug(f"🔍 Checking {len(user_folders)} user folders...")
            
            for folder_dict in user_folders:
                checked_count += 1
                folder_id = folder_dict.get('folder_id')
                folder_name = folder_dict.get('name')
                user_id = folder_dict.get('user_id')
                parent_folder_id = folder_dict.get('parent_folder_id')
                
                # **BULLY!** Reconstruct the folder's filesystem path
                # We need to build the full path by walking up the parent chain
                folder_path_parts = [folder_name]
                current_parent = parent_folder_id
                
                # Walk up the folder hierarchy
                parent_folders = {f.get('folder_id'): f for f in user_folders}
                while current_parent and current_parent in parent_folders:
                    parent = parent_folders[current_parent]
                    folder_path_parts.insert(0, parent.get('name'))
                    current_parent = parent.get('parent_folder_id')
                
                # Get username from user_id
                username = None
                if user_id:
                    import psycopg2
                    from ds_config import settings
                    try:
                        conn = psycopg2.connect(
                            host=settings.POSTGRES_HOST,
                            port=settings.POSTGRES_PORT,
                            user=settings.POSTGRES_USER,
                            password=settings.POSTGRES_PASSWORD,
                            database=settings.POSTGRES_DB
                        )
                        cur = conn.cursor()
                        cur.execute("SELECT username FROM users WHERE user_id = %s", (user_id,))
                        row = cur.fetchone()
                        username = row[0] if row else None
                        cur.close()
                        conn.close()
                    except Exception as e:
                        logger.error(f"❌ Failed to get username for user_id {user_id}: {e}")
                        continue
                
                if not username:
                    logger.warning(f"⚠️ No username found for folder {folder_name} (user_id: {user_id})")
                    continue
                
                # Build full filesystem path
                folder_fs_path = uploads_dir / 'Users' / username / Path(*folder_path_parts)
                
                # Check if folder exists on disk
                if not folder_fs_path.exists():
                    logger.debug(f"🗑️ Folder missing on disk: {folder_fs_path}")
                    logger.debug(f"🗑️ Deleting from database: {folder_name} ({folder_id})")
                    
                    try:
                        await self.folder_service.delete_folder(folder_id, user_id, recursive=True, current_user_role='admin')
                        cleaned_count += 1
                        logger.debug(f"✅ Deleted orphaned folder: {folder_name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to delete folder {folder_id}: {e}")
            
            # Check global folders
            global_folders = await doc_repo.get_folders_by_user(None, 'global')
            logger.debug(f"🔍 Checking {len(global_folders)} global folders...")
            
            for folder_dict in global_folders:
                checked_count += 1
                folder_id = folder_dict.get('folder_id')
                folder_name = folder_dict.get('name')
                parent_folder_id = folder_dict.get('parent_folder_id')
                
                # Reconstruct global folder path
                folder_path_parts = [folder_name]
                current_parent = parent_folder_id
                
                parent_folders = {f.get('folder_id'): f for f in global_folders}
                while current_parent and current_parent in parent_folders:
                    parent = parent_folders[current_parent]
                    folder_path_parts.insert(0, parent.get('name'))
                    current_parent = parent.get('parent_folder_id')
                
                folder_fs_path = uploads_dir / 'Global' / Path(*folder_path_parts)
                
                if not folder_fs_path.exists():
                    logger.debug(f"🗑️ Global folder missing on disk: {folder_fs_path}")
                    logger.debug(f"🗑️ Deleting from database: {folder_name} ({folder_id})")
                    
                    try:
                        await self.folder_service.delete_folder(folder_id, None, recursive=True, current_user_role='admin')
                        cleaned_count += 1
                        logger.debug(f"✅ Deleted orphaned global folder: {folder_name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to delete global folder {folder_id}: {e}")
            
            logger.debug(f"🧹 FOLDER CLEANUP COMPLETE!")
            logger.debug(f"   📊 Folders checked: {checked_count}")
            logger.debug(f"   🗑️ Orphaned folders removed: {cleaned_count}")
            
            if cleaned_count > 0:
                logger.debug("%s orphaned folders cleaned up", cleaned_count)
            
        except Exception as e:
            logger.error(f"❌ Folder cleanup failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def _cleanup_missing_files(self, uploads_dir):
        """
        Clean up document records for files that no longer exist on disk.
        
        This ensures database stays in sync with filesystem during startup.
        Respects RLS by using admin context to query all accessible documents.
        """
        try:
            from pathlib import Path
            logger.debug("🧹 Cleaning up missing files from database...")
            
            cleaned_count = 0
            checked_count = 0
            error_count = 0
            
            # Query all documents with admin context (respects RLS - gets user docs and global docs)
            from ds_db.database_manager.database_helpers import fetch_all
            
            # Use admin context to see all user documents and global documents
            # (Team documents are excluded unless admin is a member, which is correct for privacy)
            rls_context = {'user_id': '', 'user_role': 'admin'}
            
            # Get all documents (with pagination to handle large datasets)
            limit = 1000
            offset = 0
            total_checked = 0
            
            while True:
                rows = await fetch_all("""
                    SELECT document_id, filename, user_id, folder_id, collection_type, team_id
                    FROM document_metadata 
                    ORDER BY upload_date DESC 
                    LIMIT $1 OFFSET $2
                """, limit, offset, rls_context=rls_context)
                
                if not rows:
                    break
                
                logger.debug(f"🔍 Checking batch: {len(rows)} documents (offset: {offset})")
                
                for row in rows:
                    checked_count += 1
                    total_checked += 1
                    
                    document_id = row.get('document_id')
                    filename = row.get('filename')
                    user_id = row.get('user_id')
                    folder_id = row.get('folder_id')
                    collection_type = row.get('collection_type', 'user')
                    team_id = row.get('team_id')
                    
                    if not filename:
                        logger.warning(f"⚠️ Document {document_id} has no filename - skipping")
                        continue
                    
                    try:
                        # Reconstruct filesystem path using folder service
                        file_path = await self.folder_service.get_document_file_path(
                            filename=filename,
                            folder_id=folder_id,
                            user_id=user_id,
                            collection_type=collection_type,
                            team_id=team_id
                        )
                        
                        # Check if file exists on disk
                        if not file_path.exists():
                            logger.debug(f"🗑️ File missing on disk: {file_path}")
                            logger.debug(f"🗑️ Deleting from database: {filename} ({document_id})")
                            
                            # Delete from vector store
                            try:
                                await self.document_service.embedding_manager.delete_document_chunks(document_id, user_id)
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to delete embeddings for {document_id}: {e}")
                            
                            # Delete from database with proper user context
                            success = await self.document_service.document_repository.delete(document_id, user_id)
                            if success:
                                cleaned_count += 1
                                logger.debug(f"✅ Deleted orphaned document: {filename} ({document_id})")
                            else:
                                logger.warning(f"⚠️ Failed to delete document {document_id} from database")
                                error_count += 1
                        else:
                            logger.debug(f"✅ File exists: {file_path}")
                    
                    except Exception as e:
                        logger.error(f"❌ Error checking document {document_id}: {e}")
                        error_count += 1
                        import traceback
                        traceback.print_exc()
                
                # Check if we've processed all documents
                if len(rows) < limit:
                    break
                
                offset += limit
            
            logger.debug(f"🧹 FILE CLEANUP COMPLETE!")
            logger.debug(f"   📊 Documents checked: {checked_count}")
            logger.debug(f"   🗑️ Orphaned documents removed: {cleaned_count}")
            logger.debug(f"   ❌ Errors: {error_count}")
            
            if cleaned_count > 0:
                logger.debug(f"   ✅ {cleaned_count} orphaned documents cleaned up!")
            
        except Exception as e:
            logger.error(f"❌ File cleanup failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def _perform_startup_scan(self):
        """
        Scan the uploads directory on startup and create database records
        for any files that were added while the app was down.
        
        This ensures the database always matches the filesystem.
        """
        try:
            from pathlib import Path
            from ds_config import settings
            
            uploads_dir = Path(settings.UPLOAD_DIR)
            
            if not uploads_dir.exists():
                logger.info("📁 No uploads directory found - nothing to scan")
                return
            
            logger.info(f"🔍 Scanning uploads directory: {uploads_dir}")
            
            # Supported file extensions
            valid_extensions = [
                '.md', '.org', '.txt', '.pdf', '.docx', '.pptx', '.html', '.htm', '.epub', '.eml',  # Documents
                '.srt', '.vtt',
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'  # Images
            ]
            
            scanned_count = 0
            new_count = 0
            skipped_count = 0  # Already tracked files (duplicates prevented)
            error_count = 0

            # Walk the entire uploads directory tree
            for file_path in uploads_dir.rglob('*'):
                if not file_path.is_file():
                    continue
                
                # Use the same ignore logic as the file handler
                file_path_str = str(file_path)
                if self.event_handler._should_ignore_path(file_path_str, is_directory=False):
                    continue
                
                # Skip temp/hidden files
                if file_path.name.startswith('.') or file_path.name.startswith('~'):
                    continue
                
                scanned_count += 1
                
                try:
                    # Check if this file already has a database record
                    # This prevents duplicates by comparing filename + user + folder context
                    doc_info = await self.event_handler._get_document_by_path(file_path_str)
                    
                    if not doc_info:
                        # File exists on disk but NOT in database!
                        logger.debug(f"📄 NEW FILE FOUND: {file_path.relative_to(uploads_dir)}")
                        if file_path_str.lower().endswith('.metadata.json'):
                            await self.event_handler._handle_new_file(file_path_str)
                            new_count += 1
                            continue

                        await self.event_handler._handle_new_file(file_path_str)
                        new_count += 1
                    else:
                        # File already tracked - skip to prevent duplicate
                        logger.debug(f"⏭️  SKIP (already in DB): {file_path.name} (doc_id: {doc_info.get('document_id', 'unknown')})")
                        skipped_count += 1
                
                except Exception as e:
                    logger.error(f"❌ Error scanning {file_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1

            logger.info(f"🎯 Startup scan complete")
            logger.info(f"   📊 Files scanned: {scanned_count}")
            logger.info(f"   ✅ New files added: {new_count}")
            logger.info(f"   ⏭️  Already tracked (duplicates prevented): {skipped_count}")
            logger.info(f"   ❌ Errors: {error_count}")
            
            if new_count > 0:
                logger.info(f"   ✅ {new_count} files recovered and added to the database")
            if skipped_count > 0:
                logger.info(f"   ⏭️  {skipped_count} duplicates prevented")
            
            # Scan for folders on disk that aren't in database
            await self._scan_and_import_folders(uploads_dir)
            
            # Clean up folders that no longer exist on disk
            await self._cleanup_missing_folders(uploads_dir)
            
            # Clean up files that no longer exist on disk
            await self._cleanup_missing_files(uploads_dir)
            
        except Exception as e:
            logger.error(f"❌ Startup scan failed: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail startup if scan fails
            pass

    async def run_rescan(self) -> Dict[str, Any]:
        """Re-scan uploads directory and add any files on disk that are not in the database.
        Safe to call after clearing the document database so files are re-read without restart."""
        if not getattr(self, "event_handler", None):
            return {"success": False, "error": "File watcher not started", "new_count": 0, "scanned_count": 0}
        try:
            await self._perform_startup_scan()
            return {"success": True, "new_count": 0, "scanned_count": 0}
        except Exception as e:
            logger.error(f"Rescan failed: {e}")
            return {"success": False, "error": str(e), "new_count": 0, "scanned_count": 0}

    async def start(self):
        """Start watching the uploads directory"""
        try:
            logger.debug("👀 Starting File System Watcher...")
            
            # Get the current event loop for thread-safe async scheduling
            event_loop = asyncio.get_running_loop()
            
            # Create event handler FIRST (needed for startup scan)
            # Pass self reference so handler can access _get_system_admin_user_id
            self.event_handler = DocumentFileHandler(
                self.document_service, 
                self.folder_service,
                event_loop,
                file_watcher_service=self
            )
            
            # Fix any org files missing folder_id (deprecated, does nothing)
            await self._fix_org_file_folder_ids()
            
            # Create observer and start watching first so the server can accept connections
            self.observer = Observer()
            watch_path = str(Path(settings.UPLOAD_DIR).resolve())
            self.observer.schedule(
                self.event_handler,
                watch_path,
                recursive=True
            )
            self.observer.start()
            self.running = True
            logger.info(f"✅ File System Watcher started, monitoring: {watch_path} (recursive=True)")

            # Run startup scan in background so lifespan can yield and uvicorn can bind
            logger.debug("🔍 Performing startup file synchronization scan in background...")
            asyncio.create_task(self._perform_startup_scan())
            logger.info(f"📁 Watcher will detect: folder creation, deletion, move/rename operations")
            logger.info(f"📁 Watcher will detect: file creation, modification, deletion, move/rename operations")
            
            # Start debounce processor
            asyncio.create_task(self._debounce_loop())
            
        except Exception as e:
            logger.error(f"❌ Failed to start File System Watcher: {e}")
            raise
    
    async def _debounce_loop(self):
        """Background loop to process debounced events"""
        while self.running:
            try:
                await self.event_handler.process_pending_events()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"❌ Error in debounce loop: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def stop(self):
        """Stop watching"""
        try:
            if self.observer:
                logger.debug("⏹️ Stopping File System Watcher...")
                self.running = False
                self.observer.stop()
                self.observer.join(timeout=5)
                logger.debug("✅ File System Watcher stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping File System Watcher: {e}")


# Global instance
_file_watcher_instance = None


async def get_file_watcher() -> FileWatcherService:
    """Get the global File Watcher instance (uses injectable ``service_container``)."""
    global _file_watcher_instance

    if _file_watcher_instance is None:
        from shims.services.service_container import service_container

        _file_watcher_instance = FileWatcherService(
            service_container.document_service,
            service_container.folder_service,
        )

    return _file_watcher_instance


def reset_file_watcher() -> None:
    """Clear singleton (shutdown)."""
    global _file_watcher_instance
    _file_watcher_instance = None

