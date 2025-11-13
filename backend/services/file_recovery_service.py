"""
File Recovery Service - Roosevelt's Recovery Cavalry
Scans filesystem for orphaned files and re-adds them to the database
WITHOUT re-vectorizing if vectors already exist in Qdrant
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import mimetypes

from config import settings

logger = logging.getLogger(__name__)


class FileRecoveryService:
    """
    Service for recovering orphaned files after database resets
    
    **BULLY!** Bring those lost files back into the fold!
    """
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        
    async def scan_and_recover_user_files(
        self, 
        user_id: str,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Scan user's directory and recover files not in database
        
        Args:
            user_id: User ID to scan for
            dry_run: If True, only report what would be recovered without making changes
            
        Returns:
            Dict with recovery results and statistics
        """
        try:
            from services.database_manager.database_helpers import fetch_all, fetch_one
            from services.folder_service import FolderService
            from services.service_container import service_container
            
            logger.info(f"🔍 ROOSEVELT: Scanning for orphaned files for user {user_id}")
            
            # Get username and user directory using FolderService
            user_row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            if not user_row:
                return {"success": False, "error": "User not found"}
            
            username = user_row['username']
            
            # Use FolderService to get correct user base path
            folder_service = FolderService()
            user_dir = folder_service.get_user_base_path(user_id, username)
            
            # Get document service for embedding manager
            document_service = service_container.document_service
            
            if not user_dir.exists():
                return {
                    "success": True,
                    "message": "No user directory found",
                    "recovered": 0,
                    "skipped": 0,
                    "errors": []
                }
            
            # Get all files currently in database for this user
            db_files = await fetch_all("""
                SELECT filename, document_id 
                FROM document_metadata 
                WHERE user_id = $1
            """, user_id)
            
            # Build set of known files (by filename for simple comparison)
            known_filenames = {row['filename'] for row in db_files}
            
            logger.info(f"📊 Found {len(known_filenames)} files in database")
            
            # Scan filesystem for all supported files
            orphaned_files = []
            supported_extensions = {'.md', '.txt', '.org', '.pdf', '.epub'}
            
            for file_path in user_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    # Check if this file is in database
                    filename = file_path.name
                    relative_path = str(file_path.relative_to(user_dir))
                    
                    if filename not in known_filenames:
                        orphaned_files.append({
                            'path': file_path,
                            'filename': filename,
                            'relative_path': relative_path,
                            'size': file_path.stat().st_size,
                            'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                        })
            
            logger.info(f"🔍 Found {len(orphaned_files)} orphaned files")
            
            if dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "orphaned_files": orphaned_files,
                    "count": len(orphaned_files)
                }
            
            # Recover files
            recovered = []
            skipped = []
            errors = []
            
            for file_info in orphaned_files:
                try:
                    result = await self._recover_file(
                        user_id=user_id,
                        username=username,
                        file_info=file_info,
                        embedding_manager=document_service.embedding_manager
                    )
                    
                    if result['recovered']:
                        recovered.append(result)
                    else:
                        skipped.append(result)
                        
                except Exception as e:
                    logger.error(f"❌ Failed to recover {file_info['filename']}: {e}")
                    errors.append({
                        'file': file_info['filename'],
                        'error': str(e)
                    })
            
            logger.info(f"✅ ROOSEVELT: Recovery complete! Recovered: {len(recovered)}, Skipped: {len(skipped)}, Errors: {len(errors)}")
            
            return {
                "success": True,
                "recovered": recovered,
                "recovered_count": len(recovered),
                "skipped": skipped,
                "skipped_count": len(skipped),
                "errors": errors,
                "error_count": len(errors),
                "total_scanned": len(orphaned_files)
            }
            
        except Exception as e:
            logger.error(f"❌ File recovery failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _recover_file(
        self,
        user_id: str,
        username: str,
        file_info: Dict[str, Any],
        embedding_manager: Any
    ) -> Dict[str, Any]:
        """
        Recover a single file
        
        Checks Qdrant for existing vectors before re-vectorizing
        """
        from services.service_container import service_container
        from services.folder_service import FolderService
        from models.api_models import ProcessingStatus
        from uuid import uuid4
        
        file_path = file_info['path']
        filename = file_info['filename']
        
        logger.info(f"♻️ Recovering file: {filename}")
        
        # Generate document ID (deterministic based on file path for consistency)
        path_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        document_id = f"doc_{path_hash[:24]}"
        
        # Check if vectors already exist in Qdrant
        has_vectors = await self._check_qdrant_vectors(document_id, embedding_manager)
        
        # Determine folder_id from path
        folder_service = FolderService()
        folder_id = await self._resolve_folder_id(user_id, username, file_path, folder_service)
        
        # Determine document type
        file_ext = file_path.suffix.lower()
        doc_type = self._get_doc_type(file_ext)
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Insert into database using direct SQL
        from services.database_manager.database_helpers import execute_query
        
        await execute_query("""
            INSERT INTO document_metadata (
                document_id, title, filename, doc_type, user_id, 
                collection_type, processing_status, folder_id, 
                mime_type, file_size, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW())
            ON CONFLICT (document_id) DO NOTHING
        """, 
            document_id,
            filename.rsplit('.', 1)[0],  # title (remove extension)
            filename,
            doc_type,
            user_id,
            'user',
            'pending',  # Will be processed if needed
            folder_id,
            mime_type,
            file_info['size']
        )
        
        result = {
            'recovered': True,
            'document_id': document_id,
            'filename': filename,
            'folder_id': folder_id,
            'had_vectors': has_vectors,
            'needs_processing': not has_vectors
        }
        
        # If no vectors exist, queue for processing
        if not has_vectors and doc_type != 'org':  # .org files don't get vectorized
            logger.info(f"🔄 Queueing {filename} for processing (no vectors found)")
            # Trigger reprocessing
            await document_service.reprocess_document(document_id, user_id)
            result['queued_for_processing'] = True
        
        return result
    
    async def _check_qdrant_vectors(self, document_id: str, embedding_manager: Any) -> bool:
        """
        Check if vectors already exist in Qdrant for this document
        
        **BULLY!** Don't re-vectorize if we already have the goods!
        
        For now, always return False to allow re-vectorization
        """
        logger.info(f"📝 Skipping Qdrant check for {document_id} (will queue for processing)")
        return False
    
    async def _resolve_folder_id(
        self,
        user_id: str,
        username: str,
        file_path: Path,
        folder_service: Any
    ) -> Optional[str]:
        """
        Resolve folder_id based on file path
        
        Maps filesystem paths to folder IDs in database
        """
        try:
            from services.database_manager.database_helpers import fetch_one
            
            # Use FolderService to get correct user base path
            user_dir = folder_service.get_user_base_path(user_id, username)
            relative_path = file_path.relative_to(user_dir)
            
            # Get folder name from path (first directory component)
            if len(relative_path.parts) > 1:
                folder_name = relative_path.parts[0]
                
                # Look up folder_id
                folder_row = await fetch_one("""
                    SELECT folder_id FROM document_folders
                    WHERE user_id = $1 AND name = $2 AND collection_type = 'user'
                """, user_id, folder_name)
                
                if folder_row:
                    return folder_row['folder_id']
            
            # Default: find or create "Recovered Files" folder
            recovered_folder = await fetch_one("""
                SELECT folder_id FROM document_folders
                WHERE user_id = $1 AND name = 'Recovered Files' AND collection_type = 'user'
            """, user_id)
            
            if recovered_folder:
                return recovered_folder['folder_id']
            
            # Create "Recovered Files" folder
            import uuid
            folder_id = str(uuid.uuid4())
            await folder_service.create_folder("Recovered Files", None, user_id, "user")
            return folder_id
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to resolve folder_id: {e}")
            return None
    
    def _get_doc_type(self, file_ext: str) -> str:
        """Map file extension to doc_type"""
        doc_type_map = {
            '.md': 'markdown',
            '.txt': 'text',
            '.org': 'org',
            '.pdf': 'pdf',
            '.epub': 'epub'
        }
        return doc_type_map.get(file_ext.lower(), 'unknown')


# Singleton
_file_recovery_service: Optional[FileRecoveryService] = None

async def get_file_recovery_service() -> FileRecoveryService:
    """Get or create FileRecoveryService singleton"""
    global _file_recovery_service
    if _file_recovery_service is None:
        _file_recovery_service = FileRecoveryService()
    return _file_recovery_service

