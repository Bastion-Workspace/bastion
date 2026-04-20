"""
File recovery: scan filesystem for orphaned files and re-add them to the database.
When vectors are missing, queues the same Celery reprocess path as save/audit flows.
"""

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ds_config import settings

logger = logging.getLogger(__name__)

# Matches document_version_service snapshot names: v007_c35f752c.md (suffix = original file)
_VERSION_SNAPSHOT_FILENAME = re.compile(
    r"^v\d{3}_[0-9a-f]{8}\.(md|txt|org|pdf|epub)$",
    re.IGNORECASE,
)


def _is_document_version_snapshot_path(file_path: Path) -> bool:
    """Exclude .versions/{document_id}/ copies from recovery (same rule as file_watcher)."""
    if any(part == ".versions" for part in file_path.parts):
        return True
    return bool(_VERSION_SNAPSHOT_FILENAME.match(file_path.name))


class FileRecoveryService:
    """Recover orphaned files after database resets or vector loss."""
    
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
            from ds_db.database_manager.database_helpers import fetch_all, fetch_one
            from ds_services.folder_service import FolderService
            from shims.services.service_container import service_container
            
            logger.info("Scanning for orphaned files for user %s", user_id)
            
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
                if not file_path.is_file():
                    continue
                if _is_document_version_snapshot_path(file_path):
                    continue
                if file_path.suffix.lower() not in supported_extensions:
                    continue
                filename = file_path.name
                relative_path = str(file_path.relative_to(user_dir))

                if filename not in known_filenames:
                    orphaned_files.append(
                        {
                            "path": file_path,
                            "filename": filename,
                            "relative_path": relative_path,
                            "size": file_path.stat().st_size,
                            "modified": datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            ),
                        }
                    )
            
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
            
            logger.info(
                "Recovery complete: recovered=%s skipped=%s errors=%s",
                len(recovered),
                len(skipped),
                len(errors),
            )
            
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
        from ds_services.folder_service import FolderService

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
        
        # Insert into database using direct SQL (columns must match document_metadata schema)
        from ds_db.database_manager.database_helpers import execute

        await execute("""
            INSERT INTO document_metadata (
                document_id, title, filename, doc_type, user_id,
                collection_type, processing_status, folder_id,
                file_size, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), NOW())
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
            file_info['size'],
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
            logger.info("Queueing %s for processing (no vectors found)", filename)
            import asyncio

            from ds_services.collab_reprocess_helper import schedule_reprocess_after_save

            asyncio.create_task(schedule_reprocess_after_save(document_id, user_id))
            result['queued_for_processing'] = True
        
        return result
    
    async def _check_qdrant_vectors(self, document_id: str, embedding_manager: Any) -> bool:
        """Check if vectors already exist in Qdrant for this document."""
        logger.info(
            "Skipping Qdrant check for %s (will queue for processing)",
            document_id,
        )
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
            from ds_db.database_manager.database_helpers import fetch_one
            
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
        """Map file extension to document_metadata.doc_type (DocumentType enum values)."""
        doc_type_map = {
            ".md": "md",
            ".txt": "txt",
            ".org": "org",
            ".pdf": "pdf",
            ".epub": "epub",
        }
        return doc_type_map.get(file_ext.lower(), "md")


# Singleton
_file_recovery_service: Optional[FileRecoveryService] = None

async def get_file_recovery_service() -> FileRecoveryService:
    """Get or create FileRecoveryService singleton"""
    global _file_recovery_service
    if _file_recovery_service is None:
        _file_recovery_service = FileRecoveryService()
    return _file_recovery_service

