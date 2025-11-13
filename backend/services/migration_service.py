"""
Migration Service - Handles data migration from JSON to PostgreSQL
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from config import settings
from models.api_models import DocumentInfo, ProcessingStatus, DocumentType, DocumentCategory, QualityMetrics
from repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class MigrationService:
    """Service for migrating data from JSON to PostgreSQL"""
    
    def __init__(self, document_repo: DocumentRepository):
        self.document_repo = document_repo
        self.json_registry_path = Path(settings.PROCESSED_DIR) / "documents_registry.json"
    
    async def migrate_json_to_postgres(self) -> Dict[str, Any]:
        """Migrate document registry from JSON file to PostgreSQL"""
        migration_result = {
            "total_documents": 0,
            "migrated_documents": 0,
            "skipped_documents": 0,
            "failed_migrations": 0,
            "errors": []
        }
        
        try:
            logger.info("🔄 Starting migration from JSON to PostgreSQL...")
            
            # Check if JSON file exists
            if not self.json_registry_path.exists():
                logger.info("📚 No JSON registry file found, nothing to migrate")
                return migration_result
            
            # Load JSON data
            with open(self.json_registry_path, 'r') as f:
                json_data = json.load(f)
            
            migration_result["total_documents"] = len(json_data)
            logger.info(f"📚 Found {len(json_data)} documents in JSON registry")
            
            for doc_id, doc_data in json_data.items():
                try:
                    # Check if document already exists in PostgreSQL
                    existing_doc = await self.document_repo.get_by_id(doc_id)
                    if existing_doc:
                        logger.debug(f"⏭️ Document {doc_id} already exists in PostgreSQL, skipping")
                        migration_result["skipped_documents"] += 1
                        continue
                    
                    # Convert JSON data to DocumentInfo
                    doc_info = self._json_to_document_info(doc_id, doc_data)
                    
                    # Create document in PostgreSQL
                    success = await self.document_repo.create(doc_info)
                    
                    if success:
                        migration_result["migrated_documents"] += 1
                        logger.debug(f"✅ Migrated document: {doc_id}")
                    else:
                        migration_result["failed_migrations"] += 1
                        migration_result["errors"].append(f"Failed to create document {doc_id}")
                        logger.error(f"❌ Failed to migrate document: {doc_id}")
                
                except Exception as e:
                    migration_result["failed_migrations"] += 1
                    error_msg = f"Error migrating document {doc_id}: {str(e)}"
                    migration_result["errors"].append(error_msg)
                    logger.error(f"❌ {error_msg}")
            
            # Create backup of JSON file
            if migration_result["migrated_documents"] > 0:
                backup_path = self.json_registry_path.with_suffix('.json.backup')
                self.json_registry_path.rename(backup_path)
                logger.info(f"📦 Created backup of JSON registry: {backup_path}")
            
            logger.info(f"✅ Migration completed: {migration_result['migrated_documents']} migrated, "
                       f"{migration_result['skipped_documents']} skipped, "
                       f"{migration_result['failed_migrations']} failed")
            
            return migration_result
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            migration_result["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return migration_result
    
    def _json_to_document_info(self, doc_id: str, doc_data: Dict[str, Any]) -> DocumentInfo:
        """Convert JSON document data to DocumentInfo object"""
        # Handle datetime conversion
        upload_date = doc_data.get('upload_date')
        if isinstance(upload_date, str):
            upload_date = datetime.fromisoformat(upload_date)
        elif upload_date is None:
            upload_date = datetime.utcnow()
        
        # Handle quality metrics
        quality_metrics = None
        if doc_data.get('quality_metrics'):
            try:
                quality_metrics = QualityMetrics(**doc_data['quality_metrics'])
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse quality metrics for {doc_id}: {e}")
        
        # Handle category
        category = None
        if doc_data.get('category'):
            try:
                category = DocumentCategory(doc_data['category'])
            except Exception:
                logger.warning(f"⚠️ Invalid category for {doc_id}: {doc_data.get('category')}")
        
        # Handle document type
        doc_type = DocumentType.TXT  # Default
        if doc_data.get('doc_type'):
            try:
                doc_type = DocumentType(doc_data['doc_type'])
            except Exception:
                logger.warning(f"⚠️ Invalid doc_type for {doc_id}: {doc_data.get('doc_type')}")
        
        # Handle status
        status = ProcessingStatus.COMPLETED  # Default for existing documents
        if doc_data.get('status'):
            try:
                status = ProcessingStatus(doc_data['status'])
            except Exception:
                logger.warning(f"⚠️ Invalid status for {doc_id}: {doc_data.get('status')}")
        
        return DocumentInfo(
            document_id=doc_id,
            filename=doc_data.get('filename', 'unknown'),
            title=doc_data.get('title'),
            category=category,
            tags=doc_data.get('tags', []),
            description=doc_data.get('description'),
            author=doc_data.get('author'),
            language=doc_data.get('language'),
            doc_type=doc_type,
            file_size=doc_data.get('file_size', 0),
            file_hash=doc_data.get('file_hash'),
            status=status,
            upload_date=upload_date,
            quality_metrics=quality_metrics
        )
    
    async def verify_migration(self) -> Dict[str, Any]:
        """Verify that migration was successful"""
        verification_result = {
            "postgres_count": 0,
            "json_backup_exists": False,
            "migration_complete": False
        }
        
        try:
            # Count documents in PostgreSQL
            stats = await self.document_repo.get_stats()
            verification_result["postgres_count"] = stats.get("total_documents", 0)
            
            # Check if JSON backup exists
            backup_path = self.json_registry_path.with_suffix('.json.backup')
            verification_result["json_backup_exists"] = backup_path.exists()
            
            # Check if original JSON file is gone
            verification_result["migration_complete"] = (
                not self.json_registry_path.exists() and 
                verification_result["json_backup_exists"] and
                verification_result["postgres_count"] > 0
            )
            
            logger.info(f"🔍 Migration verification: {verification_result['postgres_count']} documents in PostgreSQL, "
                       f"backup exists: {verification_result['json_backup_exists']}, "
                       f"migration complete: {verification_result['migration_complete']}")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"❌ Migration verification failed: {e}")
            return verification_result
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        try:
            # Check if JSON file exists
            json_exists = self.json_registry_path.exists()
            
            # Check if backup exists
            backup_path = self.json_registry_path.with_suffix('.json.backup')
            backup_exists = backup_path.exists()
            
            # Get PostgreSQL document count
            stats = await self.document_repo.get_stats()
            postgres_count = stats.get("total_documents", 0)
            
            return {
                "json_file_exists": json_exists,
                "backup_exists": backup_exists,
                "postgres_document_count": postgres_count,
                "migration_needed": json_exists,
                "migration_completed": backup_exists and not json_exists
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get migration status: {e}")
            return {
                "json_file_exists": False,
                "backup_exists": False,
                "postgres_document_count": 0,
                "migration_needed": False,
                "migration_completed": False,
                "error": str(e)
            }
    
    async def close(self):
        """Close migration service resources"""
        # Migration service doesn't have persistent connections to close
        logger.debug("🔄 Migration Service closed")
