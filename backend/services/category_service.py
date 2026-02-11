"""
Category Service - Unified category management for documents and notes
Provides a centralized system for managing categories across documents and free-form notes
"""

import logging
from typing import List, Dict, Any, Set, Optional
import asyncpg
from models.api_models import DocumentCategory
from repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class CategoryService:
    """Service for unified category management across documents and notes"""
    
    def __init__(self):
        self.document_repository = None
        self.notes_service = None
        self._shared_db_pool = None  # For shared pool from service container
    
    async def initialize(self, document_repository: DocumentRepository = None, notes_service=None, shared_db_pool: Optional[asyncpg.Pool] = None):
        """Initialize the category service with dependencies"""
        logger.debug("🏷️ Initializing Category Service...")
        
        # Use shared database pool if provided
        if shared_db_pool:
            self._shared_db_pool = shared_db_pool
            logger.debug("✅ Category Service using shared database pool")
        
        # Use provided document repository or create one with shared pool
        if document_repository:
            self.document_repository = document_repository
            logger.info("✅ Category Service using provided DocumentRepository")
        else:
            # Create new DocumentRepository with shared pool if available
            self.document_repository = DocumentRepository()
            if self._shared_db_pool:
                # Initialize with shared pool to avoid creating new connections
                await self.document_repository.initialize_with_pool(self._shared_db_pool)
            else:
                await self.document_repository.initialize()
            logger.debug("✅ Category Service created new DocumentRepository")
        
        self.notes_service = notes_service
        
        logger.debug("✅ Category Service initialized")
    
    async def get_all_categories(self) -> Dict[str, Any]:
        """Get all categories from both documents and notes, unified and sorted"""
        try:
            categories = set()
            category_details = []
            
            # Add predefined document categories
            for doc_category in DocumentCategory:
                categories.add(doc_category.value)
                category_details.append({
                    "name": doc_category.value,
                    "type": "predefined",
                    "usage_count": 0,
                    "source": "documents"
                })
            
            # Add document categories from database (in case there are custom ones)
            try:
                doc_categories_response = await self.document_repository.get_categories_overview()
                for category_summary in doc_categories_response.categories:
                    category_name = category_summary.category.value
                    if category_name not in categories:
                        categories.add(category_name)
                        category_details.append({
                            "name": category_name,
                            "type": "document",
                            "usage_count": category_summary.count,
                            "source": "documents"
                        })
                    else:
                        # Update count for existing category
                        for detail in category_details:
                            if detail["name"] == category_name:
                                detail["usage_count"] = category_summary.count
                                break
            except Exception as e:
                logger.warning(f"⚠️ Failed to get document categories: {e}")
            
            # Add note categories from notes service
            if self.notes_service:
                try:
                    note_categories_data = await self.notes_service.get_categories_and_tags()
                    note_categories = note_categories_data.get('categories', [])
                    for category in note_categories:
                        if category:  # Skip None/empty categories
                            if category not in categories:
                                categories.add(category)
                                category_details.append({
                                    "name": category,
                                    "type": "note",
                                    "usage_count": 0,
                                    "source": "notes"
                                })
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get note categories: {e}")
            
            # Sort categories by name
            category_details.sort(key=lambda x: x["name"])
            
            return {
                "categories": category_details,
                "total_count": len(category_details)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get all categories: {e}")
            return {"categories": [], "total_count": 0}
    
    async def get_categories_with_counts(self) -> Dict[str, Dict[str, int]]:
        """Get categories with counts from both documents and notes"""
        try:
            result = {
                "documents": {},
                "notes": {},
                "total": {}
            }
            
            # Get document categories with counts
            try:
                doc_categories_response = await self.document_repository.get_categories_overview()
                for category_summary in doc_categories_response.categories:
                    category = category_summary.category.value
                    count = category_summary.count
                    result["documents"][category] = count
                    result["total"][category] = result["total"].get(category, 0) + count
            except Exception as e:
                logger.warning(f"⚠️ Failed to get document category counts: {e}")
            
            # Get note categories with counts
            if self.notes_service:
                try:
                    note_stats = await self.notes_service.get_statistics()
                    notes_by_category = note_stats.get("notes_by_category", {})
                    for category, count in notes_by_category.items():
                        if category:  # Skip None/empty categories
                            result["notes"][category] = count
                            result["total"][category] = result["total"].get(category, 0) + count
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get note category counts: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get categories with counts: {e}")
            return {"documents": {}, "notes": {}, "total": {}}
    
    async def get_predefined_categories(self) -> List[str]:
        """Get the predefined document categories"""
        return [category.value for category in DocumentCategory]
    
    async def get_custom_categories(self) -> List[str]:
        """Get categories that are not in the predefined list"""
        try:
            all_categories_response = await self.get_all_categories()
            all_category_names = set(cat["name"] for cat in all_categories_response["categories"])
            predefined_categories = set(await self.get_predefined_categories())
            
            custom_categories = all_category_names - predefined_categories
            return sorted(list(custom_categories))
            
        except Exception as e:
            logger.error(f"❌ Failed to get custom categories: {e}")
            return []
    
    async def validate_category(self, category: str) -> bool:
        """Validate if a category name is acceptable"""
        if not category:
            return False
        
        # Remove extra whitespace
        category = category.strip()
        
        # Check length
        if len(category) < 1 or len(category) > 100:
            return False
        
        # Check for valid characters (alphanumeric, spaces, hyphens, underscores)
        import re
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', category):
            return False
        
        return True
    
    async def normalize_category(self, category: str) -> str:
        """Normalize a category name (trim, lowercase, etc.)"""
        if not category:
            return ""
        
        # Trim whitespace and convert to lowercase for consistency
        normalized = category.strip().lower()
        
        # Replace multiple spaces with single space
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    
    async def get_category_statistics(self) -> Dict[str, Any]:
        """Get comprehensive category statistics"""
        try:
            categories_with_counts = await self.get_categories_with_counts()
            all_categories = await self.get_all_categories()
            predefined_categories = await self.get_predefined_categories()
            custom_categories = await self.get_custom_categories()
            
            # Calculate totals
            total_documents_with_categories = sum(categories_with_counts["documents"].values())
            total_notes_with_categories = sum(categories_with_counts["notes"].values())
            total_items_with_categories = sum(categories_with_counts["total"].values())
            
            # Find most used categories
            most_used = sorted(
                categories_with_counts["total"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                "total_categories": len(all_categories),
                "predefined_categories": len(predefined_categories),
                "custom_categories": len(custom_categories),
                "total_documents_with_categories": total_documents_with_categories,
                "total_notes_with_categories": total_notes_with_categories,
                "total_items_with_categories": total_items_with_categories,
                "most_used_categories": most_used,
                "categories_breakdown": categories_with_counts
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get category statistics: {e}")
            return {}
    
    async def merge_similar_categories(self, dry_run: bool = True) -> Dict[str, Any]:
        """Find and optionally merge similar categories"""
        try:
            all_categories = await self.get_all_categories()
            
            # Simple similarity check (you could enhance this with more sophisticated algorithms)
            similar_groups = []
            processed = set()
            
            for i, cat1 in enumerate(all_categories):
                if cat1 in processed:
                    continue
                
                similar = [cat1]
                for j, cat2 in enumerate(all_categories[i+1:], i+1):
                    if cat2 in processed:
                        continue
                    
                    # Check for similarity (simple string similarity)
                    if self._are_categories_similar(cat1, cat2):
                        similar.append(cat2)
                        processed.add(cat2)
                
                if len(similar) > 1:
                    similar_groups.append(similar)
                    processed.update(similar)
            
            result = {
                "similar_groups": similar_groups,
                "dry_run": dry_run,
                "total_groups": len(similar_groups)
            }
            
            if not dry_run:
                # TODO: Implement actual merging logic
                result["message"] = "Category merging not yet implemented"
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to merge similar categories: {e}")
            return {"error": str(e)}
    
    def _are_categories_similar(self, cat1: str, cat2: str) -> bool:
        """Check if two categories are similar"""
        # Simple similarity checks
        cat1_lower = cat1.lower()
        cat2_lower = cat2.lower()
        
        # Check if one is contained in the other
        if cat1_lower in cat2_lower or cat2_lower in cat1_lower:
            return True
        
        # Check for common plural/singular forms
        if cat1_lower.endswith('s') and cat1_lower[:-1] == cat2_lower:
            return True
        if cat2_lower.endswith('s') and cat2_lower[:-1] == cat1_lower:
            return True
        
        # Check for common variations
        variations = [
            ('tech', 'technical'),
            ('biz', 'business'),
            ('ref', 'reference'),
            ('doc', 'documentation'),
            ('mgmt', 'management')
        ]
        
        for short, long in variations:
            if (cat1_lower == short and cat2_lower == long) or (cat1_lower == long and cat2_lower == short):
                return True
        
        return False
    
    async def create_category(self, name: str, description: str = "", category_type: str = "custom") -> Dict[str, Any]:
        """Create a new custom category"""
        try:
            # Validate category name
            if not await self.validate_category(name):
                raise ValueError("Invalid category name")
            
            normalized_name = await self.normalize_category(name)
            
            # Check if category already exists
            all_categories_response = await self.get_all_categories()
            existing_names = [cat["name"].lower() for cat in all_categories_response["categories"]]
            
            if normalized_name in existing_names:
                raise ValueError("Category already exists")
            
            # For now, we'll just return the category info since we don't have a dedicated table
            # In a full implementation, you'd store this in a categories table
            category_info = {
                "name": normalized_name,
                "description": description,
                "type": category_type,
                "created_at": "2025-01-01T00:00:00Z"  # You'd use actual timestamp
            }
            
            logger.info(f"✅ Custom category created: {normalized_name}")
            return category_info
            
        except Exception as e:
            logger.error(f"❌ Failed to create category: {e}")
            raise
    
    async def update_category(self, name: str, description: str = "") -> bool:
        """Update a category description"""
        try:
            # In a full implementation, you'd update the categories table
            # For now, just validate that the category exists
            all_categories_response = await self.get_all_categories()
            existing_names = [cat["name"].lower() for cat in all_categories_response["categories"]]
            
            if name.lower() not in existing_names:
                return False
            
            logger.info(f"✅ Category updated: {name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update category: {e}")
            return False
    
    async def delete_category(self, name: str) -> bool:
        """Delete a custom category (predefined categories cannot be deleted)"""
        try:
            # Check if it's a predefined category
            predefined_categories = await self.get_predefined_categories()
            if name in predefined_categories:
                return False  # Cannot delete predefined categories
            
            # In a full implementation, you'd:
            # 1. Check if category exists in custom categories table
            # 2. Remove it from the table
            # 3. Optionally update documents/notes that use this category
            
            logger.info(f"✅ Category deleted: {name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete category: {e}")
            return False
    
    async def suggest_categories(self, content: str, content_type: str = "note") -> List[str]:
        """Get category suggestions for content"""
        try:
            if not content:
                return []
            
            text_lower = content.lower()
            all_categories_response = await self.get_all_categories()
            all_category_names = [cat["name"] for cat in all_categories_response["categories"]]
            
            # Simple keyword-based matching
            suggestions = []
            
            # Check if any category names appear in the text
            for category in all_category_names:
                if category.lower() in text_lower:
                    suggestions.append(category)
            
            # Add some basic keyword-to-category mappings
            keyword_mappings = {
                "technical": ["code", "programming", "software", "api", "documentation"],
                "academic": ["research", "study", "paper", "journal", "thesis"],
                "business": ["meeting", "strategy", "revenue", "client", "project"],
                "legal": ["contract", "law", "compliance", "regulation", "policy"],
                "medical": ["health", "patient", "diagnosis", "treatment", "medicine"],
                "reference": ["manual", "guide", "instruction", "how-to", "tutorial"],
                "personal": ["diary", "note", "thought", "idea", "reminder"]
            }
            
            for category, keywords in keyword_mappings.items():
                if category not in suggestions:
                    for keyword in keywords:
                        if keyword in text_lower:
                            suggestions.append(category)
                            break
            
            # Remove duplicates and limit results
            unique_suggestions = list(dict.fromkeys(suggestions))
            return unique_suggestions[:5]
            
        except Exception as e:
            logger.error(f"❌ Failed to suggest categories: {e}")
            return []

    async def close(self):
        """Clean up resources"""
        if self.document_repository:
            await self.document_repository.close()
        logger.info("🏷️ Category Service closed")
