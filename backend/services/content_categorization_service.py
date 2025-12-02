"""
Content Categorization Service
Uses LLM to automatically categorize content into relevant categories
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)


class ContentCategorizationService:
    """Service for LLM-based content categorization"""
    
    def __init__(self):
        self.openrouter_client = None
        self.initialized = False
        
        # Predefined category mappings for RSS content
        self.rss_categories = {
            "news": ["news", "current_events", "breaking_news", "politics", "world_news"],
            "technology": ["technology", "tech", "software", "hardware", "ai", "machine_learning", "programming"],
            "science": ["science", "research", "scientific", "discovery", "study", "experiment"],
            "business": ["business", "finance", "economics", "market", "investment", "startup"],
            "health": ["health", "medical", "medicine", "healthcare", "wellness", "fitness"],
            "entertainment": ["entertainment", "movies", "music", "tv", "celebrity", "arts"],
            "sports": ["sports", "athletics", "games", "competition", "fitness"],
            "education": ["education", "learning", "academic", "school", "university", "training"],
            "environment": ["environment", "climate", "sustainability", "green", "ecology"],
            "lifestyle": ["lifestyle", "fashion", "food", "travel", "culture", "social"],
            "opinion": ["opinion", "editorial", "commentary", "analysis", "perspective"],
            "local": ["local", "community", "regional", "city", "neighborhood"]
        }
        
    async def initialize(self):
        """Initialize the content categorization service"""
        try:
            # Initialize OpenRouter client for categorization with automatic reasoning support
            from utils.openrouter_client import get_openrouter_client
            self.openrouter_client = get_openrouter_client()
            
            self.initialized = True
            logger.info("✅ Content Categorization Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Content Categorization Service: {e}")
            raise
    
    async def categorize_content(
        self, 
        title: str,
        content: str,
        source_url: str = None,
        feed_category: str = None
    ) -> List[str]:
        """
        Categorize content using LLM to determine relevant categories
        
        Args:
            title: Article title
            content: Article content (can be truncated for efficiency)
            source_url: Source URL for context
            feed_category: RSS feed category if available
            
        Returns:
            List of up to 5 relevant categories
        """
        if not self.initialized:
            raise RuntimeError("Content Categorization Service not initialized")
        
        start_time = time.time()
        
        try:
            logger.info(f"🏷️ Categorizing content: '{title[:100]}...'")
            
            # Truncate content for efficiency (first 2000 characters should be sufficient)
            truncated_content = content[:2000] + "..." if len(content) > 2000 else content
            
            # Create categorization prompt
            categorization_prompt = f"""Analyze this content and assign up to 5 relevant categories from the provided list.

CONTENT TITLE: "{title}"
CONTENT PREVIEW: "{truncated_content}"
SOURCE URL: "{source_url or 'Unknown'}"
RSS FEED CATEGORY: "{feed_category or 'None'}"

AVAILABLE CATEGORIES:
{self._format_categories_for_prompt()}

CATEGORIZATION RULES:
1. Choose the most relevant categories (up to 5)
2. Consider both title and content
3. Use the RSS feed category as a hint but don't rely solely on it
4. Be specific and accurate
5. If content spans multiple domains, include all relevant categories
6. For technology/science overlap, include both if appropriate

Respond with ONLY valid JSON:
{{
    "categories": ["category1", "category2", "category3"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of categorization"
}}"""

            # Check if we have an enabled model for categorization
            from services.settings_service import settings_service
            enabled_models = await settings_service.get_enabled_models()
            
            if not enabled_models:
                logger.warning("⚠️ No models enabled for content categorization, using fallback")
                return self._fallback_categorization(title, content, feed_category)
            
            # Use first enabled model for categorization (fast and efficient)
            categorization_model = enabled_models[0]
            logger.debug(f"🧠 Using model {categorization_model} for content categorization")
            
            response = await self.openrouter_client.chat.completions.create(
                model=categorization_model,
                messages=[{
                    "role": "user", 
                    "content": categorization_prompt
                }],
                max_tokens=300,
                temperature=0.1  # Low temperature for consistent categorization
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                categorization = json.loads(response_content)
                
                # Validate required fields
                if "categories" not in categorization:
                    raise ValueError("Missing required field: categories")
                
                categories = categorization["categories"]
                
                # Ensure categories are valid
                valid_categories = self._get_all_valid_categories()
                validated_categories = []
                
                for category in categories:
                    # Check exact match first
                    if category in valid_categories:
                        validated_categories.append(category)
                    else:
                        # Try to find closest match
                        closest_match = self._find_closest_category(category, valid_categories)
                        if closest_match:
                            validated_categories.append(closest_match)
                            logger.debug(f"🔍 Mapped category '{category}' to '{closest_match}'")
                
                # Limit to 5 categories and remove duplicates
                final_categories = list(dict.fromkeys(validated_categories))[:5]
                
                # Add metadata
                categorization_time = time.time() - start_time
                
                logger.info(f"✅ Content categorized: {final_categories} (confidence: {categorization.get('confidence', 0.0):.2f}, time: {categorization_time:.2f}s)")
                
                return final_categories
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse categorization response: {e}")
                return []  # No categories if LLM fails
                
        except Exception as e:
            logger.error(f"❌ Content categorization failed: {e}")
            return []  # No categories if LLM fails
    
    def _format_categories_for_prompt(self) -> str:
        """Format categories for the LLM prompt"""
        formatted = []
        for main_category, subcategories in self.rss_categories.items():
            formatted.append(f"- {main_category}: {', '.join(subcategories)}")
        return "\n".join(formatted)
    
    def _get_all_valid_categories(self) -> List[str]:
        """Get all valid category names"""
        categories = []
        for main_category, subcategories in self.rss_categories.items():
            categories.append(main_category)
            categories.extend(subcategories)
        return categories
    
    def _find_closest_category(self, category: str, valid_categories: List[str]) -> Optional[str]:
        """Find the closest matching category using simple string matching"""
        category_lower = category.lower()
        
        # Exact match
        for valid_cat in valid_categories:
            if valid_cat.lower() == category_lower:
                return valid_cat
        
        # Partial match
        for valid_cat in valid_categories:
            if category_lower in valid_cat.lower() or valid_cat.lower() in category_lower:
                return valid_cat
        
        # Word-based match
        category_words = set(category_lower.split())
        for valid_cat in valid_categories:
            valid_words = set(valid_cat.lower().split())
            if category_words & valid_words:  # Intersection
                return valid_cat
        
        return None
    
    
    
    async def close(self):
        """Close the service"""
        if self.openrouter_client:
            await self.openrouter_client.close()
        logger.info("✅ Content Categorization Service closed")


# Global instance
content_categorization_service = ContentCategorizationService()


async def get_content_categorization_service() -> ContentCategorizationService:
    """Get the global content categorization service instance"""
    return content_categorization_service 