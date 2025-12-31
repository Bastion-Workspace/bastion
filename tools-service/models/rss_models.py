"""
RSS Models for Structured Data Handling
Pydantic models for RSS feeds, articles, and subscriptions
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl
import hashlib


class RSSFeed(BaseModel):
    """RSS Feed model for structured feed data"""
    feed_id: str = Field(..., description="Unique identifier for the RSS feed")
    feed_url: str = Field(..., description="URL of the RSS feed")
    feed_name: str = Field(..., description="Human-readable name for the feed")
    category: str = Field(..., description="Primary category for the feed content")
    tags: List[str] = Field(default=[], description="Tags for the feed")
    check_interval: int = Field(default=3600, description="Interval in seconds between feed checks")
    last_check: Optional[datetime] = Field(None, description="Timestamp of last feed check")
    is_active: bool = Field(default=True, description="Whether the feed is currently being monitored")
    user_id: Optional[str] = Field(None, description="User ID for user-specific feeds, NULL for global feeds")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    @validator('feed_url')
    def validate_feed_url(cls, v):
        """Validate RSS feed URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Feed URL must start with http:// or https://')
        return v

    @validator('check_interval')
    def validate_check_interval(cls, v):
        """Validate check interval is reasonable"""
        if v < 300:  # Minimum 5 minutes
            raise ValueError('Check interval must be at least 300 seconds (5 minutes)')
        if v > 86400:  # Maximum 24 hours
            raise ValueError('Check interval must be at most 86400 seconds (24 hours)')
        return v


class RSSArticle(BaseModel):
    """RSS Article model for structured article data"""
    article_id: str = Field(..., description="Unique identifier for the RSS article")
    feed_id: str = Field(..., description="Reference to the RSS feed")
    title: str = Field(..., description="Article title from RSS feed")
    description: Optional[str] = Field(None, description="Article description/summary from RSS feed")
    full_content: Optional[str] = Field(None, description="Full article content extracted by Crawl4AI")
    full_content_html: Optional[str] = Field(None, description="Original HTML content with images in position")
    images: Optional[List[Dict[str, Any]]] = Field(None, description="Images extracted from the article")
    link: str = Field(..., description="URL to the full article")
    published_date: Optional[datetime] = Field(None, description="Publication date from RSS feed")
    is_processed: bool = Field(default=False, description="Whether the full article has been downloaded and processed")
    is_read: bool = Field(default=False, description="Whether the user has marked this article as read")
    content_hash: Optional[str] = Field(None, description="Hash of content for duplicate detection")
    user_id: Optional[str] = Field(None, description="User ID for user-specific articles, NULL for global articles")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    @validator('link')
    def validate_article_link(cls, v):
        """Validate article link URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Article link must start with http:// or https://')
        return v

    def generate_content_hash(self) -> str:
        """Generate content hash for duplicate detection"""
        content = f"{self.title}:{self.description or ''}:{self.link}"
        return hashlib.sha256(content.encode()).hexdigest()


class RSSFeedSubscription(BaseModel):
    """RSS Feed Subscription model for user-feed relationships"""
    subscription_id: str = Field(..., description="Unique identifier for the subscription")
    feed_id: str = Field(..., description="Reference to the RSS feed")
    user_id: str = Field(..., description="User ID for the subscription")
    is_active: bool = Field(default=True, description="Whether the subscription is active")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class RSSFeedCreate(BaseModel):
    """Model for creating new RSS feeds"""
    feed_url: str = Field(..., description="URL of the RSS feed")
    feed_name: str = Field(..., description="Human-readable name for the feed")
    category: str = Field(..., description="Primary category for the feed content")
    tags: List[str] = Field(default=[], description="Tags for the feed")
    check_interval: int = Field(default=3600, description="Interval in seconds between feed checks")
    user_id: Optional[str] = Field(None, description="User ID for user-specific feeds")

    @validator('feed_url')
    def validate_feed_url(cls, v):
        """Validate RSS feed URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Feed URL must start with http:// or https://')
        return v


class RSSArticleImport(BaseModel):
    """Model for importing RSS articles"""
    article_id: str = Field(..., description="Article ID to import")
    collection_name: Optional[str] = Field(None, description="Target collection name")
    user_id: str = Field(..., description="User ID importing the article")


class RSSFeedPollResult(BaseModel):
    """Structured result from RSS feed polling"""
    feed_id: str = Field(..., description="Feed ID that was polled")
    status: str = Field(..., description="Polling status: success, error, no_new_articles")
    articles_found: int = Field(default=0, description="Number of new articles found")
    articles_added: int = Field(default=0, description="Number of articles actually added")
    error_message: Optional[str] = Field(None, description="Error message if polling failed")
    polled_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of polling")


class RSSArticleProcessResult(BaseModel):
    """Structured result from RSS article processing"""
    article_id: str = Field(..., description="Article ID that was processed")
    status: str = Field(..., description="Processing status: success, error, already_processed")
    content_downloaded: bool = Field(default=False, description="Whether full content was downloaded")
    content_length: Optional[int] = Field(None, description="Length of downloaded content")
    embeddings_generated: bool = Field(default=False, description="Whether embeddings were generated")
    collection_added: Optional[str] = Field(None, description="Collection where article was added")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of processing")


class RSSFeedHealth(BaseModel):
    """RSS Feed health and performance metrics"""
    feed_id: str = Field(..., description="Feed ID")
    is_healthy: bool = Field(..., description="Whether the feed is responding properly")
    last_successful_poll: Optional[datetime] = Field(None, description="Last successful poll timestamp")
    consecutive_failures: int = Field(default=0, description="Number of consecutive polling failures")
    average_response_time: Optional[float] = Field(None, description="Average response time in seconds")
    articles_per_day: Optional[float] = Field(None, description="Average articles per day")
    last_article_date: Optional[datetime] = Field(None, description="Date of most recent article")


class RSSUserPreferences(BaseModel):
    """User preferences for RSS feed management"""
    user_id: str = Field(..., description="User ID")
    default_import_collection: Optional[str] = Field(None, description="Default collection for imported articles")
    auto_mark_read: bool = Field(default=False, description="Automatically mark articles as read when viewed")
    notification_enabled: bool = Field(default=True, description="Enable notifications for new articles")
    max_articles_per_feed: int = Field(default=100, description="Maximum articles to keep per feed")
    preferred_categories: List[str] = Field(default=[], description="Preferred article categories")
    auto_import_keywords: List[str] = Field(default=[], description="Keywords for automatic article import")
