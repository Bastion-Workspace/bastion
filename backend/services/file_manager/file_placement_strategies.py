"""
File Placement Strategies for FileManager Service
Defines how different source types should be placed in the folder structure
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models.file_placement_models import SourceType, FilePlacementRequest

logger = logging.getLogger(__name__)


class FilePlacementStrategy:
    """Base class for file placement strategies"""
    
    def get_folder_path(self, request: FilePlacementRequest) -> List[str]:
        """Get the folder path for a file placement request"""
        raise NotImplementedError
    
    def get_filename(self, request: FilePlacementRequest) -> str:
        """Get the filename for a file placement request"""
        raise NotImplementedError
    
    def get_tags(self, request: FilePlacementRequest) -> List[str]:
        """Get additional tags for a file placement request"""
        return request.tags
    
    def get_description(self, request: FilePlacementRequest) -> Optional[str]:
        """Get the description for a file placement request"""
        return request.description


class RSSPlacementStrategy(FilePlacementStrategy):
    """Strategy for placing RSS articles"""
    
    def get_folder_path(self, request: FilePlacementRequest) -> List[str]:
        """RSS articles go to: Web Sources / [Feed Name]"""
        feed_name = request.source_metadata.get("feed_name", "Unknown Feed")
        return ["Web Sources", feed_name]
    
    def get_filename(self, request: FilePlacementRequest) -> str:
        """Generate Markdown filename from title with date"""
        # Clean title for filename
        clean_title = "".join(c for c in request.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:50]  # Limit length
        
        # Add date if available
        published_date = request.source_metadata.get("published_date")
        if published_date:
            try:
                date_str = datetime.fromisoformat(published_date.replace('Z', '+00:00')).strftime("%Y-%m-%d")
                return f"{date_str}_{clean_title}.md"
            except:
                pass
        
        return f"{clean_title}.md"
    
    def get_tags(self, request: FilePlacementRequest) -> List[str]:
        """Add RSS-specific tags"""
        tags = request.tags.copy()
        tags.extend(["rss", "imported"])
        
        feed_name = request.source_metadata.get("feed_name")
        if feed_name:
            tags.append(feed_name.lower().replace(" ", "-"))
        
        return tags
    
    def get_description(self, request: FilePlacementRequest) -> Optional[str]:
        """Get description from source metadata or generate one"""
        if request.description:
            return request.description
        
        feed_name = request.source_metadata.get("feed_name", "Unknown Feed")
        return f"Article from {feed_name}"


class ChatPlacementStrategy(FilePlacementStrategy):
    """Strategy for placing chat-generated content"""
    
    def get_folder_path(self, request: FilePlacementRequest) -> List[str]:
        """Chat content goes to: Chat Responses / [Date]"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return ["Chat Responses", date_str]
    
    def get_filename(self, request: FilePlacementRequest) -> str:
        """Generate filename from title with timestamp"""
        clean_title = "".join(c for c in request.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:30]  # Limit length
        
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{timestamp}_{clean_title}.txt"
    
    def get_tags(self, request: FilePlacementRequest) -> List[str]:
        """Add chat-specific tags"""
        tags = request.tags.copy()
        tags.extend(["chat", "generated"])
        
        conversation_id = request.source_metadata.get("conversation_id")
        if conversation_id:
            tags.append(f"conversation-{conversation_id}")
        
        return tags
    
    def get_description(self, request: FilePlacementRequest) -> Optional[str]:
        """Get description or generate one"""
        if request.description:
            return request.description
        
        return "Content generated from chat conversation"


class CodingPlacementStrategy(FilePlacementStrategy):
    """Strategy for placing coding-generated content"""
    
    def get_folder_path(self, request: FilePlacementRequest) -> List[str]:
        """Coding content goes to: Code Generated / [Language] / [Date]"""
        language = request.source_metadata.get("language", "unknown")
        date_str = datetime.now().strftime("%Y-%m-%d")
        return ["Code Generated", language, date_str]
    
    def get_filename(self, request: FilePlacementRequest) -> str:
        """Generate filename with language extension"""
        clean_title = "".join(c for c in request.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:30]  # Limit length
        
        language = request.source_metadata.get("language", "txt")
        extension = self._get_extension_for_language(language)
        
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{timestamp}_{clean_title}.{extension}"
    
    def _get_extension_for_language(self, language: str) -> str:
        """Get file extension for programming language"""
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "go": "go",
            "rust": "rs",
            "php": "php",
            "ruby": "rb",
            "swift": "swift",
            "kotlin": "kt",
            "scala": "scala",
            "html": "html",
            "css": "css",
            "sql": "sql",
            "bash": "sh",
            "powershell": "ps1"
        }
        return extensions.get(language.lower(), "txt")
    
    def get_tags(self, request: FilePlacementRequest) -> List[str]:
        """Add coding-specific tags"""
        tags = request.tags.copy()
        tags.extend(["code", "generated"])
        
        language = request.source_metadata.get("language")
        if language:
            tags.append(f"language-{language.lower()}")
        
        return tags
    
    def get_description(self, request: FilePlacementRequest) -> Optional[str]:
        """Get description or generate one"""
        if request.description:
            return request.description
        
        language = request.source_metadata.get("language", "unknown")
        return f"Code generated in {language}"


class UploadPlacementStrategy(FilePlacementStrategy):
    """Strategy for placing uploaded files"""
    
    def get_folder_path(self, request: FilePlacementRequest) -> List[str]:
        """Uploaded files go to: Uploads / [Date]"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return ["Uploads", date_str]
    
    def get_filename(self, request: FilePlacementRequest) -> str:
        """Use provided filename or generate one"""
        if request.filename:
            return request.filename
        
        clean_title = "".join(c for c in request.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:30]  # Limit length
        
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{timestamp}_{clean_title}.txt"
    
    def get_tags(self, request: FilePlacementRequest) -> List[str]:
        """Add upload-specific tags"""
        tags = request.tags.copy()
        tags.extend(["upload", "user-provided"])
        return tags


class WebScrapingPlacementStrategy(FilePlacementStrategy):
    """Strategy for placing web-scraped content"""
    
    def get_folder_path(self, request: FilePlacementRequest) -> List[str]:
        """Web-scraped content goes to: Web Sources / Scraped / [Domain]"""
        domain = request.source_metadata.get("domain", "unknown")
        return ["Web Sources", "Scraped", domain]
    
    def get_filename(self, request: FilePlacementRequest) -> str:
        """Generate Markdown filename from title with domain"""
        clean_title = "".join(c for c in request.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:30]  # Limit length
        
        domain = request.source_metadata.get("domain", "unknown")
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{timestamp}_{domain}_{clean_title}.md"
    
    def get_tags(self, request: FilePlacementRequest) -> List[str]:
        """Add web scraping-specific tags"""
        tags = request.tags.copy()
        tags.extend(["web-scraped", "imported"])
        
        domain = request.source_metadata.get("domain")
        if domain:
            tags.append(f"domain-{domain}")
        
        return tags


class CalibrePlacementStrategy(FilePlacementStrategy):
    """Strategy for placing Calibre-imported content"""
    
    def get_folder_path(self, request: FilePlacementRequest) -> List[str]:
        """Calibre content goes to: Calibre Library / [Author]"""
        author = request.author or "Unknown Author"
        return ["Calibre Library", author]
    
    def get_filename(self, request: FilePlacementRequest) -> str:
        """Generate filename from title and author"""
        clean_title = "".join(c for c in request.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:30]  # Limit length
        
        author = request.author or "Unknown"
        clean_author = "".join(c for c in author if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_author = clean_author[:20]  # Limit length
        
        return f"{clean_author}_{clean_title}.txt"
    
    def get_tags(self, request: FilePlacementRequest) -> List[str]:
        """Add Calibre-specific tags"""
        tags = request.tags.copy()
        tags.extend(["calibre", "book", "imported"])
        
        if request.author:
            tags.append(f"author-{request.author.lower().replace(' ', '-')}")
        
        return tags


class ManualPlacementStrategy(FilePlacementStrategy):
    """Strategy for manually placed files"""
    
    def get_folder_path(self, request: FilePlacementRequest) -> List[str]:
        """Manual files go to: Manual / [Date]"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return ["Manual", date_str]
    
    def get_filename(self, request: FilePlacementRequest) -> str:
        """Use provided filename or generate one"""
        if request.filename:
            return request.filename
        
        clean_title = "".join(c for c in request.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title[:30]  # Limit length
        
        timestamp = datetime.now().strftime("%H%M%S")
        return f"{timestamp}_{clean_title}.txt"
    
    def get_tags(self, request: FilePlacementRequest) -> List[str]:
        """
        Return tags from request (folder inheritance happens elsewhere)
        
        **ROOSEVELT FIX**: Don't add "manual" tags - let folder inheritance work!
        """
        return request.tags.copy()


class FilePlacementStrategyFactory:
    """Factory for creating file placement strategies"""
    
    _strategies = {
        SourceType.RSS: RSSPlacementStrategy(),
        SourceType.CHAT: ChatPlacementStrategy(),
        SourceType.CODING: CodingPlacementStrategy(),
        SourceType.UPLOAD: UploadPlacementStrategy(),
        SourceType.WEB_SCRAPING: WebScrapingPlacementStrategy(),
        SourceType.CALIBRE: CalibrePlacementStrategy(),
        SourceType.MANUAL: ManualPlacementStrategy(),
    }
    
    @classmethod
    def get_strategy(cls, source_type: SourceType) -> FilePlacementStrategy:
        """Get the appropriate strategy for a source type"""
        strategy = cls._strategies.get(source_type)
        if not strategy:
            logger.warning(f"⚠️ No strategy found for source type {source_type}, using manual strategy")
            return cls._strategies[SourceType.MANUAL]
        
        return strategy
    
    @classmethod
    def register_strategy(cls, source_type: SourceType, strategy: FilePlacementStrategy):
        """Register a custom strategy for a source type"""
        cls._strategies[source_type] = strategy
        logger.info(f"✅ Registered custom strategy for source type {source_type}")
