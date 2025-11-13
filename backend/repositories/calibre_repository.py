"""
Calibre Repository - Database operations for Calibre library metadata
Uses calibredb command-line tool for reliable Calibre library access
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class CalibreBook:
    """Calibre book metadata"""
    id: int
    title: str
    authors: str
    path: str
    timestamp: datetime
    series: Optional[str] = None
    series_index: Optional[float] = None
    tags: List[str] = None
    comments: Optional[str] = None
    publisher: Optional[str] = None
    pubdate: Optional[datetime] = None
    isbn: Optional[str] = None
    formats: List[str] = None
    size: Optional[int] = None
    rating: Optional[int] = None
    
    def to_citation_dict(self) -> Dict[str, Any]:
        """Convert to citation format for search results"""
        return {
            "id": f"calibre_{self.id}",
            "title": self.title,
            "authors": self.authors,
            "type": "calibre_book",
            "metadata": {
                "series": self.series,
                "series_index": self.series_index,
                "publisher": self.publisher,
                "pubdate": self.pubdate.isoformat() if self.pubdate else None,
                "isbn": self.isbn,
                "tags": self.tags or [],
                "rating": self.rating,
                "formats": self.formats or [],
                "size": self.size,
                "path": self.path,
                "timestamp": self.timestamp.isoformat()
            },
            "snippet": self.comments[:200] + "..." if self.comments and len(self.comments) > 200 else self.comments,
            "score": 0.8  # Default relevance score
        }


class CalibreRepository:
    """Repository for Calibre database operations using calibredb"""
    
    def __init__(self):
        self.library_path: Optional[str] = None
        self._initialized = False
        self._calibredb_available = False
    
    async def initialize(self):
        """Initialize the Calibre repository"""
        try:
            logger.info("📚 Initializing Calibre Repository with calibredb...")
            
            self.library_path = settings.CALIBRE_LIBRARY_PATH
            
            # Check if Calibre is enabled and paths exist
            if not settings.CALIBRE_ENABLED:
                logger.info("📚 Calibre integration disabled")
                return
            
            if not os.path.exists(self.library_path):
                logger.warning(f"⚠️ Calibre library path not found: {self.library_path}")
                return
            
            # Check if calibredb is available
            if not await self._check_calibredb():
                logger.error("❌ calibredb command not available")
                return
            
            # Test library connection
            await self._test_library_connection()
            self._initialized = True
            
            logger.info("✅ Calibre Repository initialized with calibredb")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Calibre Repository: {e}")
            self._initialized = False
    
    async def _check_calibredb(self) -> bool:
        """Check if calibredb command is available"""
        try:
            result = await asyncio.create_subprocess_exec(
                "calibredb", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                version = stdout.decode().strip()
                logger.info(f"📚 calibredb version: {version}")
                self._calibredb_available = True
                return True
            else:
                logger.error(f"❌ calibredb not available: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to check calibredb: {e}")
            return False
    
    async def _test_library_connection(self):
        """Test Calibre library connection using calibredb"""
        try:
            # List books to test connection
            result = await asyncio.create_subprocess_exec(
                "calibredb", "list", "--library-path", self.library_path, "--limit", "1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logger.info("📚 Calibre library connection successful")
            else:
                error_msg = stderr.decode().strip()
                logger.error(f"❌ Calibre library connection failed: {error_msg}")
                raise Exception(f"Library connection failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"❌ Failed to test library connection: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Calibre integration is available"""
        return (settings.CALIBRE_ENABLED and 
                self._initialized and 
                self._calibredb_available and
                self.library_path and 
                os.path.exists(self.library_path))
    
    async def search_books(self, query: str, limit: int = 50, include_full_text: bool = True) -> List[CalibreBook]:
        """Search books using calibredb search command"""
        if not self.is_available():
            logger.warning("📚 Calibre not available for search")
            return []
        
        try:
            logger.info(f"📚 Searching Calibre library for: '{query}' (limit: {limit})")
            
            # Build calibredb search command
            cmd = [
                "calibredb", "list",
                "--library-path", self.library_path,
                "--search", query,
                "--limit", str(limit),
                "--fields", "id,title,authors,series,series_index,tags,comments,publisher,pubdate,isbn,rating,formats,size,timestamp,path"
            ]
            
            # Execute search
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode().strip()
                logger.error(f"❌ calibredb search failed: {error_msg}")
                return []
            
            # Parse CSV output
            output = stdout.decode().strip()
            if not output:
                logger.info(f"📚 No results found for query: '{query}'")
                return []
            
            # Parse the CSV output
            books = await self._parse_calibredb_output(output)
            
            logger.info(f"📚 Found {len(books)} Calibre books for query: '{query}'")
            return books
            
        except Exception as e:
            logger.error(f"❌ Calibre search failed: {e}")
            return []
    
    async def _parse_calibredb_output(self, output: str) -> List[CalibreBook]:
        """Parse calibredb CSV output into CalibreBook objects"""
        try:
            lines = output.strip().split('\n')
            if len(lines) < 2:  # Need header + at least one data row
                return []
            
            # Parse header to get column indices
            header = lines[0].split(',')
            column_map = {col.strip(): i for i, col in enumerate(header)}
            
            books = []
            for line in lines[1:]:  # Skip header
                if not line.strip():
                    continue
                
                # Simple CSV parsing (handles basic cases)
                fields = self._parse_csv_line(line)
                
                if len(fields) < len(header):
                    logger.warning(f"📚 Skipping malformed line: {line}")
                    continue
                
                try:
                    # Extract fields using column map
                    book_id = int(fields[column_map.get('id', 0)])
                    title = fields[column_map.get('title', 1)].strip('"')
                    authors = fields[column_map.get('authors', 2)].strip('"')
                    series = fields[column_map.get('series', 3)].strip('"') if column_map.get('series') is not None else None
                    series_index_str = fields[column_map.get('series_index', 4)].strip('"') if column_map.get('series_index') is not None else None
                    tags_str = fields[column_map.get('tags', 5)].strip('"') if column_map.get('tags') is not None else None
                    comments = fields[column_map.get('comments', 6)].strip('"') if column_map.get('comments') is not None else None
                    publisher = fields[column_map.get('publisher', 7)].strip('"') if column_map.get('publisher') is not None else None
                    pubdate_str = fields[column_map.get('pubdate', 8)].strip('"') if column_map.get('pubdate') is not None else None
                    isbn = fields[column_map.get('isbn', 9)].strip('"') if column_map.get('isbn') is not None else None
                    rating_str = fields[column_map.get('rating', 10)].strip('"') if column_map.get('rating') is not None else None
                    formats_str = fields[column_map.get('formats', 11)].strip('"') if column_map.get('formats') is not None else None
                    size_str = fields[column_map.get('size', 12)].strip('"') if column_map.get('size') is not None else None
                    timestamp_str = fields[column_map.get('timestamp', 13)].strip('"') if column_map.get('timestamp') is not None else None
                    path = fields[column_map.get('path', 14)].strip('"') if column_map.get('path') is not None else ""
                    
                    # Parse optional fields
                    series_index = float(series_index_str) if series_index_str and series_index_str != 'None' else None
                    rating = int(rating_str) if rating_str and rating_str != 'None' else None
                    size = int(size_str) if size_str and size_str != 'None' else None
                    
                    # Parse tags
                    tags = []
                    if tags_str and tags_str != 'None':
                        tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                    
                    # Parse formats
                    formats = []
                    if formats_str and formats_str != 'None':
                        formats = [fmt.strip() for fmt in formats_str.split(',') if fmt.strip()]
                    
                    # Parse dates
                    timestamp = datetime.now()
                    if timestamp_str and timestamp_str != 'None':
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        except:
                            pass
                    
                    pubdate = None
                    if pubdate_str and pubdate_str != 'None':
                        try:
                            pubdate = datetime.fromisoformat(pubdate_str.replace('Z', '+00:00'))
                        except:
                            pass
                    
                    book = CalibreBook(
                        id=book_id,
                        title=title or "Unknown Title",
                        authors=authors or "Unknown Author",
                        path=path,
                        timestamp=timestamp,
                        series=series,
                        series_index=series_index,
                        tags=tags,
                        comments=comments,
                        publisher=publisher,
                        pubdate=pubdate,
                        isbn=isbn,
                        formats=formats,
                        size=size,
                        rating=rating
                    )
                    books.append(book)
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"📚 Failed to parse book data: {e}")
                    continue
            
            return books
            
        except Exception as e:
            logger.error(f"❌ Failed to parse calibredb output: {e}")
            return []
    
    def _parse_csv_line(self, line: str) -> List[str]:
        """Simple CSV parsing that handles quoted fields"""
        fields = []
        current_field = ""
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                fields.append(current_field)
                current_field = ""
            else:
                current_field += char
        
        fields.append(current_field)  # Add the last field
        return fields
    
    async def get_book_by_id(self, book_id: int) -> Optional[CalibreBook]:
        """Get a specific book by ID using calibredb"""
        if not self.is_available():
            return None
        
        try:
            cmd = [
                "calibredb", "list",
                "--library-path", self.library_path,
                "--search", f"id:{book_id}",
                "--fields", "id,title,authors,series,series_index,tags,comments,publisher,pubdate,isbn,rating,formats,size,timestamp,path"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                return None
            
            output = stdout.decode().strip()
            if not output:
                return None
            
            books = await self._parse_calibredb_output(output)
            return books[0] if books else None
            
        except Exception as e:
            logger.error(f"❌ Failed to get book {book_id}: {e}")
            return None
    
    async def get_book_file_path(self, book_id: int, format_type: str = "epub") -> Optional[str]:
        """Get the file path for a specific book format"""
        if not self.is_available():
            return None
        
        try:
            # Get book details to find the path
            book = await self.get_book_by_id(book_id)
            if not book or not book.path:
                return None
            
            # Construct the file path
            book_dir = os.path.join(self.library_path, book.path)
            if not os.path.exists(book_dir):
                return None
            
            # Look for the requested format
            for filename in os.listdir(book_dir):
                if filename.lower().endswith(f".{format_type.lower()}"):
                    return os.path.join(book_dir, filename)
            
            # If specific format not found, return any available format
            for filename in os.listdir(book_dir):
                if '.' in filename:
                    return os.path.join(book_dir, filename)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get book file path for {book_id}: {e}")
            return None
    
    async def get_library_stats(self) -> Dict[str, Any]:
        """Get Calibre library statistics using calibredb"""
        if not self.is_available():
            return {"available": False}
        
        try:
            # Get total book count
            result = await asyncio.create_subprocess_exec(
                "calibredb", "list", "--library-path", self.library_path, "--limit", "1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                return {"available": False, "error": stderr.decode()}
            
            # Count total books by getting all IDs
            count_result = await asyncio.create_subprocess_exec(
                "calibredb", "list", "--library-path", self.library_path, "--fields", "id",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            count_stdout, count_stderr = await count_result.communicate()
            
            book_count = 0
            if count_result.returncode == 0:
                lines = count_stdout.decode().strip().split('\n')
                book_count = max(0, len(lines) - 1)  # Subtract header line
            
            return {
                "available": True,
                "books": book_count,
                "library_path": self.library_path,
                "calibredb_available": self._calibredb_available
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get library stats: {e}")
            return {"available": False, "error": str(e)}
    
    async def search_books_advanced(self, query: str, limit: int = 50, search_fields: List[str] = None) -> List[CalibreBook]:
        """Advanced search with field-specific queries"""
        if not self.is_available():
            return []
        
        try:
            # Build advanced search query
            if search_fields:
                # Use field-specific search syntax
                field_queries = []
                for field in search_fields:
                    if field == "title":
                        field_queries.append(f"title:{query}")
                    elif field == "authors":
                        field_queries.append(f"authors:{query}")
                    elif field == "series":
                        field_queries.append(f"series:{query}")
                    elif field == "tags":
                        field_queries.append(f"tags:{query}")
                    elif field == "comments":
                        field_queries.append(f"comments:{query}")
                
                if field_queries:
                    search_query = " or ".join(field_queries)
                else:
                    search_query = query
            else:
                search_query = query
            
            logger.info(f"📚 Advanced search: '{search_query}'")
            
            # Use the standard search method with the advanced query
            return await self.search_books(search_query, limit)
            
        except Exception as e:
            logger.error(f"❌ Advanced search failed: {e}")
            return []
