"""
Audiobookshelf Music Client
Implementation of BaseMusicClient for Audiobookshelf servers
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import quote
import httpx

from .base_client import BaseMusicClient

logger = logging.getLogger(__name__)


def _parse_time_string_to_seconds(value: str) -> Optional[float]:
    """Parse Audiobookshelf duration strings (e.g. H:MM:SS, MM:SS) or numeric strings to seconds."""
    s = value.strip()
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            nums = [float(p) for p in parts if p.strip() != ""]
        except ValueError:
            return None
        if len(nums) == 3:
            return nums[0] * 3600 + nums[1] * 60 + nums[2]
        if len(nums) == 2:
            return nums[0] * 60 + nums[1]
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _coerce_positive_seconds(raw: Any) -> Optional[float]:
    """Convert a scalar duration from the API to seconds (handles ms heuristics)."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        return _parse_time_string_to_seconds(raw)
    try:
        n = float(raw)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    # Large integers are often milliseconds (e.g. 3_600_000)
    if n > 86400 and n == int(n):
        n = n / 1000.0
    return n


def _episode_duration_seconds(episode: Dict[str, Any]) -> int:
    """Best-effort episode length for podcast UI (ABS uses several shapes and string durations)."""
    candidates: List[float] = []

    def consider(val: Any) -> None:
        sec = _coerce_positive_seconds(val)
        if sec is not None:
            candidates.append(sec)

    consider(episode.get("duration"))
    consider(episode.get("durationS"))
    consider(episode.get("runtime"))

    audio_file = episode.get("audioFile") or {}
    if isinstance(audio_file, dict):
        consider(audio_file.get("duration"))
        br = audio_file.get("bitRate") or audio_file.get("bitrate")
        sz = audio_file.get("size")
        try:
            if br and sz:
                br_i, sz_i = int(br), int(sz)
                if br_i > 0 and sz_i > 0:
                    candidates.append((sz_i * 8) / br_i)
        except (TypeError, ValueError):
            pass

    audio_track = episode.get("audioTrack") or {}
    if isinstance(audio_track, dict):
        consider(audio_track.get("duration"))
        for nested in (audio_track.get("meta"), audio_track.get("metadata")):
            if isinstance(nested, dict):
                consider(nested.get("duration"))

    media = episode.get("media") or {}
    if isinstance(media, dict):
        consider(media.get("duration"))

    if not candidates:
        return 0
    best = max(candidates)
    return int(round(best)) if best > 0 else 0


class AudiobookshelfClient(BaseMusicClient):
    """Client for Audiobookshelf servers"""
    
    def __init__(self, server_url: str, username: str, password: str, **kwargs):
        """
        Initialize Audiobookshelf client
        
        Args:
            server_url: Audiobookshelf server URL
            username: Username (not used, kept for compatibility)
            password: API token for authentication
        """
        super().__init__(server_url, username, password, **kwargs)
        self.api_token = password  # For Audiobookshelf, password is the API token
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers for Audiobookshelf API"""
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
    
    def _build_url(self, endpoint: str) -> str:
        """Build Audiobookshelf API URL"""
        base_url = f"{self.server_url}/api/{endpoint.lstrip('/')}"
        return base_url
    
    def _extract_author_from_item(self, item: Dict[str, Any]) -> str:
        """
        Extract author from an item (book) from AudioBookShelf API response
        
        Args:
            item: Item dict from AudioBookShelf API
            
        Returns:
            Author name as string, or empty string if not found
        """
        # AudioBookShelf stores metadata in item["media"]["metadata"]
        # Try multiple locations for author information
        metadata = None
        
        # First check media.metadata (detailed item response)
        if item.get("media") and isinstance(item["media"], dict):
            metadata = item["media"].get("metadata", {})
        
        # Fallback to top-level metadata (some API responses)
        if not metadata or not isinstance(metadata, dict):
            metadata = item.get("metadata", {})
        
        # Debug: Log first item's structure
        if not hasattr(self, '_logged_sample'):
            logger.debug(f"Sample item keys: {list(item.keys())[:20]}")
            logger.debug(f"Sample metadata keys: {list(metadata.keys())[:20] if metadata else 'No metadata'}")
            if metadata:
                logger.debug(f"Sample metadata.author: {metadata.get('author')}")
                logger.debug(f"Sample metadata.authors: {metadata.get('authors')}")
                logger.debug(f"Sample metadata structure: {json.dumps(metadata, default=str)[:500]}")
            self._logged_sample = True
        
        author = None
        
        # Check top-level item fields first (some APIs put author here)
        if not author:
            author = item.get("author") or item.get("authorName")
        
        # Check metadata object
        if not author and metadata:
            # Direct author field
            author = metadata.get("author")
            
            # Check authorName
            if not author:
                author = metadata.get("authorName")
            
            # Check authors array (can be list of strings or list of objects)
            if not author and metadata.get("authors"):
                authors = metadata.get("authors")
                if isinstance(authors, list) and len(authors) > 0:
                    # If list of objects, get name from first object
                    if isinstance(authors[0], dict):
                        author = authors[0].get("name") or authors[0].get("author")
                    # If list of strings, use first string
                    elif isinstance(authors[0], str):
                        author = authors[0]
                elif isinstance(authors, str):
                    author = authors
            
            # Check authorLF (Last, First format)
            if not author:
                author = metadata.get("authorLF")
            
            # Check nested metadata (some APIs nest metadata)
            if not author:
                nested_metadata = metadata.get("metadata", {})
                if isinstance(nested_metadata, dict):
                    author = (nested_metadata.get("author") or 
                             nested_metadata.get("authorName") or
                             nested_metadata.get("authorLF"))
                    if not author and nested_metadata.get("authors"):
                        nested_authors = nested_metadata.get("authors")
                        if isinstance(nested_authors, list) and len(nested_authors) > 0:
                            if isinstance(nested_authors[0], dict):
                                author = nested_authors[0].get("name") or nested_authors[0].get("author")
                            elif isinstance(nested_authors[0], str):
                                author = nested_authors[0]
                        elif isinstance(nested_authors, str):
                            author = nested_authors
        
        # Clean up author string
        if author:
            author = str(author).strip()
            if not author or author.lower() == "unknown":
                author = ""
        
        # Log warning for first few items without authors
        if not author:
            if not hasattr(self, '_warned_count'):
                self._warned_count = 0
            if self._warned_count < 5:
                item_name = item.get("name", "Unknown")
                logger.warning(f"Book '{item_name}' has no author extracted. Item keys: {list(item.keys())[:10]}, Metadata keys: {list(metadata.keys())[:10] if metadata else 'No metadata'}")
                self._warned_count += 1
        
        return author or ""
    
    async def _fetch_item_details(self, item_id: str, original_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch detailed item information from AudioBookShelf API
        
        Args:
            item_id: Item ID to fetch
            original_item: Original item from list response (for fallback)
            
        Returns:
            Dict with 'author', 'title', 'cover_art', and 'metadata' keys
        """
        try:
            headers = self._get_headers()
            # Add expanded=1 or include query parameter to get full metadata
            # Try with expanded parameter (common pattern in REST APIs)
            item_url = self._build_url(f"items/{item_id}?expanded=1&include=metadata")
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(item_url, headers=headers)
                response.raise_for_status()
                item_detail = response.json()
                
                # Debug: Log first detailed response to see structure
                if not hasattr(self, '_logged_detail_sample'):
                    logger.info(f"Sample detailed item response keys: {list(item_detail.keys())[:30]}")
                    if item_detail.get("metadata"):
                        logger.info(f"Sample detailed item metadata keys: {list(item_detail['metadata'].keys())[:30]}")
                    if item_detail.get("media"):
                        logger.info(f"Sample detailed item media keys: {list(item_detail['media'].keys())[:30]}")
                    logger.info(f"Sample detailed item structure (first 1000 chars): {json.dumps(item_detail, default=str)[:1000]}")
                    self._logged_detail_sample = True
                
                # Extract author from detailed response
                author = self._extract_author_from_item(item_detail)
                
                # Extract title from media.metadata
                title = None
                if item_detail.get("media") and isinstance(item_detail["media"], dict):
                    media_metadata = item_detail["media"].get("metadata", {})
                    if isinstance(media_metadata, dict):
                        title = media_metadata.get("title")
                
                # Fallback to top-level fields
                if not title:
                    title = (item_detail.get("name") or 
                            item_detail.get("title") or 
                            original_item.get("name", "Unknown"))
                
                # Extract cover art from media.coverPath
                cover_art = None
                if item_detail.get("media") and isinstance(item_detail["media"], dict):
                    cover_art = item_detail["media"].get("coverPath")
                
                # Fallback to top-level fields
                if not cover_art:
                    cover_art = (item_detail.get("coverPath") or 
                                item_detail.get("cover") or 
                                original_item.get("coverPath", ""))
                
                return {
                    "author": author,
                    "title": title,
                    "cover_art": cover_art,
                    "metadata": item_detail
                }
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error fetching details for item {item_id}: {e.response.status_code}")
            # Return original item data as fallback
            return {
                "author": "",
                "title": original_item.get("name", "Unknown"),
                "cover_art": original_item.get("coverPath", ""),
                "metadata": original_item
            }
        except Exception as e:
            logger.debug(f"Failed to fetch details for item {item_id}: {e}")
            # Return original item data as fallback
            return {
                "author": "",
                "title": original_item.get("name", "Unknown"),
                "cover_art": original_item.get("coverPath", ""),
                "metadata": original_item
            }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Audiobookshelf server"""
        try:
            # Use the /api/me endpoint to test authentication
            # This endpoint returns user info if authenticated, 401 if not
            test_url = self._build_url("me")
            headers = self._get_headers()
            
            logger.info("Testing Audiobookshelf connection")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(test_url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Check if we got user info back (indicates successful auth)
                if data.get("user") or data.get("id"):
                    return {"success": True, "auth_method_used": "bearer_token", "message": "Connection successful"}
                else:
                    return {"success": False, "error": "Unexpected response format - no user data received"}
                    
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return {"success": False, "error": "Authentication failed - invalid API token"}
            elif e.response.status_code == 404:
                # Try alternative endpoint if /api/me doesn't exist (fallback to libraries)
                try:
                    libraries_url = self._build_url("libraries")
                    async with httpx.AsyncClient(timeout=10.0) as fallback_client:
                        fallback_response = await fallback_client.get(libraries_url, headers=headers)
                        fallback_response.raise_for_status()
                        return {"success": True, "auth_method_used": "bearer_token", "message": "Connection successful"}
                except Exception as fallback_error:
                    logger.warning(f"Fallback endpoint also failed: {fallback_error}")
                    return {"success": False, "error": f"Endpoint not found (404) - check server URL. Tried /api/me and /api/libraries"}
            else:
                error_text = e.response.text[:200] if e.response.text else "No error details"
                return {"success": False, "error": f"HTTP {e.response.status_code}: {error_text}"}
        except httpx.TimeoutException:
            return {"success": False, "error": "Connection timeout - server did not respond"}
        except Exception as e:
            logger.error(f"Audiobookshelf connection test error: {e}", exc_info=True)
            return {"success": False, "error": f"Connection error: {str(e)}"}
    
    async def get_albums(self) -> List[Dict[str, Any]]:
        """Fetch all books (albums) from Audiobookshelf server"""
        logger.info("AudiobookshelfClient.get_albums() called")
        # Reset instance variables for this call
        if hasattr(self, '_logged_sample'):
            delattr(self, '_logged_sample')
        if hasattr(self, '_warned_count'):
            delattr(self, '_warned_count')
        if hasattr(self, '_logged_detail_sample'):
            delattr(self, '_logged_detail_sample')
        try:
            # Get all libraries first
            libraries_url = self._build_url("libraries")
            headers = self._get_headers()
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get libraries
                libraries_response = await client.get(libraries_url, headers=headers)
                libraries_response.raise_for_status()
                libraries_data = libraries_response.json()
                
                libraries = libraries_data.get("libraries", [])
                if not libraries:
                    logger.warning("No libraries found in Audiobookshelf")
                    return []
                
                # Get items from all libraries with pagination
                all_books = []
                for library in libraries:
                    library_id = library.get("id")
                    if not library_id:
                        continue
                    
                    # Fetch library items with pagination
                    library_books = []
                    page = 0
                    limit = 500  # Fetch in batches of 500
                    
                    while True:
                        # Get library items (books) with pagination
                        items_url = self._build_url(f"libraries/{library_id}/items?limit={limit}&page={page}")
                        logger.info(f"Fetching books from library {library_id} (page: {page}, limit: {limit})")
                        items_response = await client.get(items_url, headers=headers)
                        items_response.raise_for_status()
                        items_data = items_response.json()
                        
                        items = items_data.get("results", [])
                        total = items_data.get("total", 0)
                        batch_count = len(items)
                        
                        logger.info(f"Audiobookshelf returned {batch_count} items in this batch (page: {page}, total: {total})")
                        
                        if not items:
                            break
                        
                        library_books.extend(items)
                        
                        # Check if we've fetched all items
                        if batch_count < limit or len(library_books) >= total:
                            logger.info(f"Library {library_id} pagination complete: {len(library_books)} books fetched")
                            break
                        
                        page += 1
                        
                        # Safety check
                        if page > 100:
                            logger.warning(f"Reached safety limit of 100 pages for library {library_id}, stopping")
                            break
                    
                    # Process all books from this library
                    # The list endpoint returns minimal data (no metadata, no name), so we need to fetch details for all books
                    book_items = [item for item in library_books if item.get("mediaType") == "book"]
                    
                    if not book_items:
                        continue
                    
                    logger.info(f"Fetching detailed metadata for {len(book_items)} books from library {library_id}")
                    
                    # Fetch detailed info for all books in batches
                    # Use larger batches and proper concurrency control
                    batch_size = 20  # Fetch 20 at a time for better throughput
                    books_by_id = {}  # Track books by ID for easy lookup
                    total_batches = (len(book_items) + batch_size - 1) // batch_size
                    
                    for batch_num, i in enumerate(range(0, len(book_items), batch_size), 1):
                        if batch_num % 5 == 0 or batch_num == total_batches:
                            logger.info(f"Processing batch {batch_num}/{total_batches} for library {library_id} ({len(books_by_id)} books processed so far)")
                        batch = book_items[i:i + batch_size]
                        # Fetch details concurrently for this batch
                        detail_tasks = []
                        for item in batch:
                            item_id = item.get("id")
                            if item_id:
                                detail_tasks.append(self._fetch_item_details(item_id, item))
                        
                        if detail_tasks:
                            try:
                                detail_results = await asyncio.gather(*detail_tasks, return_exceptions=True)
                                
                                # Process results and create normalized books
                                for idx, item in enumerate(batch):
                                    if idx < len(detail_results):
                                        result = detail_results[idx]
                                        item_id = item.get("id", "")
                                        
                                        # Handle exceptions from gather
                                        if isinstance(result, Exception):
                                            logger.warning(f"Failed to fetch details for item {item_id}: {result}")
                                            # Create book with minimal info from list response
                                            normalized = self.normalize_album({
                                                "id": item_id,
                                                "title": item.get("name", "Unknown"),
                                                "artist": "",
                                                "cover_art_id": item.get("coverPath", ""),
                                                "metadata": item
                                            })
                                            books_by_id[item_id] = normalized
                                            continue
                                        
                                        # Extract data from detailed response
                                        if isinstance(result, dict):
                                            detail_item = result.get("metadata", item)
                                            author = result.get("author", "")
                                            title = result.get("title", item.get("name", "Unknown"))
                                            cover_art = result.get("cover_art", item.get("coverPath", ""))
                                            
                                            normalized = self.normalize_album({
                                                "id": item_id,
                                                "title": title,
                                                "artist": author or "",
                                                "cover_art_id": cover_art,
                                                "metadata": detail_item  # Use detailed metadata
                                            })
                                            books_by_id[item_id] = normalized
                            except Exception as e:
                                logger.error(f"Error processing batch for library {library_id}: {e}", exc_info=True)
                                # Fallback: create books with minimal info
                                for item in batch:
                                    item_id = item.get("id", "")
                                    if item_id and item_id not in books_by_id:
                                        normalized = self.normalize_album({
                                            "id": item_id,
                                            "title": item.get("name", "Unknown"),
                                            "artist": "",
                                            "cover_art_id": item.get("coverPath", ""),
                                            "metadata": item
                                        })
                                        books_by_id[item_id] = normalized
                    
                    # Add all books from this library
                    all_books.extend(books_by_id.values())
                    
                    if books_by_id:
                        authors_found = sum(1 for book in books_by_id.values() if book.get("artist"))
                        logger.info(f"Processed {len(books_by_id)} books from library {library_id} ({authors_found} with authors)")
                
                logger.info(f"Normalized {len(all_books)} books")
                return all_books
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching books: {e.response.status_code} - {e.response.text[:500]}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch books: {e}", exc_info=True)
            return []
    
    async def get_artists(self) -> List[Dict[str, Any]]:
        """Fetch all authors (artists) from Audiobookshelf server"""
        try:
            # Reuse the albums (books) we fetch to extract authors
            # This avoids duplicate API calls
            books = await self.get_albums()
            
            # Extract unique authors from books
            authors_map = {}
            books_without_author = 0
            for book in books:
                # Get author from normalized book data (already extracted in normalize_album)
                author = book.get("artist") or ""
                
                # If not found, check metadata directly (fallback)
                if not author:
                    metadata = book.get("metadata", {})
                    # Check both top-level metadata and nested metadata
                    item_metadata = metadata.get("metadata", {}) if isinstance(metadata.get("metadata"), dict) else {}
                    author = (metadata.get("author") or 
                             metadata.get("authorName") or
                             item_metadata.get("author") or
                             item_metadata.get("authorName") or
                             (metadata.get("authors") and metadata.get("authors")[0] if isinstance(metadata.get("authors"), list) and len(metadata.get("authors")) > 0 else None) or
                             (item_metadata.get("authors") and item_metadata.get("authors")[0] if isinstance(item_metadata.get("authors"), list) and len(item_metadata.get("authors")) > 0 else None) or
                             "")
                
                if not author or not author.strip():
                    books_without_author += 1
                
                # Only add non-empty authors
                if author and author.strip() and author not in authors_map:
                    # Create a stable ID from author name
                    # Clean up author name for ID generation
                    author_clean = author.lower().replace(' ', '_').replace(',', '').replace('.', '').replace('&', 'and').replace("'", "").replace('"', '')[:50]
                    author_id = f"author_{author_clean}"
                    authors_map[author] = {
                        "id": author_id,
                        "name": author,
                        "metadata": {"author": author}
                    }
            
            authors = list(authors_map.values())
            logger.info(f"Normalized {len(authors)} authors from {len(books)} books ({books_without_author} books without authors)")
            if books_without_author > 0 and len(authors) == 0:
                logger.warning(f"⚠️ All {books_without_author} books have no author extracted - author extraction may be failing")
            return [self.normalize_artist(author) for author in authors]
                
        except Exception as e:
            logger.error(f"Failed to fetch authors: {e}", exc_info=True)
            return []
    
    async def get_playlists(self) -> List[Dict[str, Any]]:
        """Fetch all podcasts (playlists) from Audiobookshelf server"""
        try:
            # Get all libraries first
            libraries_url = self._build_url("libraries")
            headers = self._get_headers()
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get libraries
                libraries_response = await client.get(libraries_url, headers=headers)
                libraries_response.raise_for_status()
                libraries_data = libraries_response.json()
                
                libraries = libraries_data.get("libraries", [])
                if not libraries:
                    return []
                
                # Get podcasts from all libraries with pagination
                all_podcasts = []
                for library in libraries:
                    library_id = library.get("id")
                    if not library_id:
                        continue
                    
                    # Fetch library items with pagination (same as books)
                    library_podcasts = []
                    page = 0
                    limit = 500  # Fetch in batches of 500
                    
                    while True:
                        # Get library items (podcasts) with pagination
                        items_url = self._build_url(f"libraries/{library_id}/items?limit={limit}&page={page}")
                        logger.info(f"Fetching podcasts from library {library_id} (page: {page}, limit: {limit})")
                        items_response = await client.get(items_url, headers=headers)
                        items_response.raise_for_status()
                        items_data = items_response.json()
                        
                        items = items_data.get("results", [])
                        total = items_data.get("total", 0)
                        batch_count = len(items)
                        
                        logger.info(f"Audiobookshelf returned {batch_count} items in this batch (page: {page}, total: {total})")
                        
                        if not items:
                            break
                        
                        library_podcasts.extend(items)
                        
                        # Check if we've fetched all items
                        if batch_count < limit or len(library_podcasts) >= total:
                            logger.info(f"Library {library_id} pagination complete: {len(library_podcasts)} podcasts fetched")
                            break
                        
                        page += 1
                        
                        # Safety check
                        if page > 100:
                            logger.warning(f"Reached safety limit of 100 pages for library {library_id}, stopping")
                            break
                    
                    for item in library_podcasts:
                        # Only include podcasts
                        if item.get("mediaType") == "podcast":
                            # Get detailed podcast info (includes episodes and full metadata)
                            item_id = item.get("id")
                            episode_count = 0
                            podcast_name = item.get("name", "Unknown")
                            podcast_metadata = item
                            
                            if item_id:
                                try:
                                    item_detail_url = self._build_url(f"items/{item_id}?expanded=1&include=metadata")
                                    item_detail_response = await client.get(item_detail_url, headers=headers)
                                    item_detail_response.raise_for_status()
                                    item_detail = item_detail_response.json()
                                    
                                    # Get episode count
                                    episodes = item_detail.get("episodes", [])
                                    episode_count = len(episodes) if isinstance(episodes, list) else 0
                                    
                                    # Extract podcast name from media.metadata
                                    if item_detail.get("media") and isinstance(item_detail["media"], dict):
                                        media_metadata = item_detail["media"].get("metadata", {})
                                        if isinstance(media_metadata, dict):
                                            podcast_name = media_metadata.get("title") or media_metadata.get("name") or podcast_name
                                    
                                    # Use detailed metadata
                                    podcast_metadata = item_detail
                                except Exception as e:
                                    logger.debug(f"Failed to fetch podcast details for {item_id}: {e}")
                            
                            normalized = self.normalize_playlist({
                                "id": item_id or "",
                                "name": podcast_name,
                                "track_count": episode_count,
                                "metadata": podcast_metadata
                            })
                            all_podcasts.append(normalized)
                
                logger.info(f"Normalized {len(all_podcasts)} podcasts")
                return all_podcasts
                
        except Exception as e:
            logger.error(f"Failed to fetch podcasts: {e}", exc_info=True)
            return []
    
    async def get_album_tracks(self, album_id: str) -> List[Dict[str, Any]]:
        """Fetch chapters (tracks) for a book"""
        try:
            headers = self._get_headers()
            item_url = self._build_url(f"items/{album_id}?expanded=1&include=metadata")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(item_url, headers=headers)
                response.raise_for_status()
                item_data = response.json()
                
                # Extract book title and author from media.metadata
                book_title = item_data.get("name", "")
                book_author = ""
                book_cover = item_data.get("coverPath", "")
                
                media = item_data.get("media", {})
                if isinstance(media, dict):
                    media_metadata = media.get("metadata", {})
                    if isinstance(media_metadata, dict):
                        book_title = media_metadata.get("title") or book_title
                        # Extract first author
                        authors = media_metadata.get("authors", [])
                        if isinstance(authors, list) and len(authors) > 0:
                            if isinstance(authors[0], dict):
                                book_author = authors[0].get("name", "")
                            elif isinstance(authors[0], str):
                                book_author = authors[0]
                    # Get cover from media if available
                    if media.get("coverPath"):
                        book_cover = media["coverPath"]
                
                # Get media files (chapters)
                tracks = media.get("tracks", [])
                if not isinstance(tracks, list):
                    tracks = [tracks] if tracks else []
                
                # Debug: Log first track structure
                if tracks and len(tracks) > 0 and not hasattr(self, '_logged_track_sample'):
                    logger.info(f"Sample track keys: {list(tracks[0].keys())}")
                    logger.info(f"Sample track duration: {tracks[0].get('duration')}")
                    logger.info(f"Sample track structure: {json.dumps(tracks[0], default=str)[:500]}")
                    self._logged_track_sample = True
                
                result = []
                for idx, track in enumerate(tracks):
                    # Extract duration - check multiple locations
                    duration = (track.get("duration") or 
                               track.get("durationS") or
                               track.get("runtime") or
                               0)
                    
                    # AudioBookShelf might store duration in milliseconds
                    if duration:
                        if duration > 86400:  # Likely milliseconds if > 1 day in seconds
                            duration = duration / 1000  # Convert to seconds
                        elif duration < 1:  # Very small, might be in hours or wrong format
                            duration = 0
                    
                    normalized = self.normalize_track({
                        "id": track.get("index", idx),
                        "title": track.get("title", f"Chapter {idx + 1}"),
                        "artist": book_author,
                        "album": book_title,
                        "duration": int(duration) if duration else 0,
                        "track_number": track.get("index", idx + 1),
                        "cover_art_id": book_cover,
                        "metadata": track
                    }, parent_id=album_id)
                    result.append(normalized)
                
                return result
        except Exception as e:
            logger.error(f"Failed to fetch book chapters: {e}")
            return []
    
    async def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Fetch episodes (tracks) for a podcast"""
        try:
            headers = self._get_headers()
            item_url = self._build_url(f"items/{playlist_id}?expanded=1&include=metadata")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(item_url, headers=headers)
                response.raise_for_status()
                item_data = response.json()
                
                # Extract podcast name from media.metadata
                podcast_name = item_data.get("name", "")
                podcast_cover = item_data.get("coverPath", "")
                
                if item_data.get("media") and isinstance(item_data["media"], dict):
                    media_metadata = item_data["media"].get("metadata", {})
                    if isinstance(media_metadata, dict):
                        podcast_name = media_metadata.get("title") or media_metadata.get("name") or podcast_name
                    # Get cover from media if available
                    if item_data["media"].get("coverPath"):
                        podcast_cover = item_data["media"]["coverPath"]
                
                # Get episodes - AudioBookShelf stores them in different locations
                episodes = []
                
                # Try top-level episodes first
                if item_data.get("episodes"):
                    episodes = item_data.get("episodes", [])
                # Try media.episodes
                elif item_data.get("media") and isinstance(item_data["media"], dict):
                    episodes = item_data["media"].get("episodes", [])
                
                if not isinstance(episodes, list):
                    episodes = [episodes] if episodes else []
                
                logger.info(f"Found {len(episodes)} episodes for podcast {playlist_id}")
                
                # Debug: Log first episode structure
                if episodes and len(episodes) > 0 and not hasattr(self, '_logged_episode_sample'):
                    logger.info(f"Sample episode keys: {list(episodes[0].keys())}")
                    logger.info(f"Sample episode full structure: {json.dumps(episodes[0], default=str)[:1000]}")
                    self._logged_episode_sample = True
                
                result = []
                for episode in episodes:
                    duration = _episode_duration_seconds(episode if isinstance(episode, dict) else {})
                    
                    # Extract published date - check multiple field names
                    published_date = (episode.get("publishedAt") or 
                                    episode.get("published") or 
                                    episode.get("pubDate") or
                                    episode.get("publishedDate") or
                                    episode.get("datePublished") or
                                    episode.get("pubdate") or
                                    None)
                    
                    # If still not found, check in nested metadata
                    if not published_date and isinstance(episode.get("metadata"), dict):
                        metadata = episode.get("metadata", {})
                        published_date = (metadata.get("publishedAt") or
                                        metadata.get("published") or
                                        metadata.get("pubDate") or
                                        None)
                    
                    # Normalize to ISO string for frontend (handle numeric timestamps)
                    if published_date is not None:
                        if isinstance(published_date, (int, float)):
                            if published_date > 1e12:
                                published_date = published_date / 1000.0
                            published_date = datetime.utcfromtimestamp(published_date).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                        elif not isinstance(published_date, str):
                            published_date = str(published_date)
                    
                    # Store published date in metadata (both keys for frontend compatibility)
                    episode_metadata = episode.copy() if isinstance(episode, dict) else {}
                    if published_date:
                        episode_metadata["published_date"] = published_date
                        episode_metadata["publishedAt"] = published_date
                    
                    raw_track = {
                        "id": episode.get("id", ""),
                        "title": episode.get("title", ""),
                        "artist": podcast_name,
                        "album": podcast_name,
                        "duration": int(duration) if duration else 0,
                        "track_number": episode.get("index", 0),
                        "cover_art_id": podcast_cover,
                        "metadata": episode_metadata,
                    }
                    if published_date:
                        raw_track["published_date"] = published_date
                        raw_track["publishedAt"] = published_date
                    normalized = self.normalize_track(raw_track, parent_id=playlist_id)
                    result.append(normalized)
                
                return result
        except Exception as e:
            logger.error(f"Failed to fetch podcast episodes: {e}")
            return []
    
    async def get_stream_url(self, track_id: str, parent_id: Optional[str] = None) -> Optional[str]:
        """Generate authenticated stream URL for a track/chapter/episode.

        Uses ABS static file serving (/s/item/...) which returns audio bytes via GET,
        not the play session endpoint which returns JSON.
        """
        try:
            item_id = parent_id or track_id
            headers = self._get_headers()
            item_url = self._build_url(f"items/{item_id}?expanded=1&include=metadata")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(item_url, headers=headers)
                response.raise_for_status()
                item_data = response.json()

            media = item_data.get("media") or {}
            if not isinstance(media, dict):
                logger.warning("Item has no media object")
                return None

            base_url = self.server_url.rstrip("/")

            # Book chapter: track_id is chapter index (integer)
            try:
                chapter_index = int(track_id)
                tracks = media.get("tracks", [])
                if not isinstance(tracks, list):
                    tracks = [tracks] if tracks else []
                for track in tracks:
                    if track.get("index") == chapter_index:
                        content_url = track.get("contentUrl")
                        if content_url:
                            return f"{base_url}{content_url}?token={self.api_token}"
                # Fallback: use index as 0-based list position
                if 0 <= chapter_index < len(tracks):
                    content_url = tracks[chapter_index].get("contentUrl")
                    if content_url:
                        return f"{base_url}{content_url}?token={self.api_token}"
                logger.warning(f"No track found for chapter index {chapter_index}")
                return None
            except ValueError:
                pass

            # Podcast episode: track_id is episode UUID
            episodes = item_data.get("episodes") or (media.get("episodes") or [])
            if not isinstance(episodes, list):
                episodes = [episodes] if episodes else []
            for episode in episodes:
                if episode.get("id") == track_id:
                    # ABS serves podcast files as /s/item/{libraryItemId}/{relPathOrFilename}
                    # (see audioTrack.contentUrl in API) — never insert episode id in the path.
                    audio_track = episode.get("audioTrack") or {}
                    if isinstance(audio_track, dict):
                        content_url = audio_track.get("contentUrl")
                        if content_url:
                            path = (
                                content_url
                                if str(content_url).startswith("/")
                                else f"/{content_url}"
                            )
                            return f"{base_url}{path}?token={self.api_token}"

                    audio_file = episode.get("audioFile") or {}
                    filename = None
                    if isinstance(audio_file, dict):
                        meta = audio_file.get("metadata") or {}
                        filename = meta.get("filename") or meta.get("relPath")
                    if filename:
                        filename_encoded = quote(filename, safe="")
                        return f"{base_url}/s/item/{item_id}/{filename_encoded}?token={self.api_token}"
                    logger.warning(
                        "Episode %s has no audioTrack.contentUrl or audio file path",
                        track_id,
                    )
                    return None

            # Fallback: no parent_id and track_id is the item ID — use first track (e.g. single-file book)
            tracks = media.get("tracks", [])
            if isinstance(tracks, list) and len(tracks) > 0:
                content_url = tracks[0].get("contentUrl")
                if content_url:
                    return f"{base_url}{content_url}?token={self.api_token}"

            logger.warning(f"No track or episode found for track_id={track_id}, parent_id={parent_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate stream URL: {e}", exc_info=True)
            return None
    
    async def get_stream_url_with_parent(self, track_id: str, parent_id: Optional[str] = None) -> Optional[str]:
        """Generate stream URL with explicit parent_id (for AudioBookShelf)"""
        return await self.get_stream_url(track_id, parent_id)

