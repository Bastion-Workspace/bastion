"""
Sonarr Service
API client for Sonarr media manager integration
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional
import httpx

logger = logging.getLogger(__name__)


class SonarrService:
    """
    Service for interacting with Sonarr API
    
    Handles series and episode fetching, data transformation, and metadata extraction
    """
    
    def __init__(self, api_url: str, api_key: str):
        """
        Initialize Sonarr service
        
        Args:
            api_url: Base URL of Sonarr instance (e.g., http://localhost:8989)
            api_key: Sonarr API key
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.base_path = f"{self.api_url}/api/v3"
        self.timeout = 30.0
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Sonarr API requests"""
        return {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Sonarr API
        
        Returns:
            Dict with success status, message, and version if available
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_path}/system/status",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                data = response.json()
                
                version = data.get("version", "unknown")
                return {
                    "success": True,
                    "message": f"Successfully connected to Sonarr {version}",
                    "version": version
                }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return {
                    "success": False,
                    "message": "Authentication failed - invalid API key"
                }
            return {
                "success": False,
                "message": f"HTTP error: {e.response.status_code}"
            }
        except httpx.RequestError as e:
            return {
                "success": False,
                "message": f"Connection error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Sonarr connection test failed: {e}")
            return {
                "success": False,
                "message": f"Unexpected error: {str(e)}"
            }
    
    async def fetch_all_series(self) -> List[Dict[str, Any]]:
        """
        Fetch all TV series from Sonarr
        
        Returns:
            List of series dictionaries with full metadata
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_path}/series",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                series = response.json()
                
                logger.info(f"Fetched {len(series)} series from Sonarr")
                return series
        except Exception as e:
            logger.error(f"Failed to fetch series from Sonarr: {e}")
            raise
    
    async def fetch_series_episodes(self, series_id: int) -> List[Dict[str, Any]]:
        """
        Fetch all episodes for a specific series
        
        Args:
            series_id: Sonarr series ID
            
        Returns:
            List of episode dictionaries
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_path}/episode",
                    params={"seriesId": series_id},
                    headers=self._get_headers()
                )
                response.raise_for_status()
                episodes = response.json()
                
                logger.info(f"Fetched {len(episodes)} episodes for series {series_id}")
                return episodes
        except Exception as e:
            logger.error(f"Failed to fetch episodes for series {series_id}: {e}")
            raise
    
    async def get_series_details(self, series_id: int) -> Dict[str, Any]:
        """
        Get detailed information for a specific series
        
        Args:
            series_id: Sonarr series ID
            
        Returns:
            Series dictionary with full details
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_path}/series/{series_id}",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch series {series_id} from Sonarr: {e}")
            raise
    
    def transform_series_to_text(self, series: Dict[str, Any]) -> str:
        """
        Transform Sonarr series data into structured text format
        
        Args:
            series: Series dictionary from Sonarr API
            
        Returns:
            Formatted text string for vectorization
        """
        lines = []
        
        # Title
        title = series.get("title", "Unknown")
        lines.append(f"# {title}")
        lines.append("")
        
        # Overview
        overview = series.get("overview", "")
        if overview:
            lines.append("## Summary")
            lines.append(overview)
            lines.append("")
        
        # Network
        network = series.get("network")
        if network:
            lines.append(f"**Network**: {network}")
        
        # Status
        status = series.get("status", "")
        if status:
            lines.append(f"**Status**: {status}")
        
        # Genres
        genres = series.get("genres", [])
        if genres:
            genre_strs = [str(g) for g in genres if g is not None]
            if genre_strs:
                lines.append(f"**Genre**: {', '.join(genre_strs)}")
        
        # Ratings
        ratings = series.get("ratings", {})
        if ratings:
            value = ratings.get("value")
            if value:
                lines.append(f"**Rating**: {value}/10")
        
        # Year (from first air date)
        year = series.get("year")
        if year:
            lines.append(f"**First Aired**: {year}")
        
        # Seasons count
        seasons = series.get("seasons", [])
        if seasons:
            lines.append(f"**Seasons**: {len(seasons)}")
        
        # TVDB ID
        tvdb_id = series.get("tvdbId")
        if tvdb_id:
            lines.append(f"**TVDB ID**: {tvdb_id}")
        
        return "\n".join(lines)
    
    def transform_episode_to_text(self, episode: Dict[str, Any], series: Dict[str, Any]) -> str:
        """
        Transform Sonarr episode data into structured text format
        
        Args:
            episode: Episode dictionary from Sonarr API
            series: Parent series dictionary
            
        Returns:
            Formatted text string for vectorization
        """
        lines = []
        
        # Title with series context
        series_title = series.get("title", "Unknown")
        season_num = episode.get("seasonNumber", 0)
        episode_num = episode.get("episodeNumber", 0)
        episode_title = episode.get("title", "Untitled")
        
        lines.append(f"# {series_title} - S{season_num:02d}E{episode_num:02d} - {episode_title}")
        lines.append("")
        
        # Series reference
        lines.append(f"**Series**: {series_title}")
        lines.append(f"**Season**: {season_num}")
        lines.append(f"**Episode**: {episode_num}")
        lines.append("")
        
        # Episode overview
        overview = episode.get("overview", "")
        if overview:
            lines.append("## Summary")
            lines.append(overview)
            lines.append("")
        
        # Air date
        air_date = episode.get("airDate")
        if air_date:
            lines.append(f"**Air Date**: {air_date}")
        
        # Ratings
        ratings = episode.get("ratings", {})
        if ratings:
            value = ratings.get("value")
            if value:
                lines.append(f"**Rating**: {value}/10")
        
        # TVDB ID
        tvdb_id = episode.get("tvdbId")
        if tvdb_id:
            lines.append(f"**TVDB ID**: {tvdb_id}")
        
        return "\n".join(lines)
    
    def extract_tags(self, series: Dict[str, Any], episode: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Extract tags for vector filtering from series/episode data
        
        Args:
            series: Series dictionary from Sonarr API
            episode: Optional episode dictionary
            
        Returns:
            List of tag strings
        """
        if episode:
            # Episode tags
            tags = ["tv_episode", "entertainment", "sonarr_synced"]
            
            # Series reference
            series_title = series.get("title", "")
            if series_title:
                series_tag = series_title.lower().replace(" ", "_").replace("-", "_")
                tags.append(f"series_{series_tag}")
            
            # Season tag
            season_num = episode.get("seasonNumber")
            if season_num is not None:
                tags.append(f"season_{season_num}")
            
            # TVDB ID
            tvdb_id = episode.get("tvdbId")
            if tvdb_id:
                tags.append(f"tvdb_{tvdb_id}")
        else:
            # Series tags
            tags = ["tv_show", "entertainment", "sonarr_synced"]
            
            # TVDB ID
            tvdb_id = series.get("tvdbId")
            if tvdb_id:
                tags.append(f"tvdb_{tvdb_id}")
            
            # Genre tags
            genres = series.get("genres", [])
            for genre in genres:
                tag = genre.lower().replace(" ", "_").replace("-", "_")
                tags.append(tag)
            
            # Network tag
            network = series.get("network")
            if network:
                network_tag = network.lower().replace(" ", "_").replace("-", "_")
                tags.append(f"network_{network_tag}")
            
            # Status tag
            status = series.get("status", "").lower()
            if status:
                tags.append(f"status_{status}")
            
            # Year/decade tag
            year = series.get("year")
            if year:
                decade = (year // 10) * 10
                tags.append(f"{decade}s")
        
        return list(set(tags))  # Remove duplicates
    
    def calculate_metadata_hash(self, item: Dict[str, Any]) -> str:
        """
        Calculate hash of series/episode metadata for change detection
        
        Args:
            item: Series or episode dictionary from Sonarr API
            
        Returns:
            SHA256 hash string
        """
        # Include fields that would indicate a meaningful change
        hash_fields = {
            "title": item.get("title"),
            "overview": item.get("overview"),
            "status": item.get("status"),
        }
        
        # Series-specific fields
        if "network" in item:
            hash_fields["network"] = item.get("network")
            hash_fields["genres"] = sorted(item.get("genres", []))
            hash_fields["year"] = item.get("year")
            hash_fields["tvdbId"] = item.get("tvdbId")
        
        # Episode-specific fields
        if "seasonNumber" in item:
            hash_fields["seasonNumber"] = item.get("seasonNumber")
            hash_fields["episodeNumber"] = item.get("episodeNumber")
            hash_fields["airDate"] = item.get("airDate")
            hash_fields["tvdbId"] = item.get("tvdbId")
        
        # Ratings
        ratings = item.get("ratings", {})
        if ratings:
            hash_fields["ratings"] = ratings
        
        # Create hash
        hash_string = json.dumps(hash_fields, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def validate_series_data(self, series: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate series data before processing
        
        Args:
            series: Series dictionary from Sonarr API
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not series:
            return False, "Series data is empty or None"
        
        # Required fields
        if 'id' not in series or series.get('id') is None:
            return False, "Series missing required field: 'id'"
        
        if 'title' not in series or not series.get('title'):
            return False, f"Series {series.get('id', 'unknown')} missing required field: 'title'"
        
        # Validate data types
        series_id = series.get('id')
        if not isinstance(series_id, (int, str)):
            return False, f"Series ID must be int or str, got {type(series_id)}"
        
        return True, None
    
    def validate_episode_data(self, episode: Dict[str, Any], series: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate episode data before processing
        
        Args:
            episode: Episode dictionary from Sonarr API
            series: Parent series dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not episode:
            return False, "Episode data is empty or None"
        
        if not series:
            return False, "Episode missing parent series data"
        
        # Required fields
        if 'id' not in episode or episode.get('id') is None:
            return False, "Episode missing required field: 'id'"
        
        if 'title' not in episode:
            return False, f"Episode {episode.get('id', 'unknown')} missing required field: 'title'"
        
        if 'seasonNumber' not in episode or episode.get('seasonNumber') is None:
            return False, f"Episode {episode.get('id', 'unknown')} missing required field: 'seasonNumber'"
        
        if 'episodeNumber' not in episode or episode.get('episodeNumber') is None:
            return False, f"Episode {episode.get('id', 'unknown')} missing required field: 'episodeNumber'"
        
        # Validate data types
        episode_id = episode.get('id')
        if not isinstance(episode_id, (int, str)):
            return False, f"Episode ID must be int or str, got {type(episode_id)}"
        
        return True, None

