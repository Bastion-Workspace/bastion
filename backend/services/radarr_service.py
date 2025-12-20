"""
Radarr Service
API client for Radarr media manager integration
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional
import httpx

logger = logging.getLogger(__name__)


class RadarrService:
    """
    Service for interacting with Radarr API
    
    Handles movie fetching, data transformation, and metadata extraction
    """
    
    def __init__(self, api_url: str, api_key: str):
        """
        Initialize Radarr service
        
        Args:
            api_url: Base URL of Radarr instance (e.g., http://localhost:7878)
            api_key: Radarr API key
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.base_path = f"{self.api_url}/api/v3"
        self.timeout = 30.0
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Radarr API requests"""
        return {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Radarr API
        
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
                    "message": f"Successfully connected to Radarr {version}",
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
            logger.error(f"Radarr connection test failed: {e}")
            return {
                "success": False,
                "message": f"Unexpected error: {str(e)}"
            }
    
    async def fetch_all_movies(self) -> List[Dict[str, Any]]:
        """
        Fetch all movies from Radarr
        
        Returns:
            List of movie dictionaries with full metadata
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_path}/movie",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                movies = response.json()
                
                logger.info(f"Fetched {len(movies)} movies from Radarr")
                return movies
        except Exception as e:
            logger.error(f"Failed to fetch movies from Radarr: {e}")
            raise
    
    async def get_movie_details(self, movie_id: int) -> Dict[str, Any]:
        """
        Get detailed information for a specific movie
        
        Args:
            movie_id: Radarr movie ID
            
        Returns:
            Movie dictionary with full details
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_path}/movie/{movie_id}",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch movie {movie_id} from Radarr: {e}")
            raise
    
    def transform_movie_to_text(self, movie: Dict[str, Any]) -> str:
        """
        Transform Radarr movie data into structured text format
        
        Args:
            movie: Movie dictionary from Radarr API
            
        Returns:
            Formatted text string for vectorization
        """
        lines = []
        
        # Title and year
        title = movie.get("title", "Unknown")
        year = movie.get("year")
        if year:
            lines.append(f"# {title} ({year})")
        else:
            lines.append(f"# {title}")
        
        lines.append("")
        
        # Overview/plot
        overview = movie.get("overview", "")
        if overview:
            lines.append("## Summary")
            lines.append(overview)
            lines.append("")
        
        # Genres
        genres = movie.get("genres", [])
        if genres:
            genre_strs = [str(g) for g in genres if g is not None]
            if genre_strs:
                lines.append(f"**Genre**: {', '.join(genre_strs)}")
        
        # Ratings
        ratings = movie.get("ratings", {})
        if ratings:
            value = ratings.get("value")
            if value:
                lines.append(f"**Rating**: {value}/10")
        
        # Runtime
        runtime = movie.get("runtime")
        if runtime:
            lines.append(f"**Runtime**: {runtime} minutes")
        
        # Release date
        release_date = movie.get("inCinemas") or movie.get("physicalRelease") or movie.get("digitalRelease")
        if release_date:
            lines.append(f"**Release Date**: {release_date}")
        
        # Studio
        studio = movie.get("studio")
        if studio:
            lines.append(f"**Studio**: {studio}")
        
        # Director (from crew)
        crew = movie.get("movieCredits", {}).get("crew", [])
        directors = [p.get("name") for p in crew if p.get("job") == "Director"]
        if directors:
            director_strs = [str(d) for d in directors if d is not None]
            if director_strs:
                lines.append(f"**Director**: {', '.join(director_strs)}")
        
        # Cast
        cast = movie.get("movieCredits", {}).get("cast", [])
        if cast:
            lines.append("")
            lines.append("## Cast")
            for actor in cast[:10]:  # Limit to top 10
                name = actor.get("name", "Unknown")
                character = actor.get("character", "")
                if character:
                    lines.append(f"- **{name}** as {character}")
                else:
                    lines.append(f"- **{name}**")
        
        # Tags (from Radarr)
        tags = movie.get("tags", [])
        if tags:
            tag_strs = [str(t) for t in tags if t is not None]
            if tag_strs:
                lines.append("")
                lines.append(f"**Tags**: {', '.join(tag_strs)}")
        
        # TMDB ID
        tmdb_id = movie.get("tmdbId")
        if tmdb_id:
            lines.append(f"**TMDB ID**: {tmdb_id}")
        
        return "\n".join(lines)
    
    def extract_tags(self, movie: Dict[str, Any]) -> List[str]:
        """
        Extract tags for vector filtering from movie data
        
        Args:
            movie: Movie dictionary from Radarr API
            
        Returns:
            List of tag strings
        """
        tags = ["movie", "entertainment", "radarr_synced"]
        
        # TMDB ID tag
        tmdb_id = movie.get("tmdbId")
        if tmdb_id:
            tags.append(f"tmdb_{tmdb_id}")
        
        # Genre tags
        genres = movie.get("genres", [])
        for genre in genres:
            tag = genre.lower().replace(" ", "_").replace("-", "_")
            tags.append(tag)
        
        # Decade tag
        year = movie.get("year")
        if year:
            decade = (year // 10) * 10
            tags.append(f"{decade}s")
        
        # Rating tag (rounded)
        ratings = movie.get("ratings", {})
        value = ratings.get("value")
        if value:
            rating_floor = int(float(value))
            tags.append(f"rating_{rating_floor}")
        
        # Studio tag
        studio = movie.get("studio")
        if studio:
            studio_tag = studio.lower().replace(" ", "_").replace("-", "_")
            tags.append(f"studio_{studio_tag}")
        
        # Status tag
        status = movie.get("status", "").lower()
        if status:
            tags.append(f"status_{status}")
        
        return list(set(tags))  # Remove duplicates
    
    def calculate_metadata_hash(self, movie: Dict[str, Any]) -> str:
        """
        Calculate hash of movie metadata for change detection
        
        Args:
            movie: Movie dictionary from Radarr API
            
        Returns:
            SHA256 hash string
        """
        # Include fields that would indicate a meaningful change
        hash_fields = {
            "title": movie.get("title"),
            "year": movie.get("year"),
            "overview": movie.get("overview"),
            "genres": sorted(movie.get("genres", [])),
            "ratings": movie.get("ratings", {}),
            "runtime": movie.get("runtime"),
            "studio": movie.get("studio"),
            "status": movie.get("status"),
            "tmdbId": movie.get("tmdbId"),
        }
        
        # Include cast/crew for change detection
        cast = movie.get("movieCredits", {}).get("cast", [])
        if cast:
            hash_fields["cast"] = [(a.get("name"), a.get("character")) for a in cast[:5]]
        
        crew = movie.get("movieCredits", {}).get("crew", [])
        if crew:
            directors = [p.get("name") for p in crew if p.get("job") == "Director"]
            if directors:
                hash_fields["directors"] = sorted(directors)
        
        # Create hash
        hash_string = json.dumps(hash_fields, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def validate_movie_data(self, movie: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate movie data before processing
        
        Args:
            movie: Movie dictionary from Radarr API
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not movie:
            return False, "Movie data is empty or None"
        
        # Required fields
        if 'id' not in movie or movie.get('id') is None:
            return False, "Movie missing required field: 'id'"
        
        if 'title' not in movie or not movie.get('title'):
            return False, f"Movie {movie.get('id', 'unknown')} missing required field: 'title'"
        
        # Validate data types
        movie_id = movie.get('id')
        if not isinstance(movie_id, (int, str)):
            return False, f"Movie ID must be int or str, got {type(movie_id)}"
        
        # Warn about missing but recommended fields
        warnings = []
        if not movie.get('overview'):
            warnings.append("missing 'overview'")
        if not movie.get('year'):
            warnings.append("missing 'year'")
        
        if warnings:
            logger.debug(f"Movie {movie_id} ({movie.get('title')}) has warnings: {', '.join(warnings)}")
        
        return True, None

