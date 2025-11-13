"""
Entertainment Knowledge Graph Extractor - Roosevelt's "Entertainment Intelligence" Service
Domain-specific entity and relationship extraction for movies and TV shows
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from models.api_models import DocumentInfo

logger = logging.getLogger(__name__)


class EntertainmentKGExtractor:
    """
    Extract entertainment-specific entities and relationships from documents
    
    **BULLY!** Domain-scoped extraction for movies and TV shows!
    """
    
    def __init__(self):
        self.entity_patterns = self._build_entity_patterns()
        self.relationship_patterns = self._build_relationship_patterns()
    
    def _build_entity_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Build regex patterns for entertainment entity extraction"""
        return {
            "DIRECTOR": [
                re.compile(r'\*\*Director\*\*:\s*([A-Z][a-zA-Z\s\.,]+?)(?:\s*\n|\s*\*\*)', re.MULTILINE),
                re.compile(r'Directed by\s+([A-Z][a-zA-Z\s\.,]+?)(?:\s*\n|\s*\()', re.MULTILINE),
            ],
            "CREATOR": [
                re.compile(r'\*\*Creator\*\*:\s*([A-Z][a-zA-Z\s\.,]+?)(?:\s*\n|\s*\*\*)', re.MULTILINE),
                re.compile(r'Created by\s+([A-Z][a-zA-Z\s\.,]+?)(?:\s*\n|\s*\()', re.MULTILINE),
            ],
            "WRITER": [
                re.compile(r'\*\*Writer[s]?\*\*:\s*([A-Z][a-zA-Z\s\.,]+?)(?:\s*\n|\s*\*\*)', re.MULTILINE),
                re.compile(r'Written by\s+([A-Z][a-zA-Z\s\.,]+?)(?:\s*\n|\s*\()', re.MULTILINE),
            ],
            "ACTOR": [
                re.compile(r'\*\*([A-Z][a-zA-Z\s\.]+?)\*\*\s+as\s+([A-Z][a-zA-Z\s]+)', re.MULTILINE),
                re.compile(r'-\s+\*\*([A-Z][a-zA-Z\s\.]+?)\*\*\s+as', re.MULTILINE),
            ],
            "STARS": [
                re.compile(r'\*\*Stars\*\*:\s*([A-Z][a-zA-Z\s\.,]+?)(?:\s*\n|\s*\*\*)', re.MULTILINE),
            ],
            "GENRE": [
                re.compile(r'\*\*Genre\*\*:\s*([A-Za-z\s\,/]+?)(?:\s*\n|\s*\*\*)', re.MULTILINE),
            ],
            "NETWORK": [
                re.compile(r'\*\*Original Network\*\*:\s*([A-Z][A-Za-z\s]+?)(?:\s*\n|\s*\*\*)', re.MULTILINE),
                re.compile(r'\*\*Network\*\*:\s*([A-Z][A-Za-z\s]+?)(?:\s*\n|\s*\*\*)', re.MULTILINE),
            ],
            "STUDIO": [
                re.compile(r'\*\*Production Compan(?:y|ies)\*\*:\s*([A-Z][A-Za-z\s\.,]+?)(?:\s*\n|\s*\*\*)', re.MULTILINE),
                re.compile(r'\*\*Studio\*\*:\s*([A-Z][A-Za-z\s\.,]+?)(?:\s*\n|\s*\*\*)', re.MULTILINE),
            ],
        }
    
    def _build_relationship_patterns(self) -> Dict[str, str]:
        """Map entity types to relationship types"""
        return {
            "DIRECTOR": "DIRECTED",
            "CREATOR": "CREATED",
            "WRITER": "WROTE",
            "ACTOR": "ACTED_IN",
            "STARS": "ACTED_IN",
            "NETWORK": "AIRED_ON",
            "STUDIO": "PRODUCED_BY",
        }
    
    def should_extract_from_document(self, doc_info: DocumentInfo) -> bool:
        """
        Determine if we should extract entertainment entities from this document
        
        **ROOSEVELT'S TAG SCOPING**: Only extract from entertainment-tagged documents!
        """
        # Check category
        if doc_info.category and doc_info.category.value == "entertainment":
            return True
        
        # Check tags
        entertainment_tags = {"movie", "tv_show", "tv_episode"}
        if doc_info.tags:
            if any(tag in entertainment_tags for tag in doc_info.tags):
                return True
        
        return False
    
    def extract_entities_and_relationships(
        self, content: str, doc_info: DocumentInfo
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entertainment entities and relationships from document content
        
        Returns: (entities, relationships)
        """
        if not self.should_extract_from_document(doc_info):
            logger.info(f"⏭️  Skipping entertainment KG extraction for {doc_info.filename} (not entertainment-tagged)")
            return [], []
        
        logger.info(f"🎬 Extracting entertainment entities from {doc_info.filename}")
        
        entities = []
        relationships = []
        
        # Extract title from document
        work_title = self._extract_work_title(content, doc_info)
        work_type = self._determine_work_type(doc_info)
        
        if work_title:
            # Create the main work entity (Movie/TVShow/Episode)
            work_entity = {
                "name": work_title,
                "type": work_type,
                "label": f"Entertainment{work_type}",
                "confidence": 1.0,
                "properties": self._extract_work_properties(content, doc_info)
            }
            entities.append(work_entity)
            
            # Extract related entities
            self._extract_people_entities(content, work_title, entities, relationships)
            self._extract_organization_entities(content, work_title, entities, relationships)
            self._extract_genre_entities(content, work_title, entities, relationships)
        
        logger.info(f"✅ Extracted {len(entities)} entertainment entities, {len(relationships)} relationships")
        return entities, relationships
    
    def _extract_work_title(self, content: str, doc_info: DocumentInfo) -> Optional[str]:
        """Extract the main work title from document"""
        # Try to get from first heading
        heading_match = re.search(r'^#\s+(.+?)(?:\s*\n|$)', content, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()
        
        # Fallback to filename (remove extension and episode markers)
        title = doc_info.filename.replace('.md', '').replace('.txt', '')
        # Remove episode markers like " - S01E01"
        title = re.sub(r'\s*-\s*S\d+E\d+$', '', title)
        return title
    
    def _determine_work_type(self, doc_info: DocumentInfo) -> str:
        """Determine if this is a Movie, TVShow, or Episode"""
        if doc_info.tags:
            if "movie" in doc_info.tags:
                return "Movie"
            if "tv_show" in doc_info.tags:
                return "TVShow"
            if "tv_episode" in doc_info.tags:
                return "Episode"
        
        # Check filename for episode pattern
        if re.search(r'S\d+E\d+', doc_info.filename):
            return "Episode"
        
        # Default to Movie
        return "Movie"
    
    def _extract_work_properties(self, content: str, doc_info: DocumentInfo) -> Dict[str, Any]:
        """Extract properties of the work (year, rating, runtime, etc.)"""
        properties = {}
        
        # Extract year
        year_match = re.search(r'\((\d{4})\)', content[:500])  # Look in first 500 chars
        if year_match:
            properties["year"] = int(year_match.group(1))
        
        # Extract rating
        rating_match = re.search(r'\*\*Rating\*\*:\s*([\d\.]+)/10', content)
        if rating_match:
            properties["rating"] = float(rating_match.group(1))
        
        # Extract runtime
        runtime_match = re.search(r'\*\*Runtime\*\*:\s*(\d+)\s*minutes?', content)
        if runtime_match:
            properties["runtime"] = int(runtime_match.group(1))
        
        # Extract seasons/episodes for TV shows
        seasons_match = re.search(r'\*\*Seasons\*\*:\s*(\d+)', content)
        if seasons_match:
            properties["seasons"] = int(seasons_match.group(1))
        
        episodes_match = re.search(r'\*\*Total Episodes\*\*:\s*(\d+)', content)
        if episodes_match:
            properties["total_episodes"] = int(episodes_match.group(1))
        
        return properties
    
    def _extract_people_entities(
        self, content: str, work_title: str, 
        entities: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]]
    ):
        """Extract people (directors, actors, writers) and their relationships"""
        
        for entity_type, patterns in self.entity_patterns.items():
            if entity_type in ["DIRECTOR", "CREATOR", "WRITER", "ACTOR", "STARS"]:
                for pattern in patterns:
                    matches = pattern.findall(content)
                    
                    for match in matches:
                        # Handle tuple match from ACTOR pattern (actor, character)
                        if isinstance(match, tuple):
                            person_name = match[0].strip()
                            character_name = match[1].strip() if len(match) > 1 else None
                        else:
                            person_name = match.strip()
                            character_name = None
                        
                        # Split on commas for multiple names
                        names = [n.strip() for n in person_name.split(',') if n.strip()]
                        
                        for name in names:
                            # Clean up the name
                            name = name.strip(' .')
                            if not name or len(name) < 3:
                                continue
                            
                            # Create person entity
                            person_entity = {
                                "name": name,
                                "type": entity_type,
                                "label": f"EntertainmentPerson:{entity_type.capitalize()}",
                                "confidence": 0.9
                            }
                            
                            # Avoid duplicates
                            if not any(e["name"] == name and e["type"] == entity_type for e in entities):
                                entities.append(person_entity)
                            
                            # Create relationship
                            rel_type = self.relationship_patterns.get(entity_type, "RELATED_TO")
                            relationship = {
                                "from_name": name,
                                "from_type": entity_type,
                                "to_name": work_title,
                                "to_type": "WORK",
                                "relationship_type": rel_type,
                                "properties": {"character": character_name} if character_name else {}
                            }
                            relationships.append(relationship)
    
    def _extract_organization_entities(
        self, content: str, work_title: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ):
        """Extract organizations (studios, networks) and their relationships"""
        
        for entity_type in ["NETWORK", "STUDIO"]:
            patterns = self.entity_patterns.get(entity_type, [])
            
            for pattern in patterns:
                matches = pattern.findall(content)
                
                for match in matches:
                    # Split on commas for multiple organizations
                    org_names = [n.strip() for n in match.split(',') if n.strip()]
                    
                    for org_name in org_names:
                        org_name = org_name.strip(' .')
                        if not org_name or len(org_name) < 2:
                            continue
                        
                        # Create organization entity
                        org_entity = {
                            "name": org_name,
                            "type": entity_type,
                            "label": f"EntertainmentOrg:{entity_type.capitalize()}",
                            "confidence": 0.9
                        }
                        
                        if not any(e["name"] == org_name and e["type"] == entity_type for e in entities):
                            entities.append(org_entity)
                        
                        # Create relationship
                        rel_type = self.relationship_patterns.get(entity_type, "RELATED_TO")
                        relationship = {
                            "from_name": work_title,
                            "from_type": "WORK",
                            "to_name": org_name,
                            "to_type": entity_type,
                            "relationship_type": rel_type,
                            "properties": {}
                        }
                        relationships.append(relationship)
    
    def _extract_genre_entities(
        self, content: str, work_title: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ):
        """Extract genres and their relationships"""
        
        patterns = self.entity_patterns.get("GENRE", [])
        
        for pattern in patterns:
            matches = pattern.findall(content)
            
            for match in matches:
                # Split on commas and slashes
                genres = re.split(r'[,/]', match)
                
                for genre in genres:
                    genre_name = genre.strip()
                    if not genre_name or len(genre_name) < 3:
                        continue
                    
                    # Create genre entity
                    genre_entity = {
                        "name": genre_name,
                        "type": "GENRE",
                        "label": "EntertainmentGenre",
                        "confidence": 1.0
                    }
                    
                    if not any(e["name"] == genre_name and e["type"] == "GENRE" for e in entities):
                        entities.append(genre_entity)
                    
                    # Create relationship
                    relationship = {
                        "from_name": work_title,
                        "from_type": "WORK",
                        "to_name": genre_name,
                        "to_type": "GENRE",
                        "relationship_type": "HAS_GENRE",
                        "properties": {}
                    }
                    relationships.append(relationship)


# Singleton instance
_entertainment_kg_extractor = None

def get_entertainment_kg_extractor() -> EntertainmentKGExtractor:
    """Get singleton instance of entertainment KG extractor"""
    global _entertainment_kg_extractor
    if _entertainment_kg_extractor is None:
        _entertainment_kg_extractor = EntertainmentKGExtractor()
    return _entertainment_kg_extractor

