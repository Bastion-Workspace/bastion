"""
Domain Detector Service - Roosevelt's "Domain Intelligence" 
Detects which domain(s) a document belongs to based on tags and category
"""

import logging
from typing import Set, List, Dict, Any
from models.api_models import DocumentInfo, DocumentCategory

logger = logging.getLogger(__name__)


class DomainConfig:
    """Configuration for a specific domain"""
    
    def __init__(
        self,
        name: str,
        tags: Set[str],
        categories: Set[str],
        extractor_module: str = None,
        extractor_class: str = None
    ):
        self.name = name
        self.tags = tags
        self.categories = categories
        self.extractor_module = extractor_module
        self.extractor_class = extractor_class


class DomainDetector:
    """
    Detects document domains and identifies when domain changes occur
    
    **BULLY!** Extensible for entertainment, business, research, and future domains!
    """
    
    def __init__(self):
        self.domains = self._initialize_domains()
    
    def _initialize_domains(self) -> Dict[str, DomainConfig]:
        """Initialize domain configurations"""
        return {
            "entertainment": DomainConfig(
                name="entertainment",
                tags={"movie", "tv_show", "tv_episode"},
                categories={"entertainment"},
                extractor_module="services.entertainment_kg_extractor",
                extractor_class="get_entertainment_kg_extractor"
            ),
            # Future domains can be added here:
            # "business": DomainConfig(
            #     name="business",
            #     tags={"financial", "corporate", "legal", "worldcom", "enron"},
            #     categories={"business", "legal"},
            #     extractor_module="services.business_kg_extractor",
            #     extractor_class="get_business_kg_extractor"
            # ),
            # "research": DomainConfig(
            #     name="research",
            #     tags={"academic", "paper", "research", "study"},
            #     categories={"academic", "research"},
            #     extractor_module="services.research_kg_extractor",
            #     extractor_class="get_research_kg_extractor"
            # ),
        }
    
    def detect_domains(
        self, 
        tags: List[str] = None, 
        category: str = None
    ) -> Set[str]:
        """
        Detect which domains a document belongs to based on tags and category
        
        Returns set of domain names (e.g., {"entertainment", "business"})
        """
        tags_set = set(tags) if tags else set()
        detected_domains = set()
        
        for domain_name, domain_config in self.domains.items():
            # Check if any domain-specific tags match
            if tags_set & domain_config.tags:
                detected_domains.add(domain_name)
                continue
            
            # Check if category matches
            if category and category in domain_config.categories:
                detected_domains.add(domain_name)
                continue
        
        return detected_domains
    
    def detect_domains_from_doc_info(self, doc_info: DocumentInfo) -> Set[str]:
        """Detect domains from DocumentInfo object"""
        category = doc_info.category.value if doc_info.category else None
        return self.detect_domains(doc_info.tags, category)
    
    def has_domain_changed(
        self,
        old_tags: List[str],
        old_category: str,
        new_tags: List[str],
        new_category: str
    ) -> bool:
        """
        Detect if domain membership has changed
        
        Returns True if any domain was added or removed
        """
        old_domains = self.detect_domains(old_tags, old_category)
        new_domains = self.detect_domains(new_tags, new_category)
        
        if old_domains != new_domains:
            added = new_domains - old_domains
            removed = old_domains - new_domains
            
            if added:
                logger.info(f"📊 Domains ADDED: {added}")
            if removed:
                logger.info(f"📊 Domains REMOVED: {removed}")
            
            return True
        
        return False
    
    def get_domain_changes(
        self,
        old_tags: List[str],
        old_category: str,
        new_tags: List[str],
        new_category: str
    ) -> Dict[str, Any]:
        """
        Get detailed information about domain changes
        
        Returns dict with:
        - changed: bool
        - old_domains: Set[str]
        - new_domains: Set[str]
        - added: Set[str]
        - removed: Set[str]
        """
        old_domains = self.detect_domains(old_tags, old_category)
        new_domains = self.detect_domains(new_tags, new_category)
        
        return {
            "changed": old_domains != new_domains,
            "old_domains": old_domains,
            "new_domains": new_domains,
            "added": new_domains - old_domains,
            "removed": old_domains - new_domains
        }
    
    def get_extractor_for_domain(self, domain_name: str):
        """
        Get the KG extractor instance for a specific domain
        
        Returns the extractor object or None if domain not found
        """
        domain_config = self.domains.get(domain_name)
        if not domain_config:
            logger.warning(f"⚠️  Unknown domain: {domain_name}")
            return None
        
        if not domain_config.extractor_module or not domain_config.extractor_class:
            logger.warning(f"⚠️  No extractor configured for domain: {domain_name}")
            return None
        
        try:
            # Dynamically import the extractor
            import importlib
            module = importlib.import_module(domain_config.extractor_module)
            extractor_getter = getattr(module, domain_config.extractor_class)
            return extractor_getter()
        except Exception as e:
            logger.error(f"❌ Failed to load extractor for {domain_name}: {e}")
            return None
    
    def get_all_extractors_for_document(
        self, 
        tags: List[str] = None, 
        category: str = None
    ) -> Dict[str, Any]:
        """
        Get all applicable extractors for a document
        
        Returns dict mapping domain_name -> extractor_instance
        """
        domains = self.detect_domains(tags, category)
        extractors = {}
        
        for domain_name in domains:
            extractor = self.get_extractor_for_domain(domain_name)
            if extractor:
                extractors[domain_name] = extractor
        
        return extractors


# Singleton instance
_domain_detector = None

def get_domain_detector() -> DomainDetector:
    """Get singleton instance of domain detector"""
    global _domain_detector
    if _domain_detector is None:
        _domain_detector = DomainDetector()
    return _domain_detector

