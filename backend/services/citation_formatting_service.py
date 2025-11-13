"""
Citation Formatting Service - Roosevelt's Citation Reform System
Ensures consistent numbered citations with proper in-line format and source mapping
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urlparse
from models.agent_response_models import CitationSource

logger = logging.getLogger(__name__)


class CitationFormattingService:
    """
    **BULLY!** Roosevelt's Citation Reform Service!
    Converts research findings to use numbered in-line citations (1), (2) format
    """
    
    def __init__(self):
        self.citation_counter = 0
        self.citation_map = {}  # Maps source identifiers to citation numbers
        
    def format_research_response_with_citations(
        self, 
        findings: str, 
        tool_results: List[Dict[str, Any]] = None,
        raw_citations: List[Dict[str, Any]] = None
    ) -> Tuple[str, List[CitationSource]]:
        """
        **Roosevelt's Master Citation Formatter!**
        
        Takes research findings and converts them to numbered citation format.
        
        Args:
            findings: Raw research findings text
            tool_results: Results from search tools with citation data
            raw_citations: Pre-extracted citation data
            
        Returns:
            Tuple of (formatted_findings, numbered_citations)
        """
        try:
            self.citation_counter = 0
            self.citation_map = {}
            
            # Step 1: Extract and standardize all citations
            all_citations = self._extract_all_citations(tool_results, raw_citations)
            
            # Step 2: Process findings text to add numbered citations
            formatted_findings = self._process_findings_text(findings, all_citations)
            
            # Step 3: Create numbered citation list
            numbered_citations = self._create_numbered_citations(all_citations)
            
            logger.info(f"✅ CITATION FORMATTING: {len(numbered_citations)} citations processed")
            return formatted_findings, numbered_citations
            
        except Exception as e:
            logger.error(f"❌ Citation formatting failed: {e}")
            return findings, []
    
    def _extract_all_citations(
        self, 
        tool_results: List[Dict[str, Any]] = None,
        raw_citations: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract citations from all available sources"""
        all_citations = []
        
        # Extract from tool results
        if tool_results:
            for tool_result in tool_results:
                citations = self._extract_citations_from_tool_result(tool_result)
                all_citations.extend(citations)
        
        # Extract from raw citations
        if raw_citations:
            for citation in raw_citations:
                standardized = self._standardize_citation(citation)
                if standardized:
                    all_citations.append(standardized)
        
        # Deduplicate citations by URL or title
        return self._deduplicate_citations(all_citations)
    
    def _extract_citations_from_tool_result(self, tool_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations from individual tool result"""
        citations = []
        
        try:
            # Handle different tool result formats
            if "results" in tool_result:
                # Structured search results
                for result in tool_result.get("results", []):
                    citation = self._create_citation_from_search_result(result)
                    if citation:
                        citations.append(citation)
            
            elif "content" in tool_result:
                # Text-based tool results - extract URLs and references
                content = tool_result.get("content", "")
                extracted = self._extract_citations_from_text(content)
                citations.extend(extracted)
            
        except Exception as e:
            logger.error(f"❌ Failed to extract citations from tool result: {e}")
        
        return citations
    
    def _create_citation_from_search_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create standardized citation from search result"""
        try:
            # Determine source type and extract relevant info
            url = result.get("url") or result.get("source_url")
            title = (
                result.get("title") or 
                result.get("document_title") or 
                result.get("filename") or
                result.get("source_title", "Unknown Source")
            )
            
            # Determine citation type
            citation_type = "document"
            if url and url.startswith("http"):
                citation_type = "webpage"
            elif result.get("source_collection") == "calibre":
                citation_type = "book"
            
            return {
                "title": title,
                "type": citation_type,
                "url": url,
                "author": result.get("author"),
                "date": result.get("date") or result.get("published_date"),
                "excerpt": result.get("content") or result.get("snippet"),
                "source_key": url or title  # For deduplication
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create citation from search result: {e}")
            return None
    
    def _extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract URLs and references from text content"""
        citations = []
        
        # Extract URLs
        url_pattern = r'https?://[^\s\)>\]"}]+'
        urls = re.findall(url_pattern, text)
        
        for url in urls:
            try:
                domain = urlparse(url).netloc
                citations.append({
                    "title": f"Web source from {domain}",
                    "type": "webpage",
                    "url": url,
                    "author": None,
                    "date": None,
                    "excerpt": None,
                    "source_key": url
                })
            except:
                continue
        
        return citations
    
    def _standardize_citation(self, citation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Standardize citation format"""
        try:
            return {
                "title": citation.get("title") or citation.get("source_title", "Unknown Source"),
                "type": citation.get("type", "document"),
                "url": citation.get("url") or citation.get("source_url"),
                "author": citation.get("author"),
                "date": citation.get("date"),
                "excerpt": citation.get("excerpt") or citation.get("quote_text"),
                "source_key": citation.get("url") or citation.get("title")
            }
        except Exception as e:
            logger.error(f"❌ Failed to standardize citation: {e}")
            return None
    
    def _deduplicate_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate citations based on URL or title"""
        seen_keys = set()
        unique_citations = []
        
        for citation in citations:
            source_key = citation.get("source_key", "")
            if source_key and source_key not in seen_keys:
                seen_keys.add(source_key)
                unique_citations.append(citation)
        
        return unique_citations
    
    def _process_findings_text(self, findings: str, citations: List[Dict[str, Any]]) -> str:
        """Process findings text to add numbered citations"""
        if not citations:
            return findings
        
        processed_text = findings
        
        # Strategy 1: Replace URLs with numbered citations
        for citation in citations:
            url = citation.get("url")
            if url:
                citation_num = self._get_citation_number(citation)
                # Replace URL with citation number
                processed_text = processed_text.replace(url, f"({citation_num})")
        
        # Strategy 2: Add citations for mentioned titles/sources
        for citation in citations:
            title = citation.get("title", "")
            if title and len(title) > 10:
                # Look for title mentions and add citation
                if title.lower() in processed_text.lower():
                    citation_num = self._get_citation_number(citation)
                    # Add citation after title mention (if not already there)
                    pattern = re.compile(re.escape(title), re.IGNORECASE)
                    if f"({citation_num})" not in processed_text:
                        processed_text = pattern.sub(f"{title} ({citation_num})", processed_text, count=1)
        
        return processed_text
    
    def _get_citation_number(self, citation: Dict[str, Any]) -> int:
        """Get or assign citation number"""
        source_key = citation.get("source_key", "")
        
        if source_key in self.citation_map:
            return self.citation_map[source_key]
        
        self.citation_counter += 1
        self.citation_map[source_key] = self.citation_counter
        return self.citation_counter
    
    def _create_numbered_citations(self, citations: List[Dict[str, Any]]) -> List[CitationSource]:
        """Create final numbered citation list"""
        numbered_citations = []
        
        for citation in citations:
            citation_num = self._get_citation_number(citation)
            
            citation_source = CitationSource(
                id=citation_num,
                title=citation.get("title", "Unknown Source"),
                type=citation.get("type", "document"),
                url=citation.get("url"),
                author=citation.get("author"),
                date=citation.get("date"),
                excerpt=citation.get("excerpt")
            )
            
            numbered_citations.append(citation_source)
        
        # Sort by citation number
        numbered_citations.sort(key=lambda x: x.id)
        
        return numbered_citations


# Global service instance
_citation_service = None

def get_citation_formatting_service() -> CitationFormattingService:
    """Get global citation formatting service instance"""
    global _citation_service
    if _citation_service is None:
        _citation_service = CitationFormattingService()
    return _citation_service
