"""
Universal Content Extractor Service

**BULLY!** This service uses the most proven content extraction libraries to focus on main content!
**By George!** We're using the same algorithms that power Firefox Reader Mode and major RSS readers!

Uses:
- readability-lxml: Mozilla's Readability algorithm (Firefox Reader Mode)
- newspaper3k: Excellent for news articles and blog posts
- trafilatura: Modern, fast content extraction
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


class UniversalContentExtractor:
    """
    Universal content extraction service using the most proven libraries
    
    **Trust busting for website chrome!** This service focuses on main content!
    """
    
    def __init__(self):
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize available content extraction libraries"""
        logger.info("🎯 Universal Content Extractor initializing with proven libraries...")
        
        # Check for Readability (Mozilla's algorithm - same as Firefox Reader Mode)
        try:
            from readability import Document
            self.readability_available = True
            logger.info("✅ Readability-lxml available (Mozilla's algorithm)")
        except ImportError:
            self.readability_available = False
            logger.warning("⚠️ Readability-lxml not available")
        
        # Check for Newspaper3k (excellent for news articles)
        try:
            import newspaper
            self.newspaper_available = True
            logger.info("✅ Newspaper3k available (excellent for news articles)")
        except ImportError:
            self.newspaper_available = False
            logger.warning("⚠️ Newspaper3k not available")
        
        # Check for Trafilatura (modern, fast)
        try:
            import trafilatura
            self.trafilatura_available = True
            logger.info("✅ Trafilatura available (modern, fast)")
        except ImportError:
            self.trafilatura_available = False
            logger.warning("⚠️ Trafilatura not available")
        
        logger.info("🎯 Universal Content Extractor ready with proven extraction methods")
    
    async def extract_main_content(self, html_content: str, url: str = None) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Extract main content using the most proven method available
        
        Returns:
            Tuple of (cleaned_text, original_html, images)
        """
        if not html_content:
            return "", "", []
        
        # Try Readability first (Mozilla's algorithm - most proven)
        if self.readability_available:
            try:
                text, html, images = await self._extract_with_readability(html_content, url)
                if text and len(text.strip()) > 100:
                    logger.info(f"✅ Readability extracted {len(text)} characters and {len(images)} images")
                    return text, html, images
            except Exception as e:
                logger.warning(f"⚠️ Readability extraction failed: {e}")
        
        # Try Newspaper3k (excellent for news articles)
        if self.newspaper_available:
            try:
                text, html, images = await self._extract_with_newspaper(html_content, url)
                if text and len(text.strip()) > 100:
                    logger.info(f"✅ Newspaper3k extracted {len(text)} characters and {len(images)} images")
                    return text, html, images
            except Exception as e:
                logger.warning(f"⚠️ Newspaper3k extraction failed: {e}")
        
        # Try Trafilatura (modern, fast)
        if self.trafilatura_available:
            try:
                text, html, images = await self._extract_with_trafilatura(html_content, url)
                if text and len(text.strip()) > 100:
                    logger.info(f"✅ Trafilatura extracted {len(text)} characters and {len(images)} images")
                    return text, html, images
            except Exception as e:
                logger.warning(f"⚠️ Trafilatura extraction failed: {e}")
        
        logger.warning("⚠️ All content extractors failed, returning empty result")
        return "", "", []
    
    async def _extract_with_readability(self, html_content: str, url: str = None) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Extract content using Mozilla's Readability algorithm (same as Firefox Reader Mode)"""
        try:
            from readability import Document
            
            # Create Readability document
            doc = Document(html_content)
            
            # Extract main content
            extracted_html = doc.summary()
            extracted_text = doc.title() + "\n\n" + doc.summary(html_partial=True)
            
            if not extracted_text:
                return "", "", []
            
            # Extract images from the HTML content
            images = self._extract_images_from_html(html_content, url)
            
            # Clean the extracted text
            cleaned_text = self._clean_extracted_text(extracted_text)
            
            logger.info(f"✅ Readability extracted {len(cleaned_text)} chars and {len(images)} images")
            return cleaned_text, extracted_html, images
            
        except Exception as e:
            logger.error(f"❌ Readability extraction failed: {e}")
            return "", "", []
    
    async def _extract_with_newspaper(self, html_content: str, url: str = None) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Extract content using Newspaper3k (excellent for news articles)"""
        try:
            import newspaper
            from newspaper import Article
            
            # Create a temporary article object
            article = Article(url or 'http://example.com')
            article.download(input_html=html_content)
            article.parse()
            
            # Extract main content
            main_text = article.text
            if not main_text:
                return "", "", []
            
            # Extract images from the article
            images = []
            for img in article.images:
                images.append({
                    'src': img,
                    'alt': '',
                    'title': '',
                    'caption': '',
                    'width': None,
                    'height': None,
                    'type': 'content'
                })
            
            # Clean the extracted text
            cleaned_text = self._clean_extracted_text(main_text)
            
            logger.info(f"✅ Newspaper3k extracted {len(cleaned_text)} chars and {len(images)} images")
            return cleaned_text, main_text, images
            
        except Exception as e:
            logger.error(f"❌ Newspaper3k extraction failed: {e}")
            return "", "", []
    
    def _score_content(self, text: str, html: str) -> float:
        """
        Score content quality based on various metrics
        
        Higher scores indicate better content quality
        """
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length score (prefer longer content, but not too long)
        length = len(text)
        if 500 <= length <= 10000:
            score += 0.3
        elif 1000 <= length <= 5000:
            score += 0.4
        elif length > 5000:
            score += 0.2
        
        # Paragraph density (good content has paragraphs)
        paragraphs = text.count('\n\n')
        if 3 <= paragraphs <= 20:
            score += 0.2
        
        # Sentence structure (good content has proper sentences)
        sentences = len(re.findall(r'[.!?]+', text))
        if 5 <= sentences <= 100:
            score += 0.2
        
        # Word diversity (avoid repetitive content)
        words = text.lower().split()
        unique_words = len(set(words))
        if len(words) > 0:
            diversity = unique_words / len(words)
            if 0.6 <= diversity <= 0.9:
                score += 0.2
        
        # Penalize common website chrome indicators
        chrome_indicators = [
            'cookie', 'privacy', 'terms', 'subscribe', 'newsletter', 'follow us',
            'share this', 'related articles', 'popular posts', 'recent posts',
            'comments', 'leave a comment', 'posted by', 'published on',
            'navigation', 'menu', 'sidebar', 'footer', 'header'
        ]
        
        text_lower = text.lower()
        chrome_count = sum(1 for indicator in chrome_indicators if indicator in text_lower)
        if chrome_count > 0:
            score -= min(0.3, chrome_count * 0.05)
        
        return max(0.0, min(1.0, score))
    
    async def _extract_with_trafilatura(self, html_content: str, url: str = None) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Extract content using Trafilatura (modern, fast)"""
        try:
            import trafilatura
            
            # Extract main content
            extracted_text = trafilatura.extract(html_content, include_formatting=True, include_links=True)
            
            if not extracted_text:
                return "", "", []
            
            # Extract images from the HTML content
            images = self._extract_images_from_html(html_content, url)
            
            # Clean the extracted text
            cleaned_text = self._clean_extracted_text(extracted_text)
            
            logger.info(f"✅ Trafilatura extracted {len(cleaned_text)} chars and {len(images)} images")
            return cleaned_text, extracted_text, images
            
        except Exception as e:
            logger.error(f"❌ Trafilatura extraction failed: {e}")
            return "", "", []
    
    def _extract_images_from_html(self, html_content: str, base_url: str = None) -> List[Dict[str, Any]]:
        """Extract images from HTML content - more aggressive approach"""
        try:
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin, urlparse
            
            soup = BeautifulSoup(html_content, 'html.parser')
            images = []
            seen_srcs = set()
            
            # First, look for images in article content areas (most likely to be content)
            for article in soup.find_all(['article', 'div'], class_=re.compile(r'(entry-content|post-content|article-content|content-area|article-body|post-body)', re.I)):
                for img in article.find_all('img'):
                    src = img.get('src')
                    if src and src not in seen_srcs:
                        seen_srcs.add(src)
                        
                        # Make URL absolute if it's relative
                        if base_url and not src.startswith(('http://', 'https://')):
                            src = urljoin(base_url, src)
                        
                        # Skip only the most obvious non-content images
                        src_lower = src.lower()
                        if any(pattern in src_lower for pattern in ['favicon', 'logo', 'spacer', 'pixel', 'tracking']):
                            continue
                        
                        image_data = {
                            'src': src,
                            'alt': img.get('alt', ''),
                            'title': img.get('title', ''),
                            'caption': '',
                            'width': img.get('width'),
                            'height': img.get('height'),
                            'type': 'content'
                        }
                        images.append(image_data)
            
            # Then look for images in figure tags (common in articles)
            for figure in soup.find_all('figure'):
                img = figure.find('img')
                if img:
                    src = img.get('src')
                    if src and src not in seen_srcs:
                        seen_srcs.add(src)
                        
                        if base_url and not src.startswith(('http://', 'https://')):
                            src = urljoin(base_url, src)
                        
                        # Extract caption if available
                        caption = ""
                        figcaption = figure.find('figcaption')
                        if figcaption:
                            caption = figcaption.get_text().strip()
                        
                        image_data = {
                            'src': src,
                            'alt': img.get('alt', ''),
                            'title': img.get('title', ''),
                            'caption': caption,
                            'width': img.get('width'),
                            'height': img.get('height'),
                            'type': 'content'
                        }
                        images.append(image_data)
            
            # Finally, look for any remaining img tags that might be content
            for img in soup.find_all('img'):
                src = img.get('src')
                if src and src not in seen_srcs:
                    seen_srcs.add(src)
                    
                    # Make URL absolute if it's relative
                    if base_url and not src.startswith(('http://', 'https://')):
                        src = urljoin(base_url, src)
                    
                    # Skip only the most obvious non-content images
                    src_lower = src.lower()
                    if any(pattern in src_lower for pattern in ['favicon', 'logo', 'spacer', 'pixel', 'tracking', 'analytics']):
                        continue
                    
                    # Check if it's likely content by looking at parent context
                    parent_text = ""
                    parent = img.parent
                    for _ in range(2):  # Check up to 2 levels up
                        if parent and hasattr(parent, 'get_text'):
                            parent_text += parent.get_text() + " "
                        if parent:
                            parent = parent.parent
                    
                    # If parent has substantial text, it's likely content
                    if len(parent_text.strip()) > 50:
                        image_data = {
                            'src': src,
                            'alt': img.get('alt', ''),
                            'title': img.get('title', ''),
                            'caption': '',
                            'width': img.get('width'),
                            'height': img.get('height'),
                            'type': 'content'
                        }
                        images.append(image_data)
            
            logger.info(f"📸 Extracted {len(images)} images from HTML")
            return images
            
        except Exception as e:
            logger.error(f"❌ Image extraction failed: {e}")
            return []
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text with minimal, universal patterns"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # **BULLY!** Remove advertisement JSON widget configurations first
        # These are the pesky ruamupr.com and adcovery widgets that slip through
        text = re.sub(r'\{[^}]*"client_callback_domain"[^}]*\}', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\{[^}]*"widget_type"[^}]*\}', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\{[^}]*"publisher_website_id"[^}]*\}', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\{[^}]*"target_selector"[^}]*\}', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\{[^}]*"widget_div_id"[^}]*\}', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\{[^}]*"adcovery"[^}]*\}', ' ', text, flags=re.IGNORECASE)
        
        # Remove any JSON objects containing common ad network domains
        text = re.sub(r'\{[^}]*ruamupr\.com[^}]*\}', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\{[^}]*doubleclick[^}]*\}', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\{[^}]*googlesyndication[^}]*\}', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\{[^}]*adsystem[^}]*\}', ' ', text, flags=re.IGNORECASE)
        
        # Remove leftover ad domain references
        text = re.sub(r"ruamupr\.com\S*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"adcovery[^\s]*", " ", text, flags=re.IGNORECASE)
        
        # Remove only the most obvious navigation artifacts
        artifacts_to_remove = [
            r'Share this article', r'Follow us on', r'Subscribe to', 
            r'Skip to content', r'Cookie Policy', r'Privacy Policy',
            r'Terms of Service', r'Contact Us', r'About Us',
            r'Search', r'Login', r'Sign up', r'Subscribe', r'Newsletter',
            r'Follow', r'Share', r'Like', r'Comment', r'Related Articles',
            r'Recommended', r'Popular', r'Trending', r'Most Read', r'Latest',
            r'Previous', r'Next', r'Back to top', r'Return to top',
            r'Advertisement', r'Ad', r'Sponsored', r'Promoted',
            r'Menu', r'Navigation', r'Breadcrumb', r'Pagination',
            r'Footer', r'Header', r'Sidebar', r'Widget'
        ]
        
        for artifact in artifacts_to_remove:
            text = re.sub(artifact, '', text, flags=re.IGNORECASE)
        
        # Clean up any remaining excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Limit content length to prevent database issues
        if len(text) > 50000:
            text = text[:50000] + "..."
        
        return text


# Global instance
_universal_extractor = None

async def get_universal_content_extractor() -> UniversalContentExtractor:
    """Get global universal content extractor instance"""
    global _universal_extractor
    if _universal_extractor is None:
        _universal_extractor = UniversalContentExtractor()
    return _universal_extractor
