"""
Markdown Formatter Service
Formats web content and RSS articles as structured Markdown files with YAML frontmatter

**BULLY!** The Roosevelt "Square Deal" approach to content preservation!
"""

import logging
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class MarkdownFormatterService:
    """
    Formats web content as structured Markdown files
    
    **BULLY!** Preserves content with the reliability of a cavalry charge!
    """
    
    def __init__(self):
        logger.info("📝 Markdown Formatter Service initialized")
    
    def format_rss_article(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        document_id: str,
        title: str,
        source_url: str = None,
        feed_name: str = None,
        author: str = None,
        published_date: datetime = None,
        folder_id: str = None,
        images: List[Dict[str, Any]] = None
    ) -> str:
        """
        Format RSS article as Markdown with YAML frontmatter
        
        **By George!** Creates a beautiful, structured document!
        """
        try:
            # Create YAML frontmatter
            frontmatter = {
                'title': title,
                'source_url': source_url,
                'source_type': 'rss',
                'feed_name': feed_name,
                'author': author,
                'published_date': published_date.isoformat() if published_date else None,
                'imported_date': datetime.now().isoformat(),
                'document_id': document_id,
                'collection_type': metadata.get('collection_type', 'global'),
                'folder_id': folder_id,
                'tags': metadata.get('tags', []),
                'category': metadata.get('category', 'news'),
                'content_length': len(content),
                'chunk_count': metadata.get('chunk_count', 0)
            }
            
            # Add RSS-specific metadata
            if metadata.get('rss_article_id'):
                frontmatter['rss_article_id'] = metadata['rss_article_id']
            if metadata.get('rss_feed_id'):
                frontmatter['rss_feed_id'] = metadata['rss_feed_id']
            
            # Clean and format content
            cleaned_content = self._clean_html_content(content)
            
            # Build markdown document
            markdown_content = f"""---
{yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)}---

# {title}

{cleaned_content}

"""
            
            # Add images section if images exist
            if images:
                markdown_content += self._format_images_section(images, document_id)
            
            # Add source information footer
            markdown_content += self._format_source_footer(
                source_url, feed_name, published_date, datetime.now()
            )
            
            logger.info(f"📝 Formatted RSS article as Markdown: {len(markdown_content)} characters")
            return markdown_content
            
        except Exception as e:
            logger.error(f"❌ Failed to format RSS article as Markdown: {e}")
            raise
    
    def format_web_content(
        self,
        content: str,
        metadata: Dict[str, Any], 
        document_id: str,
        title: str,
        source_url: str,
        folder_id: str = None,
        images: List[Dict[str, Any]] = None
    ) -> str:
        """
        Format scraped web content as Markdown with YAML frontmatter
        
        **BULLY!** Perfect for general web scraping!
        """
        try:
            # Create YAML frontmatter
            frontmatter = {
                'title': title,
                'source_url': source_url,
                'source_type': 'web_scraped',
                'domain': self._extract_domain(source_url),
                'scraped_date': datetime.now().isoformat(),
                'document_id': document_id,
                'collection_type': metadata.get('collection_type', 'user'),
                'folder_id': folder_id,
                'tags': metadata.get('tags', []),
                'category': metadata.get('category', 'web_content'),
                'content_length': len(content),
                'chunk_count': metadata.get('chunk_count', 0)
            }
            
            # Clean and format content
            cleaned_content = self._clean_html_content(content)
            
            # Build markdown document
            markdown_content = f"""---
{yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)}---

# {title}

{cleaned_content}

"""
            
            # Add images section if images exist
            if images:
                markdown_content += self._format_images_section(images, document_id)
            
            # Add source information footer
            markdown_content += self._format_source_footer(
                source_url, None, None, datetime.now()
            )
            
            logger.info(f"📝 Formatted web content as Markdown: {len(markdown_content)} characters")
            return markdown_content
            
        except Exception as e:
            logger.error(f"❌ Failed to format web content as Markdown: {e}")
            raise
    
    def _clean_html_content(self, content: str) -> str:
        """Clean HTML content and convert to clean Markdown"""
        try:
            # Remove HTML tags but preserve structure
            import re
            
            # Convert common HTML elements to Markdown
            content = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<br[^>]*/?>', '\n', content, flags=re.IGNORECASE)
            content = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', content, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove remaining HTML tags
            content = re.sub(r'<[^>]+>', '', content)
            
            # Clean up whitespace
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple newlines to double
            content = re.sub(r'^\s+|\s+$', '', content, flags=re.MULTILINE)  # Trim lines
            content = content.strip()
            
            return content
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean HTML content: {e}")
            return content  # Return original if cleaning fails
    
    def _format_images_section(self, images: List[Dict[str, Any]], document_id: str = None) -> str:
        """Format images as Markdown section with local image saving"""
        if not images:
            return ""
        
        try:
            import aiohttp
            import asyncio
            from pathlib import Path
            from config import settings
            
            images_md = "\n## Images\n\n"
            saved_images = []
            
            # Create images directory
            upload_dir = Path(settings.UPLOAD_DIR)
            images_dir = upload_dir / "web_sources" / "images" / document_id if document_id else upload_dir / "web_sources" / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            # Download and save images
            for i, img in enumerate(images):
                alt_text = img.get('alt', 'Image')
                url = img.get('src', img.get('url', ''))
                
                if url:
                    try:
                        # Generate local filename
                        file_extension = self._get_image_extension(url)
                        local_filename = f"image_{i+1}{file_extension}"
                        local_path = images_dir / local_filename
                        
                        # Download image
                        async def download_image():
                            async with aiohttp.ClientSession() as session:
                                async with session.get(url, timeout=30) as response:
                                    if response.status == 200:
                                        content = await response.read()
                                        with open(local_path, 'wb') as f:
                                            f.write(content)
                                        return str(local_path)
                                    return None
                        
                        # Run download (this is a sync method, so we'll use asyncio.run)
                        try:
                            saved_path = asyncio.run(download_image())
                            if saved_path:
                                # Use static URL path in markdown
                                static_path = f"/static/images/{document_id}/{local_filename}" if document_id else f"/static/images/{local_filename}"
                                images_md += f"- ![{alt_text}]({static_path})\n"
                                saved_images.append(saved_path)
                                logger.info(f"✅ Saved image: {saved_path}")
                            else:
                                # Fallback to original URL
                                images_md += f"- ![{alt_text}]({url})\n"
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to download image {url}: {e}")
                            # Fallback to original URL
                            images_md += f"- ![{alt_text}]({url})\n"
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process image {url}: {e}")
                        # Fallback to original URL
                        images_md += f"- ![{alt_text}]({url})\n"
            
            if saved_images:
                logger.info(f"✅ Saved {len(saved_images)} images for document {document_id}")
            
            return images_md + "\n"
            
        except Exception as e:
            logger.error(f"❌ Failed to format images section: {e}")
            # Fallback to simple image links
            images_md = "\n## Images\n\n"
            for img in images:
                alt_text = img.get('alt', 'Image')
                url = img.get('src', img.get('url', ''))
                if url:
                    images_md += f"- ![{alt_text}]({url})\n"
            return images_md + "\n"
    
    def _get_image_extension(self, url: str) -> str:
        """Get image file extension from URL"""
        try:
            # Extract extension from URL
            if '.' in url.split('/')[-1]:
                ext = url.split('.')[-1].split('?')[0].lower()
                if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg']:
                    return f".{ext}"
        except:
            pass
        return ".jpg"  # Default extension
    
    def _format_source_footer(
        self, 
        source_url: str, 
        feed_name: str = None, 
        published_date: datetime = None,
        imported_date: datetime = None
    ) -> str:
        """Format source information footer"""
        footer = "\n---\n\n## Source Information\n\n"
        
        if feed_name:
            footer += f"- **Source**: {feed_name} RSS Feed\n"
        else:
            footer += f"- **Source**: Web Scraped Content\n"
            
        if source_url:
            footer += f"- **Original URL**: {source_url}\n"
        if published_date:
            footer += f"- **Published**: {published_date.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        if imported_date:
            footer += f"- **Imported**: {imported_date.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        
        return footer
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return "unknown"
    
    def generate_filename(
        self, 
        document_id: str, 
        title: str, 
        source_type: str = "rss",
        feed_name: str = None
    ) -> str:
        """
        Generate safe filename for markdown file
        
        **BULLY!** Creates clean, organized filenames!
        """
        try:
            # Sanitize title for filename
            safe_title = re.sub(r'[^\w\s-]', '', title.lower())
            safe_title = re.sub(r'[-\s]+', '_', safe_title)
            safe_title = safe_title[:50]  # Limit length
            
            # Add source prefix
            if source_type == "rss" and feed_name:
                safe_feed = re.sub(r'[^\w\s-]', '', feed_name.lower())
                safe_feed = re.sub(r'[-\s]+', '_', safe_feed)
                prefix = f"rss_{safe_feed}"
            elif source_type == "web":
                prefix = "web"
            else:
                prefix = "content"
            
            filename = f"{document_id}_{prefix}_{safe_title}.md"
            
            logger.debug(f"📝 Generated filename: {filename}")
            return filename
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate filename: {e}")
            return f"{document_id}_content.md"


# Global instance
_markdown_formatter_instance = None

async def get_markdown_formatter() -> MarkdownFormatterService:
    """Get or create global markdown formatter instance"""
    global _markdown_formatter_instance
    if _markdown_formatter_instance is None:
        _markdown_formatter_instance = MarkdownFormatterService()
    return _markdown_formatter_instance
