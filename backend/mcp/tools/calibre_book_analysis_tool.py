"""
Calibre Book Analysis Tool - MCP Tool for analyzing and processing Calibre book content
Extracts, analyzes, and prepares book content for LLM processing and research
"""

import asyncio
import logging
import os
import tempfile
import time
import re
from typing import List, Dict, Any, Optional

from mcp.schemas.tool_schemas import (
    CalibreBookAnalysisInput,
    CalibreBookAnalysisOutput,
    ToolResponse,
    ToolError
)
from repositories.calibre_repository import CalibreRepository

logger = logging.getLogger(__name__)


class CalibreBookAnalysisTool:
    """MCP tool for analyzing and processing Calibre book content for research"""
    
    def __init__(self, calibre_repo: CalibreRepository = None):
        """Initialize with Calibre repository"""
        self.calibre_repo = calibre_repo or CalibreRepository()
        self.name = "analyze_calibre_book"
        self.description = "Extract, analyze, and process content from a specific Calibre book for research and LLM processing"
        
    async def initialize(self):
        """Initialize the Calibre book analysis tool"""
        try:
            if not self.calibre_repo._initialized:
                await self.calibre_repo.initialize()
            
            logger.info("📚 CalibreBookAnalysisTool initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CalibreBookAnalysisTool: {e}")
            raise
    
    async def execute(self, input_data: CalibreBookAnalysisInput) -> ToolResponse:
        """Execute Calibre book analysis with the given parameters"""
        start_time = time.time()
        
        try:
            logger.info(f"📚 Analyzing Calibre book ID: {input_data.book_id}")
            
            # Check if Calibre is available
            if not self.calibre_repo.is_available():
                logger.warning("📚 Calibre library not available")
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="CALIBRE_NOT_AVAILABLE",
                        error_message="Calibre library is not available",
                        details={"book_id": input_data.book_id}
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Get book metadata
            book = await self.calibre_repo.get_book_by_id(input_data.book_id)
            if not book:
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="BOOK_NOT_FOUND",
                        error_message=f"Book with ID {input_data.book_id} not found",
                        details={"book_id": input_data.book_id}
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Analyze book content based on type
            analysis_result = await self._analyze_book_content(book, input_data)
            
            # Create output
            output = CalibreBookAnalysisOutput(
                book_id=input_data.book_id,
                title=book.title,
                authors=book.authors,
                analysis_type=input_data.analysis_type,
                content_preview=analysis_result.get("content_preview", ""),
                file_path=analysis_result.get("file_path", ""),
                file_size=analysis_result.get("file_size", 0),
                available_formats=book.formats or [],
                metadata={
                    "series": book.series,
                    "series_index": book.series_index,
                    "publisher": book.publisher,
                    "pubdate": book.pubdate.isoformat() if book.pubdate else None,
                    "isbn": book.isbn,
                    "tags": book.tags or [],
                    "rating": book.rating,
                    "comments": book.comments
                },
                analysis_summary=analysis_result.get("summary", ""),
                extraction_successful=analysis_result.get("success", False),
                content_segments=analysis_result.get("content_segments", []),
                total_segments=analysis_result.get("total_segments", 0),
                total_content_length=analysis_result.get("total_content_length", 0)
            )
            
            return ToolResponse(
                success=True,
                output=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing Calibre book: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="ANALYSIS_ERROR",
                    error_message=f"Error analyzing Calibre book: {str(e)}",
                    details={"book_id": input_data.book_id}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _analyze_book_content(self, book, input_data: CalibreBookAnalysisInput) -> Dict[str, Any]:
        """Analyze book content based on the specified type"""
        try:
            # Get book file path
            file_path = await self.calibre_repo.get_book_file_path(input_data.book_id, input_data.preferred_format)
            if not file_path:
                return {
                    "success": False,
                    "summary": f"Could not find {input_data.preferred_format} format for book '{book.title}'",
                    "content_preview": "",
                    "file_path": "",
                    "file_size": 0
                }
            
            # Get file size
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            # Route to appropriate analysis method
            if input_data.analysis_type == "metadata":
                return await self._analyze_metadata(book, file_path, file_size)
            elif input_data.analysis_type == "content_extract":
                return await self._extract_content(book, file_path, file_size, input_data.max_content_length)
            elif input_data.analysis_type == "format_info":
                return await self._analyze_format_info(book, file_path, file_size)
            elif input_data.analysis_type == "llm_ready":
                return await self._extract_llm_ready_content(book, file_path, file_size, input_data)
            else:
                return {
                    "success": False,
                    "summary": f"Unknown analysis type: {input_data.analysis_type}",
                    "content_preview": "",
                    "file_path": file_path,
                    "file_size": file_size
                }
                
        except Exception as e:
            logger.error(f"❌ Error in book content analysis: {e}")
            return {
                "success": False,
                "summary": f"Error analyzing book content: {str(e)}",
                "content_preview": "",
                "file_path": "",
                "file_size": 0
            }
    
    async def _analyze_metadata(self, book, file_path: str, file_size: int) -> Dict[str, Any]:
        """Analyze book metadata"""
        metadata_summary = f"Book '{book.title}' by {book.authors}"
        
        if book.series:
            metadata_summary += f" (Series: {book.series}"
            if book.series_index:
                metadata_summary += f" #{book.series_index}"
            metadata_summary += ")"
        
        if book.publisher:
            metadata_summary += f" - Published by {book.publisher}"
        
        if book.pubdate:
            metadata_summary += f" in {book.pubdate.year}"
        
        if book.tags:
            metadata_summary += f" - Tags: {', '.join(book.tags)}"
        
        if book.rating:
            metadata_summary += f" - Rating: {book.rating}/5"
        
        metadata_summary += f" - File size: {file_size} bytes"
        
        return {
            "success": True,
            "summary": metadata_summary,
            "content_preview": metadata_summary,
            "file_path": file_path,
            "file_size": file_size
        }
    
    async def _extract_content(self, book, file_path: str, file_size: int, max_length: int = 2000) -> Dict[str, Any]:
        """Extract book content with basic segmentation"""
        try:
            # Extract content using calibredb
            cmd = [
                "calibredb", "show",
                "--library-path", self.calibre_repo.library_path,
                "--as-html",
                str(book.id)
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error(f"❌ calibredb show failed: {stderr.decode()}")
                return {
                    "success": False,
                    "summary": f"Failed to extract content from '{book.title}'",
                    "content_preview": "",
                    "file_path": file_path,
                    "file_size": file_size
                }
            
            # Clean HTML content
            content = stdout.decode('utf-8', errors='ignore')
            content = self._clean_html_content(content)
            
            # Create LLM-friendly content segments
            segments = self._create_content_segments(content, max_length)
            
            # Create preview (first segment)
            content_preview = segments[0] if segments else ""
            
            summary = f"Extracted {len(content)} characters of content from '{book.title}' in {len(segments)} segments"
            if len(content) > max_length:
                summary += f" (showing first {max_length} characters in preview)"
            
            return {
                "success": True,
                "summary": summary,
                "content_preview": content_preview,
                "content_segments": segments,
                "total_segments": len(segments),
                "total_content_length": len(content),
                "file_path": file_path,
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting content: {e}")
            return {
                "success": False,
                "summary": f"Error extracting content: {str(e)}",
                "content_preview": "",
                "file_path": file_path,
                "file_size": file_size
            }
    
    async def _extract_llm_ready_content(self, book, file_path: str, file_size: int, input_data: CalibreBookAnalysisInput) -> Dict[str, Any]:
        """Extract and process content specifically for LLM consumption"""
        try:
            # Extract content using calibredb
            cmd = [
                "calibredb", "show",
                "--library-path", self.calibre_repo.library_path,
                "--as-html",
                str(book.id)
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error(f"❌ calibredb show failed: {stderr.decode()}")
                return {
                    "success": False,
                    "summary": f"Failed to extract content from '{book.title}'",
                    "content_preview": "",
                    "file_path": file_path,
                    "file_size": file_size
                }
            
            # Clean HTML content
            content = stdout.decode('utf-8', errors='ignore')
            content = self._clean_html_content(content)
            
            # Create LLM-friendly segments with enhanced metadata
            segment_size = getattr(input_data, 'segment_size', 2000)
            overlap_size = getattr(input_data, 'overlap_size', 200)
            break_at_sentences = getattr(input_data, 'break_at_sentences', True)
            include_metadata = getattr(input_data, 'include_metadata', True)
            
            segments = self._create_llm_segments(content, segment_size, overlap_size, break_at_sentences)
            enhanced_segments = self._enhance_segments_for_llm(segments, book, include_metadata)
            
            # Create preview
            content_preview = enhanced_segments[0]["content"] if enhanced_segments else ""
            
            summary = f"Processed {len(content)} characters from '{book.title}' into {len(enhanced_segments)} LLM-ready segments"
            
            return {
                "success": True,
                "summary": summary,
                "content_preview": content_preview,
                "content_segments": enhanced_segments,
                "total_segments": len(enhanced_segments),
                "total_content_length": len(content),
                "file_path": file_path,
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing LLM content: {e}")
            return {
                "success": False,
                "summary": f"Error processing LLM content: {str(e)}",
                "content_preview": "",
                "file_path": file_path,
                "file_size": file_size
            }
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content to plain text"""
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', html_content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters and normalize
        content = content.replace('&nbsp;', ' ')
        content = content.replace('&amp;', '&')
        content = content.replace('&lt;', '<')
        content = content.replace('&gt;', '>')
        content = content.replace('&quot;', '"')
        
        return content.strip()
    
    def _create_content_segments(self, content: str, max_length: int = 2000, overlap: int = 200) -> List[str]:
        """Create overlapping content segments for LLM processing"""
        if not content:
            return []
        
        segments = []
        start = 0
        
        while start < len(content):
            # Calculate segment end
            end = start + max_length
            
            # If this isn't the last segment, try to break at a sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + max_length - 100, start)
                search_end = min(end + 50, len(content))
                
                # Find the last sentence ending in this range
                for i in range(search_end - 1, search_start - 1, -1):
                    if content[i] in '.!?':
                        end = i + 1
                        break
            
            # Extract the segment
            segment = content[start:end].strip()
            if segment:
                segments.append(segment)
            
            # Move start position for next segment (with overlap)
            start = max(start + 1, end - overlap)
            
            # Safety check to prevent infinite loops
            if start >= len(content):
                break
        
        return segments
    
    def _create_llm_segments(self, content: str, segment_size: int, overlap: int, break_at_sentences: bool = True) -> List[str]:
        """Create LLM-friendly content segments with sentence-aware breaking"""
        if not content:
            return []
        
        segments = []
        start = 0
        
        while start < len(content):
            # Calculate segment end
            end = start + segment_size
            
            # If this isn't the last segment and we want sentence-aware breaks
            if end < len(content) and break_at_sentences:
                # Look for sentence endings within the last 100 characters
                search_start = max(start + segment_size - 100, start)
                search_end = min(end + 50, len(content))
                
                # Find the last sentence ending in this range
                for i in range(search_end - 1, search_start - 1, -1):
                    if content[i] in '.!?':
                        end = i + 1
                        break
            
            # Extract the segment
            segment = content[start:end].strip()
            if segment:
                segments.append(segment)
            
            # Move start position for next segment (with overlap)
            start = max(start + 1, end - overlap)
            
            # Safety check to prevent infinite loops
            if start >= len(content):
                break
        
        return segments
    
    def _enhance_segments_for_llm(self, segments: List[str], book, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Enhance segments with metadata for LLM context"""
        enhanced_segments = []
        
        for i, segment in enumerate(segments):
            enhanced_segment = {
                "segment_id": i + 1,
                "content": segment,
                "length": len(segment),
                "book_title": book.title,
                "book_author": book.authors,
                "book_id": book.id,
                "segment_number": i + 1,
                "total_segments": len(segments)
            }
            
            # Add optional metadata
            if include_metadata:
                if book.series:
                    enhanced_segment["series"] = book.series
                    if book.series_index:
                        enhanced_segment["series_index"] = book.series_index
                
                if book.publisher:
                    enhanced_segment["publisher"] = book.publisher
                
                if book.tags:
                    enhanced_segment["tags"] = book.tags
                
                if book.rating:
                    enhanced_segment["rating"] = book.rating
            
            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments
    
    async def _analyze_format_info(self, book, file_path: str, file_size: int) -> Dict[str, Any]:
        """Analyze book format information"""
        format_info = f"Book '{book.title}' format analysis:\n"
        format_info += f"- Available formats: {', '.join(book.formats) if book.formats else 'None'}\n"
        format_info += f"- File path: {file_path}\n"
        format_info += f"- File size: {file_size} bytes ({file_size / 1024:.1f} KB)\n"
        
        if book.size:
            format_info += f"- Calibre size: {book.size} bytes\n"
        
        return {
            "success": True,
            "summary": format_info,
            "content_preview": format_info,
            "file_path": file_path,
            "file_size": file_size
        }
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """Get the tool schema for MCP registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "integer",
                        "description": "Calibre book ID to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["metadata", "content_extract", "format_info", "llm_ready"],
                        "description": "Type of analysis to perform",
                        "default": "metadata"
                    },
                    "preferred_format": {
                        "type": "string",
                        "description": "Preferred book format for content extraction",
                        "default": "epub"
                    },
                    "max_content_length": {
                        "type": "integer",
                        "description": "Maximum content length for extraction",
                        "default": 2000
                    },
                    "segment_size": {
                        "type": "integer",
                        "description": "Size of each content segment for LLM processing",
                        "default": 2000
                    },
                    "overlap_size": {
                        "type": "integer",
                        "description": "Overlap between segments",
                        "default": 200
                    },
                    "break_at_sentences": {
                        "type": "boolean",
                        "description": "Break segments at sentence boundaries",
                        "default": True
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include book metadata in segments",
                        "default": True
                    }
                },
                "required": ["book_id"]
            }
        }
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get the tool definition for MCP registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.get_tool_schema()["inputSchema"]
        }


# Convenience function for direct usage
async def analyze_calibre_book(
    book_id: int, 
    analysis_type: str = "metadata",
    preferred_format: str = "epub",
    max_content_length: int = 2000,
    segment_size: int = 2000,
    overlap_size: int = 200,
    break_at_sentences: bool = True,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """Convenience function to analyze a Calibre book"""
    tool = CalibreBookAnalysisTool()
    await tool.initialize()
    
    input_data = CalibreBookAnalysisInput(
        book_id=book_id,
        analysis_type=analysis_type,
        preferred_format=preferred_format,
        max_content_length=max_content_length,
        segment_size=segment_size,
        overlap_size=overlap_size,
        break_at_sentences=break_at_sentences,
        include_metadata=include_metadata
    )
    
    result = await tool.execute(input_data)
    return result.output.dict() if result.success else {"error": result.error.error_message} 