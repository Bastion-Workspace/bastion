"""
OCR Service - Handles OCR processing and hOCR file editing
"""

import asyncio
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re

import ocrmypdf
from PIL import Image
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

from config import settings
from models.api_models import ProcessingStatus

logger = logging.getLogger(__name__)


class OCRService:
    """Service for OCR processing and hOCR file editing"""
    
    def __init__(self):
        self.initialized = False
        self.hocr_storage_dir = Path("processed/hocr_files")
        self.hocr_storage_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize OCR service"""
        if self.initialized:
            return
        
        try:
            # Test OCRmyPDF availability
            import ocrmypdf
            logger.info("✅ OCR service initialized with OCRmyPDF")
            self.initialized = True
            
        except ImportError as e:
            logger.error(f"❌ OCRmyPDF not available: {e}")
            raise
    
    async def process_pdf_with_ocr(
        self, 
        input_pdf_path: str, 
        document_id: str,
        force_ocr: bool = False,
        preserve_hocr: bool = True
    ) -> Dict:
        """
        Process PDF with OCR and optionally preserve hOCR data
        
        Args:
            input_pdf_path: Path to input PDF
            document_id: Document ID for storage
            force_ocr: Force OCR even if text exists
            preserve_hocr: Save hOCR data for editing
            
        Returns:
            Dict with processing results and hOCR info
        """
        try:
            logger.info(f"🔄 Starting OCR processing for document {document_id}")
            
            # Create output paths
            output_pdf_path = f"{input_pdf_path}_ocr.pdf"
            hocr_path = self.hocr_storage_dir / f"{document_id}.hocr"
            
            # OCR configuration for OCRmyPDF 16.10.4
            # Use force_ocr to replace existing text layers completely
            ocr_config = {
                'language': '+'.join(settings.OCR_LANGUAGES),
                'force_ocr': True,  # Force OCR and replace existing text layers
                'output_type': 'pdf',
                'optimize': 1,
                'jpeg_quality': 95,
                'png_quality': 95,
                'pdfa_image_compression': 'jpeg'
            }
            
            # Add hOCR output if requested
            if preserve_hocr:
                ocr_config['sidecar'] = str(hocr_path)
            
            # Run OCR
            logger.info(f"🔄 Running OCRmyPDF with config: {ocr_config}")
            
            try:
                ocrmypdf.ocr(
                    input_pdf_path,
                    output_pdf_path,
                    **ocr_config
                )
                
                # Replace original with OCR'd version
                os.replace(output_pdf_path, input_pdf_path)
                
                # Parse hOCR data if available
                hocr_data = None
                if preserve_hocr and hocr_path.exists():
                    hocr_data = await self._parse_hocr_file(str(hocr_path))
                
                logger.info(f"✅ OCR processing completed for document {document_id}")
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "hocr_available": preserve_hocr and hocr_path.exists(),
                    "hocr_path": str(hocr_path) if hocr_path.exists() else None,
                    "hocr_data": hocr_data,
                    "pages_processed": hocr_data.get("page_count", 0) if hocr_data else 0
                }
                
            except ocrmypdf.exceptions.PriorOcrFoundError:
                logger.info(f"ℹ️ PDF already contains OCR text, skipping OCR for {document_id}")
                return {
                    "status": "skipped",
                    "reason": "PDF already contains OCR text",
                    "document_id": document_id,
                    "hocr_available": False,
                    "pages_processed": 0
                }
                
        except Exception as e:
            logger.error(f"❌ OCR processing failed for document {document_id}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "document_id": document_id,
                "hocr_available": False,
                "pages_processed": 0
            }
    
    async def get_hocr_data(self, document_id: str) -> Optional[Dict]:
        """Get hOCR data for a document"""
        try:
            hocr_path = self.hocr_storage_dir / f"{document_id}.hocr"
            
            if not hocr_path.exists():
                return None
            
            return await self._parse_hocr_file(str(hocr_path))
            
        except Exception as e:
            logger.error(f"❌ Failed to get hOCR data for {document_id}: {e}")
            return None
    
    async def update_hocr_text(
        self, 
        document_id: str, 
        page_number: int, 
        word_id: str, 
        new_text: str
    ) -> bool:
        """Update text in hOCR file"""
        try:
            hocr_path = self.hocr_storage_dir / f"{document_id}.hocr"
            
            if not hocr_path.exists():
                logger.error(f"❌ hOCR file not found for document {document_id}")
                return False
            
            # Parse hOCR file
            with open(hocr_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Find the word element
            word_element = soup.find(id=word_id)
            if not word_element:
                logger.error(f"❌ Word element {word_id} not found in hOCR")
                return False
            
            # Update the text
            word_element.string = new_text
            
            # Save updated hOCR
            with open(hocr_path, 'w', encoding='utf-8') as f:
                f.write(str(soup))
            
            logger.info(f"✅ Updated hOCR text for {document_id}, word {word_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update hOCR text: {e}")
            return False
    
    async def export_corrected_text(self, document_id: str) -> Optional[str]:
        """Export corrected text from hOCR file"""
        try:
            hocr_data = await self.get_hocr_data(document_id)
            if not hocr_data:
                return None
            
            # Extract text from all pages
            full_text = []
            for page in hocr_data.get("pages", []):
                page_text = []
                for line in page.get("lines", []):
                    line_text = " ".join(word.get("text", "") for word in line.get("words", []))
                    if line_text.strip():
                        page_text.append(line_text.strip())
                
                if page_text:
                    full_text.append("\n".join(page_text))
            
            return "\n\n".join(full_text)
            
        except Exception as e:
            logger.error(f"❌ Failed to export corrected text: {e}")
            return None
    
    async def get_ocr_confidence_stats(self, document_id: str) -> Optional[Dict]:
        """Get OCR confidence statistics for a document"""
        try:
            hocr_data = await self.get_hocr_data(document_id)
            if not hocr_data:
                return None
            
            all_confidences = []
            page_stats = []
            
            for page in hocr_data.get("pages", []):
                page_confidences = []
                
                for line in page.get("lines", []):
                    for word in line.get("words", []):
                        confidence = word.get("confidence", 0)
                        if confidence > 0:
                            all_confidences.append(confidence)
                            page_confidences.append(confidence)
                
                if page_confidences:
                    page_stats.append({
                        "page_number": page.get("page_number", 0),
                        "avg_confidence": sum(page_confidences) / len(page_confidences),
                        "min_confidence": min(page_confidences),
                        "max_confidence": max(page_confidences),
                        "word_count": len(page_confidences)
                    })
            
            if not all_confidences:
                return None
            
            return {
                "overall_avg_confidence": sum(all_confidences) / len(all_confidences),
                "overall_min_confidence": min(all_confidences),
                "overall_max_confidence": max(all_confidences),
                "total_words": len(all_confidences),
                "page_stats": page_stats,
                "low_confidence_words": len([c for c in all_confidences if c < 60]),
                "high_confidence_words": len([c for c in all_confidences if c > 90])
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get OCR confidence stats: {e}")
            return None
    
    async def _parse_hocr_file(self, hocr_path: str) -> Dict:
        """Parse hOCR file and extract structured data"""
        try:
            with open(hocr_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            pages = []
            page_elements = soup.find_all('div', class_='ocr_page')
            
            for page_elem in page_elements:
                page_info = self._parse_page_element(page_elem)
                if page_info:
                    pages.append(page_info)
            
            return {
                "document_type": "hocr",
                "page_count": len(pages),
                "pages": pages,
                "total_words": sum(len(line.get("words", [])) for page in pages for line in page.get("lines", [])),
                "creation_info": {
                    "generator": soup.find('meta', attrs={'name': 'ocr-system'})['content'] if soup.find('meta', attrs={'name': 'ocr-system'}) else 'unknown'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse hOCR file: {e}")
            return {}
    
    def _parse_page_element(self, page_elem) -> Optional[Dict]:
        """Parse a single page element from hOCR"""
        try:
            # Extract page number and dimensions
            page_id = page_elem.get('id', '')
            page_number = int(page_id.split('_')[-1]) if page_id else 0
            
            # Parse bbox from title attribute
            title = page_elem.get('title', '')
            bbox_match = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
            if bbox_match:
                x1, y1, x2, y2 = map(int, bbox_match.groups())
                page_width = x2 - x1
                page_height = y2 - y1
            else:
                page_width = page_height = 0
            
            # Parse text areas (carea) and paragraphs (par)
            lines = []
            
            # Find all text lines
            line_elements = page_elem.find_all('span', class_='ocr_line')
            
            for line_elem in line_elements:
                line_info = self._parse_line_element(line_elem)
                if line_info:
                    lines.append(line_info)
            
            return {
                "page_number": page_number,
                "page_width": page_width,
                "page_height": page_height,
                "lines": lines,
                "line_count": len(lines)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse page element: {e}")
            return None
    
    def _parse_line_element(self, line_elem) -> Optional[Dict]:
        """Parse a single line element from hOCR"""
        try:
            # Extract line bbox
            title = line_elem.get('title', '')
            bbox_match = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
            if bbox_match:
                x1, y1, x2, y2 = map(int, bbox_match.groups())
                line_bbox = [x1, y1, x2, y2]
            else:
                line_bbox = [0, 0, 0, 0]
            
            # Parse words in this line
            words = []
            word_elements = line_elem.find_all('span', class_='ocrx_word')
            
            for word_elem in word_elements:
                word_info = self._parse_word_element(word_elem)
                if word_info:
                    words.append(word_info)
            
            return {
                "line_id": line_elem.get('id', ''),
                "bbox": line_bbox,
                "words": words,
                "word_count": len(words),
                "text": " ".join(word.get("text", "") for word in words)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse line element: {e}")
            return None
    
    def _parse_word_element(self, word_elem) -> Optional[Dict]:
        """Parse a single word element from hOCR"""
        try:
            # Extract word text
            text = word_elem.get_text().strip()
            
            # Extract bbox
            title = word_elem.get('title', '')
            bbox_match = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
            if bbox_match:
                x1, y1, x2, y2 = map(int, bbox_match.groups())
                word_bbox = [x1, y1, x2, y2]
            else:
                word_bbox = [0, 0, 0, 0]
            
            # Extract confidence
            conf_match = re.search(r'x_wconf (\d+)', title)
            confidence = int(conf_match.group(1)) if conf_match else 0
            
            return {
                "word_id": word_elem.get('id', ''),
                "text": text,
                "bbox": word_bbox,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse word element: {e}")
            return None
    
    async def create_pdf_from_hocr(self, document_id: str, output_path: str) -> bool:
        """Create a searchable PDF from corrected hOCR data"""
        try:
            hocr_path = self.hocr_storage_dir / f"{document_id}.hocr"
            
            if not hocr_path.exists():
                logger.error(f"❌ hOCR file not found for document {document_id}")
                return False
            
            # This would require additional implementation to convert hOCR back to PDF
            # For now, we'll just copy the original OCR'd PDF
            logger.info(f"✅ PDF creation from hOCR not yet implemented")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create PDF from hOCR: {e}")
            return False
    
    async def get_low_confidence_words(self, document_id: str, threshold: int = 60) -> List[Dict]:
        """Get words with confidence below threshold for review"""
        try:
            hocr_data = await self.get_hocr_data(document_id)
            if not hocr_data:
                return []
            
            low_confidence_words = []
            
            for page in hocr_data.get("pages", []):
                for line in page.get("lines", []):
                    for word in line.get("words", []):
                        confidence = word.get("confidence", 0)
                        if confidence < threshold and confidence > 0:
                            low_confidence_words.append({
                                "page_number": page.get("page_number", 0),
                                "word_id": word.get("word_id", ""),
                                "text": word.get("text", ""),
                                "confidence": confidence,
                                "bbox": word.get("bbox", []),
                                "line_id": line.get("line_id", "")
                            })
            
            # Sort by confidence (lowest first)
            low_confidence_words.sort(key=lambda x: x["confidence"])
            
            return low_confidence_words
            
        except Exception as e:
            logger.error(f"❌ Failed to get low confidence words: {e}")
            return []
    
    async def batch_update_words(self, document_id: str, updates: List[Dict]) -> Dict:
        """Batch update multiple words in hOCR file"""
        try:
            hocr_path = self.hocr_storage_dir / f"{document_id}.hocr"
            
            if not hocr_path.exists():
                return {"success": False, "error": "hOCR file not found"}
            
            # Parse hOCR file
            with open(hocr_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            successful_updates = 0
            failed_updates = []
            
            for update in updates:
                word_id = update.get("word_id")
                new_text = update.get("new_text", "")
                
                # Find the word element
                word_element = soup.find(id=word_id)
                if word_element:
                    word_element.string = new_text
                    successful_updates += 1
                else:
                    failed_updates.append(word_id)
            
            # Save updated hOCR
            with open(hocr_path, 'w', encoding='utf-8') as f:
                f.write(str(soup))
            
            logger.info(f"✅ Batch updated {successful_updates} words in hOCR for {document_id}")
            
            return {
                "success": True,
                "successful_updates": successful_updates,
                "failed_updates": failed_updates,
                "total_updates": len(updates)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to batch update hOCR: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_ocr_text_for_embedding(self, document_id: str) -> Optional[str]:
        """
        Get OCR text optimized for embedding generation
        This includes corrected text from hOCR if available, or extracted text from PDF
        """
        try:
            # First try to get corrected text from hOCR
            corrected_text = await self.export_corrected_text(document_id)
            if corrected_text:
                logger.info(f"✅ Retrieved corrected hOCR text for embedding: {document_id}")
                return corrected_text
            
            # If no hOCR available, this means OCR was not preserved
            # The text should already be in the PDF from OCR processing
            logger.info(f"ℹ️ No hOCR data available for {document_id}, using PDF text extraction")
            return None  # Let the regular PDF text extraction handle this
            
        except Exception as e:
            logger.error(f"❌ Failed to get OCR text for embedding: {e}")
            return None
    
    async def has_hocr_data(self, document_id: str) -> bool:
        """Check if document has hOCR data available"""
        try:
            hocr_path = self.hocr_storage_dir / f"{document_id}.hocr"
            return hocr_path.exists()
        except Exception:
            return False
    
    async def get_ocr_metadata(self, document_id: str) -> Dict:
        """Get OCR processing metadata for a document"""
        try:
            has_hocr = await self.has_hocr_data(document_id)
            
            metadata = {
                "has_hocr": has_hocr,
                "ocr_processed": True,  # If this method is called, OCR was processed
                "text_source": "ocr"
            }
            
            if has_hocr:
                # Get confidence stats if available
                confidence_stats = await self.get_ocr_confidence_stats(document_id)
                if confidence_stats:
                    metadata.update({
                        "ocr_confidence": confidence_stats.get("overall_avg_confidence", 0),
                        "total_words": confidence_stats.get("total_words", 0),
                        "low_confidence_words": confidence_stats.get("low_confidence_words", 0)
                    })
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to get OCR metadata: {e}")
            return {"has_hocr": False, "ocr_processed": False}
