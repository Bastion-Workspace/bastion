"""
OCR Service - Lean Version
Simplified to handle only essential OCR text extraction for vectorization.
Removed coordinate-based tracking and hOCR editing.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import ocrmypdf
import pdfplumber

from config import settings

logger = logging.getLogger(__name__)


class OCRService:
    """Simplified service for OCR text extraction fallback"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize OCR service"""
        if self.initialized:
            return
        
        try:
            # Test OCRmyPDF availability
            import ocrmypdf
            logger.info("✅ OCR service initialized with OCRmyPDF (fallback mode)")
            self.initialized = True
            
        except ImportError as e:
            logger.error(f"❌ OCRmyPDF not available: {e}")
            raise
    
    async def process_pdf_with_ocr(
        self, 
        input_pdf_path: str, 
        document_id: str,
        force_ocr: bool = False
    ) -> Dict:
        """
        Process PDF with OCR to extract text for vectorization
        """
        try:
            logger.info(f"🔄 Starting OCR fallback processing for document {document_id}")
            
            # Create output paths
            output_pdf_path = f"{input_pdf_path}_ocr.pdf"
            
            # Simplified OCR configuration
            ocr_config = {
                'language': '+'.join(settings.OCR_LANGUAGES),
                'force_ocr': force_ocr,
                'output_type': 'pdf',
                'optimize': 1,
                'skip_text': not force_ocr  # Only OCR if no text exists
            }
            
            try:
                # Run OCR
                ocrmypdf.ocr(
                    input_pdf_path,
                    output_pdf_path,
                    **ocr_config
                )
                
                # Replace original with OCR'd version
                os.replace(output_pdf_path, input_pdf_path)
                
                # Extract text for vectorization
                text = ""
                with pdfplumber.open(input_pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                logger.info(f"✅ OCR text extraction completed for document {document_id}")
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "text": text
                }
                
            except ocrmypdf.exceptions.PriorOcrFoundError:
                logger.info(f"ℹ️ PDF already contains text, skipping OCR for {document_id}")
                return {
                    "status": "skipped",
                    "document_id": document_id
                }
                
        except Exception as e:
            logger.error(f"❌ OCR processing failed for document {document_id}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "document_id": document_id
            }
    
    async def get_ocr_text_for_embedding(self, document_id: str) -> Optional[str]:
        """Keep method signature for compatibility but returning None as we now process on-the-fly"""
        return None
    
    async def get_ocr_metadata(self, document_id: str) -> Dict:
        """Keep method signature for compatibility"""
        return {"ocr_confidence": 1.0}
