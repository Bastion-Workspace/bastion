"""
Enhanced PDF Segmentation Service
Direct PDF manipulation without image dependency
"""

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import time
import json

import fitz  # PyMuPDF
import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from PIL import Image
import numpy as np

from models.segmentation_models import (
    PDFPageInfo, SegmentInfo, SegmentRelationship,
    CreateSegmentRequest, UpdateSegmentRequest, CreateRelationshipRequest,
    PDFExtractionRequest, PDFExtractionResponse,
    PageSegmentationResponse, DocumentSegmentationResponse,
    SegmentSearchRequest, SegmentSearchResponse,
    SegmentationStats, SegmentType, RelationshipType, ProcessingStatus,
    SegmentBounds, PDFTextExtractionRequest, PDFTextExtractionResponse,
    PDFSegmentCropRequest, PDFSegmentCropResponse,
    MultipleSelectionRequest, MultipleSelectionResponse,
    TextEditRequest, TextEditResponse, PDFRegionInfo
)
from repositories.document_repository import DocumentRepository
from services.embedding_service_wrapper import get_embedding_service

logger = logging.getLogger(__name__)


class EnhancedPDFSegmentationService:
    """Enhanced service for PDF segmentation working directly with PDF content"""
    
    def __init__(self, document_repository: DocumentRepository, embedding_manager):
        self.document_repository = document_repository
        self.embedding_manager = embedding_manager
        self.cropped_pdfs_path = Path("processed/cropped_pdfs")
        self.cropped_pdfs_path.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize the enhanced PDF segmentation service"""
        try:
            logger.info("🔄 Initializing Enhanced PDF Segmentation Service...")
            
            # Ensure storage directories exist
            self.cropped_pdfs_path.mkdir(parents=True, exist_ok=True)
            
            # Test database connection
            test_query = "SELECT COUNT(*) FROM pdf_pages LIMIT 1"
            await self.document_repository.execute_query(test_query)
            
            logger.info("✅ Enhanced PDF Segmentation Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced PDF Segmentation Service: {e}")
            raise
    
    async def extract_pdf_info(self, request: PDFExtractionRequest) -> PDFExtractionResponse:
        """Extract PDF information without creating images"""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Extracting PDF info: {request.document_id}")
            
            # Get document info
            doc_info = await self.document_repository.get_document_by_id(request.document_id)
            if not doc_info:
                raise ValueError(f"Document not found: {request.document_id}")
            
            # Get the PDF file path
            pdf_path = self._get_document_file_path(doc_info)
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            pages_extracted = []
            
            # Use PyMuPDF for basic page info
            with fitz.open(pdf_path) as pdf_document:
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    
                    # Get page dimensions in PDF points
                    rect = page.rect
                    page_width = int(rect.width)
                    page_height = int(rect.height)
                    
                    # Store page info in database (no image path)
                    page_info = await self._store_page_info(
                        document_id=request.document_id,
                        page_number=page_num + 1,
                        image_path="",  # No image needed
                        width=page_width,
                        height=page_height
                    )
                    
                    pages_extracted.append(page_info)
                    logger.info(f"📄 Processed page {page_num + 1}: {page_width}x{page_height} points")
            
            processing_time = time.time() - start_time
            logger.info(f"✅ PDF info extraction complete: {len(pages_extracted)} pages in {processing_time:.2f}s")
            
            return PDFExtractionResponse(
                document_id=request.document_id,
                pages_extracted=len(pages_extracted),
                pages=pages_extracted,
                extraction_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ PDF info extraction failed: {e}")
            raise
    
    async def extract_text_from_region(self, request: PDFTextExtractionRequest) -> PDFTextExtractionResponse:
        """Extract text from a specific PDF region"""
        try:
            logger.info(f"🔄 Extracting text from region: {request.document_id}, page {request.page_number}")
            
            # Get document info
            doc_info = await self.document_repository.get_document_by_id(request.document_id)
            if not doc_info:
                raise ValueError(f"Document not found: {request.document_id}")
            
            pdf_path = self._get_document_file_path(doc_info)
            
            # Try direct text extraction first
            extracted_text, confidence, method = await self._extract_text_direct(
                pdf_path, request.page_number, request.bounds
            )
            
            # Fallback to OCR if needed and requested
            if (not extracted_text.strip() or confidence < 0.5) and request.use_ocr_fallback:
                ocr_text, ocr_confidence = await self._extract_text_ocr(
                    pdf_path, request.page_number, request.bounds
                )
                if ocr_confidence > confidence:
                    extracted_text = ocr_text
                    confidence = ocr_confidence
                    method = "ocr" if method == "direct" else "hybrid"
            
            word_count = len(extracted_text.split()) if extracted_text else 0
            has_selectable_text = method in ["direct", "hybrid"]
            
            return PDFTextExtractionResponse(
                extracted_text=extracted_text,
                text_confidence=confidence,
                extraction_method=method,
                word_count=word_count,
                has_selectable_text=has_selectable_text
            )
            
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {e}")
            raise
    
    async def _extract_text_direct(self, pdf_path: str, page_number: int, bounds: SegmentBounds) -> Tuple[str, float, str]:
        """Extract text directly from PDF using pdfplumber and PyMuPDF"""
        try:
            # Try pdfplumber first (better for structured text)
            with pdfplumber.open(pdf_path) as pdf:
                if page_number <= len(pdf.pages):
                    page = pdf.pages[page_number - 1]
                    
                    # Convert bounds to pdfplumber format (x0, y0, x1, y1)
                    # Note: pdfplumber uses bottom-left origin, PyMuPDF uses top-left
                    page_height = page.height
                    crop_box = (
                        bounds.x,
                        page_height - bounds.y - bounds.height,  # Convert Y coordinate
                        bounds.x + bounds.width,
                        page_height - bounds.y
                    )
                    
                    # Crop and extract text
                    cropped_page = page.crop(crop_box)
                    text = cropped_page.extract_text()
                    
                    if text and text.strip():
                        return text.strip(), 0.9, "direct"
            
            # Fallback to PyMuPDF
            with fitz.open(pdf_path) as pdf_doc:
                if page_number <= len(pdf_doc):
                    page = pdf_doc[page_number - 1]
                    
                    # Create clip rectangle
                    clip_rect = fitz.Rect(bounds.x, bounds.y, bounds.x + bounds.width, bounds.y + bounds.height)
                    
                    # Extract text from clipped region
                    text = page.get_text(clip=clip_rect)
                    
                    if text and text.strip():
                        return text.strip(), 0.8, "direct"
            
            return "", 0.0, "direct"
            
        except Exception as e:
            logger.warning(f"⚠️ Direct text extraction failed: {e}")
            return "", 0.0, "direct"
    
    async def _extract_text_ocr(self, pdf_path: str, page_number: int, bounds: SegmentBounds) -> Tuple[str, float]:
        """Extract text using OCR as fallback"""
        try:
            # Create a temporary image of the region
            with fitz.open(pdf_path) as pdf_doc:
                if page_number <= len(pdf_doc):
                    page = pdf_doc[page_number - 1]
                    
                    # Create clip rectangle
                    clip_rect = fitz.Rect(bounds.x, bounds.y, bounds.x + bounds.width, bounds.y + bounds.height)
                    
                    # Render region to image
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    
                    # Here you would use your preferred OCR library (tesseract, etc.)
                    # For now, return empty as OCR implementation depends on your setup
                    return "", 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ OCR text extraction failed: {e}")
            return "", 0.0
    
    async def create_segment_with_text_extraction(self, request: CreateSegmentRequest) -> SegmentInfo:
        """Create a segment and automatically extract text"""
        try:
            logger.info(f"🔄 Creating segment with text extraction: {request.segment_type}")
            
            # Get page info to get document_id
            page_query = "SELECT document_id FROM pdf_pages WHERE id = $1"
            page_result = await self.document_repository.execute_query(page_query, request.page_id)
            
            if not page_result:
                raise ValueError(f"Page not found: {request.page_id}")
            
            document_id = page_result[0]['document_id']
            
            # Get page number
            page_num_query = "SELECT page_number FROM pdf_pages WHERE id = $1"
            page_num_result = await self.document_repository.execute_query(page_num_query, request.page_id)
            page_number = page_num_result[0]['page_number']
            
            # Extract text if not provided
            extracted_text = request.manual_text or ""
            if not extracted_text:
                text_request = PDFTextExtractionRequest(
                    document_id=document_id,
                    page_number=page_number,
                    bounds=request.bounds,
                    use_ocr_fallback=True
                )
                text_response = await self.extract_text_from_region(text_request)
                extracted_text = text_response.extracted_text
            
            # Generate unique segment ID
            segment_id = f"seg_{uuid.uuid4().hex[:12]}"
            
            # Store segment in database
            query = """
                INSERT INTO pdf_segments 
                (segment_id, page_id, segment_type, x_coordinate, y_coordinate, width, height, 
                 manual_text, tags, metadata_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id, created_at, updated_at
            """
            
            result = await self.document_repository.execute_query(
                query,
                segment_id,
                request.page_id,
                request.segment_type.value,
                request.bounds.x,
                request.bounds.y,
                request.bounds.width,
                request.bounds.height,
                extracted_text,
                request.tags or [],
                request.metadata or {}
            )
            
            if result:
                row = result[0]
                
                # Create embedding if text exists
                if extracted_text and extracted_text.strip():
                    await self._create_segment_embedding(segment_id, extracted_text)
                
                logger.info(f"✅ Created segment with text: {segment_id}")
                
                return SegmentInfo(
                    id=row['id'],
                    segment_id=segment_id,
                    page_id=request.page_id,
                    segment_type=request.segment_type,
                    bounds=request.bounds,
                    manual_text=extracted_text,
                    confidence_score=1.0,
                    tags=request.tags or [],
                    metadata=request.metadata or {},
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            else:
                raise Exception("Failed to create segment")
                
        except Exception as e:
            logger.error(f"❌ Failed to create segment with text extraction: {e}")
            raise
    
    async def create_multiple_selections(self, request: MultipleSelectionRequest) -> MultipleSelectionResponse:
        """Create multiple segments from selections"""
        try:
            logger.info(f"🔄 Creating multiple selections: {len(request.selections)} selections")
            
            created_segments = []
            failed_selections = []
            
            for i, selection in enumerate(request.selections):
                try:
                    # Create segment request from selection
                    segment_request = CreateSegmentRequest(
                        page_id=request.page_id,
                        segment_type=SegmentType(selection.get('segment_type', 'article')),
                        bounds=SegmentBounds(**selection['bounds']),
                        manual_text=selection.get('manual_text'),
                        tags=selection.get('tags', []),
                        metadata=selection.get('metadata', {})
                    )
                    
                    # Create segment with automatic text extraction if enabled
                    if request.auto_extract_text:
                        segment = await self.create_segment_with_text_extraction(segment_request)
                    else:
                        segment = await self.create_segment(segment_request)
                    
                    created_segments.append(segment)
                    
                except Exception as e:
                    failed_selections.append({
                        'selection_index': i,
                        'selection_data': selection,
                        'error': str(e)
                    })
                    logger.warning(f"⚠️ Failed to create selection {i}: {e}")
            
            logger.info(f"✅ Created {len(created_segments)} segments, {len(failed_selections)} failed")
            
            return MultipleSelectionResponse(
                created_segments=created_segments,
                failed_selections=failed_selections,
                total_created=len(created_segments)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create multiple selections: {e}")
            raise
    
    async def crop_segment_to_pdf(self, request: PDFSegmentCropRequest) -> PDFSegmentCropResponse:
        """Crop a segment to a new PDF document"""
        try:
            logger.info(f"🔄 Cropping segment to PDF: {request.segment_id}")
            
            # Get segment info
            segment_query = """
                SELECT ps.*, pp.document_id, pp.page_number
                FROM pdf_segments ps
                JOIN pdf_pages pp ON ps.page_id = pp.id
                WHERE ps.segment_id = $1
            """
            segment_result = await self.document_repository.execute_query(segment_query, request.segment_id)
            
            if not segment_result:
                raise ValueError(f"Segment not found: {request.segment_id}")
            
            segment_row = segment_result[0]
            
            # Get document info
            doc_info = await self.document_repository.get_document_by_id(segment_row['document_id'])
            pdf_path = self._get_document_file_path(doc_info)
            
            # Create output filename
            output_filename = request.output_filename or f"segment_{request.segment_id}.pdf"
            output_path = self.cropped_pdfs_path / output_filename
            
            # Crop PDF using pypdf
            with open(pdf_path, 'rb') as input_file:
                reader = PdfReader(input_file)
                writer = PdfWriter()
                
                # Get the specific page
                page = reader.pages[segment_row['page_number'] - 1]
                
                # Set crop box (pypdf uses bottom-left origin)
                page_height = float(page.mediabox.height)
                crop_box = [
                    segment_row['x_coordinate'],  # left
                    page_height - segment_row['y_coordinate'] - segment_row['height'],  # bottom
                    segment_row['x_coordinate'] + segment_row['width'],  # right
                    page_height - segment_row['y_coordinate']  # top
                ]
                
                page.cropbox.lower_left = (crop_box[0], crop_box[1])
                page.cropbox.upper_right = (crop_box[2], crop_box[3])
                
                writer.add_page(page)
                
                # Add metadata if requested
                if request.include_metadata:
                    metadata = {
                        '/Title': f'Segment {request.segment_id}',
                        '/Subject': f'Cropped from {doc_info.get("filename", "document")}',
                        '/Creator': 'Enhanced PDF Segmentation Service',
                        '/Producer': 'Enhanced PDF Segmentation Service',
                        '/SegmentType': segment_row['segment_type'],
                        '/OriginalDocument': segment_row['document_id']
                    }
                    writer.add_metadata(metadata)
                
                # Write to file
                with open(output_path, 'wb') as output_file:
                    writer.write(output_file)
            
            # Get file size
            file_size = output_path.stat().st_size
            
            # Get page dimensions
            page_dimensions = (segment_row['width'], segment_row['height'])
            
            logger.info(f"✅ Cropped segment to PDF: {output_path}")
            
            return PDFSegmentCropResponse(
                cropped_pdf_path=str(output_path),
                file_size=file_size,
                page_dimensions=page_dimensions
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to crop segment to PDF: {e}")
            raise
    
    async def edit_segment_text(self, request: TextEditRequest) -> TextEditResponse:
        """Edit the text content of a segment"""
        try:
            logger.info(f"🔄 Editing segment text: {request.segment_id}")
            
            # Get current segment
            segment_query = "SELECT * FROM pdf_segments WHERE segment_id = $1"
            segment_result = await self.document_repository.execute_query(segment_query, request.segment_id)
            
            if not segment_result:
                raise ValueError(f"Segment not found: {request.segment_id}")
            
            segment_row = segment_result[0]
            old_text = segment_row['manual_text'] or ""
            
            # Calculate changes
            text_changes = {
                'old_text': old_text,
                'new_text': request.edited_text,
                'character_diff': len(request.edited_text) - len(old_text),
                'word_diff': len(request.edited_text.split()) - len(old_text.split()),
                'edit_type': 'manual_edit'
            }
            
            # Update segment
            update_query = """
                UPDATE pdf_segments 
                SET manual_text = $1, updated_at = CURRENT_TIMESTAMP
                WHERE segment_id = $2
                RETURNING *
            """
            
            result = await self.document_repository.execute_query(
                update_query, request.edited_text, request.segment_id
            )
            
            if result:
                row = result[0]
                
                # Update embedding
                if request.edited_text and request.edited_text.strip():
                    await self._create_segment_embedding(request.segment_id, request.edited_text)
                else:
                    await self._delete_segment_embedding(request.segment_id)
                
                updated_segment = SegmentInfo(
                    id=row['id'],
                    segment_id=row['segment_id'],
                    page_id=row['page_id'],
                    segment_type=SegmentType(row['segment_type']),
                    bounds=SegmentBounds(
                        x=row['x_coordinate'],
                        y=row['y_coordinate'],
                        width=row['width'],
                        height=row['height']
                    ),
                    manual_text=row['manual_text'],
                    confidence_score=row['confidence_score'],
                    tags=row['tags'] or [],
                    metadata=row['metadata_json'] or {},
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                
                logger.info(f"✅ Updated segment text: {request.segment_id}")
                
                return TextEditResponse(
                    updated_segment=updated_segment,
                    text_changes=text_changes
                )
            else:
                raise Exception("Failed to update segment")
                
        except Exception as e:
            logger.error(f"❌ Failed to edit segment text: {e}")
            raise
    
    async def get_pdf_region_info(self, document_id: str, page_number: int, bounds: SegmentBounds) -> PDFRegionInfo:
        """Get detailed information about a PDF region"""
        try:
            logger.info(f"🔄 Getting PDF region info: {document_id}, page {page_number}")
            
            # Get document info
            doc_info = await self.document_repository.get_document_by_id(document_id)
            if not doc_info:
                raise ValueError(f"Document not found: {document_id}")
            
            pdf_path = self._get_document_file_path(doc_info)
            
            # Extract text
            text_request = PDFTextExtractionRequest(
                document_id=document_id,
                page_number=page_number,
                bounds=bounds,
                use_ocr_fallback=True
            )
            text_response = await self.extract_text_from_region(text_request)
            
            # Analyze content with pdfplumber
            has_images = False
            has_tables = False
            font_info = []
            text_blocks = []
            
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    if page_number <= len(pdf.pages):
                        page = pdf.pages[page_number - 1]
                        
                        # Convert bounds
                        page_height = page.height
                        crop_box = (
                            bounds.x,
                            page_height - bounds.y - bounds.height,
                            bounds.x + bounds.width,
                            page_height - bounds.y
                        )
                        
                        cropped_page = page.crop(crop_box)
                        
                        # Check for images
                        images = cropped_page.images
                        has_images = len(images) > 0
                        
                        # Check for tables
                        tables = cropped_page.find_tables()
                        has_tables = len(tables) > 0
                        
                        # Get font information
                        chars = cropped_page.chars
                        fonts = {}
                        for char in chars:
                            font_name = char.get('fontname', 'Unknown')
                            font_size = char.get('size', 0)
                            font_key = f"{font_name}_{font_size}"
                            if font_key not in fonts:
                                fonts[font_key] = {
                                    'name': font_name,
                                    'size': font_size,
                                    'count': 0
                                }
                            fonts[font_key]['count'] += 1
                        
                        font_info = list(fonts.values())
                        
                        # Get text blocks (simplified)
                        if text_response.extracted_text:
                            text_blocks = [{'text': text_response.extracted_text, 'type': 'paragraph'}]
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to analyze PDF region details: {e}")
            
            return PDFRegionInfo(
                bounds=bounds,
                text_content=text_response.extracted_text,
                has_images=has_images,
                has_tables=has_tables,
                font_info=font_info,
                text_blocks=text_blocks
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get PDF region info: {e}")
            raise
    
    # Legacy methods for compatibility (delegate to new methods)
    async def create_segment(self, request: CreateSegmentRequest) -> SegmentInfo:
        """Create a segment (legacy compatibility)"""
        return await self.create_segment_with_text_extraction(request)
    
    async def update_segment(self, segment_id: str, request: UpdateSegmentRequest) -> SegmentInfo:
        """Update an existing segment"""
        try:
            logger.info(f"🔄 Updating segment: {segment_id}")
            
            # Build dynamic update query
            update_fields = []
            params = []
            param_count = 1
            
            if request.segment_type is not None:
                update_fields.append(f"segment_type = ${param_count}")
                params.append(request.segment_type.value)
                param_count += 1
            
            if request.bounds is not None:
                update_fields.extend([
                    f"x_coordinate = ${param_count}",
                    f"y_coordinate = ${param_count + 1}",
                    f"width = ${param_count + 2}",
                    f"height = ${param_count + 3}"
                ])
                params.extend([request.bounds.x, request.bounds.y, request.bounds.width, request.bounds.height])
                param_count += 4
            
            if request.manual_text is not None:
                update_fields.append(f"manual_text = ${param_count}")
                params.append(request.manual_text)
                param_count += 1
            
            if request.tags is not None:
                update_fields.append(f"tags = ${param_count}")
                params.append(request.tags)
                param_count += 1
            
            if request.metadata is not None:
                update_fields.append(f"metadata_json = ${param_count}")
                params.append(request.metadata)
                param_count += 1
            
            if not update_fields:
                raise ValueError("No fields to update")
            
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            params.append(segment_id)
            
            query = f"""
                UPDATE pdf_segments 
                SET {', '.join(update_fields)}
                WHERE segment_id = ${param_count}
                RETURNING *
            """
            
            result = await self.document_repository.execute_query(query, *params)
            
            if result:
                row = result[0]
                
                # Update embedding if text changed
                if request.manual_text is not None and request.manual_text.strip():
                    await self._create_segment_embedding(segment_id, request.manual_text)
                
                logger.info(f"✅ Updated segment: {segment_id}")
                
                return SegmentInfo(
                    id=row['id'],
                    segment_id=row['segment_id'],
                    page_id=row['page_id'],
                    segment_type=SegmentType(row['segment_type']),
                    bounds=SegmentBounds(
                        x=row['x_coordinate'],
                        y=row['y_coordinate'],
                        width=row['width'],
                        height=row['height']
                    ),
                    manual_text=row['manual_text'],
                    confidence_score=row['confidence_score'],
                    tags=row['tags'] or [],
                    metadata=row['metadata_json'] or {},
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            else:
                raise Exception(f"Segment not found: {segment_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to update segment: {e}")
            raise
    
    async def delete_segment(self, segment_id: str) -> bool:
        """Delete a segment"""
        try:
            logger.info(f"🔄 Deleting segment: {segment_id}")
            
            # Delete from database (relationships will cascade)
            query = "DELETE FROM pdf_segments WHERE segment_id = $1"
            await self.document_repository.execute_query(query, segment_id)
            
            # Delete embedding if exists
            await self._delete_segment_embedding(segment_id)
            
            logger.info(f"✅ Deleted segment: {segment_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete segment: {e}")
            return False
    
    # Helper methods
    async def _store_page_info(self, document_id: str, page_number: int, 
                              image_path: str, width: int, height: int) -> PDFPageInfo:
        """Store page information in database"""
        try:
            query = """
                INSERT INTO pdf_pages (document_id, page_number, page_image_path, page_width, page_height, processing_status)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (document_id, page_number) 
                DO UPDATE SET 
                    page_image_path = EXCLUDED.page_image_path,
                    page_width = EXCLUDED.page_width,
                    page_height = EXCLUDED.page_height,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id, created_at, updated_at
            """
            
            result = await self.document_repository.execute_query(
                query, document_id, page_number, image_path, width, height, ProcessingStatus.UPLOADING.value
            )
            
            if result:
                row = result[0]
                return PDFPageInfo(
                    id=row['id'],
                    document_id=document_id,
                    page_number=page_number,
                    page_image_path=image_path,
                    page_width=width,
                    page_height=height,
                    processing_status=ProcessingStatus.UPLOADING,
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            else:
                raise Exception("Failed to store page info")
                
        except Exception as e:
            logger.error(f"❌ Failed to store page info: {e}")
            raise
    
    def _get_document_file_path(self, doc_info: Dict[str, Any]) -> str:
        """Get the file path for a document"""
        uploads_dir = Path("uploads")
        return str(uploads_dir / f"{doc_info['document_id']}_{doc_info['filename']}")
    
    async def _create_segment_embedding(self, segment_id: str, text: str):
        """Create embedding for segment text"""
        try:
            if self.embedding_manager:
                chunk_data = {
                    'chunk_id': segment_id,
                    'content': text,
                    'metadata': {'source_type': 'pdf_segment'}
                }
                await self.embedding_manager.create_embedding(chunk_data)
                logger.debug(f"📊 Created embedding for segment: {segment_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create embedding for segment {segment_id}: {e}")
    
    async def _delete_segment_embedding(self, segment_id: str):
        """Delete embedding for segment"""
        try:
            if self.embedding_manager:
                await self.embedding_manager.delete_embedding(segment_id)
                logger.debug(f"🗑️ Deleted embedding for segment: {segment_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete embedding for segment {segment_id}: {e}")
    
    # Additional methods for compatibility with existing API
    async def get_page_segmentation(self, page_id: int) -> PageSegmentationResponse:
        """Get all segments for a specific page"""
        try:
            # Get page info
            page_query = "SELECT * FROM pdf_pages WHERE id = $1"
            page_result = await self.document_repository.execute_query(page_query, page_id)
            
            if not page_result:
                raise ValueError(f"Page not found: {page_id}")
            
            page_row = page_result[0]
            page_info = PDFPageInfo(
                id=page_row['id'],
                document_id=page_row['document_id'],
                page_number=page_row['page_number'],
                page_image_path=page_row['page_image_path'],
                page_width=page_row['page_width'],
                page_height=page_row['page_height'],
                processing_status=ProcessingStatus(page_row['processing_status']),
                created_at=page_row['created_at'],
                updated_at=page_row['updated_at']
            )
            
            # Get segments
            segments_query = "SELECT * FROM pdf_segments WHERE page_id = $1 ORDER BY created_at"
            segments_result = await self.document_repository.execute_query(segments_query, page_id)
            
            segments = []
            for row in segments_result:
                segments.append(SegmentInfo(
                    id=row['id'],
                    segment_id=row['segment_id'],
                    page_id=row['page_id'],
                    segment_type=SegmentType(row['segment_type']),
                    bounds=SegmentBounds(
                        x=row['x_coordinate'],
                        y=row['y_coordinate'],
                        width=row['width'],
                        height=row['height']
                    ),
                    manual_text=row['manual_text'],
                    confidence_score=row['confidence_score'],
                    tags=row['tags'] or [],
                    metadata=row['metadata_json'] or {},
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                ))
            
            # Get relationships
            relationships_query = """
                SELECT sr.* FROM segment_relationships sr
                JOIN pdf_segments ps1 ON sr.parent_segment_id = ps1.segment_id
                JOIN pdf_segments ps2 ON sr.child_segment_id = ps2.segment_id
                WHERE ps1.page_id = $1 OR ps2.page_id = $1
            """
            relationships_result = await self.document_repository.execute_query(relationships_query, page_id)
            
            relationships = []
            for row in relationships_result:
                relationships.append(SegmentRelationship(
                    id=row['id'],
                    parent_segment_id=row['parent_segment_id'],
                    child_segment_id=row['child_segment_id'],
                    relationship_type=RelationshipType(row['relationship_type']),
                    created_at=row['created_at']
                ))
            
            return PageSegmentationResponse(
                page=page_info,
                segments=segments,
                relationships=relationships
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get page segmentation: {e}")
            raise
    
    async def get_document_segmentation(self, document_id: str) -> DocumentSegmentationResponse:
        """Get all pages and segments for a document"""
        try:
            # Get all pages for document
            pages_query = "SELECT id FROM pdf_pages WHERE document_id = $1 ORDER BY page_number"
            pages_result = await self.document_repository.execute_query(pages_query, document_id)
            
            pages = []
            total_segments = 0
            segment_types = {}
            
            for page_row in pages_result:
                page_segmentation = await self.get_page_segmentation(page_row['id'])
                pages.append(page_segmentation)
                
                total_segments += len(page_segmentation.segments)
                
                # Count segment types
                for segment in page_segmentation.segments:
                    segment_type = segment.segment_type.value
                    segment_types[segment_type] = segment_types.get(segment_type, 0) + 1
            
            return DocumentSegmentationResponse(
                document_id=document_id,
                pages=pages,
                total_segments=total_segments,
                segment_types=segment_types
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get document segmentation: {e}")
            raise
    
    async def get_segmentation_stats(self) -> SegmentationStats:
        """Get statistics about segmentation progress"""
        try:
            # Count documents with segmentation
            docs_query = """
                SELECT COUNT(DISTINCT pp.document_id) as total_documents
                FROM pdf_pages pp
            """
            docs_result = await self.document_repository.execute_query(docs_query)
            total_documents = docs_result[0]['total_documents'] if docs_result else 0
            
            # Count pages
            pages_query = "SELECT COUNT(*) as total_pages FROM pdf_pages"
            pages_result = await self.document_repository.execute_query(pages_query)
            total_pages = pages_result[0]['total_pages'] if pages_result else 0
            
            # Count segments
            segments_query = "SELECT COUNT(*) as total_segments FROM pdf_segments"
            segments_result = await self.document_repository.execute_query(segments_query)
            total_segments = segments_result[0]['total_segments'] if segments_result else 0
            
            # Count by segment type
            types_query = """
                SELECT segment_type, COUNT(*) as count
                FROM pdf_segments
                GROUP BY segment_type
            """
            types_result = await self.document_repository.execute_query(types_query)
            segments_by_type = {row['segment_type']: row['count'] for row in types_result}
            
            # Count pages by status
            status_query = """
                SELECT processing_status, COUNT(*) as count
                FROM pdf_pages
                GROUP BY processing_status
            """
            status_result = await self.document_repository.execute_query(status_query)
            pages_by_status = {row['processing_status']: row['count'] for row in status_result}
            
            # Calculate average segments per page
            avg_segments_per_page = total_segments / max(1, total_pages)
            
            return SegmentationStats(
                total_documents=total_documents,
                total_pages=total_pages,
                total_segments=total_segments,
                segments_by_type=segments_by_type,
                pages_by_status=pages_by_status,
                avg_segments_per_page=avg_segments_per_page
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get segmentation stats: {e}")
            raise


