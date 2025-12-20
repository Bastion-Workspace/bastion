"""
Document Processor - Handles document text extraction and chunking
"""

import asyncio
import email
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Dict, Any
import time
import re

import PyPDF2
import pdfplumber
from docx import Document as DocxDocument
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import ocrmypdf
from PIL import Image
import spacy
from langdetect import detect
import textstat
from openai import AsyncOpenAI

from config import settings
from models.api_models import Chunk, QualityMetrics, ProcessingResult, Entity
from services.ocr_service import OCRService

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents and extracts text with quality assessment"""
    
    _instance = None
    _initialized = False
    _initialization_lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize instance variables only once
            cls._instance.nlp = None
            cls._instance.ocr_service = None
            # Note: _initialized is managed at class level, not instance level
        return cls._instance
    
    def __init__(self):
        # __init__ is called every time, but we only want to initialize once
        # The actual initialization happens in the initialize() method
        pass
    
    async def initialize(self):
        """Initialize NLP models and OCR service (singleton pattern)"""
        async with self._initialization_lock:
            if self._initialized:
                logger.debug("🔄 DocumentProcessor already initialized, skipping")
                return
                
            try:
                logger.info("🔧 Initializing DocumentProcessor singleton...")
                
                # Try to load spaCy model, but don't fail if not available
                try:
                    import spacy
                    import spacy.util
                    
                    model_name = "en_core_web_lg"
                    logger.info(f"🔍 Loading spaCy model: {model_name}")
                    
                    if spacy.util.is_package(model_name):
                        self.nlp = spacy.load(model_name)
                        logger.info(f"✅ spaCy model loaded successfully: {model_name}")
                    else:
                        logger.warning(f"⚠️  spaCy model {model_name} not installed")
                        logger.warning("🔍 Install with: python -m spacy download en_core_web_lg")
                        self.nlp = None
                        
                except ImportError as e:
                    logger.warning(f"⚠️  spaCy import failed (ImportError): {e}")
                    self.nlp = None
                except Exception as e:
                    logger.warning(f"⚠️  spaCy loading failed (unexpected error): {e}")
                    self.nlp = None
                
                # Initialize OCR service
                try:
                    self.ocr_service = OCRService()
                    await self.ocr_service.initialize()
                    logger.info("✅ OCR service initialized in document processor")
                except Exception as e:
                    logger.warning(f"⚠️  OCR service initialization failed: {e}")
                    self.ocr_service = None
                
                self._initialized = True
                logger.info("✅ DocumentProcessor singleton initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ DocumentProcessor initialization failed: {e}")
                self._initialized = True  # Continue without advanced features
    
    @property
    def initialized(self) -> bool:
        """Check if the DocumentProcessor is initialized"""
        return self._initialized
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance"""
        return cls()
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (mainly for testing)"""
        cls._instance = None
        cls._initialized = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the DocumentProcessor"""
        return {
            "initialized": self._initialized,
            "spacy_loaded": self.nlp is not None,
            "ocr_service_loaded": self.ocr_service is not None,
            "instance_id": id(self),
            "spacy_model": "en_core_web_lg" if self.nlp else None
        }
    
    async def process_document(self, file_path: str, doc_type: str, document_id: str = None) -> ProcessingResult:
        """Process a document and return chunks with quality metrics
        
        Args:
            file_path: Path to the document file
            doc_type: Type of document (pdf, docx, etc.)
            document_id: UUID of the document (if None, derived from filename for backward compatibility)
        """
        start_time = time.time()
        
        # **ROOSEVELT FIX**: Use provided document_id instead of deriving from filename
        if document_id is None:
            document_id = Path(file_path).stem.split('_')[0]
            logger.warning(f"⚠️ No document_id provided, deriving from filename: {document_id}")
        
        try:
            logger.info(f"🔄 Processing {doc_type} document: {file_path} (doc_id: {document_id})")
            
            # ROOSEVELT DOCTRINE: Org Mode files use structured mechanisms, not vectorization!
            if doc_type == 'org':
                logger.info(f"⏭️  BULLY! Skipping vectorization for Org Mode file (structured data, not prose)")
                logger.info(f"📋 Org file will be handled by llm-orchestrator (OrgInboxAgent and OrgProjectAgent migrated)")
                
                # Return empty processing result - file is stored but not vectorized
                result = ProcessingResult(
                    document_id=document_id,
                    chunks=[],  # No chunks - structural queries only
                    entities=[],  # No entities - org-specific metadata used instead
                    quality_metrics=QualityMetrics(
                        overall_score=1.0,
                        ocr_confidence=1.0,
                        language_confidence=1.0,  # Default for structured data
                        vocabulary_score=1.0,  # Default for structured data
                        pattern_score=1.0  # Default for structured data
                    ),
                    processing_time=time.time() - start_time
                )
                logger.info(f"✅ Org file registered for structured access (no vectorization)")
                return result
            
            # ROOSEVELT FIX: Images are stored but NOT vectorized!
            if doc_type == 'image':
                logger.info(f"⏭️  BULLY! Skipping vectorization for image file (binary data, not text)")
                logger.info(f"📷 Image stored for reference but not embedded")
                
                # Return empty processing result - file is stored but not vectorized
                result = ProcessingResult(
                    document_id=document_id,
                    chunks=[],  # No chunks - images not vectorized
                    entities=[],  # No entities
                    quality_metrics=QualityMetrics(
                        overall_score=1.0,
                        ocr_confidence=1.0,
                        language_confidence=1.0,
                        vocabulary_score=1.0,
                        pattern_score=1.0
                    ),
                    processing_time=time.time() - start_time
                )
                logger.info(f"✅ Image file registered for storage (no vectorization)")
                return result
            
            # Extract text based on document type
            if doc_type == 'pdf':
                text, ocr_confidence = await self._process_pdf(file_path, document_id)
            elif doc_type == 'docx':
                text, ocr_confidence = await self._process_docx(file_path), 1.0
            elif doc_type == 'epub':
                text, ocr_confidence = await self._process_epub(file_path), 1.0
            elif doc_type == 'txt':
                text, ocr_confidence = await self._process_text(file_path), 1.0
            elif doc_type == 'md':
                text, ocr_confidence = await self._process_text(file_path), 1.0
            elif doc_type == 'html':
                text, ocr_confidence = await self._process_html(file_path), 1.0
            elif doc_type == 'eml':
                text, ocr_confidence = await self._process_eml(file_path), 1.0
            elif doc_type == 'zip':
                text, ocr_confidence = await self._process_zip(file_path), 1.0
            elif doc_type == 'srt':
                text, ocr_confidence = await self._process_srt(file_path), 1.0
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")
            
            # Assess text quality
            quality_metrics = await self._assess_quality(text, ocr_confidence)
            
            # Chunk the text
            chunks = await self._chunk_text(text, file_path, document_id)
            
            # Extract entities from the text
            entities = await self._extract_entities(text, chunks)
            
            # Create processing result
            result = ProcessingResult(
                document_id=document_id,  # **ROOSEVELT FIX**: Use provided UUID, not filename
                chunks=chunks,
                entities=entities,
                quality_metrics=quality_metrics,
                processing_time=time.time() - start_time
            )
            
            logger.info(f"✅ Document processed: {len(chunks)} chunks, quality: {quality_metrics.overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            raise
    
    async def _process_pdf(self, file_path: str, document_id: str) -> tuple[str, float]:
        """Process PDF document with OCR integration
        
        Args:
            file_path: Path to the PDF file
            document_id: UUID of the document for OCR lookup
        """
        text = ""
        ocr_confidence = 1.0
        
        try:
            # First check if we have OCR text available (from previous OCR processing)
            if self.ocr_service:
                ocr_text = await self.ocr_service.get_ocr_text_for_embedding(document_id)
                if ocr_text:
                    logger.info(f"✅ Using OCR text for document {document_id}")
                    # Get OCR metadata for confidence
                    ocr_metadata = await self.ocr_service.get_ocr_metadata(document_id)
                    ocr_confidence = ocr_metadata.get("ocr_confidence", 0.8)
                    return ocr_text, ocr_confidence
            
            # If no OCR text available, try standard text extraction
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + "\n"
            
            # If no text extracted, try pdfplumber
            if not text.strip():
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            
            # If still no text, try OCR as fallback
            if not text.strip():
                text, ocr_confidence = await self._ocr_pdf(file_path)
            
        except Exception as e:
            logger.warning(f"⚠️  PDF processing failed, trying OCR: {e}")
            text, ocr_confidence = await self._ocr_pdf(file_path)
        
        return text, ocr_confidence
    
    async def _ocr_pdf(self, file_path: str) -> tuple[str, float]:
        """Perform OCR on PDF"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run OCR
            ocrmypdf.ocr(
                file_path, 
                temp_path,
                language='+'.join(settings.OCR_LANGUAGES),
                force_ocr=True,
                skip_text=True
            )
            
            # Extract text from OCR'd PDF
            text = ""
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # Clean up
            os.unlink(temp_path)
            
            # Estimate OCR confidence (simplified)
            confidence = min(1.0, len([c for c in text if c.isalnum()]) / max(1, len(text)) * 2)
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"❌ OCR failed: {e}")
            return "", 0.0
    
    async def _process_docx(self, file_path: str) -> str:
        """Process DOCX document"""
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"❌ DOCX processing failed: {e}")
            return ""
    
    async def _process_epub(self, file_path: str) -> str:
        """Process EPUB document"""
        try:
            book = epub.read_epub(file_path)
            text = ""
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                    text += soup.get_text() + "\n"
            
            return text
        except Exception as e:
            logger.error(f"❌ EPUB processing failed: {e}")
            return ""
    
    async def _process_text(self, file_path: str) -> str:
        """Process plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            logger.error(f"❌ Could not decode text file: {file_path}")
            return ""
        except Exception as e:
            logger.error(f"❌ Text processing failed: {e}")
            return ""
    
    async def _process_html(self, file_path: str) -> str:
        """Process HTML document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"❌ HTML processing failed: {e}")
            return ""
    
    async def _process_eml(self, file_path: str) -> str:
        """Process EML email file with Unicode sanitization"""
        try:
            with open(file_path, 'rb') as file:
                msg = email.message_from_bytes(file.read())
            
            # Extract email metadata
            headers = []
            for header in ['From', 'To', 'Cc', 'Subject', 'Date']:
                value = msg.get(header)
                if value:
                    # Sanitize header value
                    clean_value = self._sanitize_unicode(str(value))
                    headers.append(f"{header}: {clean_value}")
            
            # Extract email body
            body_parts = []
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == 'text/plain':
                        try:
                            decoded_content = part.get_payload(decode=True).decode('utf-8', errors='replace')
                            clean_content = self._sanitize_unicode(decoded_content)
                            body_parts.append(clean_content)
                        except:
                            try:
                                decoded_content = part.get_payload(decode=True).decode('latin-1', errors='replace')
                                clean_content = self._sanitize_unicode(decoded_content)
                                body_parts.append(clean_content)
                            except:
                                continue
                    elif content_type == 'text/html':
                        try:
                            html_content = part.get_payload(decode=True).decode('utf-8', errors='replace')
                            soup = BeautifulSoup(html_content, 'html.parser')
                            text_content = soup.get_text()
                            clean_content = self._sanitize_unicode(text_content)
                            body_parts.append(clean_content)
                        except:
                            continue
            else:
                # Single part message
                try:
                    content = msg.get_payload(decode=True)
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                    
                    if msg.get_content_type() == 'text/html':
                        soup = BeautifulSoup(content, 'html.parser')
                        text_content = soup.get_text()
                        clean_content = self._sanitize_unicode(text_content)
                        body_parts.append(clean_content)
                    else:
                        clean_content = self._sanitize_unicode(str(content))
                        body_parts.append(clean_content)
                except:
                    raw_payload = str(msg.get_payload())
                    clean_content = self._sanitize_unicode(raw_payload)
                    body_parts.append(clean_content)
            
            # Combine headers and body
            full_text = "\n".join(headers) + "\n\n" + "\n".join(body_parts)
            
            # Final sanitization of the complete text
            full_text = self._sanitize_unicode(full_text)
            
            logger.info(f"✅ EML processed: {len(headers)} headers, {len(body_parts)} body parts")
            return full_text
            
        except Exception as e:
            logger.error(f"❌ EML processing failed: {e}")
            return ""
    
    async def _process_srt(self, file_path: str) -> str:
        """Process SRT subtitle file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse SRT format
            # SRT format:
            # 1
            # 00:00:01,000 --> 00:00:04,000
            # This is the first subtitle
            #
            # 2
            # 00:00:05,000 --> 00:00:08,000
            # This is the second subtitle
            
            subtitles = []
            blocks = content.strip().split('\n\n')
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # Skip the sequence number (first line)
                    # Skip the timestamp (second line)
                    # Extract the subtitle text (remaining lines)
                    subtitle_text = '\n'.join(lines[2:])
                    if subtitle_text.strip():
                        subtitles.append(subtitle_text.strip())
            
            # Combine all subtitles with proper spacing
            combined_text = '\n\n'.join(subtitles)
            
            logger.info(f"✅ SRT processed: {len(subtitles)} subtitles extracted")
            return combined_text
            
        except UnicodeDecodeError:
            # Try different encodings for SRT files
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    
                    # Parse SRT format (same logic as above)
                    subtitles = []
                    blocks = content.strip().split('\n\n')
                    
                    for block in blocks:
                        lines = block.strip().split('\n')
                        if len(lines) >= 3:
                            subtitle_text = '\n'.join(lines[2:])
                            if subtitle_text.strip():
                                subtitles.append(subtitle_text.strip())
                    
                    combined_text = '\n\n'.join(subtitles)
                    logger.info(f"✅ SRT processed with {encoding}: {len(subtitles)} subtitles extracted")
                    return combined_text
                    
                except UnicodeDecodeError:
                    continue
            
            logger.error(f"❌ Could not decode SRT file: {file_path}")
            return ""
            
        except Exception as e:
            logger.error(f"❌ SRT processing failed: {e}")
            return ""
    
    async def _process_zip(self, file_path: str) -> str:
        """Process ZIP file by extracting and processing contained files with caching"""
        try:
            extracted_texts = []
            
            # Create a cache key for this ZIP file to avoid reprocessing
            zip_cache_key = f"zip_{Path(file_path).stem}"
            
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                # Get list of files in ZIP
                file_list = zip_file.namelist()
                logger.info(f"📦 ZIP contains {len(file_list)} files")
                
                # Filter to only supported files
                supported_files = []
                supported_extensions = {
                    '.pdf': 'pdf',
                    '.txt': 'txt',
                    '.docx': 'docx',
                    '.html': 'html',
                    '.htm': 'html',
                    '.eml': 'eml',
                    '.srt': 'srt'
                }
                
                for file_name in file_list:
                    # Skip directories and hidden files
                    if file_name.endswith('/') or file_name.startswith('.'):
                        continue
                    
                    file_ext = Path(file_name).suffix.lower()
                    if file_ext in supported_extensions:
                        supported_files.append((file_name, supported_extensions[file_ext]))
                    else:
                        logger.debug(f"⏭️  Skipping unsupported file: {file_name}")
                
                logger.info(f"📦 Processing {len(supported_files)} supported files from ZIP")
                
                # Process each supported file
                for file_name, doc_type in supported_files:
                    try:
                        # Extract file to temporary location
                        with tempfile.NamedTemporaryFile(suffix=Path(file_name).suffix, delete=False) as temp_file:
                            temp_path = temp_file.name
                            temp_file.write(zip_file.read(file_name))
                        
                        # Process the extracted file with reduced logging
                        logger.debug(f"📄 Processing {file_name} as {doc_type}")
                        
                        if doc_type == 'pdf':
                            text, _ = await self._process_pdf(temp_path)
                        elif doc_type == 'txt':
                            text = await self._process_text(temp_path)
                        elif doc_type == 'docx':
                            text = await self._process_docx(temp_path)
                        elif doc_type == 'html':
                            text = await self._process_html(temp_path)
                        elif doc_type == 'eml':
                            text = await self._process_eml(temp_path)
                        elif doc_type == 'srt':
                            text = await self._process_srt(temp_path)
                        else:
                            text = ""
                        
                        if text.strip():
                            # Add file separator and filename
                            file_header = f"\n{'='*50}\nFILE: {file_name}\n{'='*50}\n"
                            extracted_texts.append(file_header + text)
                            logger.debug(f"✅ Extracted {len(text)} characters from {file_name}")
                        
                        # Clean up temporary file
                        os.unlink(temp_path)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process {file_name}: {e}")
                        continue
            
            # Combine all extracted texts
            combined_text = "\n\n".join(extracted_texts)
            
            logger.info(f"✅ ZIP processing complete: {len(extracted_texts)} files processed, {len(combined_text)} total characters")
            return combined_text
            
        except Exception as e:
            logger.error(f"❌ ZIP processing failed: {e}")
            return ""
    
    async def _assess_quality(self, text: str, ocr_confidence: float) -> QualityMetrics:
        """Assess the quality of extracted text"""
        try:
            # Language detection confidence
            try:
                language = detect(text)
                lang_confidence = 0.9 if language == 'en' else 0.7
            except:
                lang_confidence = 0.5
            
            # Vocabulary coherence (ratio of dictionary words)
            words = text.split()
            if len(words) > 0:
                # Simple heuristic: check for reasonable word length distribution
                avg_word_length = sum(len(word) for word in words) / len(words)
                vocab_score = min(1.0, max(0.0, (avg_word_length - 2) / 10))
            else:
                vocab_score = 0.0
            
            # Text pattern analysis (punctuation, capitalization, etc.)
            if len(text) > 0:
                capital_ratio = sum(1 for c in text if c.isupper()) / len(text)
                punct_ratio = sum(1 for c in text if c in '.,!?;:') / len(text)
                pattern_score = min(1.0, (capital_ratio * 10 + punct_ratio * 20))
            else:
                pattern_score = 0.0
            
            # Overall score
            overall_score = (
                ocr_confidence * 0.3 +
                lang_confidence * 0.3 +
                vocab_score * 0.2 +
                pattern_score * 0.2
            )
            
            return QualityMetrics(
                ocr_confidence=ocr_confidence,
                language_confidence=lang_confidence,
                vocabulary_score=vocab_score,
                pattern_score=pattern_score,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"❌ Quality assessment failed: {e}")
            return QualityMetrics(
                ocr_confidence=ocr_confidence,
                language_confidence=0.5,
                vocabulary_score=0.5,
                pattern_score=0.5,
                overall_score=0.5
            )
    
    async def _chunk_text(self, text: str, file_path: str, document_id: str) -> List[Chunk]:
        """Universal chunking system that adapts to document type and structure"""
        try:
            logger.info(f"📝 Text length: {len(text)} characters")
            logger.info(f"📝 First 200 chars: {text[:200]}...")
            
            # Detect document structure and choose appropriate strategy
            doc_structure = self._analyze_document_structure(text)
            logger.info(f"📝 Document structure: {doc_structure}")
            
            chunks = []
            
            if doc_structure == "book":
                chunks = self._chunk_book_content(text, file_path, document_id)
            elif doc_structure == "email":
                chunks = self._chunk_email_content(text, file_path, document_id)
            elif doc_structure == "academic_paper":
                chunks = self._chunk_academic_content(text, file_path, document_id)
            elif doc_structure == "article":
                chunks = self._chunk_article_content(text, file_path, document_id)
            else:
                # Default hierarchical chunking
                chunks = self._chunk_hierarchical(text, file_path, document_id)
            
            # Post-process: ensure optimal chunk sizes
            final_chunks = self._optimize_chunk_sizes(chunks, file_path, document_id)
            
            logger.info(f"📝 Created {len(final_chunks)} total chunks using {doc_structure} strategy")
            return final_chunks
            
        except Exception as e:
            logger.error(f"❌ Text chunking failed: {e}")
            return []
    
    def _analyze_document_structure(self, text: str) -> str:
        """Analyze document to determine its structure type"""
        text_lower = text.lower()
        lines = text.split('\n')
        
        # Email detection
        email_indicators = ['from:', 'to:', 'subject:', '@', 'sent:', 'cc:']
        if sum(1 for line in lines[:10] if any(ind in line.lower() for ind in email_indicators)) >= 2:
            return "email"
        
        # Book detection
        book_indicators = ['chapter', 'table of contents', 'preface', 'introduction', 'epilogue']
        if any(indicator in text_lower for indicator in book_indicators):
            # Check for chapter-like structure
            chapter_patterns = ['chapter ', 'part ', 'section ']
            if sum(1 for line in lines if any(pattern in line.lower() for pattern in chapter_patterns)) >= 2:
                return "book"
        
        # Academic paper detection
        academic_indicators = ['abstract', 'introduction', 'methodology', 'results', 'conclusion', 'references', 'bibliography']
        if sum(1 for indicator in academic_indicators if indicator in text_lower) >= 3:
            return "academic_paper"
        
        # Article detection (news, blog, etc.)
        if len(text) > 1000 and len(text) < 50000:  # Medium length
            paragraphs = [p for p in text.split('\n\n') if len(p.strip()) > 100]
            if len(paragraphs) >= 3:
                return "article"
        
        return "general"
    
    def _chunk_book_content(self, text: str, file_path: str, document_id: str) -> List[Chunk]:
        """Chunk book content by chapters, sections, and paragraphs"""
        chunks = []
        lines = text.split('\n')
        
        current_chapter = ""
        current_section = ""
        current_content = ""
        chunk_index = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_type = self._classify_book_line(line)
            
            if line_type == "chapter_title":
                # Save previous content
                if current_content.strip():
                    chunks.append(self._create_chunk(
                        f"{document_id}_ch_{chunk_index}",
                        document_id,
                        current_content.strip(),
                        chunk_index,
                        "chapter_content",
                        {"chapter": current_chapter, "section": current_section}
                    ))
                    chunk_index += 1
                
                current_chapter = line
                current_section = ""
                current_content = ""
                
                # Chapter title as its own chunk
                chunks.append(self._create_chunk(
                    f"{Path(file_path).stem}_title_{chunk_index}",
                    document_id,
                    line,
                    chunk_index,
                    "chapter_title",
                    {"chapter": line}
                ))
                chunk_index += 1
                
            elif line_type == "section_title":
                # Save previous section content
                if current_content.strip():
                    chunks.append(self._create_chunk(
                        f"{Path(file_path).stem}_sec_{chunk_index}",
                        document_id,
                        current_content.strip(),
                        chunk_index,
                        "section_content",
                        {"chapter": current_chapter, "section": current_section}
                    ))
                    chunk_index += 1
                
                current_section = line
                current_content = line + "\n"
                
            else:
                current_content += line + "\n"
                
                # Split if content gets too long (increased threshold)
                if len(current_content) > 1500:
                    chunks.append(self._create_chunk(
                        f"{Path(file_path).stem}_para_{chunk_index}",
                        document_id,
                        current_content.strip(),
                        chunk_index,
                        "paragraph_group",
                        {"chapter": current_chapter, "section": current_section}
                    ))
                    chunk_index += 1
                    current_content = ""
        
        # Add final content
        if current_content.strip():
            chunks.append(self._create_chunk(
                f"{Path(file_path).stem}_final_{chunk_index}",
                document_id,
                current_content.strip(),
                chunk_index,
                "final_content",
                {"chapter": current_chapter, "section": current_section}
            ))
        
        return chunks
    
    def _chunk_email_content(self, text: str, file_path: str, document_id: str) -> List[Chunk]:
        """Chunk email content with aggressive size limits and token awareness"""
        chunks = []
        
        # Split the massive email collection into individual emails first
        email_files = text.split('\n==================================================\nFILE: ')
        
        chunk_index = 0
        
        for i, email_content in enumerate(email_files):
            if not email_content.strip():
                continue
            
            # Add back the file separator for non-first emails
            if i > 0:
                email_content = 'FILE: ' + email_content
            
            # Extract filename from the content if present
            lines = email_content.split('\n')
            file_name = "unknown"
            content_start = 0
            
            # Look for the filename in the first few lines
            for j, line in enumerate(lines[:3]):
                if line.startswith('FILE: ') or 'FILE:' in line:
                    file_name = line.replace('FILE:', '').strip()
                    content_start = j + 1
                    break
                elif line.startswith('=================================================='):
                    content_start = j + 1
                    break
            
            # Process the actual email content (skip file headers)
            email_body = '\n'.join(lines[content_start:])
            
            # Now chunk this individual email with strict token limits
            email_chunks = self._chunk_single_email(email_body, file_path, chunk_index, file_name)
            chunks.extend(email_chunks)
            chunk_index += len(email_chunks)
        
        return chunks
    
    def _chunk_single_email(self, email_text: str, file_path: str, start_index: int, file_name: str) -> List[Chunk]:
        """Chunk a single email with very strict token limits"""
        chunks = []
        lines = email_text.split('\n')
        
        current_content = ""
        current_type = "content"
        chunk_index = start_index
        max_tokens_per_chunk = 2500  # Reasonable limit that preserves semantic context
        
        # Process line by line with strict token monitoring
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip system messages and artifacts
            if self._is_system_message(line):
                continue
            
            # Determine content type
            line_type = self._classify_line_type(line)
            
            # Check if adding this line would exceed token limits
            test_content = current_content + "\n" + line if current_content else line
            estimated_tokens = self._estimate_chunk_tokens(test_content)
            
            # If adding this line would exceed limits, save current chunk and start new one
            if estimated_tokens > max_tokens_per_chunk and current_content.strip():
                chunks.append(self._create_chunk(
                    f"{Path(file_path).stem}_{current_type}_{chunk_index}",
                    document_id,
                    current_content.strip(),
                    chunk_index,
                    current_type,
                    {
                        "content_type": current_type,
                        "source_file": file_name,
                        "estimated_tokens": self._estimate_chunk_tokens(current_content)
                    }
                ))
                chunk_index += 1
                current_content = line
                current_type = line_type
            
            # If the content type changes significantly, start a new chunk
            elif current_type != line_type and current_content.strip() and len(current_content) > 100:
                chunks.append(self._create_chunk(
                    f"{Path(file_path).stem}_{current_type}_{chunk_index}",
                    document_id,
                    current_content.strip(),
                    chunk_index,
                    current_type,
                    {
                        "content_type": current_type,
                        "source_file": file_name,
                        "estimated_tokens": self._estimate_chunk_tokens(current_content)
                    }
                ))
                chunk_index += 1
                current_content = line
                current_type = line_type
            
            else:
                # Add line to current chunk
                current_content = test_content
                current_type = line_type
        
        # Add final chunk if exists
        if current_content.strip():
            chunks.append(self._create_chunk(
                f"{Path(file_path).stem}_{current_type}_{chunk_index}",
                document_id,
                current_content.strip(),
                chunk_index,
                current_type,
                {
                    "content_type": current_type,
                    "source_file": file_name,
                    "estimated_tokens": self._estimate_chunk_tokens(current_content)
                }
            ))
        
        return chunks
    
    def _is_system_message(self, line: str) -> bool:
        """Check if line is a system message to skip"""
        line_lower = line.lower()
        skip_patterns = [
            'microsoft office', 'prevented automatic', 'to help protect',
            'download of this picture', 'from the internet'
        ]
        return any(pattern in line_lower for pattern in skip_patterns)
    
    def _is_quote_line(self, line: str) -> bool:
        """Check if line looks like a quote"""
        # Must be reasonable length and end with punctuation
        if len(line) < 20 or len(line) > 200:
            return False
        
        # Must end with sentence-ending punctuation
        if not any(line.endswith(p) for p in ['.', '!', '?']):
            return False
        
        # Skip if it looks like email metadata
        line_lower = line.lower()
        if any(pattern in line_lower for pattern in ['from:', 'to:', 'subject:', '@', 'sent:']):
            return False
        
        return True
    
    def _chunk_academic_content(self, text: str, file_path: str, document_id: str) -> List[Chunk]:
        """Chunk academic papers by sections"""
        chunks = []
        sections = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion', 'references']
        
        current_section = ""
        current_content = ""
        chunk_index = 0
        
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if this paragraph starts a new section
            para_lower = paragraph.lower()
            found_section = None
            for section in sections:
                if para_lower.startswith(section) or f"\n{section}" in para_lower:
                    found_section = section
                    break
            
            if found_section:
                # Save previous section
                if current_content.strip():
                    chunks.append(self._create_chunk(
                        f"{Path(file_path).stem}_{current_section}_{chunk_index}",
                        document_id,
                        current_content.strip(),
                        chunk_index,
                        f"academic_{current_section}",
                        {"section": current_section}
                    ))
                    chunk_index += 1
                
                current_section = found_section
                current_content = paragraph + "\n"
            else:
                current_content += paragraph + "\n"
                
                # Split long sections
                if len(current_content) > 1000:
                    chunks.append(self._create_chunk(
                        f"{Path(file_path).stem}_{current_section}_{chunk_index}",
                        document_id,
                        current_content.strip(),
                        chunk_index,
                        f"academic_{current_section}",
                        {"section": current_section}
                    ))
                    chunk_index += 1
                    current_content = ""
        
        # Add final content
        if current_content.strip():
            chunks.append(self._create_chunk(
                f"{Path(file_path).stem}_{current_section}_{chunk_index}",
                document_id,
                current_content.strip(),
                chunk_index,
                f"academic_{current_section}",
                {"section": current_section}
            ))
        
        return chunks
    
    def _chunk_article_content(self, text: str, file_path: str, document_id: str) -> List[Chunk]:
        """Chunk articles by paragraphs with context"""
        chunks = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 50:  # Skip very short paragraphs
                chunk_type = "title" if i == 0 and len(paragraph) < 200 else "paragraph"
                
                chunks.append(self._create_chunk(
                    f"{Path(file_path).stem}_para_{i}",
                    document_id,
                    paragraph,
                    i,
                    chunk_type,
                    {"paragraph_number": i, "total_paragraphs": len(paragraphs)}
                ))
        
        return chunks
    
    def _chunk_hierarchical(self, text: str, file_path: str, document_id: str) -> List[Chunk]:
        """Default hierarchical chunking for general documents"""
        chunks = []
        
        # Try paragraph-based first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) > 1:
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 30:
                    chunks.append(self._create_chunk(
                        f"{Path(file_path).stem}_para_{i}",
                        document_id,
                        paragraph,
                        i,
                        "paragraph",
                        {}
                    ))
        else:
            # Fall back to fixed-size chunking
            chunk_size = 600
            overlap = 100
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                if chunk_text.strip():
                    chunks.append(self._create_chunk(
                        f"{Path(file_path).stem}_fixed_{i // (chunk_size - overlap)}",
                        document_id,
                        chunk_text.strip(),
                        i // (chunk_size - overlap),
                        "fixed_size",
                        {"start_pos": i, "end_pos": i + chunk_size}
                    ))
        
        return chunks
    
    def _classify_book_line(self, line: str) -> str:
        """Classify lines in book content"""
        line_lower = line.lower().strip()
        
        # Chapter titles
        if (line_lower.startswith('chapter ') or 
            line_lower.startswith('part ') or
            (len(line) < 100 and line.isupper())):
            return "chapter_title"
        
        # Section titles (often numbered or short)
        if (len(line) < 150 and 
            (line_lower.startswith(('1.', '2.', '3.', '4.', '5.')) or
             (len(line.split()) < 8 and not line.endswith('.')))):
            return "section_title"
        
        return "content"
    
    def _create_chunk(self, chunk_id: str, document_id: str, content: str, 
                     index: int, chunk_type: str, metadata: dict) -> Chunk:
        """Helper to create a chunk object with Unicode sanitization"""
        # Sanitize content to prevent Unicode encoding issues
        clean_content = self._sanitize_unicode(content)
        
        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=clean_content,
            chunk_index=index,
            quality_score=0.8,
            method="adaptive",
            metadata={
                "word_count": len(clean_content.split()),
                "char_count": len(clean_content),
                "chunk_type": chunk_type,
                **metadata
            }
        )
    
    def _optimize_chunk_sizes(self, chunks: List[Chunk], file_path: str, document_id: str) -> List[Chunk]:
        """Optimize chunk sizes for better embedding performance with very strict limits"""
        optimized_chunks = []
        
        for chunk in chunks:
            # Very aggressive size limits to prevent token overflow
            # Using conservative token estimation: 1 token per 2 characters
            estimated_tokens = self._estimate_chunk_tokens(chunk.content)
            
            if estimated_tokens > 4000:  # Balanced limit for better semantic context
                sub_chunks = self._split_large_chunk(chunk, document_id)
                optimized_chunks.extend(sub_chunks)
            elif len(chunk.content) < 50:  # Merge very small chunks
                # Try to merge with previous chunk if compatible
                if (optimized_chunks and 
                    self._estimate_chunk_tokens(optimized_chunks[-1].content) < 2000 and  # Safe merge threshold
                    optimized_chunks[-1].metadata.get('chunk_type') == chunk.metadata.get('chunk_type')):
                    
                    # Merge with previous chunk
                    prev_chunk = optimized_chunks[-1]
                    merged_content = prev_chunk.content + "\n" + chunk.content
                    
                    # Check if merged content is still within safe token limits
                    if self._estimate_chunk_tokens(merged_content) <= 3500:  # Reasonable merge limit
                        merged_chunk = Chunk(
                            chunk_id=f"{prev_chunk.chunk_id}_merged",
                            document_id=prev_chunk.document_id,
                            content=merged_content,
                            chunk_index=prev_chunk.chunk_index,
                            quality_score=prev_chunk.quality_score,
                            method="merged",
                            metadata={
                                "word_count": len(merged_content.split()),
                                "char_count": len(merged_content),
                                "chunk_type": prev_chunk.metadata.get('chunk_type'),
                                "merged_from": [prev_chunk.chunk_id, chunk.chunk_id],
                                "estimated_tokens": self._estimate_chunk_tokens(merged_content)
                            }
                        )
                        
                        optimized_chunks[-1] = merged_chunk
                    else:
                        # Don't merge if it would create too large a chunk
                        optimized_chunks.append(chunk)
                else:
                    optimized_chunks.append(chunk)
            else:
                # Add token estimate to metadata for monitoring
                chunk.metadata["estimated_tokens"] = estimated_tokens
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _create_semantic_chunks(self, text: str) -> List[str]:
        """Create chunks based on semantic content boundaries"""
        lines = text.split('\n')
        chunks = []
        current_chunk = ""
        current_type = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_type = self._classify_line_type(line)
            
            # Start new chunk if content type changes or chunk gets too long
            if (current_type and line_type != current_type) or len(current_chunk) > 300:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line
                current_type = line_type
            else:
                current_chunk += "\n" + line if current_chunk else line
                current_type = line_type
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _classify_line_type(self, line: str) -> str:
        """Classify what type of content a line contains"""
        line_lower = line.lower()
        
        # Email headers
        if any(pattern in line_lower for pattern in ['from:', 'to:', 'cc:', 'subject:', 'sent:', '@']):
            return 'email_header'
        
        # Quotes/sayings (usually end with punctuation and are standalone)
        if len(line) > 30 and any(line.endswith(p) for p in ['.', '!', '?']) and not line_lower.startswith(('from', 'to', 'cc', 'subject')):
            return 'quote'
        
        # Title/subject lines
        if 'bob hope' in line_lower and len(line) < 100:
            return 'title'
        
        # Technical/system messages
        if any(pattern in line_lower for pattern in ['microsoft office', 'prevented', 'download', 'protect']):
            return 'system_message'
        
        # Default content
        return 'content'
    
    def _classify_chunk_type(self, chunk: str) -> str:
        """Classify the overall type of a chunk"""
        chunk_lower = chunk.lower()
        
        if 'from:' in chunk_lower or 'to:' in chunk_lower:
            return 'email_metadata'
        elif 'bob hope' in chunk_lower and len(chunk) < 200:
            return 'title_or_subject'
        elif any(chunk.endswith(p) for p in ['.', '!', '?']) and len(chunk) > 30:
            return 'quote_or_saying'
        elif 'microsoft' in chunk_lower or 'office' in chunk_lower:
            return 'system_message'
        else:
            return 'general_content'
    
    def _split_large_chunk(self, chunk: Chunk, document_id: str) -> List[Chunk]:
        """Split a large chunk into smaller pieces with very strict token limits
        
        Note: document_id parameter is for consistency but chunk.document_id is already set correctly
        """
        content = chunk.content
        sentences = self._split_by_sentences(content)
        
        sub_chunks = []
        current_content = ""
        sub_index = 0
        max_sub_chunk_tokens = 2800  # Balanced token limit for good context
        
        for sentence in sentences:
            # Check if adding this sentence would exceed token limits
            test_content = current_content + " " + sentence if current_content else sentence
            estimated_tokens = self._estimate_chunk_tokens(test_content)
            
            if estimated_tokens > max_sub_chunk_tokens and current_content:
                # Save current chunk
                sub_chunk = Chunk(
                    chunk_id=f"{chunk.chunk_id}_sub_{sub_index}",
                    document_id=chunk.document_id,
                    content=current_content.strip(),
                    chunk_index=chunk.chunk_index * 1000 + sub_index,
                    quality_score=chunk.quality_score,
                    method="sub_chunk",
                    metadata={
                        "word_count": len(current_content.split()),
                        "char_count": len(current_content),
                        "estimated_tokens": self._estimate_chunk_tokens(current_content),
                        "parent_chunk": chunk.chunk_id,
                        "chunk_type": chunk.metadata.get('chunk_type', 'unknown'),
                        "parent_chunk_index": chunk.chunk_index,
                        "sub_index": sub_index
                    }
                )
                sub_chunks.append(sub_chunk)
                logger.debug(f"📝 Sub-chunk {sub_index}: {len(current_content)} chars, ~{self._estimate_chunk_tokens(current_content)} tokens")
                current_content = sentence
                sub_index += 1
            else:
                current_content = test_content
        
        # Add final sub-chunk
        if current_content.strip():
            sub_chunk = Chunk(
                chunk_id=f"{chunk.chunk_id}_sub_{sub_index}",
                document_id=chunk.document_id,
                content=current_content.strip(),
                chunk_index=chunk.chunk_index * 1000 + sub_index,
                quality_score=chunk.quality_score,
                method="sub_chunk",
                metadata={
                    "word_count": len(current_content.split()),
                    "char_count": len(current_content),
                    "estimated_tokens": self._estimate_chunk_tokens(current_content),
                    "parent_chunk": chunk.chunk_id,
                    "chunk_type": chunk.metadata.get('chunk_type', 'unknown'),
                    "parent_chunk_index": chunk.chunk_index,
                    "sub_index": sub_index
                }
            )
            sub_chunks.append(sub_chunk)
            logger.debug(f"📝 Final sub-chunk {sub_index}: {len(current_content)} chars, ~{self._estimate_chunk_tokens(current_content)} tokens")
        
        return sub_chunks if sub_chunks else [chunk]  # Return original if no splits made
    
    def _estimate_chunk_tokens(self, text: str) -> int:
        """Estimate tokens for chunking decisions - very conservative"""
        if not text:
            return 0
        
        # Very conservative estimation: 1 token per 2 characters
        # This is more conservative than typical 3-4 chars per token
        char_count = len(text)
        base_tokens = char_count // 2
        
        # Add overhead for punctuation and special characters
        punct_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        punct_overhead = punct_chars * 0.5
        
        # Add overhead for word boundaries
        words = len(text.split())
        word_overhead = words * 0.3
        
        # Extra overhead for email content
        if any(pattern in text.lower() for pattern in ['from:', 'to:', 'subject:', '@']):
            email_overhead = char_count * 0.1
        else:
            email_overhead = 0
        
        total_estimated = int(base_tokens + punct_overhead + word_overhead + email_overhead)
        
        # Add 30% safety buffer
        return int(total_estimated * 1.3)
    
    def _clean_email_content(self, text: str) -> str:
        """Clean email headers and extract main content"""
        lines = text.split('\n')
        cleaned_lines = []
        
        # Skip email headers (lines that look like email metadata)
        skip_patterns = [
            'From:', 'To:', 'Cc:', 'Subject:', 'Sent:', 'Date:',
            '@', 'gcfl', 'pilbeam', 'Microsoft Office',
            'To help protect', 'prevented automatic download'
        ]
        
        for line in lines:
            line = line.strip()
            if line and not any(pattern.lower() in line.lower() for pattern in skip_patterns):
                # Skip very short lines that are likely metadata
                if len(line) > 10 or any(char in line for char in '.!?'):
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _sanitize_unicode(self, text: str) -> str:
        """Sanitize text to remove problematic Unicode characters that cause JSON serialization issues"""
        if not text:
            return text
        
        try:
            # Remove surrogate characters and other problematic Unicode
            # that can't be encoded in UTF-8
            sanitized = ""
            for char in text:
                try:
                    # Try to encode the character to UTF-8
                    char.encode('utf-8')
                    sanitized += char
                except UnicodeEncodeError:
                    # Replace problematic characters with a safe placeholder
                    sanitized += '?'
            
            # Additional cleanup for common email encoding issues
            sanitized = sanitized.replace('\ufffd', '?')  # Replace replacement character
            sanitized = sanitized.replace('\u0000', '')   # Remove null characters
            
            # Remove or replace other problematic characters
            import re
            # Remove control characters except newlines, tabs, and carriage returns
            sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', sanitized)
            
            return sanitized
            
        except Exception as e:
            logger.warning(f"⚠️ Unicode sanitization failed: {e}")
            # Fallback: encode/decode to remove problematic characters
            try:
                return text.encode('utf-8', errors='replace').decode('utf-8')
            except:
                return str(text)
    
    async def _extract_entities(self, text: str, chunks: List[Chunk]) -> List[Entity]:
        """Extract entities from document text using spaCy and pattern-based methods"""
        try:
            logger.info("🔍 Starting entity extraction...")
            
            entities = []
            
            # Method 1: spaCy NER (primary method)
            if self.nlp:
                spacy_entities = self._extract_spacy_entities(text)
                entities.extend(spacy_entities)
                logger.info(f"🔍 spaCy found {len(spacy_entities)} entities")
            else:
                logger.warning("🔍 spaCy not available - install with: python -m spacy download en_core_web_lg")
            
            # Method 2: Pattern-based extraction (fallback and supplement)
            pattern_entities = self._extract_pattern_entities(text)
            entities.extend(pattern_entities)
            logger.info(f"🔍 Patterns found {len(pattern_entities)} entities")
            
            # Deduplicate and score entities
            final_entities = self._deduplicate_entities(entities)
            
            logger.info(f"🔍 Final entity count: {len(final_entities)} (spaCy + patterns)")
            return final_entities
            
        except Exception as e:
            logger.error(f"❌ Entity extraction failed: {e}")
            return []
    
    def _extract_spacy_entities(self, text: str) -> List[Entity]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            return []
        
        try:
            # Process text in chunks to avoid memory issues
            max_length = 1000000  # 1M characters
            if len(text) > max_length:
                text = text[:max_length]
            
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Map spaCy labels to our entity types
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    entities.append(Entity(
                        name=ent.text.strip(),
                        entity_type=entity_type,
                        confidence=0.8,  # spaCy confidence
                        source_chunk="",
                        metadata={"source": "spacy", "context": ent.sent.text[:200] if ent.sent else ""}
                    ))
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ spaCy entity extraction failed: {e}")
            return []
    
    
    def _extract_pattern_entities(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        try:
            # Email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            for match in re.finditer(email_pattern, text):
                entities.append(Entity(
                    name=match.group(),
                    entity_type="EMAIL",
                    confidence=0.9,
                    source_chunk="",
                    metadata={"source": "pattern", "context": ""}
                ))
            
            # URLs
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            for match in re.finditer(url_pattern, text):
                entities.append(Entity(
                    name=match.group(),
                    entity_type="URL",
                    confidence=0.9,
                    source_chunk="",
                    metadata={"source": "pattern", "context": ""}
                ))
            
            # Phone numbers
            phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
            for match in re.finditer(phone_pattern, text):
                entities.append(Entity(
                    name=match.group(),
                    entity_type="PHONE",
                    confidence=0.8,
                    source_chunk="",
                    metadata={"source": "pattern", "context": ""}
                ))
            
            # Dates (simple patterns)
            date_patterns = [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{4}-\d{2}-\d{2}\b',      # YYYY-MM-DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
            ]
            
            for pattern in date_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(Entity(
                        name=match.group(),
                        entity_type="DATE",
                        confidence=0.7,
                        source_chunk="",
                        metadata={"source": "pattern", "context": ""}
                    ))
            
            # Programming languages and technologies
            tech_keywords = [
                'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Ruby', 'PHP', 'Go', 'Rust',
                'React', 'Vue', 'Angular', 'Node.js', 'Django', 'Flask', 'Spring',
                'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'MongoDB', 'PostgreSQL',
                'MySQL', 'Redis', 'Elasticsearch', 'Apache', 'Nginx', 'Git', 'GitHub'
            ]
            
            for keyword in tech_keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(Entity(
                        name=match.group(),
                        entity_type="TECHNOLOGY",
                        confidence=0.6,
                        source_chunk="",
                        metadata={"source": "pattern", "context": ""}
                    ))
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ Pattern entity extraction failed: {e}")
            return []
    
    def _map_spacy_label(self, label: str) -> str:
        """Map spaCy entity labels to our entity types"""
        mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',  # Geopolitical entity
            'LOC': 'LOCATION',
            'DATE': 'DATE',
            'TIME': 'DATE',
            'MONEY': 'MONEY',
            'PERCENT': 'PERCENT',
            'PRODUCT': 'PRODUCT',
            'EVENT': 'EVENT',
            'WORK_OF_ART': 'WORK_OF_ART',
            'LAW': 'LAW',
            'LANGUAGE': 'LANGUAGE',
            'NORP': 'GROUP',  # Nationalities, religious groups
            'FAC': 'FACILITY',  # Buildings, airports, etc.
        }
        
        return mapping.get(label, None)
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities and merge similar ones"""
        if not entities:
            return []
        
        # Group entities by normalized name
        entity_groups = {}
        
        for entity in entities:
            # Normalize entity name
            normalized_name = entity.name.lower().strip()
            
            # Skip very short or common entities
            if len(normalized_name) < 2 or normalized_name in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
                continue
            
            if normalized_name not in entity_groups:
                entity_groups[normalized_name] = []
            entity_groups[normalized_name].append(entity)
        
        # Select best entity from each group
        final_entities = []
        
        for group in entity_groups.values():
            if not group:
                continue
            
            # Sort by confidence and source priority
            source_priority = {'llm': 3, 'spacy': 2, 'pattern': 1}
            group.sort(key=lambda e: (e.confidence, source_priority.get(e.metadata.get("source", ""), 0)), reverse=True)
            
            best_entity = group[0]
            
            # Use the most common entity type if there are conflicts
            type_counts = {}
            for entity in group:
                type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1
            
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
            best_entity.entity_type = most_common_type
            
            # Average confidence across sources
            avg_confidence = sum(e.confidence for e in group) / len(group)
            best_entity.confidence = min(0.95, avg_confidence)
            
            final_entities.append(best_entity)
        
        # Sort by confidence and return top entities
        final_entities.sort(key=lambda e: e.confidence, reverse=True)
        return final_entities[:settings.MAX_ENTITY_RESULTS]  # Use configurable entity limit

    async def process_text_content(self, content: str, document_id: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Process text content directly and return chunks"""
        try:
            logger.info(f"🔄 Processing text content for document: {document_id}")
            
            # Clean and normalize text
            content = self._sanitize_unicode(content)
            
            # Assess quality
            quality_metrics = await self._assess_quality(content, 1.0)
            
            # Create chunks
            chunks = await self._chunk_text(content, f"{document_id}.txt", document_id)
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk.metadata = chunk.metadata or {}
                chunk.metadata.update(metadata or {})
                chunk.metadata["quality_metrics"] = quality_metrics.dict()
                chunk.metadata["processing_method"] = "text_content"
                chunk.metadata["document_id"] = document_id
            
            logger.info(f"✅ Processed text content: {len(chunks)} chunks for {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to process text content for {document_id}: {e}")
            return []


# Global document processor instance for lazy loading
_document_processor_instance = None


async def get_document_processor() -> DocumentProcessor:
    """Get or create a global document processor instance"""
    global _document_processor_instance
    
    if _document_processor_instance is None:
        logger.info("🔄 Creating global document processor instance...")
        _document_processor_instance = DocumentProcessor()
        await _document_processor_instance.initialize()
        logger.info("✅ Global document processor initialized")
    
    return _document_processor_instance
