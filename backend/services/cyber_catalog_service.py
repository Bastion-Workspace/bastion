"""
Cyber Data Catalog Service
JSON-first cataloging service for cyber/breach data

**BULLY!** Validate first, commit second - Roosevelt's cataloging doctrine!
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from openai import AsyncOpenAI

from config import settings
from models.cyber_catalog_models import (
    CyberDataCatalogEntry,
    CyberCatalogJSON,
    CyberCatalogConfig,
    CyberCatalogValidationResult,
    calculate_file_hash
)
from utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class CyberCatalogService:
    """
    Service for cataloging cyber/breach data with JSON-first validation
    
    **BULLY!** Systematic cataloging of unstructured cyber data!
    """
    
    def __init__(self):
        self.openrouter_client = None
        self.initialized = False
        self.document_processor = None
        
        # Cyber-specific breach types
        self.breach_types = [
            "data_breach", "ransomware", "phishing", "malware",
            "ddos", "insider_threat", "vulnerability", "credential_stuffing",
            "supply_chain", "zero_day", "other"
        ]
        
        # Cyber-specific categories
        self.cyber_categories = [
            "data_breach", "ransomware", "phishing", "malware",
            "ddos", "credentials", "vulnerability", "insider_threat",
            "supply_chain", "zero_day", "financial", "healthcare",
            "government", "education", "critical_infrastructure"
        ]
    
    async def initialize(self):
        """Initialize the cyber catalog service"""
        try:
            # Initialize OpenRouter client for LLM categorization
            self.openrouter_client = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            
            # Initialize document processor for text extraction
            self.document_processor = DocumentProcessor()
            await self.document_processor.initialize()
            
            self.initialized = True
            logger.info("✅ Cyber Catalog Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cyber Catalog Service: {e}")
            raise
    
    async def catalog_folder(
        self,
        folder_path: str,
        config: Optional[CyberCatalogConfig] = None
    ) -> Dict[str, Any]:
        """
        Catalog a folder of cyber/breach data files
        
        **BULLY!** JSON-first cataloging with validation!
        
        Args:
            folder_path: Path to folder to catalog
            config: Catalog configuration (optional)
            
        Returns:
            Dict with catalog results, JSON path, and validation info
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Default config
        if config is None:
            config = CyberCatalogConfig(
                validate_only=True,
                output_json_path=None  # Will generate default path
            )
        
        # Generate default JSON path if not provided
        json_path = config.output_json_path
        if not json_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = f"/tmp/cyber_catalog_{timestamp}.json"
        
        logger.info(f"🔍 BULLY! Starting cyber data cataloging: {folder_path}")
        logger.info(f"📝 JSON output: {json_path}")
        logger.info(f"⚙️  Validate-only mode: {config.validate_only}")
        
        try:
            # Scan folder for files
            files = await self._scan_folder(folder_path, config)
            logger.info(f"📁 Found {len(files)} files to catalog")
            
            if not files:
                return {
                    "success": False,
                    "error": "No files found to catalog",
                    "json_path": json_path
                }
            
            # Process files in batches
            entries = []
            total_files = len(files)
            
            for i in range(0, total_files, config.batch_size):
                batch = files[i:i + config.batch_size]
                batch_num = (i // config.batch_size) + 1
                total_batches = (total_files + config.batch_size - 1) // config.batch_size
                
                logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
                
                for file_path in batch:
                    try:
                        entry = await self._catalog_file(file_path, config)
                        if entry:
                            entries.append(entry)
                    except Exception as e:
                        logger.error(f"❌ Failed to catalog {file_path}: {e}")
                        continue
            
            # Create catalog JSON
            catalog = CyberCatalogJSON(
                catalog_version="1.0",
                created_at=datetime.now(),
                source_folder=folder_path,
                total_files=len(entries),
                entries=entries,
                processing_time_seconds=time.time() - start_time,
                config_used=config.dict() if config else None
            )
            
            # Save to JSON file
            catalog.to_json_file(json_path)
            logger.info(f"💾 Catalog saved to: {json_path}")
            
            # Validate catalog
            validation = catalog.validate_catalog()
            
            result = {
                "success": True,
                "json_path": json_path,
                "entry_count": len(entries),
                "validation": validation.dict(),
                "processing_time_seconds": time.time() - start_time,
                "ready_for_db": validation.is_valid and not config.validate_only
            }
            
            if validation.warnings:
                logger.warning(f"⚠️  Validation warnings: {validation.warnings}")
            if validation.errors:
                logger.error(f"❌ Validation errors: {validation.errors}")
            
            logger.info(f"✅ Cataloging complete: {len(entries)} entries in {result['processing_time_seconds']:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Cataloging failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "json_path": json_path
            }
    
    async def _scan_folder(
        self,
        folder_path: str,
        config: CyberCatalogConfig
    ) -> List[str]:
        """
        Scan folder recursively for files to catalog
        
        **BULLY!** Recursive folder scanning!
        """
        files = []
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.error(f"❌ Folder does not exist: {folder_path}")
            return files
        
        if not folder.is_dir():
            logger.error(f"❌ Path is not a directory: {folder_path}")
            return files
        
        max_size_bytes = config.max_file_size_mb * 1024 * 1024
        
        for file_path in folder.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue
            
            # Check exclude patterns
            should_exclude = False
            for pattern in config.exclude_patterns:
                if pattern in str(file_path):
                    should_exclude = True
                    break
            
            if should_exclude:
                continue
            
            # Check file size
            try:
                file_size = file_path.stat().st_size
                if file_size > max_size_bytes:
                    logger.debug(f"⏭️  Skipping large file: {file_path} ({file_size / 1024 / 1024:.1f} MB)")
                    continue
            except Exception as e:
                logger.warning(f"⚠️  Could not check size for {file_path}: {e}")
                continue
            
            files.append(str(file_path))
        
        return files
    
    async def _catalog_file(
        self,
        file_path: str,
        config: CyberCatalogConfig
    ) -> Optional[CyberDataCatalogEntry]:
        """
        Catalog a single file
        
        **BULLY!** Extract, categorize, and tag!
        """
        try:
            path = Path(file_path)
            
            # Basic file info
            file_name = path.name
            file_size = path.stat().st_size
            file_extension = path.suffix.lower()
            file_modified_at = datetime.fromtimestamp(path.stat().st_mtime)
            
            # Calculate file hash
            file_hash = calculate_file_hash(file_path)
            
            # Extract text content (if possible)
            content_preview = None
            full_text = None
            
            try:
                # Try to extract text based on file type
                # Use smart sampling for files larger than threshold (default: 1MB)
                # Even 1MB files have ~500K characters - too much for LLM!
                sampling_threshold_bytes = config.llm_sampling_threshold_mb * 1024 * 1024
                use_smart_sampling = file_size > sampling_threshold_bytes
                
                # Direct text extraction (fast, no processing needed)
                # SQL and database dump files are text-based and can be read directly
                if file_extension in ['.txt', '.md', '.json', '.csv', '.log', '.sql', '.dump', '.backup']:
                    if use_smart_sampling:
                        # Smart sampling for files > 1MB - don't load entire file
                        # This includes ALL files > 1MB, not just huge ones!
                        full_text = await self._smart_sample_text_file(
                            file_path, 
                            config.llm_sample_size_chars,
                            config.llm_sampling_strategy
                        )
                    else:
                        # Small files < 1MB - read entire file
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            full_text = f.read()
                
                # Binary SQLite database files - try to extract schema/data
                elif file_extension in ['.sqlite', '.db']:
                    # Try to read as text first (might be SQL dump with .db extension)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            full_text = f.read()
                    except (UnicodeDecodeError, Exception):
                        # It's a binary SQLite file - extract schema info
                        full_text = await self._extract_sqlite_schema(file_path)
                
                # Document processor supported formats (full processing)
                elif file_extension in ['.pdf', '.docx', '.html', '.htm', '.epub', '.eml', '.srt', '.zip']:
                    # Map extension to document type
                    doc_type_mapping = {
                        '.pdf': 'pdf',
                        '.docx': 'docx',
                        '.html': 'html',
                        '.htm': 'html',
                        '.epub': 'epub',
                        '.eml': 'eml',
                        '.srt': 'srt',
                        '.zip': 'zip'
                    }
                    
                    doc_type = doc_type_mapping.get(file_extension)
                    if doc_type:
                        processing_result = await self.document_processor.process_document(str(file_path), doc_type)
                        if processing_result.chunks:
                            full_text = " ".join(chunk.content for chunk in processing_result.chunks)
                
                # Binary/unstructured files (no text extraction, but still cataloged)
                else:
                    logger.debug(f"⚠️  Unsupported file type for text extraction: {file_extension}")
                    full_text = None  # Will still catalog with metadata only
                
                # Get preview
                if full_text:
                    content_preview = full_text[:500]
            except Exception as e:
                logger.debug(f"⚠️  Could not extract text from {file_path}: {e}")
            
            # Categorize and extract metadata using LLM
            # ALL files get LLM categorization if enabled (even large ones use sampling)
            # Only skip LLM for files larger than max_llm_file_size_mb (default: 100MB)
            breach_type = None
            affected_entities = []
            severity = None
            discovered_date = None
            breach_vector = None
            tags = []
            categories = []
            llm_confidence = None
            
            # Skip LLM only for extremely large files (default: > 100MB)
            # Files 1MB-100MB use smart sampling automatically
            max_llm_size_bytes = config.max_llm_file_size_mb * 1024 * 1024
            should_use_llm = config.enable_llm_categorization and full_text and file_size <= max_llm_size_bytes
            
            if should_use_llm:
                # Use sampled content (already limited by smart sampling for files > 1MB)
                # For files < 1MB, truncate to sample size
                sample_text = full_text[:config.llm_sample_size_chars] if full_text else ""
                
                logger.debug(
                    f"📊 LLM Processing: {file_name} "
                    f"({file_size / 1024 / 1024:.2f} MB) → "
                    f"sample: {len(sample_text)} chars"
                )
                
                categorization_result = await self._categorize_with_llm(
                    file_name,
                    sample_text,
                    config
                )
                
                breach_type = categorization_result.get("breach_type")
                affected_entities = categorization_result.get("affected_entities", [])
                severity = categorization_result.get("severity")
                discovered_date = categorization_result.get("discovered_date")
                breach_vector = categorization_result.get("breach_vector")
                tags = categorization_result.get("tags", [])
                categories = categorization_result.get("categories", [])
                llm_confidence = categorization_result.get("confidence")
            
            # Extract tags from filename if enabled
            if config.auto_tag_by_filename:
                filename_tags = self._extract_tags_from_filename(file_name)
                tags.extend(filename_tags)
            
            # Apply custom tags
            tags.extend(config.custom_tags)
            
            # Remove duplicates
            tags = list(dict.fromkeys(tags))
            
            # Extract entities if enabled
            if config.enable_entity_extraction and full_text:
                extracted_entities = await self._extract_entities(full_text[:2000])
                affected_entities.extend(extracted_entities)
                affected_entities = list(dict.fromkeys(affected_entities))  # Deduplicate
            
            # Create catalog entry
            entry = CyberDataCatalogEntry(
                file_path=str(file_path),
                file_name=file_name,
                file_hash=file_hash,
                file_size=file_size,
                file_extension=file_extension,
                breach_type=breach_type,
                affected_entities=affected_entities[:10],  # Limit to 10
                severity=severity,
                discovered_date=discovered_date,
                breach_vector=breach_vector,
                tags=tags[:20],  # Limit to 20 tags
                categories=categories[:10],  # Limit to 10 categories
                content_preview=content_preview,
                llm_confidence=llm_confidence,
                file_modified_at=file_modified_at
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"❌ Failed to catalog file {file_path}: {e}")
            return None
    
    async def _categorize_with_llm(
        self,
        file_name: str,
        content: str,
        config: CyberCatalogConfig
    ) -> Dict[str, Any]:
        """
        Use LLM to categorize cyber data
        
        **BULLY!** LLM-powered categorization!
        """
        try:
            # Get model for categorization
            model = config.categorization_model
            if not model:
                from services.settings_service import settings_service
                enabled_models = await settings_service.get_enabled_models()
                model = enabled_models[0] if enabled_models else "openai/gpt-4o-mini"
            
            prompt = f"""Analyze this cyber/breach data file and extract structured information.

FILE NAME: "{file_name}"
CONTENT PREVIEW: "{content[:2000]}"

Extract the following information:
1. Breach type (one of: {', '.join(self.breach_types)})
2. Affected entities (company/organization names)
3. Severity (critical, high, medium, low, unknown)
4. Discovered date (if mentioned)
5. Attack vector (phishing, vulnerability, insider, etc.)
6. Relevant tags (keywords)
7. Categories (from: {', '.join(self.cyber_categories)})

SPECIAL HANDLING FOR DATABASE FILES:
- If this is a SQL dump or database file, look for:
  - Table names that might indicate data types (users, passwords, emails, payments, etc.)
  - Database names that might indicate the source organization
  - INSERT statements with data that might indicate what was breached
  - Schema information that reveals data sensitivity

Respond with ONLY valid JSON:
{{
    "breach_type": "data_breach",
    "affected_entities": ["Company Inc"],
    "severity": "high",
    "discovered_date": "2024-01-15T10:30:00Z",
    "breach_vector": "vulnerability",
    "tags": ["credentials", "pii", "database", "sql"],
    "categories": ["data_breach", "financial"],
    "confidence": 0.85,
    "reasoning": "brief explanation"
}}"""

            response = await self.openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            result = json.loads(response_content)
            
            # Parse discovered_date if present
            if result.get("discovered_date") and isinstance(result["discovered_date"], str):
                try:
                    result["discovered_date"] = datetime.fromisoformat(
                        result["discovered_date"].replace('Z', '+00:00')
                    )
                except:
                    result["discovered_date"] = None
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM categorization failed: {e}")
            return {}
    
    async def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entity names (companies/organizations) from text
        
        **BULLY!** Entity extraction!
        """
        entities = []
        
        # Simple pattern-based extraction (company names often follow patterns)
        import re
        
        # Pattern: Company Name Inc/Corp/Ltd/etc.
        company_pattern = r'\b([A-Z][a-zA-Z\s&]+(?:Inc|Corp|Corporation|LLC|Ltd|Limited|Company|Co\.|Systems|Technologies|Group|Holdings))\b'
        matches = re.findall(company_pattern, text)
        entities.extend(matches)
        
        # Pattern: Standalone capitalized names (likely companies)
        capitalized_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b)'
        matches = re.findall(capitalized_pattern, text)
        
        # Filter out common false positives
        common_words = {"The", "This", "That", "There", "These", "Those", "When", "Where", "What", "Which"}
        filtered = [m for m in matches if not any(m.startswith(word) for word in common_words)]
        entities.extend(filtered[:5])  # Limit to 5
        
        return list(dict.fromkeys(entities))[:10]  # Deduplicate and limit
    
    def _extract_tags_from_filename(self, filename: str) -> List[str]:
        """
        Extract potential tags from filename
        
        **BULLY!** Filename-based tagging!
        """
        tags = []
        
        # Common cyber-related keywords
        keywords = [
            "breach", "leak", "ransomware", "phishing", "malware",
            "credentials", "pii", "financial", "healthcare", "government",
            "2024", "2023", "2022", "dump", "data", "cyber",
            "sql", "database", "db", "sqlite", "mysql", "postgres"
        ]
        
        filename_lower = filename.lower()
        for keyword in keywords:
            if keyword in filename_lower:
                tags.append(keyword)
        
        return tags
    
    async def _smart_sample_text_file(
        self,
        file_path: str,
        sample_size: int,
        strategy: str = "smart"
    ) -> str:
        """
        Smart sampling for large text files
        
        **BULLY!** Efficient sampling without loading entire file into memory!
        
        Strategies:
        - 'beginning': First N characters
        - 'smart': Beginning + end + key sections (for SQL: schema + sample data)
        - 'full': Full file (only for small files)
        
        Returns:
            Sampled text content
        """
        try:
            file_size = Path(file_path).stat().st_size
            file_extension = Path(file_path).suffix.lower()
            
            # For very large files, use beginning-only to avoid memory issues
            if file_size > 100 * 1024 * 1024:  # > 100MB
                strategy = "beginning"
            
            if strategy == "beginning":
                # Read only the first N characters
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(sample_size)
            
            elif strategy == "smart":
                # Smart sampling: beginning + end + key sections
                sample_chunk_size = sample_size // 3  # Split into 3 parts
                
                samples = []
                
                # Part 1: Beginning (first N chars)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    samples.append(f"=== BEGINNING ===\n{f.read(sample_chunk_size)}")
                
                # Part 2: End (last N chars) - seek to end
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # Seek to end - sample_size bytes from end
                        f.seek(max(0, file_size - sample_chunk_size))
                        samples.append(f"\n=== END ===\n{f.read(sample_chunk_size)}")
                except Exception:
                    pass  # Skip if can't seek
                
                # Part 3: Key sections (for SQL files, find CREATE TABLE statements)
                if file_extension in ['.sql', '.dump', '.backup']:
                    key_sections = await self._sample_sql_key_sections(file_path, sample_chunk_size)
                    if key_sections:
                        samples.append(f"\n=== KEY SECTIONS ===\n{key_sections}")
                
                return "\n".join(samples)[:sample_size]
            
            else:
                # Fallback: beginning only
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(sample_size)
                    
        except Exception as e:
            logger.warning(f"⚠️  Smart sampling failed for {file_path}: {e}")
            # Fallback: read beginning only
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(min(sample_size, 10000))  # Limit fallback
            except:
                return f"Large file ({file_size / 1024 / 1024:.1f} MB) - sampling failed"
    
    async def _sample_sql_key_sections(
        self,
        file_path: str,
        max_chars: int
    ) -> str:
        """
        Extract key sections from SQL dump files
        
        **BULLY!** Smart SQL sampling - find CREATE TABLE and database names!
        
        Extracts:
        - CREATE TABLE statements (schema)
        - Database name declarations
        - First few INSERT statements per table
        """
        try:
            import re
            
            key_sections = []
            chars_used = 0
            
            # Read file in chunks to find key sections
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first chunk to find CREATE TABLE statements
                chunk = f.read(min(100000, max_chars * 2))  # Read more to find patterns
                
                # Find CREATE TABLE statements (schema)
                create_table_pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(\w+)`?.*?(?=CREATE\s+TABLE|INSERT|$)'
                matches = re.finditer(create_table_pattern, chunk, re.IGNORECASE | re.DOTALL)
                
                for match in list(matches)[:5]:  # Limit to 5 tables
                    table_section = match.group(0)[:500]  # Limit each table schema
                    if chars_used + len(table_section) < max_chars:
                        key_sections.append(f"Schema: {table_section}")
                        chars_used += len(table_section)
                
                # Find database name
                db_pattern = r"(?:CREATE\s+DATABASE|USE)\s+`?(\w+)`?"
                db_match = re.search(db_pattern, chunk, re.IGNORECASE)
                if db_match:
                    key_sections.insert(0, f"Database: {db_match.group(1)}")
                
                # Find first INSERT statement for each table
                insert_pattern = r'INSERT\s+INTO\s+`?(\w+)`?.*?(?=INSERT\s+INTO|$)'
                insert_matches = re.finditer(insert_pattern, chunk[:50000], re.IGNORECASE | re.DOTALL)
                
                seen_tables = set()
                for match in list(insert_matches)[:3]:  # Limit to 3 INSERT samples
                    table_name = match.group(1)
                    if table_name not in seen_tables:
                        insert_section = match.group(0)[:300]  # Limit each INSERT
                        if chars_used + len(insert_section) < max_chars:
                            key_sections.append(f"Sample data from {table_name}: {insert_section}")
                            chars_used += len(insert_section)
                            seen_tables.add(table_name)
            
            return "\n\n".join(key_sections)[:max_chars]
            
        except Exception as e:
            logger.debug(f"⚠️  SQL key section sampling failed: {e}")
            return ""
    
    async def _extract_sqlite_schema(self, file_path: str) -> str:
        """
        Extract schema information from SQLite database file
        
        **BULLY!** Extract schema from binary SQLite files!
        
        Returns:
            Schema information as text (table names, column names, etc.)
        """
        try:
            import sqlite3
            
            schema_text = []
            schema_text.append(f"SQLite Database: {Path(file_path).name}\n")
            schema_text.append("=" * 60 + "\n\n")
            
            # Connect to SQLite database
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_text.append(f"Tables found: {len(tables)}\n\n")
            
            # Extract schema for each table
            for (table_name,) in tables[:10]:  # Limit to 10 tables
                schema_text.append(f"Table: {table_name}\n")
                schema_text.append("-" * 40 + "\n")
                
                # Get CREATE TABLE statement
                cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
                create_stmt = cursor.fetchone()
                if create_stmt and create_stmt[0]:
                    schema_text.append(f"Schema: {create_stmt[0]}\n")
                
                # Get row count
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                    row_count = cursor.fetchone()[0]
                    schema_text.append(f"Rows: {row_count}\n")
                    
                    # Get sample data (first 3 rows)
                    if row_count > 0:
                        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                        rows = cursor.fetchall()
                        schema_text.append(f"Sample data:\n")
                        for row in rows:
                            schema_text.append(f"  {row}\n")
                except Exception as e:
                    schema_text.append(f"  (Could not read table data: {e})\n")
                
                schema_text.append("\n")
            
            conn.close()
            
            return "\n".join(schema_text)
            
        except Exception as e:
            logger.warning(f"⚠️  Could not extract SQLite schema from {file_path}: {e}")
            return f"SQLite database file (binary format - schema extraction failed: {e})"
    
    async def close(self):
        """Close the service"""
        if self.openrouter_client:
            await self.openrouter_client.close()
        logger.info("✅ Cyber Catalog Service closed")


# Global instance
cyber_catalog_service = CyberCatalogService()


async def get_cyber_catalog_service() -> CyberCatalogService:
    """Get the global cyber catalog service instance"""
    return cyber_catalog_service

