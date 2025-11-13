"""
Cyber Data Catalog Models
Pydantic models for cataloging cyber/breach data with JSON-first validation

**BULLY!** Structured cataloging of unstructured cyber data!
"""

import hashlib
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from pathlib import Path


class CyberDataCatalogEntry(BaseModel):
    """
    Single catalog entry for a cyber/breach data file
    
    **BULLY!** Structured catalog entry with validation!
    """
    file_path: str = Field(..., description="Full path to the cataloged file")
    file_name: str = Field(..., description="Filename only")
    file_hash: str = Field(..., description="SHA256 hash of file for deduplication")
    file_size: int = Field(..., description="File size in bytes")
    file_extension: str = Field(..., description="File extension (e.g., '.txt', '.json')")
    
    # Categorization fields
    breach_type: Optional[str] = Field(None, description="Type of breach/cyber incident")
    affected_entities: List[str] = Field(default_factory=list, description="List of affected companies/organizations")
    severity: Optional[str] = Field(None, description="Severity level: critical, high, medium, low")
    discovered_date: Optional[datetime] = Field(None, description="Date breach was discovered")
    breach_vector: Optional[str] = Field(None, description="Attack vector (e.g., 'phishing', 'vulnerability', 'insider')")
    
    # Tagging
    tags: List[str] = Field(default_factory=list, description="Relevant tags for the data")
    categories: List[str] = Field(default_factory=list, description="Content categories")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional extracted metadata")
    content_preview: Optional[str] = Field(None, description="First 500 chars of content for context")
    llm_confidence: Optional[float] = Field(None, description="LLM confidence score (0.0-1.0)")
    
    # Timestamps
    cataloged_at: datetime = Field(default_factory=datetime.now, description="When this entry was cataloged")
    file_modified_at: Optional[datetime] = Field(None, description="File modification timestamp")
    
    @validator('breach_type')
    def validate_breach_type(cls, v):
        """Validate breach type against known types"""
        if v is None:
            return v
        valid_types = [
            "data_breach", "ransomware", "phishing", "malware", 
            "ddos", "insider_threat", "vulnerability", "credential_stuffing",
            "supply_chain", "zero_day", "other"
        ]
        if v not in valid_types:
            # Allow custom types but log warning
            return v
        return v
    
    @validator('severity')
    def validate_severity(cls, v):
        """Validate severity level"""
        if v is None:
            return v
        valid_severities = ["critical", "high", "medium", "low", "unknown"]
        if v not in valid_severities:
            return "unknown"
        return v
    
    @validator('file_hash')
    def validate_hash_format(cls, v):
        """Validate hash is SHA256 format (64 hex chars)"""
        if len(v) != 64:
            raise ValueError(f"Invalid hash format: expected 64 hex chars, got {len(v)}")
        try:
            int(v, 16)  # Validate hex
        except ValueError:
            raise ValueError(f"Invalid hash format: not valid hex")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/data/breaches/2024/company_breach.json",
                "file_name": "company_breach.json",
                "file_hash": "a1b2c3d4e5f6...",
                "file_size": 102400,
                "file_extension": ".json",
                "breach_type": "data_breach",
                "affected_entities": ["Company Inc", "Subsidiary Corp"],
                "severity": "high",
                "discovered_date": "2024-01-15T10:30:00Z",
                "breach_vector": "vulnerability",
                "tags": ["credentials", "pii", "financial"],
                "categories": ["data_breach", "financial"],
                "metadata": {"records_exposed": 1000000},
                "content_preview": "Company breach notification...",
                "llm_confidence": 0.85,
                "cataloged_at": "2024-01-20T08:00:00Z"
            }
        }


class CyberCatalogValidationResult(BaseModel):
    """
    Validation result for catalog JSON
    
    **BULLY!** Validate first, commit second!
    """
    is_valid: bool = Field(..., description="Whether JSON is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    entry_count: int = Field(..., description="Number of catalog entries")
    duplicate_hashes: List[str] = Field(default_factory=list, description="File hashes that appear multiple times")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "errors": [],
                "warnings": ["Missing severity for 3 entries"],
                "entry_count": 150,
                "duplicate_hashes": []
            }
        }


class CyberCatalogJSON(BaseModel):
    """
    Root JSON structure for cyber data catalog
    
    **BULLY!** Complete catalog file with validation!
    """
    catalog_version: str = Field(default="1.0", description="Catalog schema version")
    created_at: datetime = Field(default_factory=datetime.now, description="When catalog was created")
    source_folder: str = Field(..., description="Folder path that was cataloged")
    total_files: int = Field(..., description="Total number of files cataloged")
    entries: List[CyberDataCatalogEntry] = Field(default_factory=list, description="Catalog entries")
    
    # Processing metadata
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to catalog")
    config_used: Optional[Dict[str, Any]] = Field(None, description="Configuration used for cataloging")
    
    def validate_catalog(self) -> CyberCatalogValidationResult:
        """
        Validate the catalog structure and entries
        
        **BULLY!** Trust but verify!
        """
        errors = []
        warnings = []
        
        # Check total_files matches entries count
        if self.total_files != len(self.entries):
            warnings.append(
                f"total_files ({self.total_files}) doesn't match entries count ({len(self.entries)})"
            )
        
        # Check for duplicate file hashes
        seen_hashes = {}
        duplicate_hashes = []
        for i, entry in enumerate(self.entries):
            if entry.file_hash in seen_hashes:
                duplicate_hashes.append(entry.file_hash)
                errors.append(
                    f"Duplicate file hash at index {i}: {entry.file_path} "
                    f"(same as {seen_hashes[entry.file_hash]})"
                )
            else:
                seen_hashes[entry.file_hash] = entry.file_path
        
        # Check for entries missing critical fields
        missing_breach_type = sum(1 for e in self.entries if not e.breach_type)
        if missing_breach_type > 0:
            warnings.append(f"{missing_breach_type} entries missing breach_type")
        
        missing_severity = sum(1 for e in self.entries if not e.severity)
        if missing_severity > 0:
            warnings.append(f"{missing_severity} entries missing severity")
        
        is_valid = len(errors) == 0
        
        return CyberCatalogValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            entry_count=len(self.entries),
            duplicate_hashes=duplicate_hashes
        )
    
    @classmethod
    def from_json_file(cls, json_path: str) -> 'CyberCatalogJSON':
        """
        Load catalog from JSON file and validate
        
        **BULLY!** Load and validate in one step!
        """
        import json
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse datetime strings
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        
        # Parse entry datetimes
        for entry in data.get('entries', []):
            for date_field in ['discovered_date', 'cataloged_at', 'file_modified_at']:
                if date_field in entry and isinstance(entry[date_field], str):
                    entry[date_field] = datetime.fromisoformat(entry[date_field].replace('Z', '+00:00'))
        
        return cls(**data)
    
    def to_json_file(self, json_path: str) -> None:
        """
        Save catalog to JSON file
        
        **BULLY!** Export catalog to JSON!
        """
        import json
        
        # Convert to dict with datetime serialization
        data = self.dict()
        
        # Serialize datetime objects
        if isinstance(data.get('created_at'), datetime):
            data['created_at'] = data['created_at'].isoformat()
        
        # Serialize entry datetimes
        for entry in data.get('entries', []):
            for date_field in ['discovered_date', 'cataloged_at', 'file_modified_at']:
                if date_field in entry and isinstance(entry[date_field], datetime):
                    entry[date_field] = entry[date_field].isoformat()
        
        # Ensure directory exists
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    class Config:
        json_schema_extra = {
            "example": {
                "catalog_version": "1.0",
                "created_at": "2024-01-20T08:00:00Z",
                "source_folder": "/data/breaches/2024",
                "total_files": 150,
                "entries": [
                    {
                        "file_path": "/data/breaches/2024/breach1.json",
                        "file_name": "breach1.json",
                        "file_hash": "a1b2c3...",
                        "breach_type": "data_breach",
                        "severity": "high"
                    }
                ],
                "processing_time_seconds": 45.2
            }
        }


class CyberCatalogConfig(BaseModel):
    """
    Configuration for cyber catalog processing
    
    **BULLY!** Configure your cataloging campaign!
    """
    # Folder settings
    watch_folders: List[str] = Field(default_factory=list, description="Folders to scan recursively")
    exclude_patterns: List[str] = Field(
        default_factory=lambda: [".git", "__pycache__", ".DS_Store"],
        description="File/folder patterns to exclude"
    )
    max_file_size_mb: int = Field(default=100, description="Skip files larger than this (MB)")
    
    # Processing settings
    validate_only: bool = Field(default=True, description="JSON-only mode (no database)")
    output_json_path: Optional[str] = Field(None, description="Path to output JSON file")
    batch_size: int = Field(default=10, description="Files to process per batch")
    
    # LLM processing limits (cost optimization)
    max_llm_file_size_mb: int = Field(
        default=100, 
        description="Skip LLM processing for files larger than this (MB). Files 1MB-100MB use smart sampling."
    )
    llm_sample_size_chars: int = Field(
        default=5000, 
        description="Character limit for LLM context (sample size). Even 1MB files use this limit."
    )
    llm_sampling_strategy: str = Field(
        default="smart",
        description="Sampling strategy: 'beginning', 'smart', 'full' (smart = beginning + end + key sections)"
    )
    llm_sampling_threshold_mb: float = Field(
        default=1.0,
        description="Files larger than this (MB) use smart sampling instead of full file read"
    )
    
    # Categorization settings
    enable_llm_categorization: bool = Field(default=True, description="Use LLM for categorization")
    enable_entity_extraction: bool = Field(default=True, description="Extract affected entities")
    categorization_model: Optional[str] = Field(None, description="LLM model to use (defaults to FAST_MODEL)")
    
    # Tagging settings
    custom_tags: List[str] = Field(default_factory=list, description="Custom tags to apply")
    auto_tag_by_filename: bool = Field(default=True, description="Extract tags from filenames")
    
    class Config:
        json_schema_extra = {
            "example": {
                "watch_folders": ["/data/breaches", "/data/cyber"],
                "exclude_patterns": [".git", ".DS_Store"],
                "max_file_size_mb": 100,
                "validate_only": True,
                "output_json_path": "/tmp/catalog_validation.json",
                "batch_size": 10,
                "enable_llm_categorization": True,
                "enable_entity_extraction": True
            }
        }


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA256 hash of file for deduplication
    
    **BULLY!** Content-based deduplication!
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b''):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

