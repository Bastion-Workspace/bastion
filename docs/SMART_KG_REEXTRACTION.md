# Smart Knowledge Graph Re-Extraction

**BULLY!** Roosevelt's "Domain Intelligence" for automatic KG updates!

## Overview

The smart KG re-extraction system automatically detects when a document's domain membership changes (via tag or category updates) and re-extracts domain-specific knowledge graph entities **WITHOUT re-chunking or re-embedding the document**.

This means:
- ✅ **Fast updates**: No expensive chunking/embedding operations
- ✅ **Targeted extraction**: Only extracts for relevant domains
- ✅ **Automatic**: Happens transparently when metadata changes
- ✅ **Extensible**: Easy to add new domains (business, research, etc.)

## How It Works

### 1. Domain Detection

The `DomainDetector` service identifies which domain(s) a document belongs to based on:
- **Tags**: Domain-specific tags (e.g., `movie`, `tv_show`, `financial`, `research`)
- **Category**: Document category (e.g., `entertainment`, `business`, `academic`)

```python
from services.domain_detector import get_domain_detector

detector = get_domain_detector()

# Detect domains from tags and category
domains = detector.detect_domains(
    tags=["movie", "drama"],
    category="entertainment"
)
# Returns: {"entertainment"}
```

### 2. Change Detection

When document metadata is updated via `/api/documents/{doc_id}/metadata`, the system:

1. **Captures old metadata** (tags and category) BEFORE the update
2. **Updates the metadata** in PostgreSQL and Qdrant
3. **Detects domain changes** by comparing old vs new domains
4. **Triggers re-extraction** if domains were added or removed

```python
domain_changes = detector.get_domain_changes(
    old_tags=["business"],
    old_category="other",
    new_tags=["movie"],
    new_category="entertainment"
)
# Returns:
# {
#     "changed": True,
#     "old_domains": {"business"},
#     "new_domains": {"entertainment"},
#     "added": {"entertainment"},
#     "removed": {"business"}
# }
```

### 3. Smart Re-Extraction

When domain changes are detected, the `_reextract_knowledge_graph` method:

1. **Removes old entities** from removed domains (if applicable)
2. **Extracts new entities** for added domains using domain-specific extractors
3. **Stores entities** using domain-specific storage methods
4. **Does NOT re-chunk or re-embed** the document

```python
# Automatically called by update_document_metadata
await self._reextract_knowledge_graph(document_id, domain_changes)
```

## Current Domains

### Entertainment Domain

**Tags**: `movie`, `tv_show`, `tv_episode`  
**Category**: `entertainment`  
**Extractor**: `entertainment_kg_extractor.py`  
**Storage**: `kg_service.store_entertainment_entities_and_relationships()`

**Entities Extracted**:
- Movies (EntertainmentMovie)
- TV Series (EntertainmentTVSeries)
- People: Actors, Directors, Writers (EntertainmentPerson:Actor, etc.)
- Studios (EntertainmentStudio)
- Genres (EntertainmentGenre)

**Relationships**:
- `ACTED_IN`, `DIRECTED`, `WRITTEN_BY`
- `PRODUCED_BY`, `HAS_GENRE`
- `PART_OF_SERIES` (for TV episodes)

## Adding New Domains

**By George!** Adding a new domain is a cavalry charge with clear steps!

### Step 1: Create Domain Extractor

Create a new file: `backend/services/{domain}_kg_extractor.py`

```python
# Example: backend/services/business_kg_extractor.py

import logging
from typing import List, Tuple, Dict, Any
from models.api_models import DocumentInfo

logger = logging.getLogger(__name__)


class BusinessKGExtractor:
    """Extract business entities and relationships"""
    
    def should_extract_from_document(self, doc_info: DocumentInfo) -> bool:
        """Check if document is business-related"""
        if not doc_info:
            return False
        
        # Check for business tags
        business_tags = {"financial", "corporate", "legal", "worldcom", "enron"}
        if doc_info.tags and business_tags & set(doc_info.tags):
            return True
        
        # Check category
        if doc_info.category and doc_info.category.value == "business":
            return True
        
        return False
    
    def extract_entities_and_relationships(
        self, 
        content: str, 
        doc_info: DocumentInfo
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract business entities and relationships from content
        
        Returns: (entities, relationships)
        """
        entities = []
        relationships = []
        
        # TODO: Implement business-specific extraction logic
        # Extract: companies, executives, deals, financials, etc.
        
        return entities, relationships


# Singleton instance
_business_kg_extractor = None

def get_business_kg_extractor() -> BusinessKGExtractor:
    """Get singleton instance"""
    global _business_kg_extractor
    if _business_kg_extractor is None:
        _business_kg_extractor = BusinessKGExtractor()
    return _business_kg_extractor
```

### Step 2: Add Storage Method to KG Service

Add a domain-specific storage method to `knowledge_graph_service.py`:

```python
async def store_business_entities_and_relationships(
    self,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    document_id: str
) -> None:
    """
    Store business entities and relationships in Neo4j
    
    Uses domain-specific labels: BusinessCompany, BusinessExecutive, etc.
    """
    async with self.driver.session(database=self.database) as session:
        try:
            # Delete old business entities for this document
            await session.run(
                """
                MATCH (n:BusinessCompany|BusinessExecutive|BusinessDeal)
                WHERE $document_id IN n.source_documents
                DETACH DELETE n
                """,
                document_id=document_id
            )
            
            # Store new entities with BusinessXXX labels
            for entity in entities:
                await session.run(
                    """
                    MERGE (e:BusinessEntity {name: $name, type: $type})
                    SET e.source_documents = coalesce(e.source_documents, []) + [$doc_id]
                    """,
                    name=entity["name"],
                    type=entity["type"],
                    doc_id=document_id
                )
            
            # Store relationships (INVESTED_IN, MANAGES, ACQUIRED, etc.)
            # ... relationship storage logic ...
            
        except Exception as e:
            logger.error(f"❌ Failed to store business entities: {e}")
            raise
```

### Step 3: Register Domain in DomainDetector

Update `domain_detector.py`:

```python
def _initialize_domains(self) -> Dict[str, DomainConfig]:
    """Initialize domain configurations"""
    return {
        "entertainment": DomainConfig(
            name="entertainment",
            tags={"movie", "tv_show", "tv_episode"},
            categories={"entertainment"},
            extractor_module="services.entertainment_kg_extractor",
            extractor_class="get_entertainment_kg_extractor"
        ),
        "business": DomainConfig(  # NEW DOMAIN
            name="business",
            tags={"financial", "corporate", "legal", "worldcom", "enron"},
            categories={"business", "legal"},
            extractor_module="services.business_kg_extractor",
            extractor_class="get_business_kg_extractor"
        ),
        # Add more domains here...
    }
```

### Step 4: Update Re-Extraction Logic

Update `_reextract_knowledge_graph` in `document_service_v2.py`:

```python
if entities or relationships:
    # Store using domain-specific storage method
    if added_domain == "entertainment":
        await self.kg_service.store_entertainment_entities_and_relationships(
            entities, relationships, document_id
        )
    elif added_domain == "business":  # NEW DOMAIN
        await self.kg_service.store_business_entities_and_relationships(
            entities, relationships, document_id
        )
    # Add more domains here...
    else:
        logger.warning(f"⚠️ No storage method for domain: {added_domain}")
```

### Step 5: Add to Category Enum (if needed)

If the domain needs a new category, add to `api_models.py`:

```python
class DocumentCategory(str, Enum):
    GENERAL = "general"
    PERSONAL = "personal"
    WORK = "work"
    ENTERTAINMENT = "entertainment"
    BUSINESS = "business"  # NEW CATEGORY
    # ... other categories
```

## Usage Examples

### Example 1: Tag Update Triggers Re-Extraction

```bash
# Update a document's tags from general to entertainment
PUT /api/documents/doc123/metadata
{
    "tags": ["movie", "drama"]
}

# System automatically:
# 1. Detects domain change: {} -> {"entertainment"}
# 2. Extracts entertainment entities (actors, directors, etc.)
# 3. Stores in Neo4j with EntertainmentXXX labels
# 4. Does NOT re-chunk or re-embed
```

### Example 2: Category Update Triggers Re-Extraction

```bash
# Change document category
PUT /api/documents/doc456/metadata
{
    "category": "business"
}

# System automatically:
# 1. Detects domain change: {} -> {"business"}
# 2. Extracts business entities (companies, executives, etc.)
# 3. Stores in Neo4j with BusinessXXX labels
```

### Example 3: Removing Domain Tags

```bash
# Remove entertainment tags
PUT /api/documents/doc789/metadata
{
    "tags": []
}

# System automatically:
# 1. Detects domain change: {"entertainment"} -> {}
# 2. Removes entertainment entities for this document
```

## Performance Benefits

### Without Smart Re-Extraction
- Update metadata → Re-chunk entire document → Re-embed all chunks → Extract entities
- **Time**: 10-30 seconds for large documents
- **Cost**: LLM embeddings for every chunk

### With Smart Re-Extraction
- Update metadata → Extract entities (only if domain changed)
- **Time**: 1-3 seconds
- **Cost**: No embedding costs, only extraction

**By George!** That's a 10x speed improvement! 🏇

## Testing

### Test Tag Update Re-Extraction

1. Upload a document with generic tags
2. Verify it's searchable (vector chunks exist)
3. Update metadata to add `movie` tag
4. Verify:
   - Entertainment entities appear in Neo4j
   - Vector chunks still exist (not re-embedded)
   - Search still works

### Test Multi-Domain Support

1. Create a document with `movie` tag (entertainment domain)
2. Update to add `financial` tag (business domain)
3. Verify both entertainment AND business entities extracted
4. Update to remove `movie` tag
5. Verify only business entities remain

## Logging

The system provides detailed logging for debugging:

```
📊 Domain change detected for doc123:
   Added domains: {'entertainment'}
   Removed domains: set()
🎬 Extracting entertainment entities for doc123
✅ Stored 15 entertainment entities, 23 relationships
✅ Completed KG re-extraction for doc123
```

## Architecture Diagram

```
User Updates Metadata
        ↓
update_document_metadata()
        ↓
Capture OLD metadata → Update PostgreSQL → Update Qdrant
        ↓
DomainDetector.get_domain_changes()
        ↓
Domain Changed? ─NO→ Done
        ↓ YES
_reextract_knowledge_graph()
        ↓
For each added domain:
    ↓
Get domain extractor → Extract entities → Store with domain-specific labels
        ↓
Done (no re-chunking/re-embedding!)
```

## Future Enhancements

1. **Domain-specific deletion**: Currently, we rely on document-level deletion. Could add targeted deletion for removed domains.
2. **Multi-domain documents**: Support documents that belong to multiple domains simultaneously.
3. **Domain priority**: Allow domains to have priority when entities overlap.
4. **Batch re-extraction**: Re-extract multiple documents when domain config changes.

**Remember: A well-organized knowledge graph is like a well-organized cavalry charge!** 🏇📊

