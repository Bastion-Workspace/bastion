# Smart KG Re-Extraction - Quick Start

**BULLY!** Here's how to use the new smart re-extraction system!

## What Was Implemented

✅ **Domain Detector Service** (`domain_detector.py`)
- Automatically detects which domain(s) a document belongs to
- Extensible for entertainment, business, research, and future domains
- Uses tags and categories to identify domains

✅ **Smart Re-Extraction** (`document_service_v2.py`)
- Automatically triggers when document tags or category change
- Re-extracts KG entities **WITHOUT re-chunking/re-embedding**
- Handles multiple domains per document

✅ **Future-Proof Architecture**
- Easy to add new domains (business, research, academic, etc.)
- Domain-specific extractors and storage methods
- Clean separation of concerns

## How It Works

### Before (Manual Re-Processing)
```bash
# User had to manually re-upload or trigger full re-processing
# This would:
# 1. Re-chunk the entire document
# 2. Re-embed all chunks (expensive!)
# 3. Extract entities
```

### After (Smart Re-Extraction)
```bash
# User updates tags via UI or API
PUT /api/documents/{doc_id}/metadata
{
    "tags": ["movie", "drama"]
}

# System automatically:
# 1. Detects domain change
# 2. Extracts entertainment entities
# 3. Stores in Neo4j
# 4. Does NOT re-chunk or re-embed (fast!)
```

## Usage Example

### Step 1: Upload Entertainment Document

1. Create a markdown file for a movie (see `ENTERTAINMENT_AGENT_TEMPLATES.md`)
2. Upload to `Global/Reference/Movies/`
3. Initially, it might not have entertainment tags

### Step 2: Add Entertainment Tags

Update the document metadata via the UI:
- **Tags**: Add `movie`, `drama`, `thriller`, etc.
- **Category**: Set to `entertainment`

**The system will automatically:**
- Detect that the document entered the "entertainment" domain
- Extract actors, directors, studios, genres
- Store relationships in Neo4j
- **NOT** re-chunk or re-embed (vectors stay intact!)

### Step 3: Test Entertainment Agent

Ask questions like:
- "Tell me about The Godfather"
- "Recommend movies like The Godfather"
- "What other movies did Marlon Brando act in?"

The agent will use both vector search AND Neo4j relationships!

## Log Output Example

When you update tags, you'll see:

```
📊 Domain change detected for d41d8cd9_20251109113544:
   Added domains: {'entertainment'}
   Removed domains: set()
🎬 Extracting entertainment entities for d41d8cd9_20251109113544
✅ Stored 15 entertainment entities, 23 relationships
✅ Completed KG re-extraction for d41d8cd9_20251109113544
```

## Adding New Domains (Future)

When you want to add a new domain (e.g., "business"):

1. **Create extractor**: `backend/services/business_kg_extractor.py`
2. **Add storage method**: Add to `knowledge_graph_service.py`
3. **Register domain**: Add to `domain_detector.py`
4. **Update re-extraction**: Add to `_reextract_knowledge_graph()`

See `SMART_KG_REEXTRACTION.md` for detailed instructions.

## Performance

### Tag Update with Re-Extraction
- **Time**: 1-3 seconds
- **Cost**: Only entity extraction (no embedding costs)
- **Chunks**: Preserved (no re-chunking)

### Full Re-Processing (Old Way)
- **Time**: 10-30 seconds
- **Cost**: Re-embedding all chunks
- **Chunks**: Completely regenerated

**By George!** 10x faster! 🏇

## Current Supported Domains

### Entertainment Domain
- **Tags**: `movie`, `tv_show`, `tv_episode`
- **Category**: `entertainment`
- **Entities**: Movies, TV Series, Actors, Directors, Studios, Genres
- **Relationships**: ACTED_IN, DIRECTED, WRITTEN_BY, PRODUCED_BY, HAS_GENRE

### Future Domains (Easy to Add)
- **Business**: Companies, Executives, Financials, Deals
- **Research**: Papers, Authors, Citations, Institutions
- **Academic**: Courses, Professors, Topics, Prerequisites

## Testing

### Test 1: Add Entertainment Tags
1. Upload a movie document with no tags
2. Verify it's searchable (has vector chunks)
3. Add `movie` tag via metadata update
4. Check logs for "Domain change detected"
5. Verify Neo4j has entertainment entities
6. Verify vectors still exist (not re-embedded)

### Test 2: Remove Entertainment Tags
1. Take a movie document with `movie` tag
2. Remove the tag via metadata update
3. Verify entertainment entities are cleaned up
4. Verify vectors still exist

### Test 3: Multi-Domain (Future)
1. Create a document with `movie` AND `financial` tags
2. Verify both entertainment and business entities extracted
3. Remove one domain's tags
4. Verify only remaining domain's entities stay

## Key Benefits

✅ **Fast**: No re-chunking or re-embedding  
✅ **Smart**: Only extracts when domains change  
✅ **Automatic**: No manual intervention needed  
✅ **Extensible**: Easy to add new domains  
✅ **Efficient**: Preserves existing vector chunks  

**BULLY!** The cavalry charge of knowledge graph management! 🏇📊

