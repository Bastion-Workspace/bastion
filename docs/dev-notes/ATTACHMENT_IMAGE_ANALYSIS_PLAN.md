# Attachment Image Analysis - Unified CLIP-Based Architecture

## Executive Summary

**Core Principle**: Use CLIP embeddings as the universal semantic matching layer for all image analysis tasks. YOLO provides structure detection (bounding boxes), but CLIP provides semantic understanding - including user-defined custom objects.

**Key Insight**: ANY object that isn't a standard YOLO class (or even YOLO-detected objects that users have custom-tagged) should use CLIP-based similarity search against user's annotated reference library.

## Architecture Overview

```
User attaches image + asks question
    ↓
Research Agent detects attachment
    ↓
Route to Attachment Analysis Subgraph
    ↓
┌─────────────────────────────────────────────┐
│  Intent Classification (LLM)                │
│  "What is the user asking about?"           │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Parallel Processing                        │
│  1. YOLO Detection (structure)              │
│  2. CLIP Whole-Image Embedding (semantics)  │
│  3. Vision LLM Description (general)        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  CLIP-Based Matching Pipeline               │
│  For each region (YOLO) or whole image:     │
│  - Extract CLIP visual embedding            │
│  - Search user's custom collections:        │
│    * face_encodings (Qdrant)                │
│    * object_encodings (Qdrant)              │
│    * logo_encodings (Qdrant - NEW)          │
│    * tattoo_encodings (Qdrant - NEW)        │
│    * painting_encodings (Qdrant - NEW)      │
│  - Return best matches above threshold      │
└─────────────────────────────────────────────┘
    ↓
Format response with matches + context
```

## Unified CLIP Architecture

### Current State (What We Have)

1. **Face Detection & Matching** ✅
   - Image Vision Service: Face detection → 128-dim face encodings
   - Storage: Qdrant `face_encodings` collection + per-user collections
   - Matching: Cosine similarity search with 0.82 threshold

2. **YOLO Object Detection** ✅
   - Image Vision Service: YOLO v8 with COCO classes (80 classes)
   - Returns: Bounding boxes, class names, confidence scores
   - Storage: PostgreSQL `detected_objects` table

3. **CLIP Feature Extraction** ✅
   - Image Vision Service: `ExtractObjectFeatures` RPC
   - Returns: Visual embedding (512-dim), semantic embedding (512-dim), combined embedding
   - Used by: `object_encoding_service` for user-defined objects

4. **Object Encoding Service** ✅
   - Pattern: User annotates object → extract CLIP embeddings → store in Qdrant
   - Search: Query embedding → find similar objects above threshold
   - Storage: Qdrant `object_encodings` collection

### The Pattern to Replicate

**Face Detection established the pattern:**
```python
1. Detect face (or user annotates region)
2. Extract encoding (face_recognition or CLIP)
3. Store in Qdrant with metadata
4. Search via similarity (cosine distance)
5. Match above threshold → return identity
```

**Object Encoding extended the pattern to ANY user-defined object:**
```python
1. User annotates region (bbox) + text label ("Farmall tractor")
2. Extract CLIP visual + semantic embeddings
3. Store in Qdrant object_encodings
4. Future detections → extract CLIP embeddings → search → match
```

**We replicate this for:** Logos, Tattoos, Paintings, Custom Items

## Use Case Implementations

### Use Case 1: "Describe the image"

**Current State**: ❌ Not implemented

**Solution**: Vision-Language Model (VLM) integration

**Implementation**:
```python
# In attachment_analysis_subgraph
async def _vision_llm_description_node(state: AttachmentState):
    """Use vision-capable LLM to describe image"""
    image_path = state.get("attached_image_path")
    query = state.get("query")
    
    # Get vision-capable model (GPT-4V, Claude 3.5 Sonnet, Gemini Vision)
    llm = self._get_llm(temperature=0.7, state=state, vision_capable=True)
    
    # Load image as base64 or file path (depending on LLM API)
    image_content = load_image_for_vision(image_path)
    
    # Create vision message
    messages = [
        SystemMessage(content="You are an expert at describing images in detail."),
        HumanMessage(content=[
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": image_content}
        ])
    ]
    
    response = await llm.ainvoke(messages)
    
    return {
        "vision_description": response.content,
        "analysis_type": "vision_llm",
        "confidence": "high"
    }
```

**Models to Consider**:
- GPT-4 Vision (OpenAI) - excellent, but expensive
- Claude 3.5 Sonnet (Anthropic) - excellent, good pricing
- Gemini 2.0 Flash (Google) - fast, good pricing
- LLaVA (local) - free, but requires GPU

**Phase**: Phase 1 (easiest, immediate value)

---

### Use Case 2: "Who are the person(s) in the image"

**Current State**: ⚠️ Partially implemented (detection works, chat integration missing)

**Solution**: Integrate existing face detection into chat workflow

**Implementation**:
```python
async def _face_detection_node(state: AttachmentState):
    """Detect and identify faces in attached image"""
    image_path = state.get("attached_image_path")
    user_id = state.get("user_id")
    
    # Call existing Image Vision Service
    vision_client = await get_image_vision_client()
    result = await vision_client.detect_faces(
        image_path=image_path,
        document_id=state.get("conversation_id", "temp")
    )
    
    if not result or not result.get("faces"):
        return {
            "face_results": [],
            "analysis_type": "face_detection",
            "message": "No faces detected in the image."
        }
    
    # Match against known identities (existing service)
    face_service = await get_face_encoding_service()
    identified_faces = []
    
    for idx, face in enumerate(result["faces"], 1):
        match = await face_service.match_face(
            face_encoding=face["face_encoding"],
            confidence_threshold=0.82,
            user_id=user_id
        )
        
        if match:
            identified_faces.append({
                "face_index": idx,
                "identity": match["matched_identity"],
                "confidence": match["confidence"],
                "sample_count": match["sample_count"],
                "bbox": {
                    "x": face["bbox_x"],
                    "y": face["bbox_y"],
                    "width": face["bbox_width"],
                    "height": face["bbox_height"]
                }
            })
        else:
            identified_faces.append({
                "face_index": idx,
                "identity": "Unknown",
                "confidence": 0.0,
                "bbox": {
                    "x": face["bbox_x"],
                    "y": face["bbox_y"],
                    "width": face["bbox_width"],
                    "height": face["bbox_height"]
                }
            })
    
    # Format natural language response
    if identified_faces:
        known_faces = [f for f in identified_faces if f["identity"] != "Unknown"]
        unknown_count = len(identified_faces) - len(known_faces)
        
        if known_faces:
            names = [f"{f['identity']} ({f['confidence']:.0f}% confidence)" 
                     for f in known_faces]
            message = f"I found {len(identified_faces)} face(s): {', '.join(names)}"
            if unknown_count > 0:
                message += f", and {unknown_count} unknown face(s)"
        else:
            message = f"I found {len(identified_faces)} face(s), but none match known identities."
    
    return {
        "face_results": identified_faces,
        "analysis_type": "face_detection",
        "message": message,
        "total_faces": len(identified_faces),
        "known_faces": len([f for f in identified_faces if f["identity"] != "Unknown"])
    }
```

**Database Schema** (already exists):
```sql
-- known_identities table (PostgreSQL)
CREATE TABLE known_identities (
    id SERIAL PRIMARY KEY,
    identity_name TEXT NOT NULL,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- face_encodings collection (Qdrant)
{
    "identity_name": "John Smith",
    "source_document_id": "doc_123",
    "tagged_at": "2026-02-03T10:00:00Z",
    "user_id": "user_456"
}
```

**Phase**: Phase 1 (mostly exists, just needs chat integration)

---

### Use Case 3: "Identify this tattoo"

**Current State**: ❌ Not implemented

**Solution**: CLIP-based tattoo matching (replicate face/object pattern)

**Implementation**:
```python
async def _tattoo_matching_node(state: AttachmentState):
    """Match tattoo against user's annotated tattoo library"""
    image_path = state.get("attached_image_path")
    user_id = state.get("user_id")
    query = state.get("query")
    
    # Step 1: Detect if YOLO found anything (unlikely for tattoos)
    yolo_results = state.get("yolo_detections", [])
    tattoo_regions = [r for r in yolo_results if "tattoo" in r["class_name"].lower()]
    
    # Step 2: If no YOLO detection, use whole image or user-specified region
    if not tattoo_regions:
        # Option A: Use whole image
        regions_to_check = [{
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_width": state.get("image_width", 1000),
            "bbox_height": state.get("image_height", 1000),
            "source": "whole_image"
        }]
        
        # Option B: Ask user to specify region (future enhancement)
        # Could use vision LLM to detect tattoo regions: "Where is the tattoo in this image?"
    else:
        regions_to_check = tattoo_regions
    
    # Step 3: Extract CLIP embeddings for each region
    vision_client = await get_image_vision_client()
    matches = []
    
    for region in regions_to_check:
        features = await vision_client.extract_object_features(
            image_path=image_path,
            bbox=region,
            description=query  # e.g., "Celtic knot tattoo"
        )
        
        if not features or not features.get("combined_embedding"):
            continue
        
        # Step 4: Search tattoo_encodings collection
        tattoo_service = await get_tattoo_encoding_service()  # NEW SERVICE
        match = await tattoo_service.search_similar_tattoos(
            query_embedding=features["combined_embedding"],
            user_id=user_id,
            top_k=3,
            similarity_threshold=0.75
        )
        
        if match:
            matches.extend(match)
    
    # Step 5: Format results
    if matches:
        top_match = matches[0]
        message = f"This looks like: {top_match['tattoo_name']} ({top_match['similarity_score']*100:.0f}% match)"
        if len(matches) > 1:
            message += f"\n\nOther possibilities: {', '.join([m['tattoo_name'] for m in matches[1:3]])}"
    else:
        message = "I couldn't match this tattoo to any in your collection. Would you like to annotate it?"
    
    return {
        "tattoo_matches": matches,
        "analysis_type": "tattoo_matching",
        "message": message
    }
```

**New Service Required**: `tattoo_encoding_service.py`

**Database Schema**:
```python
# Qdrant collection: tattoo_encodings
{
    "tattoo_name": "Celtic knot",
    "style": "Celtic",
    "description": "Traditional Celtic knotwork",
    "source_document_id": "doc_789",
    "user_id": "user_456",
    "tagged_at": "2026-02-03T10:00:00Z",
    "body_location": "forearm",  # Optional metadata
    "artist": "Mike's Tattoo Shop"  # Optional metadata
}

# PostgreSQL table: tattoo_annotations (for metadata)
CREATE TABLE tattoo_annotations (
    id SERIAL PRIMARY KEY,
    tattoo_name TEXT NOT NULL,
    style TEXT,
    description TEXT,
    user_id TEXT NOT NULL,
    source_document_id TEXT,
    body_location TEXT,
    artist TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Phase**: Phase 2 (new collection + service, medium complexity)

---

### Use Case 4: "Identify this logo"

**Current State**: ❌ Not implemented

**Solution**: CLIP-based logo matching (replicate pattern)

**Implementation**:
```python
async def _logo_matching_node(state: AttachmentState):
    """Match logo against user's logo library using CLIP"""
    image_path = state.get("attached_image_path")
    user_id = state.get("user_id")
    
    # Step 1: Try YOLO detection first (may catch some logos as generic objects)
    yolo_results = state.get("yolo_detections", [])
    
    # Step 2: Also run open-vocabulary CLIP detection for "logo" or "brand mark"
    vision_client = await get_image_vision_client()
    logo_detection = await vision_client.detect_objects(
        image_path=image_path,
        document_id=state.get("conversation_id", "temp"),
        semantic_descriptions=["logo", "brand logo", "company logo", "brand mark"]
    )
    
    # Combine YOLO + CLIP semantic detections
    regions_to_check = []
    if logo_detection and logo_detection.get("objects"):
        regions_to_check.extend(logo_detection["objects"])
    
    # If no detection, use whole image (logo might fill entire image)
    if not regions_to_check:
        regions_to_check = [{
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_width": state.get("image_width", 1000),
            "bbox_height": state.get("image_height", 1000),
            "source": "whole_image"
        }]
    
    # Step 3: Extract CLIP embeddings and search
    matches = []
    for region in regions_to_check:
        features = await vision_client.extract_object_features(
            image_path=image_path,
            bbox=region,
            description="brand logo"
        )
        
        if not features:
            continue
        
        # Search logo_encodings collection
        logo_service = await get_logo_encoding_service()  # NEW SERVICE
        match = await logo_service.search_similar_logos(
            query_embedding=features["combined_embedding"],
            user_id=user_id,
            top_k=5,
            similarity_threshold=0.70  # Slightly lower - logos can vary
        )
        
        if match:
            matches.extend(match)
    
    # Step 4: Format results
    if matches:
        # Deduplicate and sort by score
        seen = set()
        unique_matches = []
        for m in sorted(matches, key=lambda x: x["similarity_score"], reverse=True):
            if m["brand_name"] not in seen:
                unique_matches.append(m)
                seen.add(m["brand_name"])
        
        top = unique_matches[0]
        message = f"This is the {top['brand_name']} logo ({top['similarity_score']*100:.0f}% match)"
        if len(unique_matches) > 1:
            message += f"\n\nMight also be: {', '.join([m['brand_name'] for m in unique_matches[1:3]])}"
    else:
        message = "I couldn't identify this logo. Would you like to annotate it?"
    
    return {
        "logo_matches": matches,
        "analysis_type": "logo_matching",
        "message": message
    }
```

**New Service Required**: `logo_encoding_service.py`

**Database Schema**:
```python
# Qdrant collection: logo_encodings
{
    "brand_name": "Nike",
    "logo_variant": "swoosh",
    "description": "Nike swoosh logo",
    "industry": "sportswear",
    "source_document_id": "doc_999",
    "user_id": "user_456",
    "tagged_at": "2026-02-03T10:00:00Z",
    "colors": ["black"],  # Optional
    "year_introduced": 1971  # Optional
}

# PostgreSQL table: logo_annotations
CREATE TABLE logo_annotations (
    id SERIAL PRIMARY KEY,
    brand_name TEXT NOT NULL,
    logo_variant TEXT,
    description TEXT,
    industry TEXT,
    user_id TEXT NOT NULL,
    source_document_id TEXT,
    colors TEXT[],
    year_introduced INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Phase**: Phase 2 (new collection + service, medium complexity)

---

### Use Case 5: "Identify this painting"

**Current State**: ❌ Not implemented (most complex)

**Solution**: Multi-stage CLIP matching with painting-specific heuristics

**Complexity Factors**:
1. Painting might be **photographed in environment** (framed on wall) OR **entire image is the painting**
2. Perspective distortion, lighting, reflections in photographed paintings
3. Reference images could be **professional scans** OR **user's photos**
4. User might show **detail/region** of painting vs. **whole painting**

**Implementation**:
```python
async def _painting_matching_node(state: AttachmentState):
    """Match painting using multi-stage CLIP-based approach"""
    image_path = state.get("attached_image_path")
    user_id = state.get("user_id")
    
    # STAGE 1: Detect if painting is in environment or IS the image
    vision_client = await get_image_vision_client()
    
    # Try YOLO to detect "painting" as object class
    yolo_results = state.get("yolo_detections", [])
    painting_regions = [r for r in yolo_results if "painting" in r["class_name"].lower()]
    
    # Also try semantic detection with CLIP
    semantic_result = await vision_client.detect_objects(
        image_path=image_path,
        document_id=state.get("conversation_id", "temp"),
        semantic_descriptions=["painting", "framed painting", "artwork", "canvas painting"]
    )
    
    # Decide regions to analyze
    regions_to_check = []
    
    if painting_regions or (semantic_result and semantic_result.get("objects")):
        # Painting detected in environment - use detected region
        if painting_regions:
            regions_to_check.extend(painting_regions)
        if semantic_result and semantic_result.get("objects"):
            regions_to_check.extend(semantic_result["objects"])
        logger.info(f"Detected painting in environment: {len(regions_to_check)} region(s)")
    else:
        # No painting detected - entire image is likely the painting
        regions_to_check = [{
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_width": state.get("image_width", 1000),
            "bbox_height": state.get("image_height", 1000),
            "source": "whole_image"
        }]
        logger.info("No painting detected in environment - treating whole image as painting")
    
    # STAGE 2: Extract CLIP embeddings (robust to perspective/lighting)
    matches = []
    for region in regions_to_check:
        features = await vision_client.extract_object_features(
            image_path=image_path,
            bbox=region,
            description="painting artwork"
        )
        
        if not features:
            continue
        
        # STAGE 3: Search painting_encodings collection
        painting_service = await get_painting_encoding_service()  # NEW SERVICE
        match = await painting_service.search_similar_paintings(
            query_embedding=features["combined_embedding"],
            user_id=user_id,
            top_k=5,
            similarity_threshold=0.65  # Lower - paintings vary more
        )
        
        if match:
            matches.extend(match)
    
    # STAGE 4: Post-processing and ranking
    # Deduplicate matches (same painting might match multiple times)
    seen = {}
    for m in matches:
        painting_id = m.get("painting_id") or m.get("title")
        if painting_id not in seen or m["similarity_score"] > seen[painting_id]["similarity_score"]:
            seen[painting_id] = m
    
    unique_matches = sorted(seen.values(), key=lambda x: x["similarity_score"], reverse=True)
    
    # STAGE 5: Format results with rich metadata
    if unique_matches:
        top = unique_matches[0]
        message = f"This appears to be: **{top['title']}**"
        if top.get("artist"):
            message += f" by {top['artist']}"
        if top.get("year"):
            message += f" ({top['year']})"
        message += f"\n\nMatch confidence: {top['similarity_score']*100:.0f}%"
        
        if top.get("style"):
            message += f"\nStyle: {top['style']}"
        if top.get("medium"):
            message += f"\nMedium: {top['medium']}"
        
        if len(unique_matches) > 1:
            message += "\n\n**Other possibilities:**"
            for m in unique_matches[1:3]:
                message += f"\n- {m['title']}"
                if m.get("artist"):
                    message += f" by {m['artist']}"
                message += f" ({m['similarity_score']*100:.0f}% match)"
    else:
        message = "I couldn't match this painting to any in your collection. Would you like to add it?"
    
    return {
        "painting_matches": unique_matches,
        "analysis_type": "painting_matching",
        "message": message,
        "regions_analyzed": len(regions_to_check)
    }
```

**New Service Required**: `painting_encoding_service.py`

**Database Schema**:
```python
# Qdrant collection: painting_encodings
{
    "painting_id": "paint_001",
    "title": "Starry Night",
    "artist": "Vincent van Gogh",
    "year": 1889,
    "style": "Post-Impressionism",
    "medium": "Oil on canvas",
    "dimensions": "73.7 cm × 92.1 cm",
    "description": "Swirling night sky over village with cypress tree",
    "source_document_id": "doc_painting_001",
    "user_id": "user_456",
    "reference_image_type": "photograph",  # or "scan"
    "tagged_at": "2026-02-03T10:00:00Z",
    "museum": "MoMA",  # Optional
    "collection": "Personal"  # or "Museum", "Gallery", etc.
}

# PostgreSQL table: painting_annotations
CREATE TABLE painting_annotations (
    id SERIAL PRIMARY KEY,
    painting_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    artist TEXT,
    year INTEGER,
    style TEXT,
    medium TEXT,
    dimensions TEXT,
    description TEXT,
    user_id TEXT NOT NULL,
    source_document_id TEXT,
    reference_image_type TEXT,  -- 'photograph' or 'scan'
    museum TEXT,
    collection TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Enhancement: Painting Style Detection** (if no match found):
```python
# If no specific painting match, try to identify style
style_descriptions = [
    "Renaissance painting",
    "Baroque painting", 
    "Impressionist painting",
    "Post-Impressionist painting",
    "Cubist painting",
    "Abstract painting",
    "Realist painting"
]

style_detection = await vision_client.detect_objects(
    image_path=image_path,
    semantic_descriptions=style_descriptions
)

# Return style even if specific painting not identified
```

**Phase**: Phase 3 (most complex, multi-stage)

---

## Unified CLIP Matching Architecture

### Key Principle: Universal Semantic Matching

**The Pattern**:
1. **Detection Layer** (optional): YOLO finds structure/regions
2. **Embedding Layer** (universal): CLIP extracts semantic features
3. **Storage Layer**: Qdrant vector database with metadata
4. **Matching Layer**: Cosine similarity search
5. **Post-Processing**: Ranking, deduplication, formatting

### Why CLIP is Universal

**CLIP (Contrastive Language-Image Pre-training)**:
- Trained on 400M image-text pairs
- Understands visual concepts AND text descriptions
- Robust to:
  - Perspective changes (photographed paintings)
  - Lighting variations (faces in different conditions)
  - Partial views (detail of painting)
  - Style variations (different photos of same logo)
- 512-dimensional embedding space (vs. 128 for face_recognition)

**YOLO vs. CLIP**:
| Aspect | YOLO | CLIP |
|--------|------|------|
| Purpose | Object localization | Semantic understanding |
| Output | Bounding boxes + 80 classes | 512-dim embedding |
| Custom objects | No (fixed classes) | Yes (any concept) |
| Text grounding | No | Yes (text embeddings) |
| Use in our system | Structure detection | Similarity matching |

**Our Architecture**: YOLO + CLIP Hybrid
- YOLO: "Where are the objects?" (bounding boxes)
- CLIP: "What are these objects?" (semantic matching)
- User annotations: "What do I call this?" (custom labels)

### Service Architecture

**Existing Services** (to keep):
- `face_encoding_service.py` - Face-specific (uses face_recognition library)
- `object_encoding_service.py` - Generic objects (uses CLIP)

**New Services** (to create):
- `logo_encoding_service.py` - Logo-specific matching
- `tattoo_encoding_service.py` - Tattoo-specific matching  
- `painting_encoding_service.py` - Painting-specific matching

**Why Separate Services?**
1. **Different metadata schemas** (face vs. logo vs. painting)
2. **Different similarity thresholds** (faces: 0.82, logos: 0.70, paintings: 0.65)
3. **Different post-processing** (face clustering vs. style detection)
4. **Clear separation of concerns** (each service manages its domain)

**Shared Infrastructure**:
- All use Qdrant for vector storage
- All use Image Vision Service for CLIP extraction
- All follow same search/match pattern
- All support per-user collections (user privacy)

### Qdrant Collection Architecture

**Current Collections**:
```python
face_encodings              # Global faces
user_{user_id}_face_encodings  # Per-user faces
object_encodings            # Generic objects
```

**New Collections**:
```python
logo_encodings              # Global logos
user_{user_id}_logo_encodings  # Per-user logos
tattoo_encodings            # Global tattoos
user_{user_id}_tattoo_encodings  # Per-user tattoos
painting_encodings          # Global paintings
user_{user_id}_painting_encodings  # Per-user paintings
```

**Collection Schema** (consistent across all):
```python
{
    "vectors": {
        "size": 512,  # CLIP embedding dimension
        "distance": "Cosine"
    },
    "payload_schema": {
        # Domain-specific fields (name, title, brand_name, etc.)
        "user_id": "keyword",
        "source_document_id": "keyword",
        "tagged_at": "keyword"
    }
}
```

**Hybrid Search Strategy** (like faces):
```python
# Search both global and user collections
collections_to_search = [
    "logo_encodings",  # Global (shared logos)
    f"user_{user_id}_logo_encodings"  # User's private logos
]

# Merge results, prioritize user's annotations
all_results = []
for collection in collections_to_search:
    results = await vector_service.search_vectors(
        collection_name=collection,
        query_vector=clip_embedding,
        limit=5,
        score_threshold=0.70
    )
    all_results.extend(results)

# Sort by similarity, deduplicate
```

## Research Agent Integration

### Attachment Detection Node

**New Node**: `detect_attachments_node`

```python
async def _detect_attachments_node(self, state: ResearchState) -> Dict[str, Any]:
    """
    Detect if user has attached images to their query.
    Extract attachment metadata and route to analysis if needed.
    """
    messages = state.get("messages", [])
    shared_memory = state.get("shared_memory", {})
    
    # Check for attachments in latest message
    if not messages:
        return {
            "has_attachments": False,
            "attachments": []
        }
    
    latest_message = messages[-1]
    attachments = []
    
    # Extract attachment data from message
    # Format depends on how frontend sends attachments
    # Could be: file paths, base64 data, document IDs, etc.
    
    if hasattr(latest_message, 'additional_kwargs'):
        # Check for image attachments in message metadata
        image_attachments = latest_message.additional_kwargs.get('attachments', [])
        for attachment in image_attachments:
            if attachment.get('type') == 'image':
                attachments.append({
                    "type": "image",
                    "path": attachment.get('path'),
                    "url": attachment.get('url'),
                    "document_id": attachment.get('document_id'),
                    "mime_type": attachment.get('mime_type', 'image/jpeg')
                })
    
    # Also check shared_memory for attachments (alternative path)
    if shared_memory.get("attachments"):
        attachments.extend(shared_memory["attachments"])
    
    logger.info(f"Detected {len(attachments)} attachment(s)")
    
    return {
        "has_attachments": len(attachments) > 0,
        "attachments": attachments,
        "attachment_count": len(attachments)
    }
```

### Attachment Analysis Subgraph

**New Subgraph**: `attachment_analysis_subgraph.py`

**State Definition**:
```python
class AttachmentState(TypedDict):
    # Input
    query: str  # User's question about the attachment
    attachments: List[Dict[str, Any]]  # List of attachments
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    
    # Image analysis
    attached_image_path: str  # Primary image being analyzed
    image_width: int
    image_height: int
    
    # Intent classification
    intent_type: str  # "describe", "identify_person", "identify_object", etc.
    
    # Detection results
    yolo_detections: List[Dict[str, Any]]  # YOLO-detected objects
    clip_whole_image: List[float]  # CLIP embedding of whole image
    vision_description: str  # Vision LLM description
    
    # Matching results
    face_results: List[Dict[str, Any]]
    object_matches: List[Dict[str, Any]]
    logo_matches: List[Dict[str, Any]]
    tattoo_matches: List[Dict[str, Any]]
    painting_matches: List[Dict[str, Any]]
    
    # Output
    analysis_type: str  # Which analysis was performed
    final_response: str
    confidence: str  # "high", "medium", "low"
    suggestions: List[str]  # Follow-up suggestions for user
```

**Workflow**:
```python
def build_attachment_analysis_subgraph():
    """Build attachment analysis subgraph"""
    workflow = StateGraph(AttachmentState)
    
    # Nodes
    workflow.add_node("classify_intent", _classify_intent_node)
    workflow.add_node("parallel_analysis", _parallel_analysis_node)  # YOLO + CLIP + Vision LLM
    workflow.add_node("face_detection", _face_detection_node)
    workflow.add_node("object_matching", _object_matching_node)
    workflow.add_node("logo_matching", _logo_matching_node)
    workflow.add_node("tattoo_matching", _tattoo_matching_node)
    workflow.add_node("painting_matching", _painting_matching_node)
    workflow.add_node("vision_description", _vision_llm_description_node)
    workflow.add_node("format_response", _format_response_node)
    
    # Flow
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "parallel_analysis")
    
    # Route based on intent
    workflow.add_conditional_edges(
        "parallel_analysis",
        _route_by_intent,
        {
            "describe": "vision_description",
            "identify_person": "face_detection",
            "identify_object": "object_matching",
            "identify_logo": "logo_matching",
            "identify_tattoo": "tattoo_matching",
            "identify_painting": "painting_matching",
            "general": "vision_description"  # Default fallback
        }
    )
    
    # All paths converge to format_response
    workflow.add_edge("vision_description", "format_response")
    workflow.add_edge("face_detection", "format_response")
    workflow.add_edge("object_matching", "format_response")
    workflow.add_edge("logo_matching", "format_response")
    workflow.add_edge("tattoo_matching", "format_response")
    workflow.add_edge("painting_matching", "format_response")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()
```

**Intent Classification Node**:
```python
async def _classify_intent_node(state: AttachmentState) -> Dict[str, Any]:
    """Classify user's intent about the attached image"""
    query = state.get("query", "").lower()
    
    # Simple heuristic classification (fast)
    if any(word in query for word in ["who", "person", "people", "face", "identify"]):
        intent = "identify_person"
    elif any(word in query for word in ["what", "describe", "tell me about", "what is"]):
        if "logo" in query or "brand" in query:
            intent = "identify_logo"
        elif "tattoo" in query:
            intent = "identify_tattoo"
        elif "painting" in query or "artwork" in query or "artist" in query:
            intent = "identify_painting"
        elif "object" in query or "thing" in query:
            intent = "identify_object"
        else:
            intent = "describe"  # General description
    else:
        intent = "general"
    
    logger.info(f"Classified attachment intent: {intent}")
    
    return {
        "intent_type": intent
    }
```

**Parallel Analysis Node** (run all in parallel for speed):
```python
async def _parallel_analysis_node(state: AttachmentState) -> Dict[str, Any]:
    """
    Run parallel analysis:
    1. YOLO object detection (structure)
    2. CLIP whole-image embedding (semantics)
    3. Vision LLM description (context - optional)
    """
    image_path = state.get("attached_image_path")
    
    vision_client = await get_image_vision_client()
    
    # Run in parallel
    import asyncio
    yolo_task = vision_client.detect_objects(
        image_path=image_path,
        document_id="temp",
        confidence_threshold=0.5
    )
    
    clip_task = vision_client.extract_object_features(
        image_path=image_path,
        bbox={
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_width": state.get("image_width", 1000),
            "bbox_height": state.get("image_height", 1000)
        },
        description=""
    )
    
    # Run both in parallel
    yolo_result, clip_result = await asyncio.gather(
        yolo_task,
        clip_task,
        return_exceptions=True
    )
    
    # Extract results
    yolo_detections = []
    if not isinstance(yolo_result, Exception) and yolo_result:
        yolo_detections = yolo_result.get("objects", [])
        image_width = yolo_result.get("image_width", 0)
        image_height = yolo_result.get("image_height", 0)
    
    clip_embedding = []
    if not isinstance(clip_result, Exception) and clip_result:
        clip_embedding = clip_result.get("combined_embedding", [])
    
    logger.info(f"YOLO detected {len(yolo_detections)} objects")
    logger.info(f"CLIP embedding extracted: {len(clip_embedding)} dimensions")
    
    return {
        "yolo_detections": yolo_detections,
        "clip_whole_image": clip_embedding,
        "image_width": image_width if yolo_detections else state.get("image_width", 0),
        "image_height": image_height if yolo_detections else state.get("image_height", 0)
    }
```

### Integration with Full Research Agent

**Modified Research Agent Workflow**:
```python
# In full_research_agent.py

def _build_workflow(self, checkpointer) -> StateGraph:
    workflow = StateGraph(ResearchState)
    
    # NEW: Add attachment detection as first step
    workflow.add_node("detect_attachments", self._detect_attachments_node)
    workflow.add_node("attachment_analysis", self._attachment_analysis_subgraph)
    
    # Existing nodes...
    workflow.add_node("quick_answer_check", self._quick_answer_check_node)
    workflow.add_node("cache_check", self._cache_check_node)
    # ... etc.
    
    # NEW: Entry point checks for attachments first
    workflow.set_entry_point("detect_attachments")
    
    # NEW: Route based on attachments
    workflow.add_conditional_edges(
        "detect_attachments",
        lambda state: "attachment_analysis" if state.get("has_attachments") else "quick_answer_check",
        {
            "attachment_analysis": "attachment_analysis",
            "quick_answer_check": "quick_answer_check"
        }
    )
    
    # NEW: After attachment analysis, continue to synthesis or end
    workflow.add_conditional_edges(
        "attachment_analysis",
        lambda state: "final_synthesis" if state.get("needs_synthesis") else END,
        {
            "final_synthesis": "final_synthesis",
            END: END
        }
    )
    
    # Existing flow continues...
    workflow.add_edge("quick_answer_check", "cache_check")
    # ... rest of workflow
    
    return workflow.compile(checkpointer=checkpointer)
```

## User Annotation Workflow

### Annotation UI Flow

**Frontend**: User can annotate any image to build their reference library

**Steps**:
1. User uploads/selects image
2. User draws bounding box around object/region
3. User provides label + metadata (name, description, etc.)
4. Backend extracts CLIP embeddings
5. Store in Qdrant + PostgreSQL
6. Future images → auto-match against this annotation

**API Endpoint**: `/api/annotations/create`
```python
@router.post("/annotations/create")
async def create_annotation(
    document_id: str,
    annotation_type: str,  # "face", "object", "logo", "tattoo", "painting"
    label: str,
    bbox: Dict[str, int],  # {"x": 100, "y": 200, "width": 50, "height": 50}
    metadata: Dict[str, Any] = {},
    user_id: str = Depends(get_current_user)
):
    """
    Create a new annotation for an object/region in an image.
    Extracts CLIP embeddings and stores in appropriate Qdrant collection.
    """
    # Get document path
    document = await document_repository.get_by_id(document_id, user_id)
    image_path = get_document_file_path(document)
    
    # Extract CLIP embeddings for the region
    vision_client = await get_image_vision_client()
    features = await vision_client.extract_object_features(
        image_path=image_path,
        bbox=bbox,
        description=label
    )
    
    if not features:
        raise HTTPException(status_code=500, detail="Failed to extract features")
    
    # Store based on annotation type
    if annotation_type == "face":
        service = await get_face_encoding_service()
        point_id = await service.add_face_encoding(
            identity_name=label,
            face_encoding=features["combined_embedding"],
            source_document_id=document_id,
            metadata=metadata,
            user_id=user_id
        )
    elif annotation_type == "logo":
        service = await get_logo_encoding_service()
        point_id = await service.add_logo_encoding(
            brand_name=label,
            logo_embedding=features["combined_embedding"],
            source_document_id=document_id,
            metadata=metadata,
            user_id=user_id
        )
    # ... other types
    
    return {
        "success": True,
        "point_id": point_id,
        "annotation_type": annotation_type,
        "label": label
    }
```

### Bulk Annotation Workflow

**Use Case**: User has 100 photos of logos they want to annotate

**Strategy**: Semi-automated annotation with user confirmation

**Workflow**:
1. **Batch Upload**: User uploads 100 images
2. **Auto-Detection**: System runs YOLO + CLIP to detect potential logos
3. **Suggested Annotations**: System shows detected regions + suggested labels (via CLIP text matching)
4. **User Confirms/Edits**: User confirms or corrects labels
5. **Batch Storage**: Store all confirmed annotations at once

**API Endpoint**: `/api/annotations/batch-suggest`
```python
@router.post("/annotations/batch-suggest")
async def batch_suggest_annotations(
    document_ids: List[str],
    annotation_type: str,
    user_id: str = Depends(get_current_user)
):
    """
    Auto-detect objects in batch and suggest annotations.
    User confirms/edits before final storage.
    """
    suggestions = []
    
    for doc_id in document_ids:
        document = await document_repository.get_by_id(doc_id, user_id)
        image_path = get_document_file_path(document)
        
        # Auto-detect
        vision_client = await get_image_vision_client()
        detection = await vision_client.detect_objects(
            image_path=image_path,
            document_id=doc_id,
            semantic_descriptions=["logo", "brand"] if annotation_type == "logo" else []
        )
        
        if detection and detection.get("objects"):
            for obj in detection["objects"]:
                # Extract CLIP embeddings
                features = await vision_client.extract_object_features(
                    image_path=image_path,
                    bbox=obj,
                    description=""
                )
                
                # Suggest label via CLIP text matching
                suggested_label = await suggest_label_from_embedding(
                    features["combined_embedding"],
                    annotation_type
                )
                
                suggestions.append({
                    "document_id": doc_id,
                    "bbox": obj,
                    "suggested_label": suggested_label,
                    "confidence": obj.get("confidence", 0.0),
                    "embedding": features["combined_embedding"]  # Pre-computed for faster storage
                })
    
    return {
        "suggestions": suggestions,
        "count": len(suggestions)
    }
```

## Phase-by-Phase Implementation Plan

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Get basic image analysis working in Research Agent

**Tasks**:

1. **Attachment Detection** ✅
   - Add `detect_attachments_node` to Research Agent
   - Extract image attachments from messages/shared_memory
   - Handle multiple attachment formats (file paths, base64, URLs)
   - **Files to modify**:
     - `llm-orchestrator/orchestrator/agents/full_research_agent.py`
   - **Estimated effort**: 1 day

2. **Vision LLM Integration** ✅
   - Add vision-capable model support to `BaseAgent._get_llm()`
   - Support GPT-4V, Claude 3.5 Sonnet, Gemini Vision
   - Implement `_vision_llm_description_node`
   - **Files to modify**:
     - `llm-orchestrator/orchestrator/agents/base_agent.py`
   - **New files**:
     - `llm-orchestrator/orchestrator/subgraphs/attachment_analysis_subgraph.py`
   - **Estimated effort**: 2 days

3. **Face Detection Integration** ✅
   - Integrate existing face detection into chat workflow
   - Add `_face_detection_node` to attachment subgraph
   - Format natural language responses for identified faces
   - **Files to modify**:
     - `llm-orchestrator/orchestrator/subgraphs/attachment_analysis_subgraph.py`
   - **Estimated effort**: 1 day

4. **Basic Testing** ✅
   - Test "describe this image" flow end-to-end
   - Test "who is this person" flow
   - Verify attachment detection works correctly
   - **Estimated effort**: 1 day

**Deliverables**:
- Users can attach images to Research Agent queries
- Vision LLM can describe images
- Face detection identifies known people
- Natural language responses formatted correctly

---

### Phase 2: Object/Logo/Tattoo Matching (Weeks 3-4)

**Goal**: CLIP-based matching for non-face objects

**Tasks**:

1. **Logo Encoding Service** ✅
   - Create `backend/services/logo_encoding_service.py`
   - Follow pattern from `face_encoding_service.py`
   - Implement: `add_logo_encoding()`, `search_similar_logos()`
   - Create Qdrant collection: `logo_encodings`
   - **New files**:
     - `backend/services/logo_encoding_service.py`
     - `backend/models/logo_models.py`
   - **Estimated effort**: 2 days

2. **Tattoo Encoding Service** ✅
   - Create `backend/services/tattoo_encoding_service.py`
   - Same pattern as logos
   - Create Qdrant collection: `tattoo_encodings`
   - **New files**:
     - `backend/services/tattoo_encoding_service.py`
     - `backend/models/tattoo_models.py`
   - **Estimated effort**: 2 days

3. **PostgreSQL Schema** ✅
   - Add tables: `logo_annotations`, `tattoo_annotations`
   - Add migration scripts
   - **New files**:
     - `backend/sql/migrations/036_add_logo_annotations.sql`
     - `backend/sql/migrations/037_add_tattoo_annotations.sql`
   - **Estimated effort**: 1 day

4. **Attachment Subgraph Nodes** ✅
   - Implement `_logo_matching_node`
   - Implement `_tattoo_matching_node`
   - Implement `_object_matching_node` (generic objects)
   - **Files to modify**:
     - `llm-orchestrator/orchestrator/subgraphs/attachment_analysis_subgraph.py`
   - **Estimated effort**: 2 days

5. **Annotation API** ✅
   - Create `/api/annotations/create` endpoint
   - Create `/api/annotations/batch-suggest` endpoint
   - Support all annotation types (face, object, logo, tattoo)
   - **New files**:
     - `backend/api/annotation_api.py`
   - **Estimated effort**: 2 days

6. **Testing** ✅
   - Test logo matching end-to-end
   - Test tattoo matching
   - Test user annotation workflow
   - **Estimated effort**: 1 day

**Deliverables**:
- Logo identification working
- Tattoo identification working
- Users can annotate any object type
- CLIP-based matching pipeline proven

---

### Phase 3: Painting Matching (Weeks 5-6)

**Goal**: Complex multi-stage painting identification

**Tasks**:

1. **Painting Encoding Service** ✅
   - Create `backend/services/painting_encoding_service.py`
   - Implement multi-stage matching logic
   - Handle whole-image vs. region detection
   - Create Qdrant collection: `painting_encodings`
   - **New files**:
     - `backend/services/painting_encoding_service.py`
     - `backend/models/painting_models.py`
   - **Estimated effort**: 3 days

2. **Painting Detection Pipeline** ✅
   - Stage 1: Detect if painting in environment
   - Stage 2: Extract painting region or use whole image
   - Stage 3: CLIP embedding extraction
   - Stage 4: Search painting_encodings
   - Stage 5: Post-processing and ranking
   - **Files to modify**:
     - `llm-orchestrator/orchestrator/subgraphs/attachment_analysis_subgraph.py`
   - **Estimated effort**: 3 days

3. **Style Detection Fallback** ✅
   - If no specific painting match, identify style
   - Use CLIP semantic matching for styles
   - Return: "This appears to be a Post-Impressionist painting"
   - **Estimated effort**: 2 days

4. **PostgreSQL Schema** ✅
   - Add table: `painting_annotations`
   - Rich metadata: artist, year, style, medium, dimensions
   - **New files**:
     - `backend/sql/migrations/038_add_painting_annotations.sql`
   - **Estimated effort**: 1 day

5. **Testing** ✅
   - Test painting identification (whole image)
   - Test painting identification (photographed/framed)
   - Test style detection fallback
   - Test with perspective distortions
   - **Estimated effort**: 2 days

**Deliverables**:
- Painting identification working
- Style detection as fallback
- Robust to photographed paintings
- Rich metadata in responses

---

### Phase 4: Annotation UI (Weeks 7-8)

**Goal**: Frontend for user annotations

**Tasks**:

1. **Annotation Component** ✅
   - React component for drawing bounding boxes
   - Canvas-based annotation tool
   - Support drag-to-create, resize, delete
   - **New files**:
     - `frontend/src/components/ImageAnnotation.js`
   - **Estimated effort**: 3 days

2. **Annotation Dialog** ✅
   - Modal for entering annotation metadata
   - Type selector: face, object, logo, tattoo, painting
   - Label input + metadata fields
   - Preview of annotated region
   - **New files**:
     - `frontend/src/components/AnnotationDialog.js`
   - **Estimated effort**: 2 days

3. **Batch Annotation UI** ✅
   - Grid view of images with suggested annotations
   - User confirms/edits each suggestion
   - Bulk actions: "Accept all", "Reject all"
   - **New files**:
     - `frontend/src/components/BatchAnnotation.js`
   - **Estimated effort**: 3 days

4. **Integration with Document Viewer** ✅
   - Add "Annotate" button to image documents
   - Show existing annotations as overlays
   - Allow editing/deleting annotations
   - **Files to modify**:
     - `frontend/src/components/DocumentViewer.js`
   - **Estimated effort**: 2 days

5. **Testing** ✅
   - Test annotation creation workflow
   - Test batch annotation workflow
   - Test annotation editing/deletion
   - **Estimated effort**: 1 day

**Deliverables**:
- Visual annotation tool in frontend
- Batch annotation for efficiency
- Integrated with existing document viewer
- User-friendly annotation experience

---

### Phase 5: Optimization & Polish (Weeks 9-10)

**Goal**: Performance, UX improvements, edge cases

**Tasks**:

1. **Caching Strategy** ✅
   - Cache CLIP embeddings for uploaded images
   - Avoid re-computing embeddings for same image
   - Cache in Redis or PostgreSQL
   - **Files to modify**:
     - `backend/services/logo_encoding_service.py`
     - `backend/services/tattoo_encoding_service.py`
     - `backend/services/painting_encoding_service.py`
   - **Estimated effort**: 2 days

2. **Performance Optimization** ✅
   - Parallel embedding extraction (batch CLIP)
   - Optimize Qdrant search (reduce latency)
   - Profile slow paths and optimize
   - **Estimated effort**: 3 days

3. **Edge Case Handling** ✅
   - Multiple objects in one image
   - Low-quality images
   - Partial views
   - Occlusions and overlaps
   - **Estimated effort**: 2 days

4. **Response Formatting** ✅
   - Rich formatting for match results
   - Include confidence scores, metadata
   - Suggest follow-up actions
   - Handle "no match found" gracefully
   - **Files to modify**:
     - `llm-orchestrator/orchestrator/subgraphs/attachment_analysis_subgraph.py`
   - **Estimated effort**: 2 days

5. **Documentation** ✅
   - User guide for annotation workflow
   - API documentation
   - Service architecture docs
   - **New files**:
     - `docs/IMAGE_ANALYSIS_USER_GUIDE.md`
     - `docs/IMAGE_ANALYSIS_ARCHITECTURE.md`
   - **Estimated effort**: 2 days

6. **End-to-End Testing** ✅
   - Test all use cases from user perspective
   - Test cross-browser compatibility
   - Load testing with large image collections
   - **Estimated effort**: 2 days

**Deliverables**:
- Fast, responsive image analysis
- Robust edge case handling
- Complete documentation
- Production-ready system

---

## Technical Considerations

### Performance Benchmarks

**Target Latency** (end-to-end):
- Vision LLM description: < 5 seconds
- Face detection + matching: < 3 seconds
- CLIP-based matching (logo/tattoo/object): < 2 seconds
- Painting matching (multi-stage): < 4 seconds

**Optimization Strategies**:
1. **Parallel processing**: Run YOLO + CLIP + Vision LLM in parallel
2. **Caching**: Cache CLIP embeddings for all uploaded images
3. **Batch operations**: Process multiple regions in one CLIP call
4. **Qdrant optimization**: Use filtered search to narrow candidates

### Scalability Considerations

**Qdrant Collections**:
- Global collections: Shared across all users
- Per-user collections: Private annotations
- Total vectors: Could reach millions as users annotate
- Strategy: Shard large collections, use filtered search

**Image Vision Service**:
- CPU-bound (YOLO + CLIP inference)
- Scale horizontally: Add more service instances
- Use GPU if available (10x faster inference)
- Queue long-running jobs (batch processing)

**Storage Requirements**:
- CLIP embeddings: 512 floats × 4 bytes = 2KB per image
- 1 million images = 2GB vector storage (manageable)
- PostgreSQL metadata: Minimal (few KB per annotation)

### Security Considerations

**Per-User Collections**:
- User annotations are private by default
- Global collections for shared/public annotations
- RLS (Row-Level Security) in PostgreSQL
- Qdrant search filters by user_id

**Image Access Control**:
- Verify user owns document before analyzing
- Don't leak annotations from other users
- Sanitize file paths (prevent directory traversal)

**Rate Limiting**:
- Limit annotation creation (prevent spam)
- Limit vision LLM calls (expensive API calls)
- Queue batch operations (prevent service overload)

### Error Handling

**Graceful Degradation**:
- If Image Vision Service down → fall back to vision LLM
- If Qdrant down → return "matching unavailable"
- If no matches found → suggest annotation
- Always provide useful feedback to user

**Retry Strategy**:
- Transient failures: Retry 3x with exponential backoff
- Service unavailable: Mark for later processing
- Invalid image: Return clear error message

### Monitoring & Observability

**Metrics to Track**:
- Image analysis latency (p50, p95, p99)
- Match success rate (% of queries with matches)
- Annotation growth rate (new annotations per day)
- Service health (Image Vision Service uptime)
- Qdrant search performance (query latency)

**Logging**:
- Log all image analysis requests (for debugging)
- Log match results and confidence scores
- Log failures and errors with context
- Log performance metrics (timing breakdown)

### Testing Strategy

**Unit Tests**:
- Service methods (add/search encodings)
- CLIP embedding extraction
- Intent classification logic
- Response formatting

**Integration Tests**:
- End-to-end image analysis workflow
- Annotation creation → storage → matching
- Multi-stage painting detection pipeline
- Error handling and fallbacks

**Manual Testing**:
- Real images from various sources
- Edge cases (low quality, occlusions, etc.)
- Cross-browser compatibility
- Mobile responsiveness

## Success Metrics

**Phase 1**:
- ✅ Users can attach images to queries
- ✅ Vision LLM describes images accurately
- ✅ Face detection identifies known people

**Phase 2**:
- ✅ Logo matching works with >70% accuracy
- ✅ Tattoo matching works with >75% accuracy
- ✅ Users can annotate objects easily

**Phase 3**:
- ✅ Painting matching works for common paintings
- ✅ Style detection provides useful fallback
- ✅ Handles photographed paintings correctly

**Phase 4**:
- ✅ Annotation UI is intuitive and fast
- ✅ Batch annotation saves users time
- ✅ Users actively create annotations

**Phase 5**:
- ✅ Image analysis < 5 seconds end-to-end
- ✅ No major bugs or edge case failures
- ✅ Complete documentation available

**Overall Success**:
- Users rely on image analysis regularly
- Annotation library grows organically
- High user satisfaction (survey feedback)
- Low error rate (<5% failed analyses)

## Next Steps

1. **Review this plan** with team
2. **Prioritize phases** based on user needs
3. **Allocate resources** (developers, GPU access, etc.)
4. **Start Phase 1** implementation
5. **Iterate based on feedback**

---

## Appendix: Code Examples

### Example: Logo Encoding Service

**File**: `backend/services/logo_encoding_service.py`

```python
"""
Logo Encoding Service - CLIP-based logo matching
Follows the same pattern as face_encoding_service and object_encoding_service
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from clients.vector_service_client import get_vector_service_client

logger = logging.getLogger(__name__)


class LogoEncodingService:
    """Vector database service for logo encoding storage and matching"""
    
    COLLECTION_NAME = "logo_encodings"
    VECTOR_SIZE = 512  # CLIP embedding size
    DISTANCE = "COSINE"
    
    def __init__(self):
        self.vector_service_client = None
        self._initialized = False
        self._user_collections_ensured: set = set()
    
    @staticmethod
    def _user_collection_name(user_id: str) -> str:
        return f"user_{user_id}_logo_encodings"
    
    async def initialize(self):
        """Initialize Vector Service client and ensure global collection exists"""
        if self._initialized:
            return
        
        logger.info("Initializing Logo Encoding Service (via Vector Service)...")
        
        self.vector_service_client = await get_vector_service_client(required=False)
        
        await self._ensure_collection_exists(self.COLLECTION_NAME)
        
        self._initialized = True
        logger.info("Logo Encoding Service initialized")
    
    async def _ensure_collection_exists(self, collection_name: str) -> None:
        """Create collection if it doesn't exist via Vector Service"""
        try:
            collections_result = await self.vector_service_client.list_collections()
            if not collections_result.get("success"):
                logger.warning("Failed to list collections: %s", collections_result.get("error"))
                return
            
            collection_names = [col["name"] for col in collections_result.get("collections", [])]
            
            if collection_name not in collection_names:
                create_result = await self.vector_service_client.create_collection(
                    collection_name=collection_name,
                    vector_size=self.VECTOR_SIZE,
                    distance=self.DISTANCE,
                )
                if create_result.get("success"):
                    logger.info("Created logo encodings collection: %s", collection_name)
                else:
                    error = create_result.get("error", "Unknown error")
                    logger.error("Failed to create logo encodings collection: %s", error)
                    raise Exception(f"Failed to create collection: {error}")
            else:
                logger.debug("Logo encodings collection already exists: %s", collection_name)
        except Exception as e:
            logger.error("Failed to ensure collection exists: %s", e)
            raise
    
    async def _ensure_user_collection_exists(self, user_id: str) -> None:
        """Ensure per-user logo_encodings collection exists (lazy)"""
        if user_id in self._user_collections_ensured:
            return
        name = self._user_collection_name(user_id)
        await self._ensure_collection_exists(name)
        self._user_collections_ensured.add(user_id)
    
    async def add_logo_encoding(
        self,
        brand_name: str,
        logo_embedding: List[float],
        source_document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Add a logo encoding to the vector database (global and user collection)
        
        Args:
            brand_name: Name of the brand (e.g., "Nike", "Apple")
            logo_embedding: 512-dimensional CLIP embedding vector
            source_document_id: Document ID where this logo was annotated
            metadata: Additional metadata (logo_variant, industry, colors, etc.)
            user_id: If set, also store in per-user collection
        
        Returns:
            Point ID (UUID string)
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            point_id = str(uuid.uuid4())
            
            payload = {
                "brand_name": brand_name,
                "source_document_id": source_document_id,
                "tagged_at": datetime.utcnow().isoformat(),
                **(metadata or {}),
            }
            
            if user_id:
                await self._ensure_user_collection_exists(user_id)
            
            collections_to_write = [self.COLLECTION_NAME]
            if user_id:
                collections_to_write.append(self._user_collection_name(user_id))
            
            for coll_name in collections_to_write:
                result = await self.vector_service_client.upsert_vectors(
                    collection_name=coll_name,
                    points=[{
                        "id": point_id,
                        "vector": logo_embedding,
                        "payload": payload,
                    }],
                )
                if not result.get("success"):
                    error = result.get("error", "Unknown error")
                    raise Exception(f"Vector Service upsert failed ({coll_name}): {error}")
            
            logger.info("Stored logo encoding for '%s' (point: %s)", brand_name, point_id)
            return point_id
        
        except Exception as e:
            logger.error("Failed to add logo encoding: %s", e)
            raise
    
    async def search_similar_logos(
        self,
        query_embedding: List[float],
        similarity_threshold: float = 0.70,
        top_k: int = 5,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar logos using CLIP embedding
        
        Args:
            query_embedding: 512-dimensional CLIP embedding to match
            similarity_threshold: Minimum cosine similarity (0.0-1.0)
            top_k: Number of top matches to return
            user_id: If set, also search user's logo collection
        
        Returns:
            List of matches with brand_name, similarity_score, and metadata
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            collections_to_search = [self.COLLECTION_NAME]
            if user_id:
                await self._ensure_user_collection_exists(user_id)
                collections_to_search.append(self._user_collection_name(user_id))
            
            all_results = []
            for coll_name in collections_to_search:
                results = await self.vector_service_client.search_vectors(
                    collection_name=coll_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=similarity_threshold,
                )
                all_results.extend(results)
            
            if not all_results:
                logger.debug("No logo matches found above threshold")
                return []
            
            # Sort by similarity and format results
            all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
            
            formatted_results = []
            for result in all_results[:top_k]:
                payload = result.get("payload", {})
                formatted_results.append({
                    "brand_name": payload.get("brand_name"),
                    "similarity_score": result.get("score", 0.0),
                    "logo_variant": payload.get("logo_variant"),
                    "industry": payload.get("industry"),
                    "source_document_id": payload.get("source_document_id"),
                    "colors": payload.get("colors", []),
                    "year_introduced": payload.get("year_introduced"),
                })
            
            logger.info(
                "Found %d logo matches (top: '%s' at %.1f%% similarity)",
                len(formatted_results),
                formatted_results[0]["brand_name"] if formatted_results else "N/A",
                formatted_results[0]["similarity_score"] * 100 if formatted_results else 0
            )
            
            return formatted_results
        
        except Exception as e:
            logger.error("Logo matching failed: %s", e)
            return []


# Global service instance
_logo_encoding_service: Optional[LogoEncodingService] = None


async def get_logo_encoding_service() -> LogoEncodingService:
    """Get or create global Logo Encoding Service instance"""
    global _logo_encoding_service
    if _logo_encoding_service is None:
        _logo_encoding_service = LogoEncodingService()
        await _logo_encoding_service.initialize()
    return _logo_encoding_service
```

---

## Summary

This unified CLIP-based architecture provides:

1. **Universal semantic matching** for all visual concepts
2. **User-defined custom objects** (not just YOLO classes)
3. **Consistent pattern** across all object types
4. **Scalable storage** with Qdrant + PostgreSQL
5. **Graceful degradation** with multiple fallback paths

**The key insight**: YOLO gives structure, CLIP gives meaning, user annotations give personalization.

This system will allow users to:
- Attach images to queries and get intelligent analysis
- Build their own reference library through annotations
- Match future images against their library automatically
- Get natural language explanations of what's in images

All implemented in a phased, testable, production-ready way.
