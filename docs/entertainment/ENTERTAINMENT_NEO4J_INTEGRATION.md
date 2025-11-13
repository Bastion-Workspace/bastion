# Entertainment Neo4j Knowledge Graph Integration

**BULLY!** Roosevelt's Entertainment Intelligence with Graph-Powered Recommendations! 🎬📊

## Overview

The Entertainment Agent now leverages Neo4j's knowledge graph capabilities for relationship-based recommendations and analysis. Entertainment documents are automatically processed to extract entities (actors, directors, genres) and relationships (ACTED_IN, DIRECTED, HAS_GENRE) that power intelligent recommendations.

## Key Features

### ✅ Domain-Scoped Extraction
- **Tag-based filtering**: Only processes documents tagged with `movie`, `tv_show`, or `tv_episode`
- **Category filtering**: Documents with `entertainment` category
- **Namespace isolation**: Entertainment entities use `Entertainment*` labels to avoid conflicts with business/research data

### ✅ Automatic Synchronization
- **Document upload**: Automatically extracts entertainment graph data
- **Document edit**: Deletes old relationships and creates new ones (same as vector DB)
- **File watching**: Monitors file changes and updates both vector DB AND Neo4j

### ✅ Relationship-Based Recommendations
- **Shared actors**: Find movies with common cast members
- **Shared directors**: Discover works by the same directors
- **Shared genres**: Similar genre combinations
- **Weighted scoring**: Directors weighted higher than actors, actors higher than genres

## Architecture

### Components Created/Modified

**New Files:**
1. **`backend/services/entertainment_kg_extractor.py`** (360 lines)
   - Domain-specific entity extraction
   - Relationship pattern matching
   - Tag-based scoping logic

**Modified Files:**
1. **`backend/services/knowledge_graph_service.py`**
   - Added `store_entertainment_entities_and_relationships()`
   - Added `get_entertainment_recommendations()`
   - Added `get_actor_collaborations()`

2. **`backend/services/langgraph_agents/entertainment_agent.py`**
   - Added `_get_graph_recommendations()`
   - Enhanced response generation with graph data
   - Integrated Neo4j queries for recommendation type queries

3. **`backend/services/file_watcher_service.py`**
   - Added entertainment KG extraction after vector embedding
   - Ensures updates trigger both systems

4. **`backend/main.py`**
   - Added entertainment KG extraction in document update endpoint
   - Parallel processing with vector DB updates

## Entity Types and Relationships

### Node Types (with Entertainment namespace)

```cypher
// Work nodes (Movies/TV Shows/Episodes)
(:EntertainmentMovie:EntertainmentWork {
    name: "The Godfather",
    type: "Movie",
    year: 1972,
    rating: 9.2,
    runtime: 175
})

(:EntertainmentTVShow:EntertainmentWork {
    name: "Breaking Bad",
    type: "TVShow",
    seasons: 5,
    total_episodes: 62
})

// People nodes
(:EntertainmentPerson:Director {
    name: "Francis Ford Coppola",
    type: "DIRECTOR"
})

(:EntertainmentPerson:Actor {
    name: "Al Pacino",
    type: "ACTOR"
})

// Organization nodes
(:EntertainmentOrg:Studio {
    name: "Paramount Pictures",
    type: "STUDIO"
})

(:EntertainmentOrg:Network {
    name: "AMC",
    type: "NETWORK"
})

// Genre nodes
(:EntertainmentGenre {
    name: "Crime",
    type: "GENRE"
})
```

### Relationship Types

```cypher
// Creative relationships
(Director)-[:DIRECTED]->(Movie)
(Actor)-[:ACTED_IN {character: "Michael Corleone"}]->(Movie)
(Writer)-[:WROTE]->(Movie)
(Creator)-[:CREATED]->(TVShow)

// Business relationships
(Movie)-[:PRODUCED_BY]->(Studio)
(TVShow)-[:AIRED_ON]->(Network)

// Classification
(Movie)-[:HAS_GENRE]->(Genre)

// Document reference
(Entity)-[:MENTIONED_IN]->(Document)
```

## Extraction Process

### 1. Document Upload/Edit

```
User uploads: "The Godfather (1972).md"
  ↓
Tag check: ["movie", "crime", "drama"]
  ↓
Category check: "entertainment"
  ↓
✅ Passes entertainment filter → Extract entities
```

### 2. Entity Extraction

```python
# From markdown content:
**Director**: Francis Ford Coppola
**Stars**: Marlon Brando, Al Pacino, James Caan
**Genre**: Crime, Drama
**Studio**: Paramount Pictures

# Extracted entities:
- Francis Ford Coppola (DIRECTOR)
- Marlon Brando (ACTOR)
- Al Pacino (ACTOR)
- James Caan (ACTOR)
- Crime (GENRE)
- Drama (GENRE)
- Paramount Pictures (STUDIO)
```

### 3. Relationship Creation

```cypher
// Create work node
MERGE (m:EntertainmentMovie:EntertainmentWork {name: "The Godfather"})
SET m.year = 1972, m.rating = 9.2

// Create and link director
MERGE (d:EntertainmentPerson:Director {name: "Francis Ford Coppola"})
MERGE (d)-[:DIRECTED]->(m)

// Create and link actors
MERGE (a:EntertainmentPerson:Actor {name: "Al Pacino"})
MERGE (a)-[:ACTED_IN {character: "Michael Corleone"}]->(m)

// Link to genres
MERGE (g:EntertainmentGenre {name: "Crime"})
MERGE (m)-[:HAS_GENRE]->(g)
```

## Recommendation Algorithm

### Query Flow

```
User: "Recommend movies like Inception"
  ↓
1. Intent Classification → "recommendation"
  ↓
2. Extract title: "Inception"
  ↓
3. Neo4j Query:
   - Find works sharing actors with Inception
   - Find works sharing directors with Inception
   - Find works sharing genres with Inception
  ↓
4. Calculate weighted score:
   - Shared directors × 5
   - Shared actors × 3
   - Shared genres × 2
  ↓
5. Combine with vector search results
  ↓
6. LLM generates response with explanations
```

### Example Neo4j Query

```cypher
MATCH (seed {name: "Inception"})
WHERE 'EntertainmentMovie' IN labels(seed)

// Find works sharing actors
OPTIONAL MATCH (seed)<-[:ACTED_IN]-(actor)-[:ACTED_IN]->(rec)
WHERE rec <> seed
WITH rec, count(DISTINCT actor) as shared_actors

// Find works sharing directors
OPTIONAL MATCH (seed)<-[:DIRECTED]-(director)-[:DIRECTED]->(rec)
WITH rec, shared_actors, count(DISTINCT director) as shared_directors

// Find works sharing genres
OPTIONAL MATCH (seed)-[:HAS_GENRE]->(genre)<-[:HAS_GENRE]-(rec)
WITH rec, shared_actors, shared_directors, count(DISTINCT genre) as shared_genres

// Calculate recommendation score
WITH rec, (shared_actors * 3 + shared_directors * 5 + shared_genres * 2) as score
WHERE score > 0

RETURN rec.name, rec.year, rec.rating, score,
       shared_actors, shared_directors, shared_genres
ORDER BY score DESC
LIMIT 10
```

## Update Synchronization

### File Edit Flow

```
User edits: "Breaking Bad.md"
  ↓
File watcher detects change
  ↓
1. Delete old vector chunks
2. Delete old KG entities (including entertainment relationships)
  ↓
3. Re-process content
  ↓
4. Store new vector chunks
5. Extract new entertainment entities
6. Create new entertainment relationships
  ↓
✅ Both systems synchronized
```

### API Edit Flow

```
PUT /api/documents/{doc_id}/content
  ↓
1. Save file to disk
2. Delete old vectors from Qdrant
3. Delete old entities from Neo4j
  ↓
4. Re-chunk content
5. Re-embed chunks
6. Extract entertainment entities
7. Store in Neo4j
  ↓
✅ Complete re-indexing
```

## Tag-Based Scoping

### Entertainment Documents → Entertainment Graph

```python
# Example: The Godfather (1972).md
tags: ["movie", "crime", "drama", "classic"]
category: "entertainment"

→ Extracts:
  - Work: The Godfather (Movie)
  - People: Al Pacino, Marlon Brando (Actors)
  - Genres: Crime, Drama
  
→ Creates entertainment-namespaced relationships
```

### Business Documents → Business Graph

```python
# Example: Worldcom Financial Report.pdf
tags: ["financial", "worldcom", "corporate"]
category: "business"

→ Extracts:
  - Organization: Worldcom (Corporation)
  - People: Bernie Ebbers (Executive)
  - Financial data
  
→ Creates business-namespaced relationships
→ NO entertainment extraction!
```

## Query Examples

### 1. Actor Filmography

```cypher
MATCH (actor:EntertainmentPerson:Actor {name: "Al Pacino"})
      -[:ACTED_IN]->(work)
OPTIONAL MATCH (work)<-[:DIRECTED]-(director)
RETURN work.name, work.year, work.rating, 
       collect(DISTINCT director.name) as directors
ORDER BY work.year DESC
```

### 2. Director Collaborations

```cypher
MATCH (director:EntertainmentPerson:Director {name: "Christopher Nolan"})
      -[:DIRECTED]->(movie)<-[:ACTED_IN]-(actor)
RETURN actor.name, count(movie) as collaborations
ORDER BY collaborations DESC
```

### 3. Genre Evolution

```cypher
MATCH (actor:EntertainmentPerson {name: "Tom Hanks"})
      -[:ACTED_IN]->(movie)-[:HAS_GENRE]->(genre)
RETURN genre.name, count(movie) as film_count
ORDER BY film_count DESC
```

## Benefits Over Tags Alone

| Feature | Tags Only | Neo4j Graph |
|---------|-----------|-------------|
| **Find collaborations** | ❌ | ✅ "Actors who worked with Nolan's regulars" |
| **Multi-hop queries** | ❌ | ✅ "Directors of actors in X's movies" |
| **Weighted recommendations** | ❌ | ✅ Score by relationship strength |
| **Career analysis** | ❌ | ✅ Timeline of actor/director work |
| **Network effects** | ❌ | ✅ "Six degrees of separation" |
| **Explicit relationships** | ⚠️ Implicit | ✅ ACTED_IN, DIRECTED relationships |

## Usage

### 1. Upload Entertainment Document

```bash
# Using templates from docs/ENTERTAINMENT_AGENT_TEMPLATES.md
curl -X POST /api/documents/upload \
  -F "file=@The_Godfather_1972.md" \
  -F "category=entertainment" \
  -F "tags=movie,crime,drama"
```

**What happens:**
1. File saved to disk
2. Content vectorized → Qdrant
3. Entertainment entities extracted → Neo4j
4. Relationships created → Neo4j

### 2. Query for Recommendations

```
User: "Recommend crime movies like The Godfather"

→ Entertainment Agent:
  1. Searches documents (vector search)
  2. Queries Neo4j graph (relationship search)
  3. Combines results
  4. Returns: Goodfellas, The Departed, Casino
     (Explains shared actors, directors, genres)
```

### 3. Edit Document

```bash
PUT /api/documents/{doc_id}/content
{
  "content": "# The Godfather (1972)\n\n**NEW CONTENT**..."
}
```

**What happens:**
1. Old vectors deleted
2. Old graph relationships deleted
3. New vectors created
4. New graph relationships created
5. ✅ Fully synchronized

## File Size Compliance

All files respect the 500-line limit:
- ✅ `entertainment_kg_extractor.py`: 360 lines
- ✅ `entertainment_agent.py`: 493 lines (enhanced)
- ✅ `knowledge_graph_service.py`: 641 lines (added 3 methods)

## Future Enhancements

**Potential additions:**
1. **Awards tracking**: Oscar/Emmy wins and nominations
2. **Franchise relationships**: SEQUEL_TO, PART_OF_FRANCHISE
3. **Similar works scoring**: SIMILAR_TO with ML-calculated similarity
4. **Character networks**: Track character appearances across works
5. **Temporal analysis**: Career trajectory visualization

**By George!** Entertainment recommendations now powered by the full might of Neo4j's graph traversal capabilities! 🎬📊🏇

