"""
Knowledge Graph Service - Manages entities and relationships extraction
"""

import logging
import time
from typing import List, Dict, Any, Optional

from neo4j import AsyncGraphDatabase

from config import settings
from models.api_models import Entity, Chunk

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Service for knowledge graph management"""

    def __init__(self):
        self.driver = None
        self.enabled = False
        self._last_unavailable_log_monotonic = 0.0
        self._startup_degraded_logged = False

    def is_connected(self) -> bool:
        """True when Neo4j driver is active and the service is marked enabled."""
        return bool(self.enabled and self.driver is not None)

    def _available(self) -> bool:
        return self.is_connected()

    async def _mark_unavailable(self, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_unavailable_log_monotonic > 60.0:
            self._last_unavailable_log_monotonic = now
            logger.warning("Neo4j marked unavailable: %s", reason)
        self.enabled = False
        if self.driver:
            try:
                await self.driver.close()
            except Exception:
                pass
            self.driver = None

    async def initialize(self):
        """Initialize Neo4j when enabled; degrade gracefully unless NEO4J_REQUIRED."""
        self.enabled = False
        if not getattr(settings, "NEO4J_ENABLED", True):
            logger.info("Knowledge graph disabled (NEO4J_ENABLED=false)")
            self.driver = None
            return
        logger.debug("Initializing Knowledge Graph Service")
        try:
            self.driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            await self.driver.verify_connectivity()
            await self._setup_schema()
            self.enabled = True
            logger.debug("Knowledge Graph Service initialized")
        except Exception as e:
            self.enabled = False
            if self.driver:
                try:
                    await self.driver.close()
                except Exception:
                    pass
                self.driver = None
            if getattr(settings, "NEO4J_REQUIRED", False):
                logger.error("Failed to initialize Neo4j (required): %s", e)
                raise
            if not self._startup_degraded_logged:
                self._startup_degraded_logged = True
                logger.warning(
                    "Neo4j unavailable at startup; running without knowledge graph (%s)",
                    e,
                )

    async def try_reconnect(self) -> bool:
        """Attempt to establish Neo4j after a prior failure. Returns True if connected."""
        if not getattr(settings, "NEO4J_ENABLED", True):
            return False
        if self._available():
            return True
        try:
            self.driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            await self.driver.verify_connectivity()
            await self._setup_schema()
            self.enabled = True
            logger.info("Neo4j connection restored")
            return True
        except Exception as e:
            logger.debug("Neo4j reconnect attempt failed: %s", e)
            self.enabled = False
            if self.driver:
                try:
                    await self.driver.close()
                except Exception:
                    pass
                self.driver = None
            return False
    
    async def _setup_schema(self):
        """Set up Neo4j schema with indexes and constraints"""
        async with self.driver.session() as session:
            # Create constraints and indexes
            queries = [
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX document_title IF NOT EXISTS FOR (d:Document) ON (d.title)",
                "CREATE INDEX chunk_document IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            ]
            
            for query in queries:
                try:
                    await session.run(query)
                except Exception as e:
                    # Constraint might already exist
                    logger.debug(f"Schema setup query failed (expected): {e}")

    async def _write_entities_to_neo4j(
        self,
        document_id: str,
        entities: List[Entity],
        chunks: Optional[List[Chunk]] = None,
    ) -> bool:
        """Execute MERGE graph writes. Caller must ensure driver is available."""
        try:
            async with self.driver.session() as session:
                if chunks:
                    for ch in chunks:
                        await session.run(
                            """
                            MERGE (d:Document {id: $doc_id})
                            MERGE (c:Chunk {id: $chunk_id})
                            SET c.document_id = $doc_id, c.chunk_index = $chunk_index
                            MERGE (c)-[:PART_OF]->(d)
                            """,
                            doc_id=document_id,
                            chunk_id=ch.chunk_id,
                            chunk_index=ch.chunk_index,
                        )
                for entity in entities:
                    context = (entity.metadata or {}).get("context", "") or ""
                    await session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, e.confidence = $confidence
                        WITH e
                        MERGE (d:Document {id: $doc_id})
                        MERGE (e)-[r:MENTIONED_IN]->(d)
                        SET r.context = $context
                        """,
                        name=entity.name,
                        type=entity.entity_type,
                        confidence=entity.confidence,
                        doc_id=document_id,
                        context=context[:2000] if context else "",
                    )
                    chunk_ids = (entity.metadata or {}).get("chunk_ids", [])
                    if chunks and chunk_ids:
                        for cid in chunk_ids:
                            await session.run(
                                """
                                MATCH (e:Entity {name: $name})
                                MATCH (c:Chunk {id: $chunk_id})
                                WHERE c.document_id = $doc_id
                                MERGE (e)-[:APPEARS_IN]->(c)
                                """,
                                name=entity.name,
                                chunk_id=cid,
                                doc_id=document_id,
                            )
            logger.info("Stored %s entities for document %s", len(entities), document_id)
            return True
        except Exception as e:
            logger.warning(
                "Neo4j entity storage failed for document %s (non-fatal): %s",
                document_id,
                e,
            )
            return False

    async def replay_store_entities(
        self,
        document_id: str,
        entities: List[Entity],
        chunks: Optional[List[Chunk]] = None,
    ) -> bool:
        """Replay a backlog store_entities row; no backlog enqueue on failure."""
        if not entities or not self._available():
            return False
        ok = await self._write_entities_to_neo4j(document_id, entities, chunks)
        if ok:
            logger.info("Replayed backlog entities for document %s", document_id)
        return ok

    async def replay_delete_document(self, document_id: str) -> bool:
        if not self._available():
            return False
        try:
            async with self.driver.session() as session:
                await session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    DETACH DELETE d
                    """,
                    doc_id=document_id,
                )
                await session.run(
                    """
                    MATCH (e:Entity)
                    WHERE NOT (e)-[:MENTIONED_IN]->()
                    DELETE e
                    """
                )
            return True
        except Exception as e:
            logger.warning("Neo4j replay delete failed for %s: %s", document_id, e)
            return False

    async def store_entities(
        self,
        entities: List[Entity],
        document_id: str,
        chunks: Optional[List[Chunk]] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Store extracted entities with optional chunk-level links for co-occurrence.
        When chunks are provided, creates Chunk nodes, PART_OF->Document, and APPEARS_IN
        from entities to chunks so co-occurrence can be computed at chunk level.

        Returns True on success, False on failure. Never raises — a Neo4j outage
        should not abort the wider document processing pipeline.
        """
        if not entities:
            return True
        if not getattr(settings, "NEO4J_ENABLED", True):
            return True
        from services import kg_write_backlog

        if not self._available():
            await kg_write_backlog.enqueue_store_entities(
                document_id, user_id, entities, chunks
            )
            return False
        ok = await self._write_entities_to_neo4j(document_id, entities, chunks)
        if not ok:
            await self._mark_unavailable("store_entities failed")
            await kg_write_backlog.enqueue_store_entities(
                document_id, user_id, entities, chunks
            )
        return ok

    async def get_entity_detail(
        self, entity_name: str, user_document_ids: List[str]
    ) -> Dict[str, Any]:
        """Return entity type/confidence, per-document mentions with context, and co-occurring entities."""
        if not user_document_ids:
            return {
                "name": entity_name,
                "entity_type": None,
                "confidence": None,
                "document_mentions": [],
                "co_occurring_entities": [],
            }
        if not self._available():
            return {
                "name": entity_name,
                "entity_type": None,
                "confidence": None,
                "document_mentions": [],
                "co_occurring_entities": [],
            }
        try:
            async with self.driver.session() as session:
                # Entity + per-doc mentions with context
                result = await session.run(
                    """
                    MATCH (e:Entity {name: $name})-[r:MENTIONED_IN]->(d:Document)
                    WHERE d.id IN $doc_ids
                    RETURN e.type AS entity_type, e.confidence AS confidence,
                           d.id AS document_id, r.context AS context
                    """,
                    name=entity_name,
                    doc_ids=user_document_ids,
                )
                entity_type = None
                confidence = None
                document_mentions = []
                doc_mention_map = {}
                async for record in result:
                    if entity_type is None:
                        entity_type = record["entity_type"]
                        confidence = record["confidence"]
                    doc_id = record["document_id"]
                    doc_mention_map[doc_id] = {
                        "document_id": doc_id,
                        "context": (record["context"] or "").strip(),
                        "chunks": [],
                    }

                # Chunk-level mentions (when APPEARS_IN exists)
                chunk_result = await session.run(
                    """
                    MATCH (e:Entity {name: $name})-[:APPEARS_IN]->(c:Chunk)-[:PART_OF]->(d:Document)
                    WHERE d.id IN $doc_ids
                    RETURN d.id AS document_id, c.id AS chunk_id, c.chunk_index AS chunk_index
                    ORDER BY d.id, c.chunk_index
                    """,
                    name=entity_name,
                    doc_ids=user_document_ids,
                )
                async for record in chunk_result:
                    doc_id = record["document_id"]
                    if doc_id in doc_mention_map:
                        doc_mention_map[doc_id]["chunks"].append({
                            "chunk_id": record["chunk_id"],
                            "chunk_index": record["chunk_index"],
                        })
                document_mentions = list(doc_mention_map.values())

                # Co-occurring entities (prefer same chunk, then same docs)
                co_result = await session.run(
                    """
                    MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(d:Document)<-[:MENTIONED_IN]-(co:Entity)
                    WHERE d.id IN $doc_ids AND co.name <> $name
                    WITH co.name AS name, co.type AS entity_type, count(DISTINCT d) AS shared_docs
                    RETURN name, entity_type, shared_docs
                    ORDER BY shared_docs DESC
                    LIMIT 10
                    """,
                    name=entity_name,
                    doc_ids=user_document_ids,
                )
                co_occurring = []
                async for record in co_result:
                    co_occurring.append({
                        "name": record["name"],
                        "entity_type": record["entity_type"] or "MISC",
                        "shared_docs": record["shared_docs"],
                    })

                return {
                    "name": entity_name,
                    "entity_type": entity_type,
                    "confidence": confidence,
                    "document_mentions": document_mentions,
                    "co_occurring_entities": co_occurring,
                }
        except Exception as e:
            logger.error(f"Failed to get entity detail for {entity_name}: {e}")
            return {
                "name": entity_name,
                "entity_type": None,
                "confidence": None,
                "document_mentions": [],
                "co_occurring_entities": [],
            }
    
    async def get_entities(self, entity_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get entities from the knowledge graph"""
        if not self._available():
            return []
        async with self.driver.session() as session:
            if entity_type:
                result = await session.run(
                    """
                    MATCH (e:Entity {type: $type})
                    RETURN e.name as name, e.type as type, e.confidence as confidence
                    LIMIT $limit
                    """,
                    type=entity_type,
                    limit=limit
                )
            else:
                result = await session.run(
                    """
                    MATCH (e:Entity)
                    RETURN e.name as name, e.type as type, e.confidence as confidence
                    LIMIT $limit
                    """,
                    limit=limit
                )
            
            entities = []
            async for record in result:
                entities.append({
                    "name": record["name"],
                    "type": record["type"],
                    "confidence": record["confidence"]
                })
            
            return entities
    
    async def get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get relationships for a specific entity"""
        if not self._available():
            return []
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {name: $name})-[r]->(other)
                RETURN type(r) as relationship_type, other.name as target_name, 
                       labels(other) as target_labels
                UNION
                MATCH (other)-[r]->(e:Entity {name: $name})
                RETURN type(r) as relationship_type, other.name as target_name,
                       labels(other) as target_labels
                """,
                name=entity_name
            )
            
            relationships = []
            async for record in result:
                relationships.append({
                    "relationship_type": record["relationship_type"],
                    "target_name": record["target_name"],
                    "target_labels": record["target_labels"]
                })
            
            return relationships
    
    async def find_related_entities(self, entity_name: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Find entities related to a given entity within max_hops"""
        if not self._available():
            return []
        async with self.driver.session() as session:
            result = await session.run(
                f"""
                MATCH path = (start:Entity {{name: $name}})-[*1..{max_hops}]-(related:Entity)
                WHERE start <> related
                RETURN DISTINCT related.name as name, related.type as type, 
                       length(path) as distance
                ORDER BY distance, related.name
                """,
                name=entity_name
            )
            
            related = []
            async for record in result:
                related.append({
                    "name": record["name"],
                    "type": record["type"],
                    "distance": record["distance"]
                })
            
            return related
    
    async def get_document_entities(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all entities mentioned in a specific document"""
        if not self._available():
            return []
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document {id: $doc_id})
                RETURN e.name as name, e.type as type, e.confidence as confidence
                ORDER BY e.confidence DESC
                """,
                doc_id=document_id
            )
            
            entities = []
            async for record in result:
                entities.append({
                    "name": record["name"],
                    "type": record["type"],
                    "confidence": record["confidence"]
                })
            
            return entities
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        if not self._available():
            return {
                "total_entities": 0,
                "total_documents": 0,
                "total_relationships": 0,
                "entity_types": {},
            }
        try:
            async with self.driver.session() as session:
                # Get entity count
                entity_result = await session.run("MATCH (e:Entity) RETURN count(e) as entity_count")
                entity_record = await entity_result.single()
                entity_count = entity_record["entity_count"] if entity_record else 0
                
                # Get document count
                doc_result = await session.run("MATCH (d:Document) RETURN count(d) as doc_count")
                doc_record = await doc_result.single()
                doc_count = doc_record["doc_count"] if doc_record else 0
                
                # Get relationship count
                rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_record = await rel_result.single()
                rel_count = rel_record["rel_count"] if rel_record else 0
                
                # Get entity types
                types_result = await session.run(
                    "MATCH (e:Entity) RETURN e.type as type, count(e) as count ORDER BY count DESC"
                )
                entity_types = {}
                async for record in types_result:
                    entity_types[record["type"]] = record["count"]
                
                return {
                    "total_entities": entity_count,
                    "total_documents": doc_count,
                    "total_relationships": rel_count,
                    "entity_types": entity_types
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get graph stats: {e}")
            return {
                "total_entities": 0,
                "total_documents": 0,
                "total_relationships": 0,
                "entity_types": {}
            }

    async def get_entity_graph_for_documents(
        self,
        document_ids: List[str],
        entity_limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Return entity nodes and co-occurrence edges for the given document set.
        Used by the entity graph visualization; document_ids are typically
        the current user's documents (RLS applied by caller).
        """
        if not document_ids:
            return {
                "entity_nodes": [],
                "document_ids": [],
                "co_occurrence_edges": [],
            }
        if not self._available():
            return {
                "entity_nodes": [],
                "document_ids": [],
                "co_occurrence_edges": [],
            }
        try:
            async with self.driver.session() as session:
                entity_result = await session.run(
                    """
                    MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
                    WHERE d.id IN $document_ids
                    WITH e, collect(d.id) AS doc_ids, count(d) AS doc_count
                    RETURN e.name AS name, e.type AS type, doc_ids, doc_count
                    ORDER BY doc_count DESC
                    LIMIT $entity_limit
                    """,
                    document_ids=document_ids,
                    entity_limit=entity_limit,
                )
                entity_nodes = []
                all_doc_ids = set()
                async for record in entity_result:
                    doc_ids_list = record["doc_ids"] or []
                    all_doc_ids.update(doc_ids_list)
                    entity_nodes.append({
                        "name": record["name"],
                        "type": record["type"] or "MISC",
                        "doc_ids": doc_ids_list,
                        "doc_count": record["doc_count"] or 0,
                    })
                entity_names = [n["name"] for n in entity_nodes]
                if not entity_names:
                    return {
                        "entity_nodes": [],
                        "document_ids": list(all_doc_ids),
                        "co_occurrence_edges": [],
                    }
                # Prefer chunk-level co-occurrence (entities in same chunk); fall back to document-level
                edge_result = await session.run(
                    """
                    MATCH (e1:Entity)-[:APPEARS_IN]->(c:Chunk)<-[:APPEARS_IN]-(e2:Entity)
                    WHERE c.document_id IN $document_ids AND e1.name IN $entity_names AND e2.name IN $entity_names
                    AND e1.name < e2.name
                    WITH e1.name AS source, e2.name AS target, count(DISTINCT c) AS weight
                    RETURN source, target, weight
                    ORDER BY weight DESC
                    """,
                    document_ids=document_ids,
                    entity_names=entity_names,
                )
                co_occurrence_edges = []
                async for record in edge_result:
                    co_occurrence_edges.append({
                        "source": record["source"],
                        "target": record["target"],
                        "weight": record["weight"] or 0,
                    })
                if not co_occurrence_edges:
                    doc_edge_result = await session.run(
                        """
                        MATCH (e1:Entity)-[:MENTIONED_IN]->(d:Document)<-[:MENTIONED_IN]-(e2:Entity)
                        WHERE d.id IN $document_ids AND e1.name IN $entity_names AND e2.name IN $entity_names
                        AND e1.name < e2.name
                        WITH e1.name AS source, e2.name AS target, count(d) AS weight
                        RETURN source, target, weight
                        ORDER BY weight DESC
                        """,
                        document_ids=document_ids,
                        entity_names=entity_names,
                    )
                    async for record in doc_edge_result:
                        co_occurrence_edges.append({
                            "source": record["source"],
                            "target": record["target"],
                            "weight": record["weight"] or 0,
                        })
                return {
                    "entity_nodes": entity_nodes,
                    "document_ids": list(all_doc_ids),
                    "co_occurrence_edges": co_occurrence_edges,
                }
        except Exception as e:
            logger.error(f"Failed to get entity graph for documents: {e}")
            return {
                "entity_nodes": [],
                "document_ids": [],
                "co_occurrence_edges": [],
            }

    async def check_health(self) -> bool:
        """Check Neo4j health"""
        if not self._available():
            return False
        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as health_check")
                await result.single()
                return True
        except Exception as e:
            logger.error(f"❌ Neo4j health check failed: {e}")
            return False
    
    async def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using simple keyword matching
        
        ⚠️  **DEPRECATED:** This primitive method is kept for backwards compatibility only.
        Use DocumentProcessor._extract_entities() for proper spaCy NER instead.
        
        For proper entity extraction:
        ```python
        from utils.document_processor import get_document_processor
        doc_processor = await get_document_processor()
        entities = await doc_processor._extract_entities(text, [])
        ```
        """
        logger.warning("⚠️  Using deprecated primitive entity extraction. Use DocumentProcessor for spaCy NER!")
        
        # This is a placeholder - in production you'd use a proper NER model
        entities = []
        
        # Simple keyword-based entity extraction
        # In practice, you'd use spaCy, transformers, or another NER library
        words = text.split()
        
        # Look for capitalized words (potential names)
        for word in words:
            if len(word) > 2 and word[0].isupper():
                entities.append({
                    'name': word,
                    'type': 'PERSON',  # Default type
                    'confidence': 0.8
                })
        
        return entities[:10]  # Limit to 10 entities
    
    async def find_documents_by_entities(self, entity_names: List[str]) -> List[str]:
        """Find documents that mention any of the given entities"""
        if not entity_names:
            return []
        if not self._available():
            return []
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
                WHERE e.name IN $entity_names
                RETURN DISTINCT d.id as document_id
                """,
                entity_names=entity_names
            )
            
            document_ids = []
            async for record in result:
                document_ids.append(record["document_id"])
            
            return document_ids
    
    async def find_related_documents_by_entities(self, entity_names: List[str], max_hops: int = 2) -> List[str]:
        """Find documents mentioning entities related to the given entities"""
        if not entity_names:
            return []
        if not self._available():
            return []
        async with self.driver.session() as session:
            result = await session.run(
                f"""
                MATCH (start:Entity)-[*1..{max_hops}]-(related:Entity)-[:MENTIONED_IN]->(d:Document)
                WHERE start.name IN $entity_names
                RETURN DISTINCT d.id as document_id
                """,
                entity_names=entity_names
            )
            
            document_ids = []
            async for record in result:
                document_ids.append(record["document_id"])
            
            return document_ids
    
    async def get_entity_importance_scores(self, entity_names: List[str]) -> Dict[str, float]:
        """Calculate importance scores for entities based on document frequency and centrality"""
        if not entity_names:
            return {}
        if not self._available():
            return {}
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
                WHERE e.name IN $entity_names
                WITH e, count(d) as doc_count
                MATCH (e)-[*1..2]-(related:Entity)
                WITH e, doc_count, count(DISTINCT related) as centrality
                RETURN e.name as name, doc_count, centrality
                """,
                entity_names=entity_names
            )
            
            scores = {}
            async for record in result:
                # Simple scoring: combine document frequency and centrality
                doc_freq = record["doc_count"]
                centrality = record["centrality"]
                # Normalize and combine (you can adjust this formula)
                score = (doc_freq * 0.7) + (centrality * 0.3)
                scores[record["name"]] = score
            
            return scores
    
    async def find_co_occurring_entities(self, entity_names: List[str], min_co_occurrences: int = 2) -> List[Dict[str, Any]]:
        """Find entities that frequently co-occur with the given entities"""
        if not entity_names:
            return []
        if not self._available():
            return []
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (target:Entity)-[:MENTIONED_IN]->(d:Document)<-[:MENTIONED_IN]-(cooccur:Entity)
                WHERE target.name IN $entity_names AND target <> cooccur
                WITH cooccur, count(DISTINCT d) as co_occurrence_count
                WHERE co_occurrence_count >= $min_count
                RETURN cooccur.name as name, cooccur.type as type, co_occurrence_count
                ORDER BY co_occurrence_count DESC
                """,
                entity_names=entity_names,
                min_count=min_co_occurrences
            )
            
            co_occurring = []
            async for record in result:
                co_occurring.append({
                    "name": record["name"],
                    "type": record["type"],
                    "co_occurrence_count": record["co_occurrence_count"]
                })
            
            return co_occurring
    
    async def get_document_similarity_by_entities(self, document_id: str, min_shared_entities: int = 2) -> List[Dict[str, Any]]:
        """Find documents similar to the given document based on shared entities"""
        if not self._available():
            return []
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (d1:Document {id: $doc_id})<-[:MENTIONED_IN]-(e:Entity)-[:MENTIONED_IN]->(d2:Document)
                WHERE d1 <> d2
                WITH d2, count(e) as shared_entities
                WHERE shared_entities >= $min_shared
                RETURN d2.id as document_id, shared_entities
                ORDER BY shared_entities DESC
                """,
                doc_id=document_id,
                min_shared=min_shared_entities
            )
            
            similar_docs = []
            async for record in result:
                similar_docs.append({
                    "document_id": record["document_id"],
                    "shared_entities": record["shared_entities"]
                })
            
            return similar_docs

    async def delete_document_entities(
        self, document_id: str, user_id: Optional[str] = None
    ) -> None:
        """Delete all entities and relationships for a specific document."""
        from services import kg_write_backlog

        if not getattr(settings, "NEO4J_ENABLED", True):
            return
        if not self._available():
            await kg_write_backlog.enqueue_delete_document(document_id, user_id)
            return
        try:
            async with self.driver.session() as session:
                await session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    DETACH DELETE d
                    """,
                    doc_id=document_id,
                )
                await session.run(
                    """
                    MATCH (e:Entity)
                    WHERE NOT (e)-[:MENTIONED_IN]->()
                    DELETE e
                    """
                )
                logger.info("Deleted knowledge graph entities for document %s", document_id)
        except Exception as e:
            logger.error("Failed to delete document entities from knowledge graph: %s", e)
            await self._mark_unavailable("delete_document_entities failed")
            await kg_write_backlog.enqueue_delete_document(document_id, user_id)

    async def clear_all_data(self):
        """Clear all entities, relationships, and documents from the knowledge graph"""
        if not self._available():
            logger.warning("clear_all_data skipped: Neo4j not connected")
            return
        try:
            async with self.driver.session() as session:
                # Delete all nodes and relationships
                await session.run("MATCH (n) DETACH DELETE n")
                
                logger.info("🗑️ Cleared all data from knowledge graph")
                
        except Exception as e:
            logger.error(f"❌ Failed to clear knowledge graph: {e}")
            raise

    async def store_entertainment_entities_and_relationships(
        self, 
        entities: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]],
        document_id: str
    ):
        """
        Store entertainment-specific entities and relationships in Neo4j
        
        Entertainment-scoped graph with namespaced labels.
        """
        if not self._available():
            logger.debug("store_entertainment_entities skipped: Neo4j not connected")
            return
        try:
            async with self.driver.session() as session:
                # Store entities with entertainment-specific labels
                for entity in entities:
                    entity_name = entity.get("name")
                    entity_type = entity.get("type")
                    entity_label = entity.get("label", "EntertainmentEntity")
                    confidence = entity.get("confidence", 0.8)
                    properties = entity.get("properties", {})
                    
                    # Create node with multiple labels for easy querying
                    labels_str = ":".join(entity_label.split(":"))
                    
                    # Build properties string
                    props_list = [f"name: $name", f"type: $type", f"confidence: $confidence"]
                    for key, value in properties.items():
                        props_list.append(f"{key}: ${key}")
                    props_str = ", ".join(props_list)
                    
                    query = f"""
                    MERGE (e:{labels_str} {{name: $name}})
                    SET e.type = $type, e.confidence = $confidence
                    """
                    
                    # Add additional properties
                    for key in properties.keys():
                        query += f", e.{key} = ${key}"
                    
                    query += """
                    WITH e
                    MATCH (d:Document {id: $doc_id})
                    MERGE (e)-[:MENTIONED_IN]->(d)
                    """
                    
                    params = {
                        "name": entity_name,
                        "type": entity_type,
                        "confidence": confidence,
                        "doc_id": document_id,
                        **properties
                    }
                    
                    await session.run(query, **params)
                
                # Store relationships
                for rel in relationships:
                    from_name = rel.get("from_name")
                    to_name = rel.get("to_name")
                    rel_type = rel.get("relationship_type")
                    rel_properties = rel.get("properties", {})
                    
                    # Create relationship between entities
                    query = f"""
                    MATCH (from {{name: $from_name}})
                    MATCH (to {{name: $to_name}})
                    MERGE (from)-[r:{rel_type}]->(to)
                    """
                    
                    # Add relationship properties if any
                    if rel_properties:
                        prop_sets = [f"r.{key} = ${key}" for key in rel_properties.keys()]
                        query += " SET " + ", ".join(prop_sets)
                    
                    params = {
                        "from_name": from_name,
                        "to_name": to_name,
                        **rel_properties
                    }
                    
                    await session.run(query, **params)
            
            logger.info(f"🎬 Stored {len(entities)} entertainment entities, {len(relationships)} relationships for {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store entertainment entities: {e}")
            raise
    
    async def get_entertainment_recommendations(
        self, work_title: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get entertainment recommendations based on shared actors/directors/genres
        
        Graph-based recommendations using relationship traversal.
        """
        if not self._available():
            return []
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (seed {name: $title})
                WHERE 'EntertainmentMovie' IN labels(seed) OR 'EntertainmentTVShow' IN labels(seed)
                
                // Find works sharing actors
                OPTIONAL MATCH (seed)<-[:ACTED_IN]-(actor)-[:ACTED_IN]->(rec)
                WHERE rec <> seed AND ('EntertainmentMovie' IN labels(rec) OR 'EntertainmentTVShow' IN labels(rec))
                WITH rec, count(DISTINCT actor) as shared_actors
                
                // Find works sharing directors
                OPTIONAL MATCH (seed)<-[:DIRECTED]-(director)-[:DIRECTED]->(rec)
                WHERE rec <> seed
                WITH rec, shared_actors, count(DISTINCT director) as shared_directors
                
                // Find works sharing genres
                OPTIONAL MATCH (seed)-[:HAS_GENRE]->(genre)<-[:HAS_GENRE]-(rec)
                WHERE rec <> seed
                WITH rec, shared_actors, shared_directors, count(DISTINCT genre) as shared_genres
                
                // Calculate recommendation score
                WITH rec, 
                     (shared_actors * 3 + shared_directors * 5 + shared_genres * 2) as score,
                     shared_actors, shared_directors, shared_genres
                WHERE score > 0
                
                RETURN rec.name as title, 
                       rec.type as type,
                       rec.year as year,
                       rec.rating as rating,
                       score,
                       shared_actors,
                       shared_directors,
                       shared_genres
                ORDER BY score DESC
                LIMIT $limit
                """
                
                result = await session.run(query, title=work_title, limit=limit)
                
                recommendations = []
                async for record in result:
                    recommendations.append({
                        "title": record["title"],
                        "type": record["type"],
                        "year": record.get("year"),
                        "rating": record.get("rating"),
                        "score": record["score"],
                        "shared_actors": record["shared_actors"],
                        "shared_directors": record["shared_directors"],
                        "shared_genres": record["shared_genres"]
                    })
                
                return recommendations
                
        except Exception as e:
            logger.error(f"❌ Failed to get entertainment recommendations: {e}")
            return []
    
    async def get_actor_collaborations(
        self, actor_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find all works featuring a specific actor
        
        Actor filmography via graph traversal.
        """
        if not self._available():
            return []
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (actor:EntertainmentPerson {name: $actor_name})-[:ACTED_IN]->(work)
                WHERE 'EntertainmentMovie' IN labels(work) OR 'EntertainmentTVShow' IN labels(work)
                OPTIONAL MATCH (work)<-[:DIRECTED]-(director:EntertainmentPerson)
                RETURN work.name as title,
                       work.type as type,
                       work.year as year,
                       work.rating as rating,
                       collect(DISTINCT director.name) as directors
                ORDER BY work.year DESC
                LIMIT $limit
                """
                
                result = await session.run(query, actor_name=actor_name, limit=limit)
                
                works = []
                async for record in result:
                    works.append({
                        "title": record["title"],
                        "type": record["type"],
                        "year": record.get("year"),
                        "rating": record.get("rating"),
                        "directors": record["directors"]
                    })
                
                return works
                
        except Exception as e:
            logger.error(f"❌ Failed to get actor collaborations: {e}")
            return []

    async def close(self):
        """Close Neo4j connection"""
        self.enabled = False
        if self.driver:
            await self.driver.close()
            self.driver = None
        logger.info("Knowledge Graph Service closed")
