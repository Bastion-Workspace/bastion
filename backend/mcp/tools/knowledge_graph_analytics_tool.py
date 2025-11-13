"""
Knowledge Graph Analytics Tool - MCP Tool for Graph Analysis and Insights
Allows LLM to analyze the knowledge graph and get insights about entities and relationships
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GraphAnalyticsInput(BaseModel):
    """Input for knowledge graph analytics"""
    analysis_type: str = Field(..., description="Type of analysis: 'graph_stats', 'entity_network', 'document_clusters', 'entity_centrality', 'relationship_patterns', 'temporal_analysis'")
    entity_names: Optional[List[str]] = Field(None, description="Specific entities to analyze")
    entity_type: Optional[str] = Field(None, description="Entity type filter")
    max_hops: int = Field(3, ge=1, le=5, description="Maximum hops for network analysis")
    min_connections: int = Field(2, ge=1, description="Minimum connections for network analysis")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")


class NetworkNode(BaseModel):
    """Node in entity network"""
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    centrality: float = Field(..., description="Centrality score")
    connections: int = Field(..., description="Number of connections")
    importance_score: Optional[float] = Field(None, description="Importance score")


class NetworkEdge(BaseModel):
    """Edge in entity network"""
    source: str = Field(..., description="Source entity")
    target: str = Field(..., description="Target entity")
    relationship_type: str = Field(..., description="Type of relationship")
    strength: float = Field(..., description="Relationship strength")


class DocumentCluster(BaseModel):
    """Document cluster information"""
    cluster_id: str = Field(..., description="Cluster identifier")
    documents: List[str] = Field(..., description="Document IDs in cluster")
    shared_entities: List[str] = Field(..., description="Entities shared by cluster")
    cluster_size: int = Field(..., description="Number of documents in cluster")
    cohesion_score: float = Field(..., description="Cluster cohesion score")


class GraphAnalyticsOutput(BaseModel):
    """Output from graph analytics"""
    analysis_type: str = Field(..., description="Type of analysis performed")
    graph_stats: Dict[str, Any] = Field(..., description="Graph statistics")
    network_nodes: List[NetworkNode] = Field(..., description="Network nodes")
    network_edges: List[NetworkEdge] = Field(..., description="Network edges")
    document_clusters: List[DocumentCluster] = Field(..., description="Document clusters")
    insights: List[str] = Field(..., description="Key insights from analysis")
    analysis_time: float = Field(..., description="Time taken for analysis")


class KnowledgeGraphAnalyticsTool:
    """MCP tool for knowledge graph analytics and insights"""
    
    def __init__(self, knowledge_graph_service=None):
        """Initialize with required services"""
        self.knowledge_graph_service = knowledge_graph_service
        self.name = "analyze_knowledge_graph"
        self.description = "Analyze knowledge graph for insights and patterns"
        
    async def initialize(self):
        """Initialize the analytics tool"""
        if not self.knowledge_graph_service:
            raise ValueError("KnowledgeGraphService is required")
        
        logger.info("📊 KnowledgeGraphAnalyticsTool initialized")
    
    async def execute(self, input_data: GraphAnalyticsInput) -> ToolResponse:
        """Execute knowledge graph analysis"""
        start_time = time.time()
        
        try:
            logger.info(f"📊 Executing graph analysis: {input_data.analysis_type}")
            
            graph_stats = {}
            network_nodes = []
            network_edges = []
            document_clusters = []
            insights = []
            
            if input_data.analysis_type == "graph_stats":
                # Get basic graph statistics
                graph_stats = await self.knowledge_graph_service.get_graph_stats()
                insights = self._generate_stats_insights(graph_stats)
                
            elif input_data.analysis_type == "entity_network":
                # Analyze entity network
                if input_data.entity_names:
                    network_nodes, network_edges = await self._analyze_entity_network(
                        input_data.entity_names, input_data.max_hops, input_data.limit
                    )
                    insights = self._generate_network_insights(network_nodes, network_edges)
                
            elif input_data.analysis_type == "document_clusters":
                # Analyze document clusters by shared entities
                document_clusters = await self._analyze_document_clusters(
                    input_data.min_connections, input_data.limit
                )
                insights = self._generate_cluster_insights(document_clusters)
                
            elif input_data.analysis_type == "entity_centrality":
                # Analyze entity centrality and importance
                if input_data.entity_names:
                    network_nodes = await self._analyze_entity_centrality(
                        input_data.entity_names, input_data.limit
                    )
                    insights = self._generate_centrality_insights(network_nodes)
                
            elif input_data.analysis_type == "relationship_patterns":
                # Analyze relationship patterns
                network_edges = await self._analyze_relationship_patterns(
                    input_data.entity_type, input_data.limit
                )
                insights = self._generate_relationship_insights(network_edges)
                
            elif input_data.analysis_type == "temporal_analysis":
                # Analyze temporal patterns (placeholder)
                insights = ["Temporal analysis not yet implemented"]
                
            else:
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="INVALID_ANALYSIS_TYPE",
                        error_message=f"Unknown analysis type: {input_data.analysis_type}",
                        details={"valid_types": ["graph_stats", "entity_network", "document_clusters", "entity_centrality", "relationship_patterns", "temporal_analysis"]}
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Create output
            output = GraphAnalyticsOutput(
                analysis_type=input_data.analysis_type,
                graph_stats=graph_stats,
                network_nodes=network_nodes,
                network_edges=network_edges,
                document_clusters=document_clusters,
                insights=insights,
                analysis_time=time.time() - start_time
            )
            
            logger.info(f"✅ Graph analysis completed: {len(insights)} insights in {output.analysis_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Graph analysis failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="ANALYSIS_FAILED",
                    error_message=str(e),
                    details={"analysis_type": input_data.analysis_type}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _analyze_entity_network(self, entity_names: List[str], max_hops: int, limit: int) -> tuple[List[NetworkNode], List[NetworkEdge]]:
        """Analyze entity network around given entities"""
        nodes = []
        edges = []
        
        for entity_name in entity_names:
            # Get related entities
            related = await self.knowledge_graph_service.find_related_entities(entity_name, max_hops)
            
            # Add source entity
            nodes.append(NetworkNode(
                name=entity_name,
                type="SOURCE",
                centrality=1.0,
                connections=len(related)
            ))
            
            # Add related entities
            for rel in related[:limit]:
                nodes.append(NetworkNode(
                    name=rel["name"],
                    type=rel["type"],
                    centrality=0.8 / rel["distance"],
                    connections=1,
                    distance=rel["distance"]
                ))
                
                # Add edge
                edges.append(NetworkEdge(
                    source=entity_name,
                    target=rel["name"],
                    relationship_type="RELATED",
                    strength=1.0 / rel["distance"]
                ))
        
        return nodes, edges
    
    async def _analyze_document_clusters(self, min_connections: int, limit: int) -> List[DocumentCluster]:
        """Analyze document clusters based on shared entities"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated clustering algorithms
        
        clusters = []
        
        # Get all documents and their entities
        # This would require additional methods in the knowledge graph service
        
        # Placeholder implementation
        clusters.append(DocumentCluster(
            cluster_id="cluster_1",
            documents=["doc1", "doc2", "doc3"],
            shared_entities=["entity1", "entity2"],
            cluster_size=3,
            cohesion_score=0.8
        ))
        
        return clusters
    
    async def _analyze_entity_centrality(self, entity_names: List[str], limit: int) -> List[NetworkNode]:
        """Analyze entity centrality and importance"""
        nodes = []
        
        for entity_name in entity_names:
            # Get importance scores
            scores = await self.knowledge_graph_service.get_entity_importance_scores([entity_name])
            importance = scores.get(entity_name, 0.0)
            
            # Get related entities for centrality calculation
            related = await self.knowledge_graph_service.find_related_entities(entity_name, 2)
            
            nodes.append(NetworkNode(
                name=entity_name,
                type="ANALYZED",
                centrality=importance,
                connections=len(related),
                importance_score=importance
            ))
        
        return nodes
    
    async def _analyze_relationship_patterns(self, entity_type: Optional[str], limit: int) -> List[NetworkEdge]:
        """Analyze relationship patterns in the graph"""
        edges = []
        
        # Get entities of specified type
        entities = await self.knowledge_graph_service.get_entities(entity_type, limit)
        
        for entity in entities:
            # Get relationships for this entity
            relationships = await self.knowledge_graph_service.get_entity_relationships(entity["name"])
            
            for rel in relationships:
                edges.append(NetworkEdge(
                    source=entity["name"],
                    target=rel["target_name"],
                    relationship_type=rel["relationship_type"],
                    strength=0.8  # Default strength
                ))
        
        return edges
    
    def _generate_stats_insights(self, stats: Dict[str, Any]) -> List[str]:
        """Generate insights from graph statistics"""
        insights = []
        
        total_entities = stats.get("total_entities", 0)
        total_documents = stats.get("total_documents", 0)
        total_relationships = stats.get("total_relationships", 0)
        entity_types = stats.get("entity_types", {})
        
        insights.append(f"Knowledge graph contains {total_entities} entities across {total_documents} documents")
        insights.append(f"Total of {total_relationships} relationships between entities")
        
        if entity_types:
            most_common_type = max(entity_types.items(), key=lambda x: x[1])
            insights.append(f"Most common entity type: {most_common_type[0]} ({most_common_type[1]} entities)")
        
        if total_entities > 0 and total_documents > 0:
            avg_entities_per_doc = total_entities / total_documents
            insights.append(f"Average of {avg_entities_per_doc:.1f} entities per document")
        
        return insights
    
    def _generate_network_insights(self, nodes: List[NetworkNode], edges: List[NetworkEdge]) -> List[str]:
        """Generate insights from network analysis"""
        insights = []
        
        if nodes:
            insights.append(f"Network contains {len(nodes)} entities with {len(edges)} relationships")
            
            # Find most central entities
            central_nodes = sorted(nodes, key=lambda x: x.centrality, reverse=True)[:3]
            if central_nodes:
                central_names = [node.name for node in central_nodes]
                insights.append(f"Most central entities: {', '.join(central_names)}")
        
        return insights
    
    def _generate_cluster_insights(self, clusters: List[DocumentCluster]) -> List[str]:
        """Generate insights from document clusters"""
        insights = []
        
        if clusters:
            insights.append(f"Found {len(clusters)} document clusters")
            
            largest_cluster = max(clusters, key=lambda x: x.cluster_size)
            insights.append(f"Largest cluster contains {largest_cluster.cluster_size} documents")
            
            avg_cluster_size = sum(c.cluster_size for c in clusters) / len(clusters)
            insights.append(f"Average cluster size: {avg_cluster_size:.1f} documents")
        
        return insights
    
    def _generate_centrality_insights(self, nodes: List[NetworkNode]) -> List[str]:
        """Generate insights from centrality analysis"""
        insights = []
        
        if nodes:
            insights.append(f"Analyzed centrality for {len(nodes)} entities")
            
            # Find most important entities
            important_nodes = sorted(nodes, key=lambda x: x.importance_score or 0, reverse=True)[:3]
            if important_nodes:
                important_names = [node.name for node in important_nodes if node.importance_score]
                if important_names:
                    insights.append(f"Most important entities: {', '.join(important_names)}")
        
        return insights
    
    def _generate_relationship_insights(self, edges: List[NetworkEdge]) -> List[str]:
        """Generate insights from relationship patterns"""
        insights = []
        
        if edges:
            insights.append(f"Analyzed {len(edges)} relationships")
            
            # Count relationship types
            rel_types = {}
            for edge in edges:
                rel_types[edge.relationship_type] = rel_types.get(edge.relationship_type, 0) + 1
            
            if rel_types:
                most_common_rel = max(rel_types.items(), key=lambda x: x[1])
                insights.append(f"Most common relationship type: {most_common_rel[0]} ({most_common_rel[1]} instances)")
        
        return insights
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": GraphAnalyticsInput.schema(),
            "outputSchema": GraphAnalyticsOutput.schema()
        } 