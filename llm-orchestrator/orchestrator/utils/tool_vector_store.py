"""
Tool Vector Store - gRPC Client for Knowledge Hub Tool Operations

This module acts as a lean client for the Vector Service's Tool operations.
The heavy lifting (embeddings, Qdrant storage/search) is now handled centrally
by the Vector Service "Knowledge Hub".
"""

import logging
import os
from typing import List, Dict, Any, Optional
import asyncio
import grpc

# Import vector service protos
import sys
sys.path.insert(0, '/app')
from protos import vector_service_pb2, vector_service_pb2_grpc

from config.settings import settings
from orchestrator.tools.tool_pack_registry import ToolMetadata, get_all_tools

logger = logging.getLogger(__name__)


class ToolVectorStore:
    """gRPC Client for tool vector operations in the Knowledge Hub"""
    
    def __init__(self):
        self.vector_service_stub: Optional[vector_service_pb2_grpc.VectorServiceStub] = None
        self.vector_service_channel: Optional[grpc.aio.Channel] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize connection to the centralized Vector Service"""
        try:
            # Initialize vector service gRPC client
            vector_host = settings.VECTOR_SERVICE_HOST
            vector_port = settings.VECTOR_SERVICE_PORT
            self.vector_service_channel = grpc.aio.insecure_channel(f"{vector_host}:{vector_port}")
            self.vector_service_stub = vector_service_pb2_grpc.VectorServiceStub(self.vector_service_channel)
            logger.info(f"Connected to Central Armory (Vector Service) at {vector_host}:{vector_port}")
            
            self._initialized = True
            logger.info("Tool Vector Store (gRPC Client) initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Tool Vector Store client: {e}")
            raise
    
    async def vectorize_tool(self, tool: ToolMetadata) -> bool:
        """
        Order the Knowledge Hub to vectorize and store a tool
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = vector_service_pb2.UpsertToolsRequest(
                tools=[vector_service_pb2.ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    pack=tool.pack.value,
                    keywords=tool.semantic_keywords
                )]
            )
            
            response = await self.vector_service_stub.UpsertTools(
                request, timeout=30.0
            )
            
            if response.success:
                logger.debug(f"Vectorized tool via Knowledge Hub: {tool.name}")
            else:
                logger.error(f"Knowledge Hub failed to vectorize tool {tool.name}: {response.error}")
                
            return response.success
            
        except Exception as e:
            logger.error(f"gRPC error vectorizing tool {tool.name}: {e}")
            return False
    
    async def vectorize_all_tools(self) -> Dict[str, Any]:
        """
        Vectorize all tools from the registry via the Knowledge Hub
        """
        if not self._initialized:
            await self.initialize()
        
        all_tools = get_all_tools()
        
        try:
            # Convert all tools to proto definitions
            proto_tools = [
                vector_service_pb2.ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    pack=tool.pack.value,
                    keywords=tool.semantic_keywords
                ) for tool in all_tools
            ]
            
            request = vector_service_pb2.UpsertToolsRequest(tools=proto_tools)
            response = await self.vector_service_stub.UpsertTools(request, timeout=60.0)
            
            return {
                "total": len(all_tools),
                "success": response.count if response.success else 0,
                "failures": [] if response.success else [t.name for t in all_tools]
            }
            
        except Exception as e:
            logger.error(f"Bulk vectorization via Knowledge Hub failed: {e}")
            return {"total": len(all_tools), "success": 0, "failures": [t.name for t in all_tools]}
    
    async def search_tools(
        self,
        query: str,
        top_k: int = 5,
        pack_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic tool discovery via Knowledge Hub gRPC
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            request = vector_service_pb2.SearchToolsRequest(
                query=query,
                limit=top_k,
                min_score=0.5,
                pack_filter=pack_filter or ""
            )
            
            response = await self.vector_service_stub.SearchTools(
                request, timeout=30.0
            )
            
            # Format results
            results = []
            for match in response.matches:
                results.append({
                    "tool_name": match.name,
                    "description": match.description,
                    "pack": match.pack,
                    "score": match.score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic discovery via Knowledge Hub failed: {e}")
            return []
    
    async def close(self):
        """Close gRPC channel"""
        if self.vector_service_channel:
            await self.vector_service_channel.close()
        self._initialized = False


# Global instance
_tool_vector_store: Optional[ToolVectorStore] = None


async def get_tool_vector_store() -> ToolVectorStore:
    """Get or create global tool vector store instance"""
    global _tool_vector_store
    if _tool_vector_store is None:
        _tool_vector_store = ToolVectorStore()
        await _tool_vector_store.initialize()
    return _tool_vector_store
