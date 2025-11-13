"""
Backend Tool Client - gRPC client for accessing backend data services
"""

import logging
import os
from typing import List, Dict, Any, Optional
import asyncio

import grpc
from protos import tool_service_pb2, tool_service_pb2_grpc

logger = logging.getLogger(__name__)


class BackendToolClient:
    """
    gRPC client for backend tool service
    
    Provides async methods to access backend data services:
    - Document search and retrieval
    - RSS feed operations
    - Entity operations
    - Weather data
    - Org-mode operations
    """
    
    def __init__(self, host: str = None, port: int = None):
        """
        Initialize backend tool client
        
        Args:
            host: Backend service host (defaults to env BACKEND_TOOL_SERVICE_HOST)
            port: Backend service port (defaults to env BACKEND_TOOL_SERVICE_PORT)
        """
        self.host = host or os.getenv('BACKEND_TOOL_SERVICE_HOST', 'backend')
        self.port = port or int(os.getenv('BACKEND_TOOL_SERVICE_PORT', '50052'))
        self.address = f'{self.host}:{self.port}'
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[tool_service_pb2_grpc.ToolServiceStub] = None
        
        logger.info(f"Backend Tool Client configured for {self.address}")
    
    async def connect(self):
        """Establish connection to backend tool service"""
        if self._channel is None:
            logger.info(f"Connecting to backend tool service at {self.address}...")
            # Increase message size limits for large responses (default is 4MB)
            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
            ]
            self._channel = grpc.aio.insecure_channel(self.address, options=options)
            self._stub = tool_service_pb2_grpc.ToolServiceStub(self._channel)
            logger.info(f"✅ Connected to backend tool service")
    
    async def close(self):
        """Close connection to backend tool service"""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
            logger.info("Disconnected from backend tool service")
    
    async def _ensure_connected(self):
        """Ensure connection is established"""
        if self._stub is None:
            await self.connect()
    
    # ===== Document Operations =====
    
    async def search_documents(
        self,
        query: str,
        user_id: str = "system",
        limit: int = 10,
        filters: List[str] = None
    ) -> Dict[str, Any]:
        """
        Search documents by query
        
        Args:
            query: Search query
            user_id: User ID for access control
            limit: Maximum number of results
            filters: Optional filters
            
        Returns:
            Dict with 'results' (list of documents) and 'total_count'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.SearchRequest(
                user_id=user_id,
                query=query,
                limit=limit,
                filters=filters or []
            )
            
            response = await self._stub.SearchDocuments(request)
            
            # Convert proto response to dict
            results = []
            for doc in response.results:
                results.append({
                    'document_id': doc.document_id,
                    'title': doc.title,
                    'filename': doc.filename,
                    'content_preview': doc.content_preview,
                    'relevance_score': doc.relevance_score,
                    'metadata': dict(doc.metadata)
                })
            
            return {
                'results': results,
                'total_count': response.total_count
            }
            
        except grpc.RpcError as e:
            logger.error(f"Document search failed: {e.code()} - {e.details()}")
            return {'results': [], 'total_count': 0, 'error': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in document search: {e}")
            return {'results': [], 'total_count': 0, 'error': str(e)}
    
    async def get_document(
        self,
        document_id: str,
        user_id: str = "system"
    ) -> Optional[Dict[str, Any]]:
        """
        Get document metadata
        
        Args:
            document_id: Document ID
            user_id: User ID for access control
            
        Returns:
            Document metadata dict or None
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.DocumentRequest(
                document_id=document_id,
                user_id=user_id
            )
            
            response = await self._stub.GetDocument(request)
            
            return {
                'document_id': response.document_id,
                'title': response.title,
                'filename': response.filename,
                'content_type': response.content_type,
                'metadata': dict(response.metadata)
            }
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                logger.warning(f"Document not found: {document_id}")
                return None
            logger.error(f"Get document failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting document: {e}")
            return None
    
    async def get_document_content(
        self,
        document_id: str,
        user_id: str = "system"
    ) -> Optional[str]:
        """
        Get full document content
        
        Args:
            document_id: Document ID
            user_id: User ID for access control
            
        Returns:
            Document content string or None
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.DocumentRequest(
                document_id=document_id,
                user_id=user_id
            )
            
            response = await self._stub.GetDocumentContent(request)
            
            return response.content
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                logger.warning(f"Document content not found: {document_id}")
                return None
            logger.error(f"Get document content failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting document content: {e}")
            return None
    
    # ===== Weather Operations =====
    
    async def get_weather(
        self,
        location: str,
        user_id: str = "system",
        data_types: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get weather data for location
        
        Args:
            location: Location name
            user_id: User ID
            data_types: Types of data to retrieve (e.g., ["current", "forecast"])
            
        Returns:
            Weather data dict or None
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.WeatherRequest(
                location=location,
                user_id=user_id,
                data_types=data_types or ["current"]
            )
            
            response = await self._stub.GetWeatherData(request)
            
            return {
                'location': response.location,
                'current_conditions': response.current_conditions,
                'forecast': list(response.forecast),
                'alerts': list(response.alerts),
                'metadata': dict(response.metadata)
            }
            
        except grpc.RpcError as e:
            logger.error(f"Get weather failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting weather: {e}")
            return None
    
    # ===== Entity Operations =====
    
    async def search_entities(
        self,
        query: str,
        user_id: str = "system",
        entity_types: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search entities
        
        Args:
            query: Search query
            user_id: User ID
            entity_types: Types of entities to search
            limit: Maximum results
            
        Returns:
            List of entity dicts
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.EntitySearchRequest(
                user_id=user_id,
                query=query,
                entity_types=entity_types or [],
                limit=limit
            )
            
            response = await self._stub.SearchEntities(request)
            
            entities = []
            for entity in response.entities:
                entities.append({
                    'entity_id': entity.entity_id,
                    'entity_type': entity.entity_type,
                    'name': entity.name,
                    'properties': dict(entity.properties)
                })
            
            return entities
            
        except grpc.RpcError as e:
            logger.error(f"Entity search failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching entities: {e}")
            return []
    
    async def get_entity(
        self,
        entity_id: str,
        user_id: str = "system"
    ) -> Optional[Dict[str, Any]]:
        """
        Get entity details
        
        Args:
            entity_id: Entity ID
            user_id: User ID
            
        Returns:
            Entity details dict or None
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.EntityRequest(
                entity_id=entity_id,
                user_id=user_id
            )
            
            response = await self._stub.GetEntity(request)
            
            return {
                'entity': {
                    'entity_id': response.entity.entity_id,
                    'entity_type': response.entity.entity_type,
                    'name': response.entity.name,
                    'properties': dict(response.entity.properties)
                },
                'related_documents': list(response.related_documents)
            }
            
        except grpc.RpcError as e:
            logger.error(f"Get entity failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting entity: {e}")
            return None
    
    # ===== Web Operations =====
    
    async def search_web(
        self,
        query: str,
        num_results: int = 15,
        user_id: str = "system"
    ) -> List[Dict[str, Any]]:
        """
        Search the web
        
        Args:
            query: Search query
            num_results: Maximum number of results
            user_id: User ID
            
        Returns:
            List of web search results
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.WebSearchRequest(
                query=query,
                num_results=num_results,
                user_id=user_id
            )
            
            response = await self._stub.SearchWeb(request)
            
            results = []
            for result in response.results:
                results.append({
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'content': result.snippet  # WebSearchResult doesn't have content field, use snippet
                })
            
            return results
            
        except grpc.RpcError as e:
            logger.error(f"Web search failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in web search: {e}")
            return []
    
    async def crawl_web_content(
        self,
        url: str = None,
        urls: List[str] = None,
        user_id: str = "system"
    ) -> List[Dict[str, Any]]:
        """
        Crawl web content from URLs
        
        Args:
            url: Single URL to crawl
            urls: Multiple URLs to crawl
            user_id: User ID
            
        Returns:
            List of crawled content
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.WebCrawlRequest(
                url=url if url else "",
                urls=urls if urls else [],
                user_id=user_id
            )
            
            response = await self._stub.CrawlWebContent(request)
            
            results = []
            for result in response.results:
                results.append({
                    'url': result.url,
                    'title': result.title,
                    'content': result.content,
                    'html': result.html,  # WebCrawlResponse (singular) has html field
                    'metadata': dict(result.metadata)
                })
            
            return results
            
        except grpc.RpcError as e:
            logger.error(f"Web crawl failed: {e.code()} - {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in web crawl: {e}")
            return []
    
    async def search_and_crawl(
        self,
        query: str,
        num_results: int = 10,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Combined search and crawl operation
        
        Args:
            query: Search query
            num_results: Number of results to crawl
            user_id: User ID
            
        Returns:
            Dict with 'search_results' and 'crawled_content'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.SearchAndCrawlRequest(
                query=query,
                num_results=num_results,
                user_id=user_id
            )
            
            response = await self._stub.SearchAndCrawl(request)
            
            search_results = []
            for result in response.search_results:
                search_results.append({
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'content': result.snippet  # WebSearchResult doesn't have content field, use snippet
                })
            
            crawled_content = []
            for result in response.crawl_results:
                crawled_content.append({
                    'url': result.url,
                    'title': result.title,
                    'content': result.content,
                    'html': result.content,  # WebCrawlResult doesn't have html field, use content
                    'metadata': dict(result.metadata)
                })
            
            return {
                'search_results': search_results,
                'crawled_content': crawled_content
            }
            
        except grpc.RpcError as e:
            logger.error(f"Search and crawl failed: {e.code()} - {e.details()}")
            return {'search_results': [], 'crawled_content': []}
        except Exception as e:
            logger.error(f"Unexpected error in search and crawl: {e}")
            return {'search_results': [], 'crawled_content': []}
    
    # ===== Query Enhancement =====
    
    async def expand_query(
        self,
        query: str,
        num_variations: int = 3,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Expand query with variations
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            user_id: User ID
            
        Returns:
            Dict with 'original_query', 'expanded_queries', 'key_entities', 'expansion_count'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.QueryExpansionRequest(
                query=query,
                num_variations=num_variations,
                user_id=user_id
            )
            
            response = await self._stub.ExpandQuery(request)
            
            return {
                'original_query': response.original_query,
                'expanded_queries': list(response.expanded_queries),
                'key_entities': list(response.key_entities),
                'expansion_count': response.expansion_count
            }
            
        except grpc.RpcError as e:
            logger.error(f"Query expansion failed: {e.code()} - {e.details()}")
            return {
                'original_query': query,
                'expanded_queries': [query],
                'key_entities': [],
                'expansion_count': 1
            }
        except Exception as e:
            logger.error(f"Unexpected error in query expansion: {e}")
            return {
                'original_query': query,
                'expanded_queries': [query],
                'key_entities': [],
                'expansion_count': 1
            }
    
    # ===== Conversation Cache =====
    
    async def search_conversation_cache(
        self,
        query: str,
        conversation_id: str = None,
        freshness_hours: int = 24,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Search conversation cache for previous research
        
        Args:
            query: Search query
            conversation_id: Conversation ID (optional)
            freshness_hours: How recent to search (hours)
            user_id: User ID
            
        Returns:
            Dict with 'cache_hit' and 'entries'
        """
        try:
            await self._ensure_connected()
            
            request = tool_service_pb2.CacheSearchRequest(
                query=query,
                conversation_id=conversation_id if conversation_id else "",
                freshness_hours=freshness_hours,
                user_id=user_id
            )
            
            response = await self._stub.SearchConversationCache(request)
            
            entries = []
            for entry in response.entries:
                entries.append({
                    'content': entry.content,
                    'timestamp': entry.timestamp,
                    'agent_name': entry.agent_name,
                    'relevance_score': entry.relevance_score
                })
            
            return {
                'cache_hit': response.cache_hit,
                'entries': entries
            }
            
        except grpc.RpcError as e:
            logger.error(f"Cache search failed: {e.code()} - {e.details()}")
            return {'cache_hit': False, 'entries': []}
        except Exception as e:
            logger.error(f"Unexpected error in cache search: {e}")
            return {'cache_hit': False, 'entries': []}


# Global client instance
_backend_tool_client: Optional[BackendToolClient] = None


async def get_backend_tool_client() -> BackendToolClient:
    """Get or create the global backend tool client"""
    global _backend_tool_client
    
    if _backend_tool_client is None:
        _backend_tool_client = BackendToolClient()
        await _backend_tool_client.connect()
    
    return _backend_tool_client


async def close_backend_tool_client():
    """Close the global backend tool client"""
    global _backend_tool_client
    
    if _backend_tool_client:
        await _backend_tool_client.close()
        _backend_tool_client = None

