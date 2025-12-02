"""
Embedding Manager - Handles text embedding generation and vector storage
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from qdrant_client import models
from pydantic import BaseModel, validator

from config import settings
from models.api_models import Chunk

logger = logging.getLogger(__name__)


class QueryExpansions(BaseModel):
    """Pydantic model for validating query expansion responses"""
    queries: List[str]
    
    @validator('queries')
    def validate_queries(cls, v):
        # Filter out invalid queries
        valid_queries = []
        for query in v:
            if isinstance(query, str) and len(query.strip()) > 5:
                valid_queries.append(query.strip())
        return valid_queries[:5]  # Limit to 5 expansions max


class EmbeddingManager:
    """
    Manages text embeddings and vector storage
    
    Query expansion was removed: Use QueryExpansionService and query_expansion_tool instead.
    Agents now explicitly decide when to expand queries, giving them full control.
    """
    
    def __init__(self):
        self.openai_client = None  # For embeddings
        self.qdrant_client = None
    
    async def initialize(self):
        """Initialize embedding client and vector database"""
        logger.info("🔧 Initializing Embedding Manager...")
        
        # Check if OpenAI API key is set
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        
        if settings.OPENAI_API_KEY.startswith("sk-"):
            logger.info("✅ OpenAI API key found")
        else:
            logger.warning("⚠️  OpenAI API key format may be incorrect")
        
        # Initialize OpenAI client for embeddings
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        
        # Ensure default collection exists (for backward compatibility)
        await self._ensure_collection_exists()
        
        logger.info("✅ Embedding Manager initialized")

    async def _ensure_collection_exists(self):
        """Ensure the default vector collection exists (for backward compatibility)"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if settings.VECTOR_COLLECTION_NAME not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=settings.VECTOR_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created default Qdrant collection: {settings.VECTOR_COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"❌ Failed to ensure default collection exists: {e}")
            raise

    def _get_user_collection_name(self, user_id: str) -> str:
        """Generate collection name for a specific user"""
        return f"user_{user_id}_documents"

    async def _ensure_user_collection_exists(self, user_id: str):
        """Ensure user-specific collection exists"""
        try:
            collection_name = self._get_user_collection_name(user_id)
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created user collection: {collection_name}")
        except Exception as e:
            logger.error(f"❌ Failed to ensure user collection exists: {e}")
            raise

    async def generate_embeddings(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """Generate embeddings for a list of texts with retry logic"""
        try:
            # Validate and clean input texts
            if not texts:
                logger.warning("⚠️ Empty text list provided for embedding generation")
                return []
            
            # Filter out empty or None texts
            valid_texts = []
            for i, text in enumerate(texts):
                if text is None:
                    logger.warning(f"⚠️ Text at index {i} is None, skipping")
                    continue
                if not isinstance(text, str):
                    logger.warning(f"⚠️ Text at index {i} is not a string ({type(text)}), converting")
                    text = str(text)
                if not text.strip():
                    logger.warning(f"⚠️ Text at index {i} is empty or whitespace, skipping")
                    continue
                valid_texts.append(text.strip())
            
            if not valid_texts:
                logger.error("❌ No valid texts to embed after filtering")
                return []
            
            logger.info(f"🔍 Generating embeddings for {len(valid_texts)} texts")
            logger.info(f"🔍 Valid texts: {[repr(t[:100]) + '...' if len(t) > 100 else repr(t) for t in valid_texts]}")
            
            # Check if OpenAI client is initialized
            if not self.openai_client:
                logger.error("❌ OpenAI client not initialized")
                return []
            
            # Generate embeddings with retry logic
            for attempt in range(max_retries):
                try:
                    logger.info(f"🔍 Attempt {attempt + 1}: Calling OpenAI embeddings API")
                    logger.info(f"🔍 Model: {settings.EMBEDDING_MODEL}")
                    logger.info(f"🔍 Input type: {type(valid_texts)}")
                    logger.info(f"🔍 Input length: {len(valid_texts)}")
                    
                    response = await asyncio.wait_for(
                        self.openai_client.embeddings.create(
                            model=settings.EMBEDDING_MODEL,
                            input=valid_texts,
                            encoding_format="float"
                        ),
                        timeout=30.0
                    )
                    
                    # Extract embeddings from response
                    embeddings = []
                    for data in response.data:
                        if hasattr(data, 'embedding') and data.embedding:
                            embeddings.append(data.embedding)
                        else:
                            logger.warning("⚠️ Empty embedding received from OpenAI")
                            embeddings.append([0.0] * settings.EMBEDDING_DIMENSIONS)
                    
                    logger.info(f"✅ Generated {len(embeddings)} embeddings successfully")
                    return embeddings
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ Embedding generation timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"❌ Embedding generation failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    
                    # Check if it's a non-retryable error
                    if "invalid_request_error" in error_msg.lower() or "400" in error_msg:
                        logger.error(f"❌ Non-retryable error: {error_msg}")
                        raise
                    
                    if attempt == max_retries - 1:
                        raise
                    
                    # Wait before retry with exponential backoff
                    wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                    await asyncio.sleep(wait_time)
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return []
    
    async def embed_and_store_chunks(
        self, 
        chunks: List[Chunk], 
        user_id: str = None,
        document_category: str = None,
        document_tags: List[str] = None,
        document_title: str = None,
        document_author: str = None,
        document_filename: str = None
    ):
        """
        Generate embeddings for chunks and store in user-specific vector database
        
        **ROOSEVELT METADATA DOCTRINE**: Include document category and tags in Qdrant payloads
        for intelligent search filtering and targeted agent routing!
        
        Args:
            chunks: List of text chunks to embed
            user_id: User ID for collection isolation
            document_category: Document category for filtering (e.g., "constitutional", "technical")
            document_tags: List of document tags for filtering
        """
        try:
            if not chunks:
                return
            
            # Determine collection to use
            if user_id:
                await self._ensure_user_collection_exists(user_id)
                collection_name = self._get_user_collection_name(user_id)
            else:
                collection_name = settings.VECTOR_COLLECTION_NAME
            
            # Deduplicate chunks by content hash
            unique_chunks = self._deduplicate_chunks(chunks)
            logger.info(f"📊 Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks for user {user_id or 'system'}")
            
            # Extract text content
            texts = [chunk.content for chunk in unique_chunks]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)
            
            # **BULLY!** Prepare enhanced metadata tags
            metadata_info = ""
            if document_category:
                metadata_info += f" category={document_category}"
            if document_tags:
                metadata_info += f" tags={document_tags}"  # Show actual tag values, not just count
            if document_title:
                metadata_info += f" title='{document_title}'"
            if document_author:
                metadata_info += f" author='{document_author}'"
            if document_filename:
                metadata_info += f" filename='{document_filename}'"
            
            if metadata_info:
                logger.info(f"📋 Including document metadata in vector payloads:{metadata_info}")
            
            # Prepare points for Qdrant with metadata filtering support
            points = []
            for chunk, embedding in zip(unique_chunks, embeddings):
                # Use content hash for consistent IDs
                content_hash = abs(hash(chunk.content))
                
                # **ROOSEVELT FIX**: Include category, tags, title, author, and filename for rich metadata
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "quality_score": chunk.quality_score,
                    "method": chunk.method,
                    "metadata": chunk.metadata,
                    "content_hash": content_hash,
                    "user_id": user_id  # User isolation
                }
                
                # Add category if provided
                if document_category:
                    payload["document_category"] = document_category
                
                # Add tags if provided
                if document_tags:
                    payload["document_tags"] = document_tags
                
                # Add title if provided (for meaningful document identification)
                if document_title:
                    payload["document_title"] = document_title
                
                # Add author if provided
                if document_author:
                    payload["document_author"] = document_author
                
                # Add filename if provided (fallback for title)
                if document_filename:
                    payload["document_filename"] = document_filename
                
                point = PointStruct(
                    id=content_hash,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Store in user-specific collection
            await self._store_points_with_retry(points, collection_name)
            
            collection_type = "user" if user_id else "global"
            logger.info(f"✅ Stored {len(points)} unique embeddings in {collection_type} collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to embed and store chunks for user {user_id}: {e}")
            raise
    
    async def _store_points_with_retry(self, points: List[PointStruct], collection_name: str = None, max_retries: int = None):
        """Store points with enhanced retry logic and configurable settings"""
        try:
            # Use default collection if none specified
            if collection_name is None:
                collection_name = settings.VECTOR_COLLECTION_NAME
            
            # Use configurable settings
            batch_size = settings.STORAGE_BATCH_SIZE
            max_retries = max_retries or settings.STORAGE_MAX_RETRIES
            timeout_seconds = settings.STORAGE_TIMEOUT_SECONDS
            retry_delay_base = settings.STORAGE_RETRY_DELAY_BASE
            batch_delay = settings.STORAGE_BATCH_DELAY
            
            total_stored = 0
            
            logger.info(f"📦 Storing {len(points)} points in collection '{collection_name}' in batches of {batch_size} (timeout: {timeout_seconds}s, retries: {max_retries})")
            
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                # Retry logic for each batch
                for attempt in range(max_retries):
                    try:
                        # Add configurable timeout for storage operation
                        await asyncio.wait_for(
                            self._store_single_batch_sync(batch_points, collection_name),
                            timeout=timeout_seconds
                        )
                        
                        total_stored += len(batch_points)
                        logger.info(f"📦 Stored batch {batch_num}: {len(batch_points)} embeddings ({total_stored}/{len(points)} total)")
                        break  # Success, move to next batch
                        
                    except asyncio.TimeoutError:
                        if attempt < max_retries - 1:
                            wait_time = (retry_delay_base ** attempt) * 2  # Configurable exponential backoff
                            logger.warning(f"⏳ Batch {batch_num} timed out after {timeout_seconds}s (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"❌ Batch {batch_num} failed after {max_retries} timeout attempts")
                            raise
                    
                    except Exception as e:
                        error_str = str(e)
                        if "rate limit" in error_str.lower() or "429" in error_str:
                            if attempt < max_retries - 1:
                                wait_time = (retry_delay_base ** attempt) * 5  # Longer wait for rate limits
                                logger.warning(f"⏳ Batch {batch_num} rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(f"❌ Batch {batch_num} failed after {max_retries} rate limit attempts")
                                raise
                        else:
                            logger.error(f"❌ Batch {batch_num} failed with non-retryable error: {e}")
                            raise
                
                # Configurable delay between batches to be gentle on the database
                if i + batch_size < len(points):
                    await asyncio.sleep(batch_delay)
            
            logger.info(f"✅ Successfully stored all {total_stored} embeddings in collection '{collection_name}' with enhanced resilience")
            
        except Exception as e:
            logger.error(f"❌ Failed to store points with retry in collection '{collection_name}': {e}")
            raise
    
    async def _store_single_batch_sync(self, batch_points: List[PointStruct], collection_name: str):
        """Store a single batch synchronously with proper async handling"""
        loop = asyncio.get_event_loop()
        
        # Run the Qdrant operation in a thread to avoid blocking
        await loop.run_in_executor(
            None,  # Use default thread pool
            lambda: self.qdrant_client.upsert(
                collection_name=collection_name,
                points=batch_points
            )
        )
    
    def _deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove duplicate chunks based on content similarity"""
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            # Create a normalized version for comparison
            normalized_content = self._normalize_content(chunk.content)
            content_hash = hash(normalized_content)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
            else:
                logger.debug(f"🔄 Skipping duplicate chunk: {chunk.chunk_id}")
        
        return unique_chunks
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for deduplication comparison"""
        # Remove extra whitespace and normalize
        normalized = ' '.join(content.split())
        # Remove common email artifacts that might vary
        normalized = normalized.replace('1Adam Pilbeam', '').strip()
        return normalized.lower()
    
    async def search_similar(
        self, 
        query_text: str, 
        limit: int = 500,  # Increased query results limit
        score_threshold: float = 0.7,
        user_id: str = None,
        include_adjacent_chunks: bool = False,
        filter_category: str = None,
        filter_tags: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhanced vector search - searches both global and user collections
        
        Query expansion removed: Agents now explicitly decide when to expand queries
        using the query_expansion_tool, giving them full control over search strategy.
        
        **METADATA FILTERING**: Filter search results by category and tags!
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            user_id: User ID for user-specific search
            include_adjacent_chunks: Include chunks before/after matches
            filter_category: Filter by document category (e.g., "constitutional", "technical")
            filter_tags: Filter by document tags (documents must have ALL specified tags)
        """
        try:
            if user_id:
                # Search both user and global collections
                results = await self._hybrid_search(query_text, user_id, limit, score_threshold)
            else:
                # Search only global collection
                results = await self._search_collection(settings.VECTOR_COLLECTION_NAME, query_text, limit, score_threshold)
            
            # Include adjacent chunks if requested
            if include_adjacent_chunks and results:
                results = await self._include_adjacent_chunks(results, limit, user_id)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Enhanced search failed: {e}")
            # Fallback to simple search in global collection
            return await self._simple_vector_search(query_text, limit, score_threshold, settings.VECTOR_COLLECTION_NAME)

    async def _hybrid_search(
        self,
        query_text: str,
        user_id: str,
        limit: int,
        score_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search both global and user collections, combining results"""
        try:
            logger.info(f"Hybrid search for user {user_id}: {query_text[:50]}...")
            
            import asyncio
            
            # Search both collections in parallel
            global_task = asyncio.create_task(
                self._search_collection(
                    settings.VECTOR_COLLECTION_NAME, 
                    query_text, 
                    limit // 2,  # Split limit between collections
                    score_threshold
                )
            )
            
            user_collection_name = self._get_user_collection_name(user_id)
            user_task = asyncio.create_task(
                self._search_collection(
                    user_collection_name, 
                    query_text, 
                    limit // 2, 
                    score_threshold
                )
            )
            
            # Wait for both searches to complete
            global_results, user_results = await asyncio.gather(global_task, user_task, return_exceptions=True)
            
            # Handle exceptions
            if isinstance(global_results, Exception):
                logger.warning(f"⚠️ Global search failed: {global_results}")
                global_results = []
            
            if isinstance(user_results, Exception):
                logger.warning(f"⚠️ User search failed: {user_results}")
                user_results = []
            
            # Combine and deduplicate results
            combined_results = []
            seen_chunks = set()
            
            # Add results with source annotation
            for result in global_results:
                if result['chunk_id'] not in seen_chunks:
                    result['source_collection'] = 'global'
                    combined_results.append(result)
                    seen_chunks.add(result['chunk_id'])
            
            for result in user_results:
                if result['chunk_id'] not in seen_chunks:
                    result['source_collection'] = 'user'
                    combined_results.append(result)
                    seen_chunks.add(result['chunk_id'])
            
            # Sort by score (descending)
            combined_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Limit final results
            final_results = combined_results[:limit]
            
            logger.info(f"✅ Hybrid search: {len(global_results)} global + {len(user_results)} user = {len(final_results)} total results")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            # Fallback to global search only
            return await self._search_collection(settings.VECTOR_COLLECTION_NAME, query_text, limit, score_threshold)

    async def _search_collection(
        self,
        collection_name: str,
        query_text: str,
        limit: int,
        score_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Search a specific collection with simple vector search
        
        Query expansion removed: Agents explicitly decide when to use expansion
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            # Direct vector search (no automatic expansion)
            return await self._simple_vector_search(query_text, limit, score_threshold, collection_name)
                
        except Exception as e:
            logger.error(f"❌ Collection search failed for {collection_name}: {e}")
            return []

    async def _hybrid_simple_search(
        self,
        query_text: str,
        limit: int,
        score_threshold: float
    ) -> List[Dict[str, Any]]:
        """Perform simple search across both global and user collections"""
        try:
            # Search both collections in parallel with simple search
            global_task = self._simple_vector_search(
                query_text, limit, score_threshold, settings.VECTOR_COLLECTION_NAME
            )
            user_task = self._simple_vector_search(
                query_text, limit, score_threshold, self.user_collection_name
            )
            
            # Wait for both searches to complete
            global_results, user_results = await asyncio.gather(global_task, user_task, return_exceptions=True)
            
            # Handle exceptions
            if isinstance(global_results, Exception):
                logger.warning(f"⚠️ Global simple search failed: {global_results}")
                global_results = []
            
            if isinstance(user_results, Exception):
                logger.warning(f"⚠️ User simple search failed: {user_results}")
                user_results = []
            
            # Combine and deduplicate results
            combined_results = []
            seen_chunks = set()
            
            # Add results with source annotation
            for result in global_results:
                if result['chunk_id'] not in seen_chunks:
                    result['source_collection'] = 'global'
                    combined_results.append(result)
                    seen_chunks.add(result['chunk_id'])
            
            for result in user_results:
                if result['chunk_id'] not in seen_chunks:
                    result['source_collection'] = 'user'
                    combined_results.append(result)
                    seen_chunks.add(result['chunk_id'])
            
            # Sort by score (descending)
            combined_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Limit final results
            final_results = combined_results[:limit]
            
            logger.info(f"✅ Hybrid simple search: {len(global_results)} global + {len(user_results)} user = {len(final_results)} total results")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Hybrid simple search failed: {e}")
            return []
    
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Use proper Qdrant filter format
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="chunk_id",
                        match=MatchValue(value=chunk_id)
                    )
                ]
            )
            
            results = self.qdrant_client.scroll(
                collection_name=settings.VECTOR_COLLECTION_NAME,
                scroll_filter=filter_condition,
                limit=1
            )
            
            if results[0]:  # results is (points, next_page_offset)
                point = results[0][0]
                return {
                    "chunk_id": point.payload["chunk_id"],
                    "document_id": point.payload["document_id"],
                    "content": point.payload["content"],
                    "metadata": point.payload.get("metadata", {})
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve chunk {chunk_id}: {e}")
            return None
    
    async def delete_document_chunks(self, document_id: str, user_id: str = None):
        """Delete all chunks for a specific document in user's collection"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Determine collection to use
            if user_id:
                collection_name = self._get_user_collection_name(user_id)
            else:
                collection_name = settings.VECTOR_COLLECTION_NAME
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if collection_name not in collection_names:
                logger.warning(f"⚠️ Collection {collection_name} does not exist, nothing to delete")
                return
            
            # Use proper Qdrant filter format
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=filter_condition
            )
            
            logger.info(f"🗑️  Deleted chunks for document {document_id} from collection {collection_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document chunks: {e}")
            raise

    async def get_all_document_chunks(self, document_id: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document from user's collection"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Determine collection to use
            if user_id:
                collection_name = self._get_user_collection_name(user_id)
            else:
                collection_name = settings.VECTOR_COLLECTION_NAME
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if collection_name not in collection_names:
                logger.warning(f"⚠️ Collection {collection_name} does not exist")
                return []
            
            all_chunks = []
            offset = None
            
            # Use proper Qdrant filter format
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            while True:
                # Scroll through all chunks for this document
                results = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_condition,
                    limit=100,  # Process in batches
                    offset=offset,
                    with_payload=True
                )
                
                points, next_offset = results
                
                if not points:
                    break
                
                # Add chunks to results
                for point in points:
                    chunk_data = {
                        "chunk_id": point.payload["chunk_id"],
                        "document_id": point.payload["document_id"],
                        "content": point.payload["content"],
                        "chunk_index": point.payload.get("chunk_index", 0),
                        "metadata": point.payload.get("metadata", {})
                    }
                    all_chunks.append(chunk_data)
                
                # Check if there are more results
                if next_offset is None:
                    break
                offset = next_offset
            
            # Sort by chunk index to maintain document order
            all_chunks.sort(key=lambda x: x.get("chunk_index", 0))
            
            logger.info(f"📄 Retrieved {len(all_chunks)} chunks for document {document_id} from collection {collection_name}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve all chunks for document {document_id}: {e}")
            return []

    async def get_user_collection_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's vector collection"""
        try:
            collection_name = self._get_user_collection_name(user_id)
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if collection_name not in collection_names:
                return {
                    "total_points": 0,
                    "vector_size": settings.EMBEDDING_DIMENSIONS,
                    "distance_metric": "cosine",
                    "collection_exists": False
                }
            
            # Initialize default stats
            stats = {
                "total_points": 0,
                "vector_size": settings.EMBEDDING_DIMENSIONS,
                "distance_metric": "cosine",
                "collection_exists": True
            }
            
            # Try to get points count using scroll method
            try:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_payload=False,
                    with_vectors=False
                )
                
                # Get approximate count by scrolling through all points
                total_count = 0
                offset = None
                
                while True:
                    count_result = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False
                    )
                    
                    points, next_offset = count_result
                    total_count += len(points)
                    
                    if next_offset is None or len(points) == 0:
                        break
                    offset = next_offset
                
                stats["total_points"] = total_count
                    
            except Exception as count_error:
                logger.warning(f"⚠️ Could not count points in user collection {collection_name}: {count_error}")
                stats["total_points"] = 0
            
            logger.info(f"📊 User {user_id} collection stats: {stats['total_points']} points")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get user collection stats for {user_id}: {e}")
            return {
                "total_points": 0,
                "vector_size": settings.EMBEDDING_DIMENSIONS,
                "distance_metric": "cosine",
                "collection_exists": False
            }

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the default/global vector collection"""
        try:
            collection_name = settings.VECTOR_COLLECTION_NAME
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if collection_name not in collection_names:
                return {
                    "total_points": 0,
                    "vector_size": settings.EMBEDDING_DIMENSIONS,
                    "distance_metric": "cosine",
                    "collection_exists": False
                }
            
            # Initialize default stats
            stats = {
                "total_points": 0,
                "vector_size": settings.EMBEDDING_DIMENSIONS,
                "distance_metric": "cosine",
                "collection_exists": True
            }
            
            # Try to get points count using scroll method
            try:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1,
                    with_payload=False,
                    with_vectors=False
                )
                
                # Get approximate count by scrolling through all points
                total_count = 0
                offset = None
                
                while True:
                    count_result = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False
                    )
                    
                    points, next_offset = count_result
                    total_count += len(points)
                    
                    if next_offset is None or len(points) == 0:
                        break
                    offset = next_offset
                
                stats["total_points"] = total_count
                    
            except Exception as count_error:
                logger.warning(f"⚠️ Could not count points in global collection {collection_name}: {count_error}")
                stats["total_points"] = 0
            
            logger.info(f"📊 Global collection stats: {stats['total_points']} points")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get global collection stats: {e}")
            return {
                "total_points": 0,
                "vector_size": settings.EMBEDDING_DIMENSIONS,
                "distance_metric": "cosine",
                "collection_exists": False
            }

    async def delete_user_collection(self, user_id: str) -> bool:
        """Delete a user's entire collection (for account deletion/cleanup)"""
        try:
            collection_name = self._get_user_collection_name(user_id)
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if collection_name not in collection_names:
                logger.info(f"⚠️ Collection {collection_name} does not exist, nothing to delete")
                return True
            
            # Delete the entire collection
            self.qdrant_client.delete_collection(collection_name)
            
            logger.info(f"🗑️  Deleted user collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete user collection for {user_id}: {e}")
            return False

    async def list_user_collections(self) -> List[Dict[str, Any]]:
        """List all user collections and their basic info"""
        try:
            collections = self.qdrant_client.get_collections()
            user_collections = []
            
            for collection in collections.collections:
                if collection.name.startswith("user_") and collection.name.endswith("_documents"):
                    # Extract user_id from collection name
                    user_id = collection.name.replace("user_", "").replace("_documents", "")
                    
                    # Get basic stats
                    stats = await self.get_user_collection_stats(user_id)
                    
                    user_collections.append({
                        "user_id": user_id,
                        "collection_name": collection.name,
                        "total_points": stats["total_points"],
                        "collection_exists": stats["collection_exists"]
                    })
            
            logger.info(f"📊 Found {len(user_collections)} user collections")
            return user_collections
            
        except Exception as e:
            logger.error(f"❌ Failed to list user collections: {e}")
            return []

    async def search_similar_in_documents(self, query_text: str, document_ids: List[str], limit: int = 10, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search for similar chunks within specific documents"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            
            logger.info(f"🔍 Searching in {len(document_ids)} specific documents")
            
            # Generate query embedding
            query_embedding = await self.generate_embeddings([query_text])
            if not query_embedding:
                return []
            
            # Use proper Qdrant filter format
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=document_ids)
                    )
                ]
            )
            
            # Search with document filter using Qdrant client
            search_results = self.qdrant_client.search(
                collection_name=settings.VECTOR_COLLECTION_NAME,
                query_vector=query_embedding[0],
                query_filter=filter_condition,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Convert to standard format
            results = []
            for result in search_results:
                chunk_data = {
                    'chunk_id': result.payload.get('chunk_id'),
                    'document_id': result.payload.get('document_id'),
                    'content': result.payload.get('content', ''),
                    'score': float(result.score),
                    'metadata': result.payload.get('metadata', {})
                }
                results.append(chunk_data)
            
            logger.info(f"🔍 Found {len(results)} chunks in specified documents")
            return results
            
        except Exception as e:
            logger.error(f"❌ Document-filtered search failed: {e}")
            return []

    async def update_embeddings_with_title(self, document_id: str, title: str):
        """Update existing embeddings to include title information with safe token limits"""
        try:
            logger.info(f"🔄 Updating embeddings with title for document {document_id}: '{title}'")
            
            # Get all existing chunks for this document
            existing_chunks = await self.get_all_document_chunks(document_id)
            
            if not existing_chunks:
                logger.warning(f"⚠️ No existing chunks found for document {document_id}")
                return
            
            logger.info(f"📚 Found {len(existing_chunks)} chunks to update with title")
            
            # Prepare enhanced texts with title context - with strict token limits
            enhanced_texts = []
            chunk_data = []
            title_prefix = f"Document Title: {title}\n\n"
            title_tokens = self._estimate_tokens_accurately(title_prefix)
            
            for chunk in existing_chunks:
                # Calculate safe content length to stay within token limits
                content = chunk['content']
                content_tokens = self._estimate_tokens_accurately(content)
                total_tokens = title_tokens + content_tokens
                
                # If combined content exceeds safe limit, truncate the chunk content
                max_safe_tokens = 7000  # Very conservative limit
                if total_tokens > max_safe_tokens:
                    # Calculate how much content we can safely include
                    available_tokens = max_safe_tokens - title_tokens - 100  # 100 token buffer
                    safe_content = self._truncate_text_safely(content, available_tokens)
                    logger.debug(f"⚠️ Truncated chunk content from {len(content)} to {len(safe_content)} chars for title enhancement")
                    enhanced_text = title_prefix + safe_content
                else:
                    enhanced_text = title_prefix + content
                
                # Double-check final token count
                final_tokens = self._estimate_tokens_accurately(enhanced_text)
                if final_tokens > 7500:  # Final safety check
                    # Emergency truncation
                    safe_enhanced = self._truncate_text_safely(enhanced_text, 7000)
                    logger.warning(f"⚠️ Emergency truncation: {final_tokens} -> {self._estimate_tokens_accurately(safe_enhanced)} tokens")
                    enhanced_text = safe_enhanced
                
                enhanced_texts.append(enhanced_text)
                chunk_data.append(chunk)
            
            # Generate new embeddings with title context
            logger.info(f"🔄 Generating new embeddings with title context...")
            new_embeddings = await self.generate_embeddings(enhanced_texts)
            
            # Update points in Qdrant
            points_to_update = []
            for chunk, embedding in zip(chunk_data, new_embeddings):
                # Use content hash for consistent IDs (same as original storage)
                content_hash = abs(hash(chunk['content']))
                
                # Create updated point with enhanced metadata
                updated_metadata = chunk.get('metadata', {}).copy()
                updated_metadata['document_title'] = title
                updated_metadata['title_enhanced'] = True
                
                point = PointStruct(
                    id=content_hash,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk['chunk_id'],
                        "document_id": chunk['document_id'],
                        "content": chunk['content'],  # Keep original content
                        "chunk_index": chunk.get('chunk_index', 0),
                        "quality_score": chunk.get('quality_score', 1.0),
                        "method": chunk.get('method', 'unknown'),
                        "metadata": updated_metadata,
                        "content_hash": content_hash,
                        "enhanced_with_title": title  # Track that this was enhanced
                    }
                )
                points_to_update.append(point)
            
            # Update in batches
            batch_size = 100
            total_updated = 0
            
            for i in range(0, len(points_to_update), batch_size):
                batch_points = points_to_update[i:i + batch_size]
                try:
                    self.qdrant_client.upsert(
                        collection_name=settings.VECTOR_COLLECTION_NAME,
                        points=batch_points
                    )
                    total_updated += len(batch_points)
                    logger.info(f"📦 Updated batch {i//batch_size + 1}: {len(batch_points)} embeddings ({total_updated}/{len(points_to_update)} total)")
                except Exception as e:
                    logger.error(f"❌ Failed to update batch {i//batch_size + 1}: {e}")
                    raise
            
            logger.info(f"✅ Successfully updated {total_updated} embeddings with title '{title}' for document {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update embeddings with title: {e}")
            raise

    async def update_zip_embeddings_with_title(self, parent_document_id: str, title: str):
        """Update embeddings for all sub-documents within a zip file"""
        try:
            logger.info(f"📦 Updating zip embeddings with title for parent {parent_document_id}: '{title}'")
            
            # First, try to get sub-documents from the database
            # We'll need to query the document repository to find related documents
            from repositories.document_repository import DocumentRepository
            doc_repo = DocumentRepository()
            await doc_repo.initialize()
            
            # Query to find all documents that are part of this zip
            query = """
                SELECT document_id, filename 
                FROM documents 
                WHERE parent_document_id = $1 OR document_id = $1
            """
            
            related_docs = await doc_repo.execute_query(query, parent_document_id)
            
            if not related_docs:
                logger.warning(f"⚠️ No related documents found for zip {parent_document_id}")
                # Fallback: try to update the parent document itself
                await self.update_embeddings_with_title(parent_document_id, title)
                return
            
            logger.info(f"📦 Found {len(related_docs)} documents in zip to update")
            
            # Update embeddings for each document in the zip
            updated_count = 0
            for doc in related_docs:
                doc_id = doc['document_id']
                filename = doc.get('filename', 'Unknown')
                
                try:
                    # Create enhanced title that includes both zip title and filename
                    enhanced_title = f"{title} - {filename}" if filename != title else title
                    
                    await self.update_embeddings_with_title(doc_id, enhanced_title)
                    updated_count += 1
                    logger.info(f"✅ Updated document {doc_id} ({filename}) with enhanced title")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to update document {doc_id} ({filename}): {e}")
                    continue
            
            logger.info(f"✅ Successfully updated {updated_count}/{len(related_docs)} documents in zip with title context")
            
        except Exception as e:
            logger.error(f"❌ Failed to update zip embeddings with title: {e}")
            raise

    async def batch_update_titles_for_existing_documents(self):
        """Batch update all existing documents with their titles from the database"""
        try:
            logger.info("🔄 Starting batch update of all existing documents with titles...")
            
            # Get all documents from the database
            from repositories.document_repository import DocumentRepository
            doc_repo = DocumentRepository()
            await doc_repo.initialize()
            
            # Query to get all documents with their titles
            query = """
                SELECT document_id, title, filename, parent_document_id
                FROM documents 
                WHERE title IS NOT NULL AND title != ''
                ORDER BY created_at
            """
            
            documents = await doc_repo.execute_query(query)
            
            if not documents:
                logger.info("📄 No documents with titles found to update")
                return
            
            logger.info(f"📚 Found {len(documents)} documents with titles to update")
            
            updated_count = 0
            failed_count = 0
            
            for doc in documents:
                doc_id = doc['document_id']
                title = doc['title']
                filename = doc.get('filename', '')
                parent_id = doc.get('parent_document_id')
                
                try:
                    # Check if this document has chunks in the vector database
                    existing_chunks = await self.get_all_document_chunks(doc_id)
                    
                    if not existing_chunks:
                        logger.debug(f"⏭️ Skipping {doc_id} - no chunks found")
                        continue
                    
                    # Use title if available, otherwise use filename
                    effective_title = title if title and title.strip() else filename
                    
                    if not effective_title:
                        logger.debug(f"⏭️ Skipping {doc_id} - no title or filename")
                        continue
                    
                    logger.info(f"🔄 Updating {doc_id} with title: '{effective_title}'")
                    await self.update_embeddings_with_title(doc_id, effective_title)
                    updated_count += 1
                    
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to update document {doc_id}: {e}")
                    failed_count += 1
                    continue
            
            logger.info(f"✅ Batch update completed: {updated_count} updated, {failed_count} failed")
            
        except Exception as e:
            logger.error(f"❌ Batch title update failed: {e}")
            raise

    def _estimate_tokens_accurately(self, text: str) -> int:
        """Much more conservative token estimation for embedding models"""
        if not text:
            return 0
        
        # Very conservative estimation to prevent token limit errors
        char_count = len(text)
        
        # Much more conservative ratio - assume 1 token per 2 characters for safety
        # This is more conservative than the typical 3-4 chars per token
        base_tokens = char_count // 2
        
        # Add substantial overhead for various factors
        punct_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        punct_overhead = punct_chars * 0.5  # Increased overhead
        
        words = len(text.split())
        word_overhead = words * 0.3  # Increased overhead
        
        unicode_chars = sum(1 for c in text if ord(c) > 127)
        unicode_overhead = unicode_chars * 0.7  # Increased overhead
        
        # Add extra overhead for email content which tends to have more tokens
        if any(pattern in text.lower() for pattern in ['from:', 'to:', 'subject:', '@']):
            email_overhead = char_count * 0.1
        else:
            email_overhead = 0
        
        total_estimated = int(base_tokens + punct_overhead + word_overhead + unicode_overhead + email_overhead)
        
        # Add 25% safety buffer instead of 10%
        return int(total_estimated * 1.25)
    
    def _truncate_text_safely(self, text: str, max_tokens: int) -> str:
        """Safely truncate text to fit within token limits"""
        if not text:
            return text
        
        # Quick check if already under limit
        estimated_tokens = self._estimate_tokens_accurately(text)
        if estimated_tokens <= max_tokens:
            return text
        
        # Binary search for optimal length
        left, right = 0, len(text)
        best_length = 0
        
        while left <= right:
            mid = (left + right) // 2
            candidate = text[:mid]
            
            # Try to end at a sentence boundary for better quality
            if mid < len(text):
                # Look for sentence endings within last 100 characters
                last_part = candidate[-100:] if len(candidate) > 100 else candidate
                sentence_ends = ['.', '!', '?', '\n']
                
                best_end = -1
                for end_char in sentence_ends:
                    pos = last_part.rfind(end_char)
                    if pos > best_end:
                        best_end = pos
                
                if best_end > 0:
                    # Adjust to sentence boundary
                    adjustment = len(last_part) - best_end - 1
                    candidate = text[:mid - adjustment]
            
            tokens = self._estimate_tokens_accurately(candidate)
            
            if tokens <= max_tokens:
                best_length = len(candidate)
                left = mid + 1
            else:
                right = mid - 1
        
        result = text[:best_length] if best_length > 0 else text[:max_tokens * 2]  # Fallback
        
        # Ensure we don't cut off in the middle of a word
        if best_length < len(text) and not text[best_length].isspace():
            # Find the last space
            last_space = result.rfind(' ')
            if last_space > len(result) * 0.8:  # Only if we don't lose too much
                result = result[:last_space]
        
        return result.strip()

    async def store_chunk_embedding(self, chunk_id: str, document_id: str, content: str, metadata: Dict[str, Any] = None):
        """Store a single chunk embedding (for notes) with enhanced metadata context"""
        try:
            # Enhance content with metadata for better semantic search
            enhanced_content = await self._enhance_content_with_metadata(content, metadata)
            
            # Generate embedding for the enhanced content
            embeddings = await self.generate_embeddings([enhanced_content])
            if not embeddings:
                raise Exception("Failed to generate embedding")
            
            embedding = embeddings[0]
            
            # Use content hash for consistent IDs (based on original content)
            content_hash = abs(hash(content))
            
            # Prepare point for Qdrant
            point = PointStruct(
                id=content_hash,
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "content": content,  # Store original content
                    "enhanced_content": enhanced_content,  # Store enhanced version
                    "chunk_index": 0,
                    "quality_score": 1.0,
                    "method": "manual",
                    "metadata": metadata or {},
                    "content_hash": content_hash,
                    "metadata_enhanced": True
                }
            )
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=settings.VECTOR_COLLECTION_NAME,
                points=[point]
            )
            
            logger.info(f"✅ Stored enhanced embedding for chunk: {chunk_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store chunk embedding: {e}")
            raise

    async def delete_document_embeddings(self, document_id: str):
        """Delete all embeddings for a document (alias for delete_document_chunks)"""
        try:
            await self.delete_document_chunks(document_id)
        except Exception as e:
            logger.error(f"❌ Failed to delete document embeddings: {e}")
            raise

    async def _enhance_content_with_metadata(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Enhance content with metadata (categories, tags, dates) for better semantic search"""
        try:
            if not metadata:
                return content
            
            # Build metadata context
            metadata_parts = []
            
            # Add category information
            category = metadata.get('category')
            if category and category.strip():
                metadata_parts.append(f"Category: {category}")
            
            # Add tags information
            tags = metadata.get('tags', [])
            if tags and isinstance(tags, list) and len(tags) > 0:
                # Filter out empty tags
                valid_tags = [tag for tag in tags if tag and str(tag).strip()]
                if valid_tags:
                    metadata_parts.append(f"Tags: {', '.join(valid_tags)}")
            
            # Add date information
            created_at = metadata.get('created_at')
            if created_at:
                # Try to format date nicely
                try:
                    if isinstance(created_at, str):
                        # Assume ISO format
                        from datetime import datetime
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        formatted_date = dt.strftime("%B %Y")  # e.g., "January 2025"
                        metadata_parts.append(f"Created: {formatted_date}")
                except:
                    # Fallback to string representation
                    metadata_parts.append(f"Created: {created_at}")
            
            # Add document type if available
            doc_type = metadata.get('document_type') or metadata.get('type')
            if doc_type and doc_type.strip():
                metadata_parts.append(f"Type: {doc_type}")
            
            # Add source information if available
            source = metadata.get('source')
            if source and source.strip():
                metadata_parts.append(f"Source: {source}")
            
            # Add title if available (and not already in content)
            title = metadata.get('title') or metadata.get('document_title')
            if title and title.strip() and title.lower() not in content.lower()[:100]:
                metadata_parts.append(f"Title: {title}")
            
            # Build enhanced content
            if metadata_parts:
                metadata_context = " | ".join(metadata_parts)
                enhanced_content = f"[{metadata_context}]\n\n{content}"
                
                # Ensure we don't exceed token limits
                estimated_tokens = self._estimate_tokens_accurately(enhanced_content)
                if estimated_tokens > 7000:  # Conservative limit
                    # Truncate content while preserving metadata context
                    metadata_context_tokens = self._estimate_tokens_accurately(f"[{metadata_context}]\n\n")
                    available_tokens = 7000 - metadata_context_tokens - 100  # Buffer
                    
                    if available_tokens > 0:
                        safe_content = self._truncate_text_safely(content, available_tokens)
                        enhanced_content = f"[{metadata_context}]\n\n{safe_content}"
                    else:
                        # If metadata is too long, just use original content
                        enhanced_content = content
                        logger.warning("⚠️ Metadata context too long, using original content only")
                
                logger.debug(f"📝 Enhanced content with metadata: {metadata_context}")
                return enhanced_content
            else:
                return content
                
        except Exception as e:
            logger.error(f"❌ Failed to enhance content with metadata: {e}")
            return content

    async def update_document_metadata_in_vectors(
        self,
        document_id: str,
        title: str = None,
        author: str = None,
        filename: str = None,
        category: str = None,
        tags: List[str] = None
    ):
        """
        **ROOSEVELT'S VECTOR METADATA SYNC**: Update document metadata in Qdrant vector chunks
        
        This updates the payload fields (document_title, document_author, etc.) for all chunks
        belonging to a document, ensuring vector search returns current metadata.
        """
        try:
            logger.info(f"🔄 VECTOR METADATA SYNC: Updating {document_id} - title='{title}', author='{author}', tags={tags}")
            
            # Build filter to find all chunks for this document
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            # Get all collections that might have this document
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            total_updated = 0
            
            for collection_name in collection_names:
                try:
                    # Scroll through all matching chunks in this collection
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter=filter_condition,
                        limit=1000,  # Process in batches
                        with_payload=True,
                        with_vectors=False  # We don't need vectors, just updating payload
                    )
                    
                    points_to_update = scroll_result[0]  # First element is the list of points
                    
                    if not points_to_update:
                        continue
                    
                    logger.info(f"📦 Found {len(points_to_update)} chunks in collection {collection_name}")
                    
                    # Prepare updates for each point
                    from qdrant_client.models import SetPayload
                    
                    for point in points_to_update:
                        # Build payload update with only the fields that were provided
                        payload_update = {}
                        
                        if title is not None:
                            payload_update["document_title"] = title
                        if author is not None:
                            payload_update["document_author"] = author
                        if filename is not None:
                            payload_update["document_filename"] = filename
                        if category is not None:
                            payload_update["document_category"] = category
                        if tags is not None:
                            payload_update["document_tags"] = tags
                        
                        # Update this point's payload
                        if payload_update:
                            self.qdrant_client.set_payload(
                                collection_name=collection_name,
                                payload=payload_update,
                                points=[point.id]
                            )
                    
                    total_updated += len(points_to_update)
                    logger.info(f"✅ Updated {len(points_to_update)} chunks in {collection_name}")
                    
                except Exception as col_error:
                    logger.warning(f"⚠️ Failed to update collection {collection_name}: {col_error}")
                    continue
            
            logger.info(f"✅ VECTOR METADATA SYNC COMPLETE: Updated {total_updated} total chunks for document {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update document metadata in vectors: {e}")
            raise
    
    async def update_embeddings_with_metadata(self, document_id: str, metadata: Dict[str, Any]):
        """Update existing embeddings to include metadata (categories, tags, dates)"""
        try:
            logger.info(f"🔄 Updating embeddings with metadata for document {document_id}")
            
            # Get all existing chunks for this document
            existing_chunks = await self.get_all_document_chunks(document_id)
            
            if not existing_chunks:
                logger.warning(f"⚠️ No existing chunks found for document {document_id}")
                return
            
            logger.info(f"📚 Found {len(existing_chunks)} chunks to update with metadata")
            
            # Prepare enhanced texts with metadata context
            enhanced_texts = []
            chunk_data = []
            
            for chunk in existing_chunks:
                content = chunk['content']
                
                # Merge chunk metadata with provided metadata
                chunk_metadata = chunk.get('metadata', {}).copy()
                chunk_metadata.update(metadata)
                
                # Enhance content with metadata
                enhanced_content = await self._enhance_content_with_metadata(content, chunk_metadata)
                
                enhanced_texts.append(enhanced_content)
                chunk_data.append(chunk)
            
            # Generate new embeddings with metadata context
            logger.info(f"🔄 Generating new embeddings with metadata context...")
            new_embeddings = await self.generate_embeddings(enhanced_texts)
            
            # Update points in Qdrant
            points_to_update = []
            for chunk, embedding, enhanced_content in zip(chunk_data, new_embeddings, enhanced_texts):
                # Use content hash for consistent IDs
                content_hash = abs(hash(chunk['content']))
                
                # Create updated point with enhanced metadata
                updated_metadata = chunk.get('metadata', {}).copy()
                updated_metadata.update(metadata)
                updated_metadata['metadata_enhanced'] = True
                
                point = PointStruct(
                    id=content_hash,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk['chunk_id'],
                        "document_id": chunk['document_id'],
                        "content": chunk['content'],  # Keep original content
                        "enhanced_content": enhanced_content,  # Store enhanced version
                        "chunk_index": chunk.get('chunk_index', 0),
                        "quality_score": chunk.get('quality_score', 1.0),
                        "method": chunk.get('method', 'unknown'),
                        "metadata": updated_metadata,
                        "content_hash": content_hash
                    }
                )
                points_to_update.append(point)
            
            # Update in batches
            batch_size = 100
            total_updated = 0
            
            for i in range(0, len(points_to_update), batch_size):
                batch_points = points_to_update[i:i + batch_size]
                try:
                    self.qdrant_client.upsert(
                        collection_name=settings.VECTOR_COLLECTION_NAME,
                        points=batch_points
                    )
                    total_updated += len(batch_points)
                    logger.info(f"📦 Updated batch {i//batch_size + 1}: {len(batch_points)} embeddings ({total_updated}/{len(points_to_update)} total)")
                except Exception as e:
                    logger.error(f"❌ Failed to update batch {i//batch_size + 1}: {e}")
                    raise
            
            logger.info(f"✅ Successfully updated {total_updated} embeddings with metadata for document {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update embeddings with metadata: {e}")
            raise

    async def _generate_batch_with_retry(self, batch_texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """Generate embeddings for a batch with exponential backoff retry for rate limits"""
        import re
        
        for attempt in range(max_retries):
            try:
                response = await self.openai_client.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input=batch_texts
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                return batch_embeddings
                
            except Exception as e:
                error_str = str(e)
                
                # Check if this is a rate limit error
                if "rate limit" in error_str.lower() or "429" in error_str:
                    # Extract wait time from error message if available
                    wait_time = self._extract_wait_time_from_error(error_str)
                    
                    if wait_time is None:
                        # Use exponential backoff if no specific wait time provided
                        wait_time = min(60, (2 ** attempt) + (attempt * 0.1))  # Cap at 60 seconds
                    
                    # Enforce minimum 5-second backoff for rate limits
                    wait_time = max(5.0, wait_time)
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"⏳ Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f}s before retry (minimum 5s enforced)...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ Rate limit exceeded after {max_retries} attempts")
                        raise
                
                # Check for other retryable errors
                elif any(error_type in error_str.lower() for error_type in ["timeout", "connection", "server error", "503", "502", "500"]):
                    if attempt < max_retries - 1:
                        wait_time = min(30, (2 ** attempt))  # Shorter wait for non-rate-limit errors
                        logger.warning(f"⏳ Retryable error (attempt {attempt + 1}/{max_retries}): {error_str}. Waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ Max retries exceeded for error: {error_str}")
                        raise
                
                else:
                    # Non-retryable error, fail immediately
                    logger.error(f"❌ Non-retryable error: {error_str}")
                    raise
        
        # Should never reach here, but just in case
        raise Exception(f"Failed to generate embeddings after {max_retries} attempts")
    
    def _extract_wait_time_from_error(self, error_str: str) -> float:
        """Extract recommended wait time from OpenAI rate limit error message"""
        import re
        
        try:
            # Look for patterns like "Please try again in 2.969s" or "try again in 30s"
            patterns = [
                r"try again in (\d+\.?\d*)s",
                r"try again in (\d+\.?\d*) seconds",
                r"wait (\d+\.?\d*)s",
                r"wait (\d+\.?\d*) seconds"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, error_str, re.IGNORECASE)
                if match:
                    wait_time = float(match.group(1))
                    # Add a small buffer to the recommended wait time
                    return wait_time + 0.5
            
            # If no specific time found, return None to use exponential backoff
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract wait time from error: {e}")
            return None

    async def move_document_vectors_to_global(self, document_id: str, source_user_id: str) -> bool:
        """Move all vectors for a document from user collection to global collection"""
        try:
            source_collection = self._get_user_collection_name(source_user_id)
            target_collection = settings.VECTOR_COLLECTION_NAME
            
            logger.info(f"📦 Moving vectors for document {document_id} from {source_collection} to {target_collection}")
            
            # Ensure target collection exists
            await self._ensure_default_collection_exists()
            
            # Get all points for the document from user collection
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )
            
            # Scroll through all points for this document
            points_to_move = []
            scroll_result = self.qdrant_client.scroll(
                collection_name=source_collection,
                scroll_filter=scroll_filter,
                limit=1000,  # Process in batches
                with_payload=True,
                with_vectors=True
            )
            
            points_to_move.extend(scroll_result[0])
            next_page_offset = scroll_result[1]
            
            # Continue scrolling if there are more points
            while next_page_offset:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=source_collection,
                    scroll_filter=scroll_filter,
                    limit=1000,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=True
                )
                points_to_move.extend(scroll_result[0])
                next_page_offset = scroll_result[1]
            
            if not points_to_move:
                logger.warning(f"⚠️ No vectors found for document {document_id} in user collection {source_collection}")
                return True  # Not an error, just nothing to move
            
            logger.info(f"📦 Found {len(points_to_move)} vectors to move for document {document_id}")
            
            # Prepare points for insertion into global collection
            points_for_global = []
            for point in points_to_move:
                # Update payload to indicate it's now in global collection
                payload = point.payload.copy()
                payload["collection_type"] = "global"
                payload["migrated_from_user"] = source_user_id
                payload["migrated_at"] = datetime.utcnow().isoformat()
                
                points_for_global.append(models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=payload
                ))
            
            # Insert points into global collection
            await self._store_points_with_retry(
                points_for_global,
                collection_name=target_collection
            )
            
            # Delete points from user collection
            point_ids = [point.id for point in points_to_move]
            self.qdrant_client.delete(
                collection_name=source_collection,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            
            logger.info(f"✅ Successfully moved {len(points_to_move)} vectors for document {document_id} to global collection")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to move vectors for document {document_id}: {e}")
            return False
    
    async def copy_document_vectors_to_global(self, document_id: str, source_user_id: str) -> bool:
        """Copy all vectors for a document from user collection to global collection (keeps original)"""
        try:
            source_collection = self._get_user_collection_name(source_user_id)
            target_collection = settings.VECTOR_COLLECTION_NAME
            
            logger.info(f"📋 Copying vectors for document {document_id} from {source_collection} to {target_collection}")
            
            # Ensure target collection exists
            await self._ensure_default_collection_exists()
            
            # Get all points for the document from user collection
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )
            
            # Scroll through all points for this document
            points_to_copy = []
            scroll_result = self.qdrant_client.scroll(
                collection_name=source_collection,
                scroll_filter=scroll_filter,
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
            
            points_to_copy.extend(scroll_result[0])
            next_page_offset = scroll_result[1]
            
            while next_page_offset:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=source_collection,
                    scroll_filter=scroll_filter,
                    limit=1000,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=True
                )
                points_to_copy.extend(scroll_result[0])
                next_page_offset = scroll_result[1]
            
            if not points_to_copy:
                logger.warning(f"⚠️ No vectors found for document {document_id} in user collection {source_collection}")
                return True
            
            logger.info(f"📋 Found {len(points_to_copy)} vectors to copy for document {document_id}")
            
            # Generate new IDs for global collection to avoid conflicts
            import uuid
            points_for_global = []
            for point in points_to_copy:
                # Update payload to indicate it's in global collection
                payload = point.payload.copy()
                payload["collection_type"] = "global"
                payload["copied_from_user"] = source_user_id
                payload["copied_at"] = datetime.utcnow().isoformat()
                payload["original_point_id"] = str(point.id)
                
                # Generate new UUID for global collection
                new_id = str(uuid.uuid4())
                points_for_global.append(models.PointStruct(
                    id=new_id,
                    vector=point.vector,
                    payload=payload
                ))
            
            # Insert points into global collection
            await self._store_points_with_retry(
                points_for_global,
                collection_name=target_collection
            )
            
            logger.info(f"✅ Successfully copied {len(points_to_copy)} vectors for document {document_id} to global collection")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy vectors for document {document_id}: {e}")
            return False
    
    async def remove_document_vectors_from_global(self, document_id: str) -> bool:
        """Remove all vectors for a document from global collection"""
        try:
            target_collection = settings.VECTOR_COLLECTION_NAME
            
            logger.info(f"🗑️ Removing vectors for document {document_id} from global collection")
            
            # Delete all points for this document from global collection
            self.qdrant_client.delete(
                collection_name=target_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"✅ Successfully removed vectors for document {document_id} from global collection")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to remove vectors for document {document_id} from global: {e}")
            return False

    async def _include_adjacent_chunks(self, results: List[Dict[str, Any]], limit: int, user_id: str = None) -> List[Dict[str, Any]]:
        """Include adjacent chunks (-1 and +1) for each hit to ensure good coverage"""
        try:
            logger.info(f"📎 Including adjacent chunks for {len(results)} search hits")
            
            # Determine collection to use
            if user_id:
                collection_name = self._get_user_collection_name(user_id)
            else:
                collection_name = settings.VECTOR_COLLECTION_NAME
            
            # Group results by document_id and chunk_index for efficient adjacent chunk lookup
            doc_chunks = {}
            for result in results:
                doc_id = result['document_id']
                chunk_index = result.get('metadata', {}).get('chunk_index', 0)
                if 'chunk_index' in result:
                    chunk_index = result['chunk_index']
                
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append((chunk_index, result))
            
            # For each document, find adjacent chunks
            enhanced_results = []
            seen_chunk_ids = set()
            
            # Add original results first
            for result in results:
                if result['chunk_id'] not in seen_chunk_ids:
                    enhanced_results.append(result)
                    seen_chunk_ids.add(result['chunk_id'])
            
            # Now find and add adjacent chunks
            for doc_id, chunks in doc_chunks.items():
                # Get all chunk indices we need to find adjacent chunks for
                target_indices = set()
                for chunk_index, _ in chunks:
                    target_indices.add(chunk_index - 1)  # Previous chunk
                    target_indices.add(chunk_index + 1)  # Next chunk
                
                # Remove indices we already have
                existing_indices = {chunk_index for chunk_index, _ in chunks}
                target_indices = target_indices - existing_indices
                
                # Remove negative indices
                target_indices = {idx for idx in target_indices if idx >= 0}
                
                if not target_indices:
                    continue
                
                # Query for adjacent chunks
                adjacent_chunks = await self._get_chunks_by_indices(doc_id, list(target_indices), collection_name)
                
                # Add adjacent chunks to results
                for adj_chunk in adjacent_chunks:
                    if adj_chunk['chunk_id'] not in seen_chunk_ids:
                        # Mark as adjacent chunk and give it a lower score
                        adj_chunk['score'] = adj_chunk.get('score', 0.5) * 0.8  # Reduce score for adjacent chunks
                        adj_chunk['is_adjacent'] = True
                        adj_chunk['query_source'] = 'adjacent_chunk'
                        enhanced_results.append(adj_chunk)
                        seen_chunk_ids.add(adj_chunk['chunk_id'])
            
            # Sort by score (descending) and limit results
            enhanced_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            final_results = enhanced_results[:limit]
            
            adjacent_count = len([r for r in final_results if r.get('is_adjacent', False)])
            original_count = len(final_results) - adjacent_count
            
            logger.info(f"📎 Enhanced results: {original_count} original + {adjacent_count} adjacent = {len(final_results)} total chunks")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Failed to include adjacent chunks: {e}")
            # Return original results if adjacent chunk inclusion fails
            return results[:limit]
    
    async def _get_chunks_by_indices(self, document_id: str, chunk_indices: List[int], collection_name: str) -> List[Dict[str, Any]]:
        """Get chunks by document ID and chunk indices"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
            
            if not chunk_indices:
                return []
            
            # Build filter for document_id and chunk_indices
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    ),
                    FieldCondition(
                        key="chunk_index",
                        match=MatchAny(any=chunk_indices)
                    )
                ]
            )
            
            # Scroll through results
            results = []
            offset = None
            
            while True:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_condition,
                    limit=100,
                    offset=offset,
                    with_payload=True
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                # Convert points to result format
                for point in points:
                    chunk_data = {
                        "chunk_id": point.payload["chunk_id"],
                        "document_id": point.payload["document_id"],
                        "content": point.payload["content"],
                        "chunk_index": point.payload.get("chunk_index", 0),
                        "score": 0.5,  # Default score for adjacent chunks
                        "metadata": point.payload.get("metadata", {}),
                        "query_source": "adjacent_lookup",
                        "source_query": "adjacent_chunk_lookup"
                    }
                    results.append(chunk_data)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            logger.debug(f"📎 Found {len(results)} adjacent chunks for document {document_id}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get chunks by indices for document {document_id}: {e}")
            return []

    async def close(self):
        """Close connections"""
        if self.qdrant_client:
            await self.qdrant_client.close()
        
        # Log cache statistics before closing
        logger.info("🔄 Embedding Manager closed")


# Global embedding manager instance for lazy loading
_embedding_manager_instance = None


async def get_embedding_manager() -> EmbeddingManager:
    """Get or create a global embedding manager instance"""
    global _embedding_manager_instance
    
    if _embedding_manager_instance is None:
        logger.info("🔄 Creating global embedding manager instance...")
        _embedding_manager_instance = EmbeddingManager()
        await _embedding_manager_instance.initialize()
        logger.info("✅ Global embedding manager initialized")
    
    return _embedding_manager_instance
