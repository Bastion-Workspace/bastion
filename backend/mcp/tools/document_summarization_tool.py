"""
Document Summarization Tool - MCP Tool for Intelligent Document Summarization
Provides multi-stage search and hierarchical summarization capabilities
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field
from services.settings_service import settings_service

logger = logging.getLogger(__name__)


class SummaryType(str, Enum):
    """Types of summaries that can be generated"""
    BRIEF = "brief"              # Short overview (1-2 paragraphs)
    DETAILED = "detailed"        # Comprehensive summary (3-5 paragraphs)
    KEY_POINTS = "key_points"    # Bullet point format
    EXECUTIVE = "executive"      # Executive summary format
    ACADEMIC = "academic"        # Academic abstract format


class SearchStrategy(str, Enum):
    """Search strategies for finding documents"""
    AUTO = "auto"                # Automatically determine best strategy
    FILENAME = "filename"        # Search by exact filename
    SEMANTIC = "semantic"        # Semantic content search
    METADATA = "metadata"        # Search by document metadata
    COMBINED = "combined"        # Use multiple strategies


class DocumentSummarizationInput(BaseModel):
    """Input for document summarization"""
    query: str = Field(..., description="Search query or specific document name to summarize")
    summary_type: SummaryType = Field(SummaryType.DETAILED, description="Type of summary to generate")
    search_strategy: SearchStrategy = Field(SearchStrategy.AUTO, description="Strategy for finding documents")
    max_documents: int = Field(10, ge=1, le=50, description="Maximum number of documents to include")
    max_content_length: int = Field(8000, ge=1000, le=20000, description="Maximum content length for direct summarization")
    include_sources: bool = Field(True, description="Whether to include source citations")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on in the summary")


class DocumentSource(BaseModel):
    """Source document information"""
    document_id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    filename: str = Field(..., description="Document filename")
    relevance_score: float = Field(..., description="Relevance score for this document")
    content_preview: str = Field(..., description="Preview of relevant content")


class SummarySection(BaseModel):
    """A section of the summary"""
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    sources: List[str] = Field(..., description="Source document IDs for this section")


class DocumentSummarizationOutput(BaseModel):
    """Output from document summarization"""
    summary: str = Field(..., description="The generated summary")
    summary_type: SummaryType = Field(..., description="Type of summary generated")
    sections: List[SummarySection] = Field(default_factory=list, description="Structured summary sections")
    sources: List[DocumentSource] = Field(..., description="Source documents used")
    total_documents_found: int = Field(..., description="Total documents found by search")
    total_documents_used: int = Field(..., description="Total documents used in summary")
    search_strategy_used: SearchStrategy = Field(..., description="Search strategy that was used")
    processing_time: float = Field(..., description="Total processing time")
    content_length: int = Field(..., description="Total content length processed")
    hierarchical_processing: bool = Field(..., description="Whether hierarchical processing was used")


class DocumentSummarizationTool:
    """MCP tool for intelligent document summarization"""
    
    def __init__(self, search_tool=None, filename_search_tool=None, metadata_search_tool=None, 
                 document_repository=None, embedding_manager=None):
        """Initialize with required services"""
        self.search_tool = search_tool
        self.filename_search_tool = filename_search_tool
        self.metadata_search_tool = metadata_search_tool
        self.document_repository = document_repository
        self.embedding_manager = embedding_manager
        self.name = "summarize_documents"
        self.description = "Intelligently search and summarize documents with multi-stage search and hierarchical processing"
        
    async def initialize(self):
        """Initialize the summarization tool"""
        if not self.search_tool:
            raise ValueError("SearchTool is required")
        if not self.document_repository:
            raise ValueError("DocumentRepository is required")
        
        logger.info("📋 DocumentSummarizationTool initialized")
    
    async def execute(self, input_data: DocumentSummarizationInput) -> ToolResponse:
        """Execute document summarization with intelligent search and processing"""
        start_time = time.time()
        
        try:
            logger.info(f"📋 Starting summarization: '{input_data.query}' (type: {input_data.summary_type.value})")
            
            # Stage 1: Intelligent document discovery
            documents, search_strategy_used = await self._discover_documents(input_data)
            
            if not documents:
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="NO_DOCUMENTS_FOUND",
                        error_message=f"No documents found for query: {input_data.query}",
                        details={"query": input_data.query, "strategy": search_strategy_used.value}
                    ),
                    execution_time=time.time() - start_time
                )
            
            logger.info(f"📋 Found {len(documents)} documents using {search_strategy_used.value} strategy")
            
            # Stage 2: Content aggregation and processing
            summary_result = await self._generate_summary(input_data, documents)
            
            # Stage 3: Format output
            output = DocumentSummarizationOutput(
                summary=summary_result["summary"],
                summary_type=input_data.summary_type,
                sections=summary_result.get("sections", []),
                sources=documents[:input_data.max_documents],
                total_documents_found=len(documents),
                total_documents_used=summary_result["documents_used"],
                search_strategy_used=search_strategy_used,
                processing_time=time.time() - start_time,
                content_length=summary_result["content_length"],
                hierarchical_processing=summary_result["hierarchical_processing"]
            )
            
            logger.info(f"✅ Summarization completed: {output.total_documents_used} docs, "
                       f"{output.content_length} chars in {output.processing_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="SUMMARIZATION_FAILED",
                    error_message=str(e),
                    details={"query": input_data.query}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _discover_documents(self, input_data: DocumentSummarizationInput) -> tuple[List[DocumentSource], SearchStrategy]:
        """Intelligently discover relevant documents using multi-stage search"""
        
        strategy = input_data.search_strategy
        documents = []
        
        # Auto-determine strategy if needed
        if strategy == SearchStrategy.AUTO:
            strategy = self._determine_search_strategy(input_data.query)
        
        # Execute search based on strategy
        if strategy == SearchStrategy.FILENAME:
            documents = await self._search_by_filename(input_data)
            
        elif strategy == SearchStrategy.METADATA:
            documents = await self._search_by_metadata(input_data)
            
        elif strategy == SearchStrategy.SEMANTIC:
            documents = await self._search_by_content(input_data)
            
        elif strategy == SearchStrategy.COMBINED:
            # Try multiple strategies and combine results
            filename_docs = await self._search_by_filename(input_data)
            metadata_docs = await self._search_by_metadata(input_data)
            semantic_docs = await self._search_by_content(input_data)
            
            # Combine and deduplicate
            all_docs = filename_docs + metadata_docs + semantic_docs
            seen_ids = set()
            documents = []
            for doc in all_docs:
                if doc.document_id not in seen_ids:
                    documents.append(doc)
                    seen_ids.add(doc.document_id)
        
        # If primary strategy fails, try fallback
        if not documents and strategy != SearchStrategy.SEMANTIC:
            logger.info(f"📋 Primary strategy {strategy.value} found no results, falling back to semantic search")
            documents = await self._search_by_content(input_data)
            strategy = SearchStrategy.SEMANTIC
        
        # Sort by relevance and limit
        documents.sort(key=lambda x: x.relevance_score, reverse=True)
        return documents[:input_data.max_documents], strategy
    
    def _determine_search_strategy(self, query: str) -> SearchStrategy:
        """Automatically determine the best search strategy"""
        query_lower = query.lower().strip()
        
        # Check for filename patterns
        if any(ext in query_lower for ext in ['.pdf', '.txt', '.docx', '.doc', '.epub']):
            return SearchStrategy.FILENAME
        
        # Check for specific document references
        filename_indicators = ['document', 'file', 'paper titled', 'book called', 'report named']
        if any(indicator in query_lower for indicator in filename_indicators):
            return SearchStrategy.FILENAME
        
        # Check for metadata-based queries
        metadata_indicators = ['all', 'category', 'tag', 'author', 'type', 'documents about']
        if any(indicator in query_lower for indicator in metadata_indicators):
            return SearchStrategy.METADATA
        
        # Check for summarization keywords that suggest broad search
        broad_indicators = ['summarize all', 'overview of all', 'everything about']
        if any(indicator in query_lower for indicator in broad_indicators):
            return SearchStrategy.COMBINED
        
        # Default to semantic search for content-based queries
        return SearchStrategy.SEMANTIC
    
    async def _search_by_filename(self, input_data: DocumentSummarizationInput) -> List[DocumentSource]:
        """Search documents by filename"""
        try:
            if not self.filename_search_tool:
                return []
            
            from mcp.tools.filename_search_tool import FilenameSearchInput
            
            # Extract potential filename from query
            filename = self._extract_filename_from_query(input_data.query)
            
            search_input = FilenameSearchInput(
                filename=filename,
                include_content=True,
                limit=input_data.max_documents
            )
            
            result = await self.filename_search_tool.execute(search_input)
            
            if result.success and result.data:
                documents = []
                for doc_result in result.data.results:
                    documents.append(DocumentSource(
                        document_id=doc_result.document_id,
                        title=doc_result.title or doc_result.filename,
                        filename=doc_result.filename,
                        relevance_score=doc_result.match_score,
                        content_preview=doc_result.content[:200] + "..." if doc_result.content else ""
                    ))
                return documents
            
        except Exception as e:
            logger.warning(f"⚠️ Filename search failed: {e}")
        
        return []
    
    async def _search_by_metadata(self, input_data: DocumentSummarizationInput) -> List[DocumentSource]:
        """Search documents by metadata (category, tags, etc.)"""
        try:
            # Extract metadata criteria from query
            categories, tags = self._extract_metadata_from_query(input_data.query)
            
            if not categories and not tags:
                return []
            
            # Use document repository to search by metadata
            query_parts = []
            params = []
            
            if categories:
                placeholders = ','.join(['$' + str(len(params) + i + 1) for i in range(len(categories))])
                query_parts.append(f"category = ANY(ARRAY[{placeholders}])")
                params.extend(categories)
            
            if tags:
                for tag in tags:
                    params.append(tag)
                    query_parts.append(f"tags ? ${len(params)}")
            
            if query_parts:
                where_clause = " AND ".join(query_parts)
                sql_query = f"""
                    SELECT document_id, filename, title, doc_type, category, tags, author
                    FROM document_metadata 
                    WHERE {where_clause}
                    ORDER BY upload_date DESC
                    LIMIT {input_data.max_documents}
                """
                
                results = await self.document_repository.execute_query(sql_query, *params)
                
                documents = []
                for row in results:
                    # Get content preview
                    content_preview = await self._get_document_preview(row['document_id'])
                    
                    documents.append(DocumentSource(
                        document_id=row['document_id'],
                        title=row['title'] or row['filename'],
                        filename=row['filename'],
                        relevance_score=0.9,  # High relevance for metadata matches
                        content_preview=content_preview
                    ))
                
                return documents
            
        except Exception as e:
            logger.warning(f"⚠️ Metadata search failed: {e}")
        
        return []
    
    async def _search_by_content(self, input_data: DocumentSummarizationInput) -> List[DocumentSource]:
        """Search documents by semantic content"""
        try:
            from mcp.schemas.tool_schemas import SearchDocumentsInput
            
            search_input = SearchDocumentsInput(
                query=input_data.query,
                limit=input_data.max_documents * 2,  # Get more for better selection
                similarity_threshold=0.4,  # Lower threshold for broader results
                use_expansion=True
            )
            
            result = await self.search_tool.execute(search_input)
            
            if result.success and result.data:
                documents = []
                for search_result in result.data.results:
                    documents.append(DocumentSource(
                        document_id=search_result.document_id,
                        title=search_result.document_title,
                        filename=search_result.metadata.get('filename', f"Document {search_result.document_id}"),
                        relevance_score=search_result.similarity_score,
                        content_preview=search_result.content[:200] + "..."
                    ))
                return documents
            
        except Exception as e:
            logger.warning(f"⚠️ Semantic search failed: {e}")
        
        return []
    
    async def _generate_summary(self, input_data: DocumentSummarizationInput, 
                               documents: List[DocumentSource]) -> Dict[str, Any]:
        """Generate summary from discovered documents"""
        
        # Collect all content
        all_content = []
        total_length = 0
        
        for doc in documents:
            try:
                # Get full document content
                content = await self._get_full_document_content(doc.document_id)
                if content:
                    all_content.append({
                        'document_id': doc.document_id,
                        'title': doc.title,
                        'content': content
                    })
                    total_length += len(content)
            except Exception as e:
                logger.warning(f"⚠️ Failed to get content for {doc.document_id}: {e}")
        
        # Determine processing approach
        hierarchical_processing = total_length > input_data.max_content_length
        
        if hierarchical_processing:
            logger.info(f"📋 Using hierarchical processing for {total_length} characters")
            summary = await self._hierarchical_summarization(input_data, all_content)
        else:
            logger.info(f"📋 Using direct processing for {total_length} characters")
            summary = await self._direct_summarization(input_data, all_content)
        
        return {
            "summary": summary["text"],
            "sections": summary.get("sections", []),
            "documents_used": len(all_content),
            "content_length": total_length,
            "hierarchical_processing": hierarchical_processing
        }
    
    async def _direct_summarization(self, input_data: DocumentSummarizationInput, 
                                   content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Direct summarization for smaller content sets"""
        
        # Combine all content
        combined_content = "\n\n".join([
            f"=== {doc['title']} ===\n{doc['content']}" 
            for doc in content_list
        ])
        
        # Generate summary prompt based on type
        prompt = self._build_summary_prompt(input_data, combined_content, len(content_list))
        
        # Get LLM model
        current_model = await settings_service.get_llm_model()
        if not current_model:
            raise ValueError("No LLM model available for summarization")
        
        # Generate summary
        from openai import AsyncOpenAI
        from config import settings
        
        client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Import datetime context utility
        from utils.system_prompt_utils import add_datetime_context_to_system_prompt_for_user
        
        # Try to get user_id from document_repository if available
        user_id = None
        if self.document_repository and hasattr(self.document_repository, 'current_user_id'):
            user_id = self.document_repository.current_user_id
        
        system_prompt = await add_datetime_context_to_system_prompt_for_user(
            "You are an expert document summarizer. Create clear, comprehensive summaries that capture key information and insights.",
            user_id=user_id
        )
        
        response = await client.chat.completions.create(
            model=current_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        summary_text = response.choices[0].message.content
        
        return {
            "text": summary_text,
            "sections": []  # Could parse sections from structured output
        }
    
    async def _hierarchical_summarization(self, input_data: DocumentSummarizationInput, 
                                         content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Hierarchical summarization for large content sets"""
        
        # Step 1: Summarize each document individually
        individual_summaries = []
        
        for doc in content_list:
            try:
                doc_summary = await self._summarize_single_document(
                    doc['content'], doc['title'], input_data.summary_type
                )
                individual_summaries.append({
                    'title': doc['title'],
                    'summary': doc_summary,
                    'document_id': doc['document_id']
                })
            except Exception as e:
                logger.warning(f"⚠️ Failed to summarize {doc['title']}: {e}")
        
        # Step 2: Create meta-summary from individual summaries
        combined_summaries = "\n\n".join([
            f"=== {summary['title']} ===\n{summary['summary']}"
            for summary in individual_summaries
        ])
        
        meta_prompt = self._build_meta_summary_prompt(input_data, combined_summaries, len(individual_summaries))
        
        # Generate final summary
        from openai import AsyncOpenAI
        from config import settings
        
        client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        current_model = await settings_service.get_llm_model()
        
        # Import datetime context utility
        from utils.system_prompt_utils import add_datetime_context_to_system_prompt_for_user
        
        # Try to get user_id from document_repository if available
        user_id = None
        if self.document_repository and hasattr(self.document_repository, 'current_user_id'):
            user_id = self.document_repository.current_user_id
        
        system_prompt = await add_datetime_context_to_system_prompt_for_user(
            "You are an expert at synthesizing multiple document summaries into coherent overviews.",
            user_id=user_id
        )
        
        response = await client.chat.completions.create(
            model=current_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": meta_prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        final_summary = response.choices[0].message.content
        
        return {
            "text": final_summary,
            "sections": [
                SummarySection(
                    title=summary['title'],
                    content=summary['summary'],
                    sources=[summary['document_id']]
                ) for summary in individual_summaries
            ]
        }
    
    async def _summarize_single_document(self, content: str, title: str, summary_type: SummaryType) -> str:
        """Summarize a single document"""
        
        # Truncate content if too long
        max_content = 4000  # Leave room for prompt
        if len(content) > max_content:
            content = content[:max_content] + "..."
        
        prompt = f"""Summarize the following document in {summary_type.value} style:

Title: {title}

Content:
{content}

Provide a {summary_type.value} summary that captures the key information and insights."""
        
        from openai import AsyncOpenAI
        from config import settings
        
        client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        current_model = await settings_service.get_llm_model()
        
        # Import datetime context utility
        from utils.system_prompt_utils import add_datetime_context_to_system_prompt_for_user
        
        # Try to get user_id from document_repository if available
        user_id = None
        if self.document_repository and hasattr(self.document_repository, 'current_user_id'):
            user_id = self.document_repository.current_user_id
        
        system_prompt = await add_datetime_context_to_system_prompt_for_user(
            "You are an expert document summarizer.",
            user_id=user_id
        )
        
        response = await client.chat.completions.create(
            model=current_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def _build_summary_prompt(self, input_data: DocumentSummarizationInput, 
                             content: str, doc_count: int) -> str:
        """Build prompt for direct summarization"""
        
        focus_instruction = ""
        if input_data.focus_areas:
            focus_instruction = f"\n\nPay special attention to these areas: {', '.join(input_data.focus_areas)}"
        
        type_instructions = {
            SummaryType.BRIEF: "Create a brief overview in 1-2 paragraphs.",
            SummaryType.DETAILED: "Create a comprehensive summary in 3-5 paragraphs.",
            SummaryType.KEY_POINTS: "Create a bullet-point summary of key points.",
            SummaryType.EXECUTIVE: "Create an executive summary suitable for decision-makers.",
            SummaryType.ACADEMIC: "Create an academic-style abstract with methodology and conclusions."
        }
        
        instruction = type_instructions.get(input_data.summary_type, "Create a comprehensive summary.")
        
        return f"""Please summarize the following {doc_count} document(s) based on the query: "{input_data.query}"

{instruction}{focus_instruction}

{"Include source references where relevant." if input_data.include_sources else ""}

Documents:
{content}

Summary:"""
    
    def _build_meta_summary_prompt(self, input_data: DocumentSummarizationInput, 
                                  summaries: str, doc_count: int) -> str:
        """Build prompt for meta-summary from individual summaries"""
        
        return f"""Create a comprehensive overview by synthesizing the following {doc_count} document summaries.

Query: "{input_data.query}"
Summary Type: {input_data.summary_type.value}

Individual Document Summaries:
{summaries}

Please create a coherent {input_data.summary_type.value} summary that:
1. Identifies common themes and patterns
2. Highlights key insights across documents
3. Notes any contradictions or different perspectives
4. Provides a unified understanding of the topic

Synthesized Summary:"""
    
    def _extract_filename_from_query(self, query: str) -> str:
        """Extract potential filename from query"""
        # Remove common summarization words
        clean_query = query.lower()
        for word in ['summarize', 'summary', 'overview', 'of', 'the', 'document', 'file', 'paper']:
            clean_query = clean_query.replace(word, ' ')
        
        # Clean up and return
        return clean_query.strip()
    
    def _extract_metadata_from_query(self, query: str) -> tuple[List[str], List[str]]:
        """Extract categories and tags from query"""
        query_lower = query.lower()
        
        categories = []
        tags = []
        
        # Common category patterns
        if 'economics' in query_lower or 'economic' in query_lower:
            categories.append('economics')
        if 'science' in query_lower or 'scientific' in query_lower:
            categories.append('science')
        if 'history' in query_lower or 'historical' in query_lower:
            categories.append('history')
        if 'research' in query_lower:
            categories.append('research')
        
        # Extract quoted terms as potential tags
        import re
        quoted_terms = re.findall(r'"([^"]*)"', query)
        tags.extend(quoted_terms)
        
        return categories, tags
    
    async def _get_document_preview(self, document_id: str) -> str:
        """Get a preview of document content"""
        try:
            if self.embedding_manager:
                chunks = await self.embedding_manager.get_all_document_chunks(document_id)
                if chunks:
                    return chunks[0]['content'][:200] + "..."
        except Exception as e:
            logger.warning(f"⚠️ Failed to get preview for {document_id}: {e}")
        
        return "Content preview not available"
    
    async def _get_full_document_content(self, document_id: str) -> Optional[str]:
        """Get full content of a document"""
        try:
            if self.embedding_manager:
                chunks = await self.embedding_manager.get_all_document_chunks(document_id)
                if chunks:
                    # Combine all chunks
                    return "\n\n".join([chunk['content'] for chunk in chunks])
        except Exception as e:
            logger.warning(f"⚠️ Failed to get full content for {document_id}: {e}")
        
        return None
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": DocumentSummarizationInput.schema(),
            "outputSchema": DocumentSummarizationOutput.schema()
        }
