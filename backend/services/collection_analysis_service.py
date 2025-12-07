"""
Collection Analysis Service - Advanced analysis for large document collections
Handles multi-document summarization, temporal analysis, and intelligent sampling
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from models.api_models import *
from services.embedding_service_wrapper import get_embedding_service
from services.knowledge_graph_service import KnowledgeGraphService

logger = logging.getLogger(__name__)


class CollectionAnalysisService:
    """Service for analyzing large document collections"""
    
    def __init__(self, chat_service=None):
        self.chat_service = chat_service
        self.embedding_manager = None
        self.kg_service = None
    
    async def initialize(self):
        """Initialize collection analysis service"""
        logger.info("🔧 Initializing Collection Analysis Service...")
        
        # Initialize embedding service wrapper
        self.embedding_manager = await get_embedding_service()
        
        # Initialize knowledge graph service
        self.kg_service = KnowledgeGraphService()
        await self.kg_service.initialize()
        
        logger.info("✅ Collection Analysis Service initialized")
    
    async def analyze_document_collection(
        self, 
        document_ids: List[str], 
        analysis_type: str = "comprehensive",
        session_id: str = "default"
    ) -> QueryResponse:
        """
        Analyze a collection of documents with intelligent sampling and multi-level summarization
        
        Args:
            document_ids: List of document IDs to analyze
            analysis_type: Type of analysis ('comprehensive', 'temporal', 'thematic', 'summary')
            session_id: Session ID for conversation tracking
        """
        start_time = time.time()
        
        try:
            logger.info(f"📊 Starting collection analysis for {len(document_ids)} documents")
            logger.info(f"📊 Analysis type: {analysis_type}")
            
            # Step 1: Get all document metadata and chunks
            documents_data = await self._gather_collection_data(document_ids)
            
            if not documents_data:
                return QueryResponse(
                    answer="I couldn't find any content for the specified documents. They may not have been processed yet or may have been deleted.",
                    citations=[],
                    session_id=session_id,
                    query_time=time.time() - start_time,
                    retrieval_count=0
                )
            
            logger.info(f"📊 Gathered data for {len(documents_data)} documents with {sum(len(doc['chunks']) for doc in documents_data)} total chunks")
            
            # Step 2: Perform analysis based on type
            if analysis_type == "temporal":
                analysis_result = await self._temporal_analysis(documents_data, session_id)
            elif analysis_type == "thematic":
                analysis_result = await self._thematic_analysis(documents_data, session_id)
            elif analysis_type == "summary":
                analysis_result = await self._intelligent_summary(documents_data, session_id)
            else:  # comprehensive
                analysis_result = await self._comprehensive_analysis(documents_data, session_id)
            
            query_time = time.time() - start_time
            analysis_result.query_time = query_time
            
            logger.info(f"✅ Collection analysis completed in {query_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Collection analysis failed: {e}")
            return QueryResponse(
                answer=f"I apologize, but I encountered an error while analyzing the document collection: {str(e)}. Please try again or contact support if the issue persists.",
                citations=[],
                session_id=session_id,
                query_time=time.time() - start_time,
                retrieval_count=0
            )
    
    async def _gather_collection_data(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Gather all data for the document collection"""
        documents_data = []
        
        for doc_id in document_ids:
            try:
                # Get all chunks for this document
                chunks = await self.embedding_manager.get_all_document_chunks(doc_id)
                
                if chunks:
                    # Extract metadata from first chunk (document-level info)
                    doc_metadata = chunks[0].get('metadata', {}) if chunks else {}
                    
                    documents_data.append({
                        'document_id': doc_id,
                        'chunks': chunks,
                        'metadata': doc_metadata,
                        'total_content': '\n\n'.join([chunk['content'] for chunk in chunks]),
                        'chunk_count': len(chunks)
                    })
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to gather data for document {doc_id}: {e}")
                continue
        
        return documents_data
    
    async def _temporal_analysis(self, documents_data: List[Dict], session_id: str) -> QueryResponse:
        """Analyze documents with temporal patterns (e.g., emails over time)"""
        logger.info("📅 Performing temporal analysis")
        
        # Group documents by time periods
        temporal_groups = self._group_by_time_periods(documents_data)
        
        # Analyze trends and patterns
        temporal_insights = await self._analyze_temporal_patterns(temporal_groups)
        
        # Generate temporal summary
        summary = await self._generate_temporal_summary(temporal_insights, documents_data)
        
        # Create citations from representative documents
        citations = self._create_temporal_citations(temporal_groups, documents_data)
        
        return QueryResponse(
            answer=summary,
            citations=citations,
            session_id=session_id,
            query_time=0,  # Will be set by caller
            retrieval_count=len(documents_data)
        )
    
    async def _thematic_analysis(self, documents_data: List[Dict], session_id: str) -> QueryResponse:
        """Analyze documents by themes and topics using clustering"""
        logger.info("🎯 Performing thematic analysis")
        
        # Extract themes using clustering
        theme_clusters = await self._cluster_documents_by_theme(documents_data)
        
        # Analyze each theme cluster
        theme_insights = await self._analyze_theme_clusters(theme_clusters)
        
        # Generate thematic summary
        summary = await self._generate_thematic_summary(theme_insights, documents_data)
        
        # Create citations from representative documents in each theme
        citations = self._create_thematic_citations(theme_clusters, documents_data)
        
        return QueryResponse(
            answer=summary,
            citations=citations,
            session_id=session_id,
            query_time=0,
            retrieval_count=len(documents_data)
        )
    
    async def _intelligent_summary(self, documents_data: List[Dict], session_id: str) -> QueryResponse:
        """Create intelligent summary using hierarchical approach and sampling"""
        logger.info("🧠 Performing intelligent summarization")
        
        # Step 1: Intelligent sampling to identify key documents
        key_documents = await self._intelligent_document_sampling(documents_data)
        
        # Step 2: Create individual summaries for key documents
        individual_summaries = await self._create_individual_summaries(key_documents)
        
        # Step 3: Create hierarchical summary (summary of summaries)
        final_summary = await self._create_hierarchical_summary(individual_summaries, documents_data)
        
        # Create citations from all analyzed documents
        citations = self._create_comprehensive_citations(documents_data)
        
        return QueryResponse(
            answer=final_summary,
            citations=citations,
            session_id=session_id,
            query_time=0,
            retrieval_count=len(documents_data)
        )
    
    async def _comprehensive_analysis(self, documents_data: List[Dict], session_id: str) -> QueryResponse:
        """Perform comprehensive analysis combining all approaches"""
        logger.info("🔬 Performing comprehensive analysis")
        
        # Combine temporal, thematic, and intelligent summarization
        analyses = await asyncio.gather(
            self._temporal_analysis(documents_data, session_id),
            self._thematic_analysis(documents_data, session_id),
            self._intelligent_summary(documents_data, session_id),
            return_exceptions=True
        )
        
        # Combine results
        combined_summary = await self._combine_analysis_results(analyses, documents_data)
        
        # Create comprehensive citations
        citations = self._create_comprehensive_citations(documents_data)
        
        return QueryResponse(
            answer=combined_summary,
            citations=citations,
            session_id=session_id,
            query_time=0,
            retrieval_count=len(documents_data)
        )
    
    def _group_by_time_periods(self, documents_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group documents by time periods (daily, weekly, monthly)"""
        temporal_groups = {
            'daily': defaultdict(list),
            'weekly': defaultdict(list),
            'monthly': defaultdict(list)
        }
        
        for doc in documents_data:
            # Try to extract date from metadata or content
            doc_date = self._extract_document_date(doc)
            
            if doc_date:
                # Group by different time periods
                day_key = doc_date.strftime('%Y-%m-%d')
                week_key = f"{doc_date.year}-W{doc_date.isocalendar()[1]:02d}"
                month_key = doc_date.strftime('%Y-%m')
                
                temporal_groups['daily'][day_key].append(doc)
                temporal_groups['weekly'][week_key].append(doc)
                temporal_groups['monthly'][month_key].append(doc)
        
        return temporal_groups
    
    def _extract_document_date(self, doc: Dict) -> Optional[datetime]:
        """Extract date from document metadata or content"""
        # Try metadata first
        metadata = doc.get('metadata', {})
        
        # Common date fields
        date_fields = ['date', 'created_date', 'upload_date', 'timestamp', 'sent_date']
        for field in date_fields:
            if field in metadata:
                try:
                    return datetime.fromisoformat(str(metadata[field]).replace('Z', '+00:00'))
                except:
                    continue
        
        # Try to extract from filename or content
        # This would need more sophisticated date extraction
        return None
    
    async def _analyze_temporal_patterns(self, temporal_groups: Dict) -> Dict[str, Any]:
        """Analyze patterns in temporal data"""
        insights = {
            'volume_trends': {},
            'peak_periods': {},
            'content_evolution': {}
        }
        
        # Analyze volume trends
        for period_type, groups in temporal_groups.items():
            if groups:
                volumes = {period: len(docs) for period, docs in groups.items()}
                insights['volume_trends'][period_type] = volumes
                
                # Find peak periods
                if volumes:
                    max_period = max(volumes.items(), key=lambda x: x[1])
                    insights['peak_periods'][period_type] = max_period
        
        return insights
    
    async def _cluster_documents_by_theme(self, documents_data: List[Dict]) -> Dict[int, List[Dict]]:
        """Cluster documents by themes using TF-IDF and K-means"""
        if len(documents_data) < 2:
            return {0: documents_data}
        
        try:
            # Extract text content for clustering
            texts = [doc['total_content'] for doc in documents_data]
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Determine optimal number of clusters (max 10, min 2)
            n_clusters = min(max(2, len(documents_data) // 5), 10)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Group documents by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(documents_data[i])
            
            logger.info(f"🎯 Created {len(clusters)} theme clusters")
            return dict(clusters)
            
        except Exception as e:
            logger.warning(f"⚠️ Clustering failed: {e}, using single cluster")
            return {0: documents_data}
    
    async def _analyze_theme_clusters(self, theme_clusters: Dict[int, List[Dict]]) -> Dict[int, Dict]:
        """Analyze each theme cluster to extract insights"""
        cluster_insights = {}
        
        for cluster_id, docs in theme_clusters.items():
            # Extract key terms for this cluster
            key_terms = self._extract_cluster_keywords(docs)
            
            # Calculate cluster statistics
            total_chunks = sum(doc['chunk_count'] for doc in docs)
            avg_length = total_chunks / len(docs) if docs else 0
            
            cluster_insights[cluster_id] = {
                'document_count': len(docs),
                'total_chunks': total_chunks,
                'avg_length': avg_length,
                'key_terms': key_terms,
                'representative_doc': docs[0] if docs else None  # Could be more sophisticated
            }
        
        return cluster_insights
    
    def _extract_cluster_keywords(self, docs: List[Dict]) -> List[str]:
        """Extract key terms that characterize a cluster"""
        try:
            # Combine all text from the cluster
            combined_text = ' '.join([doc['total_content'] for doc in docs])
            
            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform([combined_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top terms
            term_scores = list(zip(feature_names, scores))
            term_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [term for term, score in term_scores[:10]]
            
        except Exception as e:
            logger.warning(f"⚠️ Keyword extraction failed: {e}")
            return []
    
    async def _intelligent_document_sampling(self, documents_data: List[Dict]) -> List[Dict]:
        """Intelligently sample key documents from large collection"""
        if len(documents_data) <= 10:
            return documents_data
        
        logger.info(f"🎯 Sampling from {len(documents_data)} documents")
        
        # Strategy 1: Diversity sampling using embeddings
        diverse_docs = await self._diversity_sampling(documents_data, target_count=5)
        
        # Strategy 2: Length-based sampling (longest documents often most important)
        length_docs = sorted(documents_data, key=lambda x: x['chunk_count'], reverse=True)[:3]
        
        # Strategy 3: Temporal sampling (if dates available)
        temporal_docs = self._temporal_sampling(documents_data, target_count=2)
        
        # Combine and deduplicate
        sampled_docs = []
        seen_ids = set()
        
        for doc_list in [diverse_docs, length_docs, temporal_docs]:
            for doc in doc_list:
                if doc['document_id'] not in seen_ids:
                    sampled_docs.append(doc)
                    seen_ids.add(doc['document_id'])
        
        logger.info(f"🎯 Sampled {len(sampled_docs)} key documents")
        return sampled_docs[:15]  # Limit to 15 documents max
    
    async def _diversity_sampling(self, documents_data: List[Dict], target_count: int) -> List[Dict]:
        """Sample documents to maximize diversity using embeddings"""
        if len(documents_data) <= target_count:
            return documents_data
        
        try:
            # Get embeddings for each document (using first chunk as representative)
            embeddings = []
            valid_docs = []
            
            for doc in documents_data:
                if doc['chunks']:
                    # Use the embedding from the first chunk
                    first_chunk = doc['chunks'][0]
                    if 'embedding' in first_chunk:
                        embeddings.append(first_chunk['embedding'])
                        valid_docs.append(doc)
            
            if len(embeddings) < target_count:
                return valid_docs
            
            # Use greedy selection for diversity
            selected_indices = [0]  # Start with first document
            embeddings_array = np.array(embeddings)
            
            for _ in range(target_count - 1):
                # Find document most different from already selected
                selected_embeddings = embeddings_array[selected_indices]
                
                max_min_distance = -1
                best_idx = -1
                
                for i, embedding in enumerate(embeddings_array):
                    if i in selected_indices:
                        continue
                    
                    # Calculate minimum distance to selected documents
                    distances = [np.linalg.norm(embedding - sel_emb) for sel_emb in selected_embeddings]
                    min_distance = min(distances)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_idx = i
                
                if best_idx != -1:
                    selected_indices.append(best_idx)
            
            return [valid_docs[i] for i in selected_indices]
            
        except Exception as e:
            logger.warning(f"⚠️ Diversity sampling failed: {e}")
            return documents_data[:target_count]
    
    def _temporal_sampling(self, documents_data: List[Dict], target_count: int) -> List[Dict]:
        """Sample documents across time periods"""
        docs_with_dates = []
        
        for doc in documents_data:
            doc_date = self._extract_document_date(doc)
            if doc_date:
                docs_with_dates.append((doc, doc_date))
        
        if len(docs_with_dates) <= target_count:
            return [doc for doc, date in docs_with_dates]
        
        # Sort by date and sample evenly across time range
        docs_with_dates.sort(key=lambda x: x[1])
        
        # Sample evenly across the time range
        step = len(docs_with_dates) // target_count
        sampled = []
        
        for i in range(0, len(docs_with_dates), step):
            if len(sampled) < target_count:
                sampled.append(docs_with_dates[i][0])
        
        return sampled
    
    async def _create_individual_summaries(self, documents: List[Dict]) -> List[Dict]:
        """Create individual summaries for each document"""
        summaries = []
        
        for doc in documents:
            try:
                # Create a focused summary for this document
                summary = await self._summarize_single_document(doc)
                summaries.append({
                    'document_id': doc['document_id'],
                    'summary': summary,
                    'metadata': doc['metadata'],
                    'chunk_count': doc['chunk_count']
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to summarize document {doc['document_id']}: {e}")
                continue
        
        return summaries
    
    async def _summarize_single_document(self, doc: Dict) -> str:
        """Create a focused summary of a single document"""
        if not self.chat_service:
            return f"Document {doc['document_id']} contains {doc['chunk_count']} sections."
        
        # Use the chat service's document summarization
        try:
            response = await self.chat_service.summarize_document(doc['document_id'], "collection_analysis")
            return response.answer
        except Exception as e:
            logger.warning(f"⚠️ Chat service summarization failed: {e}")
            # Fallback to simple summary
            content = doc['total_content'][:2000]  # First 2000 chars
            return f"Document {doc['document_id']}: {content}..."
    
    async def _create_hierarchical_summary(self, individual_summaries: List[Dict], all_documents: List[Dict]) -> str:
        """Create a summary of summaries"""
        if not self.chat_service or not individual_summaries:
            return "Unable to generate hierarchical summary."
        
        try:
            # Combine individual summaries
            combined_summaries = "\n\n".join([
                f"Document {summary['document_id']} ({summary['chunk_count']} sections):\n{summary['summary']}"
                for summary in individual_summaries
            ])
            
            # Import datetime context utility
            from utils.system_prompt_utils import add_datetime_context_to_system_prompt
            
            # Create meta-summary prompt
            system_prompt = add_datetime_context_to_system_prompt(
                """You are Alex, an AI assistant specialized in analyzing large document collections. You are creating a comprehensive overview of a document collection based on individual document summaries.

Your task is to:
1. Identify overarching themes and patterns across all documents
2. Highlight key insights and findings
3. Note any temporal patterns or evolution of topics
4. Provide a coherent narrative that ties the collection together
5. Mention the scope and scale of the collection

Create a well-structured, comprehensive summary that gives someone a complete understanding of what this document collection contains and its key insights."""
            )

            user_prompt = f"""Based on the following individual document summaries, please create a comprehensive overview of this document collection:

Collection Overview:
- Total documents analyzed: {len(all_documents)}
- Key documents summarized: {len(individual_summaries)}
- Total content sections: {sum(doc['chunk_count'] for doc in all_documents)}

Individual Document Summaries:
{combined_summaries}

Please provide a comprehensive analysis that:
1. Summarizes the main themes and topics across the collection
2. Identifies key patterns, trends, or insights
3. Highlights the most important findings
4. Provides context about the collection's scope and significance
5. Notes any temporal evolution or changes over time (if applicable)

Structure your response with clear sections and use markdown formatting for readability."""

            # Call LLM for meta-summary
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await asyncio.wait_for(
                self.chat_service.openai_client.chat.completions.create(
                    model=self.chat_service.current_model,
                    messages=messages,
                    max_tokens=8000,
                    temperature=0.3
                ),
                timeout=120.0
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            else:
                return "Unable to generate comprehensive summary."
                
        except Exception as e:
            logger.error(f"❌ Hierarchical summary generation failed: {e}")
            return f"Collection of {len(all_documents)} documents analyzed. Individual summaries available but meta-summary generation failed."
    
    async def _generate_temporal_summary(self, temporal_insights: Dict, documents_data: List[Dict]) -> str:
        """Generate comprehensive temporal analysis using LLM"""
        if not self.chat_service:
            # Fallback to basic summary if no chat service
            summary_parts = [
                f"# Temporal Analysis of {len(documents_data)} Documents\n",
                "## Volume Trends"
            ]
            
            for period_type, volumes in temporal_insights['volume_trends'].items():
                if volumes:
                    total_periods = len(volumes)
                    avg_volume = sum(volumes.values()) / total_periods
                    summary_parts.append(f"- **{period_type.title()}**: {total_periods} periods, average {avg_volume:.1f} documents per period")
            
            summary_parts.append("\n## Peak Activity Periods")
            for period_type, (period, count) in temporal_insights['peak_periods'].items():
                summary_parts.append(f"- **{period_type.title()}**: {period} ({count} documents)")
            
            return "\n".join(summary_parts)
        
        try:
            # Extract temporal patterns and content for LLM analysis
            temporal_data = self._prepare_temporal_data_for_analysis(documents_data, temporal_insights)
            
            # Import datetime context utility
            from utils.system_prompt_utils import add_datetime_context_to_system_prompt
            
            system_prompt = add_datetime_context_to_system_prompt(
                """You are Alex, an AI assistant specialized in temporal analysis of document collections. You excel at identifying patterns, trends, and insights across time periods.

Your task is to analyze a collection of documents with temporal information and provide comprehensive insights about:
1. Communication patterns and volume trends over time
2. Key events, topics, or themes that emerge in different time periods
3. Evolution of content, tone, or focus over time
4. Notable peaks, gaps, or changes in activity
5. Relationships between timing and content themes

Provide a detailed, insightful analysis that helps understand the temporal dynamics of this document collection."""
            )

            user_prompt = f"""Please analyze this temporal document collection:

**Collection Overview:**
- Total Documents: {len(documents_data)}
- Time Period Analysis: {self._get_date_range_summary(documents_data)}

**Temporal Patterns:**
{temporal_data['patterns']}

**Document Content by Time Period:**
{temporal_data['content_by_period']}

**Volume Trends:**
{temporal_data['volume_analysis']}

Please provide a comprehensive temporal analysis that includes:

1. **Overview**: What does this collection represent temporally?
2. **Volume Patterns**: How does document volume change over time?
3. **Content Evolution**: How do themes, topics, or tone evolve?
4. **Key Periods**: What are the most significant time periods and why?
5. **Notable Patterns**: Any interesting temporal patterns or anomalies?
6. **Insights**: What can we learn from the temporal distribution and content?

Structure your response with clear sections and provide specific examples from the documents."""

            # Generate comprehensive temporal analysis
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await asyncio.wait_for(
                self.chat_service.openai_client.chat.completions.create(
                    model=self.chat_service.current_model,
                    messages=messages,
                    max_tokens=6000,
                    temperature=0.3
                ),
                timeout=120.0
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            else:
                return f"Temporal analysis of {len(documents_data)} documents completed, but detailed analysis generation failed."
                
        except Exception as e:
            logger.error(f"❌ Temporal summary generation failed: {e}")
            return f"Temporal analysis of {len(documents_data)} documents completed. Found {len(temporal_insights.get('volume_trends', {}))} time periods with varying activity levels."
    
    async def _generate_thematic_summary(self, theme_insights: Dict, documents_data: List[Dict]) -> str:
        """Generate summary focused on thematic analysis"""
        summary_parts = [
            f"# Thematic Analysis of {len(documents_data)} Documents\n",
            f"Identified {len(theme_insights)} main themes:\n"
        ]
        
        for cluster_id, insights in theme_insights.items():
            summary_parts.append(f"## Theme {cluster_id + 1}")
            summary_parts.append(f"- **Documents**: {insights['document_count']}")
            summary_parts.append(f"- **Key Terms**: {', '.join(insights['key_terms'][:5])}")
            summary_parts.append(f"- **Average Length**: {insights['avg_length']:.1f} sections per document\n")
        
        return "\n".join(summary_parts)
    
    async def _combine_analysis_results(self, analyses: List, documents_data: List[Dict]) -> str:
        """Combine results from multiple analysis types"""
        summary_parts = [
            f"# Comprehensive Analysis of {len(documents_data)} Documents\n",
            "This analysis combines temporal patterns, thematic clustering, and intelligent summarization.\n"
        ]
        
        # Add results from each analysis type
        for i, analysis in enumerate(analyses):
            if isinstance(analysis, Exception):
                summary_parts.append(f"Analysis {i+1} failed: {str(analysis)}\n")
            elif hasattr(analysis, 'answer'):
                summary_parts.append(f"## Analysis {i+1}\n{analysis.answer}\n")
        
        return "\n".join(summary_parts)
    
    def _create_temporal_citations(self, temporal_groups: Dict, documents_data: List[Dict]) -> List[Citation]:
        """Create citations for temporal analysis"""
        citations = []
        
        # Sample documents from different time periods
        for period_type, groups in temporal_groups.items():
            if period_type == 'monthly':  # Use monthly for citations
                for period, docs in list(groups.items())[:5]:  # Max 5 periods
                    if docs:
                        doc = docs[0]  # Representative document
                        citation = Citation(
                            document_id=doc['document_id'],
                            document_title=f"Document {doc['document_id']} ({period})",
                            chunk_id=doc['chunks'][0]['chunk_id'] if doc['chunks'] else '',
                            relevance_score=1.0,
                            snippet=f"Representative document from {period}"
                        )
                        citations.append(citation)
        
        return citations
    
    def _create_thematic_citations(self, theme_clusters: Dict, documents_data: List[Dict]) -> List[Citation]:
        """Create citations for thematic analysis"""
        citations = []
        
        for cluster_id, docs in theme_clusters.items():
            if docs:
                # Use the first document as representative
                doc = docs[0]
                citation = Citation(
                    document_id=doc['document_id'],
                    document_title=f"Document {doc['document_id']} (Theme {cluster_id + 1})",
                    chunk_id=doc['chunks'][0]['chunk_id'] if doc['chunks'] else '',
                    relevance_score=1.0,
                    snippet=f"Representative document from theme cluster {cluster_id + 1}"
                )
                citations.append(citation)
        
        return citations
    
    def _create_comprehensive_citations(self, documents_data: List[Dict]) -> List[Citation]:
        """Create comprehensive citations from all documents"""
        citations = []
        
        for doc in documents_data[:20]:  # Limit to 20 citations
            # Create enhanced snippet for better context
            extended_snippet = self._create_extended_snippet(doc['total_content'], "")
            
            citation = Citation(
                document_id=doc['document_id'],
                document_title=f"Document {doc['document_id']}",
                chunk_id=doc['chunks'][0]['chunk_id'] if doc['chunks'] else '',
                relevance_score=1.0,
                snippet=extended_snippet
            )
            citations.append(citation)
        
        return citations
    
    def _create_extended_snippet(self, content: str, query: str) -> str:
        """Create an extended snippet with more context around the relevant content"""
        try:
            # Use larger snippet length for collection analysis
            max_snippet_length = 600  # Good size for collection overviews
            
            if len(content) <= max_snippet_length:
                return content
            
            # For collection analysis, we want to show the beginning of documents
            # as they often contain the most important summary information
            
            # Try to find good breaking points (sentences) near the target length
            end_pos = max_snippet_length
            
            # Look back for sentence ending within reasonable distance
            for i in range(end_pos, max(0, end_pos - 150), -1):
                if content[i] in ['.', '!', '?', '\n']:
                    end_pos = i + 1
                    break
            
            snippet = content[:end_pos].strip()
            
            if end_pos < len(content):
                snippet = snippet + "..."
            
            return snippet
            
        except Exception as e:
            # Fallback to simple truncation with larger size
            return content[:600] + "..." if len(content) > 600 else content
    
    def _prepare_temporal_data_for_analysis(self, documents_data: List[Dict], temporal_insights: Dict) -> Dict[str, str]:
        """Prepare temporal data for LLM analysis"""
        try:
            # Extract patterns summary
            patterns = []
            for period_type, volumes in temporal_insights['volume_trends'].items():
                if volumes:
                    total_docs = sum(volumes.values())
                    periods = len(volumes)
                    avg_per_period = total_docs / periods if periods > 0 else 0
                    patterns.append(f"- {period_type.title()}: {total_docs} documents across {periods} periods (avg: {avg_per_period:.1f} per period)")
            
            # Group content by time periods for analysis
            content_by_period = []
            temporal_groups = self._group_by_time_periods(documents_data)
            
            # Sample content from different periods
            for period_type, groups in temporal_groups.items():
                if period_type == 'monthly' and groups:  # Focus on monthly for detailed analysis
                    sorted_periods = sorted(groups.items())
                    for period, docs in sorted_periods[:10]:  # Max 10 periods
                        if docs:
                            # Sample content from this period
                            sample_content = []
                            for doc in docs[:3]:  # Max 3 docs per period
                                content_preview = doc['total_content'][:300]
                                sample_content.append(f"  - Doc {doc['document_id']}: {content_preview}...")
                            
                            content_by_period.append(f"**{period}** ({len(docs)} documents):")
                            content_by_period.extend(sample_content)
                            content_by_period.append("")
            
            # Volume analysis summary
            volume_analysis = []
            for period_type, (peak_period, peak_count) in temporal_insights.get('peak_periods', {}).items():
                volume_analysis.append(f"- Peak {period_type}: {peak_period} with {peak_count} documents")
            
            return {
                'patterns': '\n'.join(patterns) if patterns else "No clear temporal patterns detected",
                'content_by_period': '\n'.join(content_by_period) if content_by_period else "No temporal content grouping available",
                'volume_analysis': '\n'.join(volume_analysis) if volume_analysis else "No volume analysis available"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare temporal data: {e}")
            return {
                'patterns': "Error extracting temporal patterns",
                'content_by_period': "Error grouping content by period", 
                'volume_analysis': "Error analyzing volume trends"
            }
    
    def _get_date_range_summary(self, documents_data: List[Dict]) -> str:
        """Get summary of date range covered by documents"""
        try:
            dates = []
            for doc in documents_data:
                doc_date = self._extract_document_date(doc)
                if doc_date:
                    dates.append(doc_date)
            
            if not dates:
                return "No temporal information available"
            
            dates.sort()
            start_date = dates[0].strftime('%Y-%m-%d')
            end_date = dates[-1].strftime('%Y-%m-%d')
            
            if start_date == end_date:
                return f"Single date: {start_date}"
            else:
                duration = dates[-1] - dates[0]
                return f"Date range: {start_date} to {end_date} (spanning {duration.days} days)"
                
        except Exception as e:
            logger.error(f"❌ Failed to get date range summary: {e}")
            return "Error determining date range"

    async def close(self):
        """Clean up resources"""
        if self.embedding_manager:
            await self.embedding_manager.close()
        logger.info("🔄 Collection Analysis Service closed")
