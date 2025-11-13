"""
Content deduplication utilities for query results
"""

import hashlib
import logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

from config import settings

logger = logging.getLogger(__name__)


class DeduplicationManager:
    """Manages content deduplication for query results"""
    
    def __init__(self):
        self.similarity_cache = {}
        self.max_cache_size = 1000
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using multiple text-based methods"""
        
        # Check cache first
        cache_key = tuple(sorted([hash(text1), hash(text2)]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Method 1: Jaccard similarity (word overlap)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union) if union else 0
        
        # Method 2: Character n-gram similarity
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text)-n+1))
        
        ngrams1 = get_ngrams(text1.lower())
        ngrams2 = get_ngrams(text2.lower())
        ngram_similarity = len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2)) if ngrams1.union(ngrams2) else 0
        
        # Method 3: Longest common subsequence ratio
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        lcs_ratio = lcs_length(text1, text2) / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 0
        
        # Weighted combination
        similarity = 0.4 * jaccard + 0.3 * ngram_similarity + 0.3 * lcs_ratio
        
        # Cache the result
        self._cache_similarity(cache_key, similarity)
        
        return similarity
    
    def _cache_similarity(self, cache_key: tuple, similarity: float):
        """Cache similarity score with size limit"""
        if len(self.similarity_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.similarity_cache.keys())[:100]
            for key in oldest_keys:
                del self.similarity_cache[key]
        
        self.similarity_cache[cache_key] = similarity
    
    def deduplicate_by_content_similarity(self, chunks: List[Dict]) -> List[Dict]:
        """Remove chunks with high content similarity using optimized algorithm"""
        if not chunks or len(chunks) <= 1:
            return chunks
        
        logger.info(f"🔄 Deduplicating {len(chunks)} chunks by content similarity (optimized)")
        
        # For large datasets, use fast pre-filtering
        if len(chunks) > settings.FAST_DEDUP_CHUNK_THRESHOLD:
            return self._fast_similarity_deduplication(chunks)
        else:
            return self._standard_similarity_deduplication(chunks)
    
    def _fast_similarity_deduplication(self, chunks: List[Dict]) -> List[Dict]:
        """Fast deduplication using word fingerprints and early termination"""
        threshold = settings.CONTENT_SIMILARITY_THRESHOLD
        result = []
        used_indices = set()
        
        # Sort by relevance score first
        sorted_chunks = sorted(enumerate(chunks), key=lambda x: x[1].get('score', 0), reverse=True)
        
        # Pre-compute word fingerprints for fast filtering
        chunk_fingerprints = []
        for i, (original_idx, chunk) in enumerate(sorted_chunks):
            content = chunk.get('content', '')
            # Create word fingerprint (set of significant words)
            words = set(word.lower() for word in content.split() if len(word) > 3)
            chunk_fingerprints.append((original_idx, chunk, words))
        
        logger.info(f"🔄 Processing {len(chunk_fingerprints)} chunks with fast fingerprint matching")
        
        for i, (original_idx, chunk1, words1) in enumerate(chunk_fingerprints):
            if original_idx in used_indices:
                continue
            
            result.append(chunk1)
            used_indices.add(original_idx)
            
            # Fast pre-filtering: only check chunks with significant word overlap
            for j in range(i + 1, len(chunk_fingerprints)):
                other_idx, chunk2, words2 = chunk_fingerprints[j]
                if other_idx in used_indices:
                    continue
                
                # Quick Jaccard check on significant words
                if not words1 or not words2:
                    continue
                
                word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                
                # Only do expensive similarity calculation if there's significant word overlap
                if word_overlap > 0.3:  # Pre-filter threshold
                    similarity = self.calculate_text_similarity(
                        chunk1.get('content', ''),
                        chunk2.get('content', '')
                    )
                    
                    if similarity >= threshold:
                        used_indices.add(other_idx)
                        logger.debug(f"📊 Removed similar chunk (similarity: {similarity:.2f})")
        
        logger.info(f"📊 Fast content similarity dedup: {len(chunks)} → {len(result)} chunks")
        return result
    
    def _standard_similarity_deduplication(self, chunks: List[Dict]) -> List[Dict]:
        """Standard deduplication for smaller datasets"""
        result = []
        used_indices = set()
        threshold = settings.CONTENT_SIMILARITY_THRESHOLD
        
        # Sort by relevance score first
        sorted_chunks = sorted(enumerate(chunks), key=lambda x: x[1].get('score', 0), reverse=True)
        
        for i, (original_idx, chunk1) in enumerate(sorted_chunks):
            if original_idx in used_indices:
                continue
            
            result.append(chunk1)
            used_indices.add(original_idx)
            
            # Check remaining chunks for similarity
            for j in range(i + 1, len(sorted_chunks)):
                other_idx, chunk2 = sorted_chunks[j]
                if other_idx not in used_indices:
                    similarity = self.calculate_text_similarity(
                        chunk1.get('content', ''),
                        chunk2.get('content', '')
                    )
                    
                    if similarity >= threshold:
                        used_indices.add(other_idx)
                        logger.debug(f"📊 Removed similar chunk (similarity: {similarity:.2f})")
        
        logger.info(f"📊 Standard content similarity dedup: {len(chunks)} → {len(result)} chunks")
        return result
    
    def deduplicate_by_document(self, chunks: List[Dict]) -> List[Dict]:
        """Limit chunks per document to prevent over-representation"""
        if not chunks:
            return chunks
        
        logger.info(f"🔄 Applying document-level deduplication")
        
        document_chunks = defaultdict(list)
        
        for chunk in chunks:
            doc_id = chunk.get('document_id', 'unknown')
            document_chunks[doc_id].append(chunk)
        
        # Sort chunks within each document by relevance score
        for doc_id in document_chunks:
            document_chunks[doc_id].sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Take top N chunks per document
        result = []
        max_per_doc = settings.MAX_CHUNKS_PER_DOCUMENT
        
        for doc_id, doc_chunks in document_chunks.items():
            selected = doc_chunks[:max_per_doc]
            result.extend(selected)
            
            if len(doc_chunks) > max_per_doc:
                logger.debug(f"📊 Limited document {doc_id}: {len(doc_chunks)} → {len(selected)} chunks")
        
        # Re-sort by overall relevance
        result.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"📊 Document dedup: {len(chunks)} → {len(result)} chunks")
        return result
    
    def deduplicate_email_threads(self, chunks: List[Dict]) -> List[Dict]:
        """Remove redundant email content from threads"""
        if not chunks or not settings.EMAIL_THREAD_DEDUP_ENABLED:
            return chunks
        
        logger.info(f"🔄 Deduplicating email threads")
        
        def extract_email_metadata(content: str) -> Dict:
            """Extract email headers and quoted content"""
            lines = content.split('\n')
            metadata = {'subject': '', 'from': '', 'quoted_lines': 0}
            
            for line in lines:
                line_lower = line.lower().strip()
                if line_lower.startswith('subject:'):
                    metadata['subject'] = line[8:].strip()
                elif line_lower.startswith('from:'):
                    metadata['from'] = line[5:].strip()
                elif line.startswith('>') or line_lower.startswith('on '):
                    metadata['quoted_lines'] += 1
            
            return metadata
        
        # Group by email thread (same subject)
        threads = defaultdict(list)
        non_email_chunks = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            if 'from:' in content.lower() or 'subject:' in content.lower():
                metadata = extract_email_metadata(content)
                subject = metadata['subject'] or 'no_subject'
                threads[subject].append((chunk, metadata))
            else:
                non_email_chunks.append(chunk)
        
        # For each thread, keep the email with least quoted content
        result = non_email_chunks.copy()
        
        for subject, thread_chunks in threads.items():
            if len(thread_chunks) == 1:
                result.append(thread_chunks[0][0])
            else:
                # Sort by quoted content (ascending) then by score (descending)
                thread_chunks.sort(key=lambda x: (x[1]['quoted_lines'], -x[0].get('score', 0)))
                result.append(thread_chunks[0][0])
                logger.debug(f"📊 Email thread '{subject}': {len(thread_chunks)} → 1 chunk")
        
        logger.info(f"📊 Email dedup: {len(chunks)} → {len(result)} chunks")
        return result
    
    def ensure_source_diversity(self, chunks: List[Dict]) -> List[Dict]:
        """Ensure diverse sources in results"""
        if not chunks:
            return chunks
        
        logger.info(f"🔄 Ensuring source diversity")
        
        source_counts = defaultdict(int)
        result = []
        max_per_source = settings.MAX_CHUNKS_PER_SOURCE
        
        # Sort by relevance score first
        sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)
        
        for chunk in sorted_chunks:
            # Determine source (could be document type, author, domain, etc.)
            metadata = chunk.get('metadata', {})
            source = (
                metadata.get('source') or 
                metadata.get('document_type') or 
                metadata.get('author') or 
                'unknown'
            )
            
            if source_counts[source] < max_per_source:
                result.append(chunk)
                source_counts[source] += 1
        
        logger.info(f"📊 Source diversity: {len(chunks)} → {len(result)} chunks")
        return result
    
    def deduplicate_by_content_hash(self, chunks: List[Dict]) -> List[Dict]:
        """Remove exact content duplicates using hash"""
        if not chunks:
            return chunks
        
        logger.info(f"🔄 Removing exact content duplicates")
        
        seen_hashes = set()
        result = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                result.append(chunk)
        
        logger.info(f"📊 Hash dedup: {len(chunks)} → {len(result)} chunks")
        return result
    
    async def deduplicate_query_results(self, chunks: List[Dict], query: str = "") -> List[Dict]:
        """Multi-stage deduplication pipeline - content similarity focused"""
        if not settings.DEDUPLICATION_ENABLED or not chunks:
            return chunks[:settings.FINAL_RESULT_LIMIT]
        
        logger.info(f"🔄 Starting content similarity deduplication for {len(chunks)} chunks")
        original_count = len(chunks)
        
        # Stage 1: Exact content deduplication (hash-based)
        chunks = self.deduplicate_by_content_hash(chunks)
        
        # Stage 2: Email thread deduplication (if applicable)
        chunks = self.deduplicate_email_threads(chunks)
        
        # Stage 3: Content similarity deduplication (main focus)
        chunks = self.deduplicate_by_content_similarity(chunks)
        
        # Stage 4: Final ranking and limit
        chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_chunks = chunks[:settings.FINAL_RESULT_LIMIT]
        
        reduction_ratio = (original_count - len(final_chunks)) / original_count if original_count > 0 else 0
        
        logger.info(f"✅ Content similarity deduplication complete: {original_count} → {len(final_chunks)} chunks ({reduction_ratio:.1%} reduction)")
        
        return final_chunks


# Global deduplication manager instance
deduplication_manager = DeduplicationManager()
