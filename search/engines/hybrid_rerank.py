"""
Two-stage hybrid search: BM25 retrieval + Vector re-ranking.

Stage 1: BM25 quickly retrieves top-N candidates (leveraging its speed and recall)
Stage 2: Vector re-ranks only those candidates (semantic refinement)

This is faster than full hybrid because vector only processes a small candidate set.
"""

import numpy as np
from typing import List, Dict
from search.core.base import BaseSearchEngine, SearchResult
from search.engines.vector_faiss import VectorSearchEngineFAISS


class HybridRerankEngine(BaseSearchEngine):
    """
    Two-stage hybrid: BM25 retrieval → Vector re-ranking.

    Leverages BM25's speed and broad recall for initial retrieval,
    then uses vector search for semantic re-ranking of candidates.
    """

    def __init__(self, retriever: BaseSearchEngine, reranker: VectorSearchEngineFAISS,
                 candidate_pool_size: int = 50):
        """
        Initialize two-stage hybrid engine.

        Args:
            retriever: Fast retrieval engine (typically BM25)
            reranker: Vector engine for re-ranking
            candidate_pool_size: How many candidates to retrieve before re-ranking
        """
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_pool_size = candidate_pool_size

    def index(self, db_path: str):
        """Index both engines."""
        print("Indexing retriever (Stage 1)...")
        self.retriever.index(db_path)
        print("Indexing reranker (Stage 2)...")
        self.reranker.index(db_path)
        print("✓ Two-stage hybrid engine ready")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Two-stage search: retrieve candidates, then re-rank.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects re-ranked by vector similarity
        """
        # Stage 1: BM25 retrieves top-N candidates (fast, broad recall)
        candidates = self.retriever.search(query, top_k=self.candidate_pool_size)

        if not candidates:
            return []

        # Stage 2: Re-rank candidates using vector similarity
        # Load model if not already loaded
        if self.reranker.model is None:
            self.reranker._load_model()

        # Encode query
        encode_kwargs = {'convert_to_numpy': True}
        if self.reranker.config.query_prompt:
            encode_kwargs['prompt_name'] = self.reranker.config.query_prompt

        query_embedding = self.reranker.model.encode([query], **encode_kwargs)

        # Get embeddings for candidate documents only
        candidate_doc_ids = [c.doc_id for c in candidates]

        # Find indices of candidates in the full document list
        candidate_indices = []
        for doc_id in candidate_doc_ids:
            try:
                idx = self.reranker.doc_ids.index(doc_id)
                candidate_indices.append(idx)
            except ValueError:
                # Document not in embeddings (shouldn't happen)
                continue

        # Get embeddings for these candidates
        candidate_embeddings = self.reranker.faiss_index.reconstruct_n(0, len(self.reranker.doc_ids))
        candidate_embeddings = candidate_embeddings[candidate_indices]

        # Calculate similarity scores
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # Re-rank candidates by vector similarity
        reranked_indices = np.argsort(similarities)[::-1][:top_k]

        # Build final results
        results = []
        for idx in reranked_indices:
            original_candidate = candidates[idx]
            vector_score = float(similarities[idx])

            results.append(SearchResult(
                doc_id=original_candidate.doc_id,
                url=original_candidate.url,
                score=vector_score,
                content_snippet=original_candidate.content_snippet,
                metadata={
                    'engine': 'hybrid-rerank',
                    'stage1_rank': idx + 1,
                    'stage1_score': original_candidate.score,
                    'stage2_score': vector_score
                }
            ))

        return results

    def get_name(self) -> str:
        """Return engine name."""
        retriever_name = self.retriever.get_name()
        reranker_name = self.reranker.get_name().replace('vector-faiss-', '')
        return f"hybrid-rerank-{retriever_name}-{reranker_name}"
