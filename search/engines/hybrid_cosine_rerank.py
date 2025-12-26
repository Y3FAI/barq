"""
Two-stage hybrid with cosine similarity reranking.

Stage 1: Hybrid-weighted fusion (BM25 + Vector) → top-N candidates
Stage 2: Cosine similarity reranking using cached embeddings

This approach is much faster than cross-encoders while still providing
semantic reranking to improve Hit@1.
"""

from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from search.core.base import BaseSearchEngine, SearchResult
from search.engines.hybrid_weighted import HybridWeightedEngine
from search.engines.vector_faiss import VectorSearchEngineFAISS


class HybridCosineRerankEngine(BaseSearchEngine):
    """
    Two-stage hybrid with cosine similarity reranking.

    Stage 1: Hybrid-weighted 0.5/0.5 (broad recall)
    Stage 2: Cosine similarity reranking (semantic precision)
    """

    def __init__(self, stage1_engine: HybridWeightedEngine,
                 reranker: VectorSearchEngineFAISS,
                 stage2_pool_size: int = 10):
        """
        Initialize hybrid with cosine reranker.

        Args:
            stage1_engine: Hybrid-weighted engine for Stage 1
            reranker: Vector engine for accessing cached embeddings
            stage2_pool_size: How many candidates to rerank (default 10)
        """
        self.stage1 = stage1_engine
        self.reranker = reranker
        self.stage2_pool_size = stage2_pool_size

    def index(self, db_path: str):
        """Index both stages."""
        print("Indexing Stage 1 (Hybrid-weighted)...")
        self.stage1.index(db_path)
        print("Indexing Stage 2 (Vector for reranking)...")
        self.reranker.index(db_path)
        print("✓ Two-stage hybrid with cosine reranking ready")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Two-stage search with cosine similarity reranking.

        Args:
            query: Search query
            top_k: Number of final results to return

        Returns:
            List of SearchResult objects reranked by cosine similarity
        """
        # Stage 1: Hybrid-weighted fusion
        candidates = self.stage1.search(query, top_k=self.stage2_pool_size)

        if not candidates:
            return []

        # If we have very few candidates, return them as-is
        if len(candidates) <= 1:
            return candidates

        # Stage 2: Cosine similarity reranking
        # Load model if needed
        if self.reranker.model is None:
            self.reranker._load_model()

        # Encode query with retrieval prompt
        encode_kwargs = {'convert_to_numpy': True, 'show_progress_bar': False}
        if self.reranker.config.query_prompt:
            encode_kwargs['prompt_name'] = self.reranker.config.query_prompt

        query_embedding = self.reranker.model.encode([query], **encode_kwargs)

        # Get cached embeddings for candidates
        candidate_indices = []
        valid_candidates = []

        for c in candidates:
            try:
                idx = self.reranker.doc_ids.index(c.doc_id)
                candidate_indices.append(idx)
                valid_candidates.append(c)
            except ValueError:
                continue

        if len(valid_candidates) == 0:
            return candidates[:top_k]  # No valid candidates

        # Reconstruct embeddings from FAISS index
        doc_embeddings = np.array([
            self.reranker.faiss_index.reconstruct(int(idx))
            for idx in candidate_indices
        ])

        # Ensure proper shapes for cosine_similarity
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if len(doc_embeddings.shape) == 1:
            doc_embeddings = doc_embeddings.reshape(1, -1)

        # Calculate cosine similarity scores
        rerank_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Sort by cosine similarity (descending)
        reranked_indices = np.argsort(rerank_scores)[::-1][:top_k]

        # Build final results
        results = []
        for new_rank, idx in enumerate(reranked_indices, 1):
            original = valid_candidates[idx]
            cosine_score = float(rerank_scores[idx])

            results.append(SearchResult(
                doc_id=original.doc_id,
                url=original.url,
                score=cosine_score,
                content_snippet=original.content_snippet,
                metadata={
                    'engine': 'hybrid-cosine-rerank',
                    'stage1_rank': idx + 1,
                    'stage1_weighted_score': original.score,
                    'stage2_cosine_score': cosine_score,
                    'stage2_rank': new_rank,
                    'rerank_method': 'cosine_similarity_cached'
                }
            ))

        return results

    def get_name(self) -> str:
        """Return engine name."""
        return "hybrid-cosine-rerank"
