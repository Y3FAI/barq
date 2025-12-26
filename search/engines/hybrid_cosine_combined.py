"""
Two-stage hybrid with combined score reranking.

Stage 1: Hybrid-weighted fusion (BM25 + Vector) → top-N candidates
Stage 2: Combine Stage 1 score with cosine similarity for final ranking

Instead of discarding Stage 1 scores, we combine them with Stage 2
cosine scores to get the best of both worlds.
"""

from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from search.core.base import BaseSearchEngine, SearchResult
from search.engines.hybrid_weighted import HybridWeightedEngine
from search.engines.vector_faiss import VectorSearchEngineFAISS


class HybridCosineCombinedEngine(BaseSearchEngine):
    """
    Two-stage hybrid with combined score reranking.

    Stage 1: Hybrid-weighted 0.5/0.5 (broad recall)
    Stage 2: Combine Stage 1 scores with cosine scores (refinement)
    """

    def __init__(self, stage1_engine: HybridWeightedEngine,
                 reranker: VectorSearchEngineFAISS,
                 stage2_pool_size: int = 10,
                 stage1_weight: float = 0.5,
                 stage2_weight: float = 0.5):
        """
        Initialize hybrid with combined score reranker.

        Args:
            stage1_engine: Hybrid-weighted engine for Stage 1
            reranker: Vector engine for accessing cached embeddings
            stage2_pool_size: How many candidates to rerank (default 10)
            stage1_weight: Weight for Stage 1 scores (default 0.5)
            stage2_weight: Weight for Stage 2 cosine scores (default 0.5)
        """
        self.stage1 = stage1_engine
        self.reranker = reranker
        self.stage2_pool_size = stage2_pool_size
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight

    def index(self, db_path: str):
        """Index both stages."""
        print("Indexing Stage 1 (Hybrid-weighted)...")
        self.stage1.index(db_path)
        print("Indexing Stage 2 (Vector for reranking)...")
        self.reranker.index(db_path)
        print("✓ Two-stage hybrid with combined score reranking ready")

    @staticmethod
    def normalize_scores(scores: List[float]) -> np.ndarray:
        """Min-max normalization to [0, 1] range."""
        scores_arr = np.array(scores)
        if len(scores_arr) == 0:
            return scores_arr
        min_score = scores_arr.min()
        max_score = scores_arr.max()
        if max_score - min_score < 1e-9:
            return np.ones_like(scores_arr)
        return (scores_arr - min_score) / (max_score - min_score)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Two-stage search with combined score reranking.

        Args:
            query: Search query
            top_k: Number of final results to return

        Returns:
            List of SearchResult objects reranked by combined scores
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
        cosine_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Normalize both Stage 1 and Stage 2 scores
        stage1_scores = [c.score for c in valid_candidates]
        norm_stage1 = self.normalize_scores(stage1_scores)
        norm_stage2 = self.normalize_scores(cosine_scores)

        # Combine scores
        combined_scores = (
            self.stage1_weight * norm_stage1 +
            self.stage2_weight * norm_stage2
        )

        # Sort by combined score (descending)
        reranked_indices = np.argsort(combined_scores)[::-1][:top_k]

        # Build final results
        results = []
        for new_rank, idx in enumerate(reranked_indices, 1):
            original = valid_candidates[idx]
            final_score = float(combined_scores[idx])

            results.append(SearchResult(
                doc_id=original.doc_id,
                url=original.url,
                score=final_score,
                content_snippet=original.content_snippet,
                metadata={
                    'engine': 'hybrid-cosine-combined',
                    'stage1_weighted_score': original.score,
                    'stage2_cosine_score': float(cosine_scores[idx]),
                    'combined_score': final_score,
                    'stage2_rank': new_rank,
                    'rerank_method': 'combined_scores'
                }
            ))

        return results

    def get_name(self) -> str:
        """Return engine name."""
        return f"hybrid-cosine-combined-{self.stage1_weight}-{self.stage2_weight}"
