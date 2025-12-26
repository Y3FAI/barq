"""
Weighted score fusion hybrid search.

Normalizes scores from both engines to [0, 1] range, then combines
with weighted average: final_score = w1 * norm(score1) + w2 * norm(score2)

Allows tuning the balance between BM25 (keyword) and Vector (semantic).
"""

import numpy as np
from typing import List, Dict
from search.core.base import BaseSearchEngine, SearchResult


class HybridWeightedEngine(BaseSearchEngine):
    """
    Hybrid search using weighted score fusion.

    Normalizes scores from both engines and combines with weights:
    final_score = w1 * normalize(bm25_score) + w2 * normalize(vector_score)

    Weights allow tuning the balance between keyword and semantic search.
    """

    def __init__(self, engine1: BaseSearchEngine, engine2: BaseSearchEngine,
                 weight1: float = 0.5, weight2: float = 0.5,
                 candidate_pool_size: int = 50):
        """
        Initialize weighted hybrid engine.

        Args:
            engine1: First search engine (e.g., BM25)
            engine2: Second search engine (e.g., Vector)
            weight1: Weight for engine1 scores (default 0.5)
            weight2: Weight for engine2 scores (default 0.5)
            candidate_pool_size: How many results to get from each engine
        """
        self.engine1 = engine1
        self.engine2 = engine2
        self.weight1 = weight1
        self.weight2 = weight2
        self.candidate_pool_size = candidate_pool_size

        # Validate weights
        if not (0 <= weight1 <= 1 and 0 <= weight2 <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if abs(weight1 + weight2 - 1.0) > 0.001:
            raise ValueError("Weights should sum to 1.0")

    def index(self, db_path: str):
        """Index both engines."""
        print("Indexing engine 1...")
        self.engine1.index(db_path)
        print("Indexing engine 2...")
        self.engine2.index(db_path)
        print("✓ Weighted hybrid engine ready")

    @staticmethod
    def normalize_scores(scores: List[float]) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: List of scores to normalize

        Returns:
            Normalized scores as numpy array
        """
        scores_arr = np.array(scores)

        if len(scores_arr) == 0:
            return scores_arr

        min_score = scores_arr.min()
        max_score = scores_arr.max()

        # Avoid division by zero
        if max_score - min_score < 1e-9:
            return np.ones_like(scores_arr)

        return (scores_arr - min_score) / (max_score - min_score)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search using weighted score fusion.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects ranked by weighted combined score
        """
        # Step 1: Get results from both engines
        results1 = self.engine1.search(query, top_k=self.candidate_pool_size)
        results2 = self.engine2.search(query, top_k=self.candidate_pool_size)

        # Step 2: Build score maps
        scores1 = {r.doc_id: r.score for r in results1}
        scores2 = {r.doc_id: r.score for r in results2}

        # Step 3: Normalize scores within each engine
        # Get all doc_ids
        all_doc_ids = set(scores1.keys()) | set(scores2.keys())

        # Build score arrays for normalization
        scores1_list = [scores1.get(doc_id, 0) for doc_id in all_doc_ids]
        scores2_list = [scores2.get(doc_id, 0) for doc_id in all_doc_ids]

        norm_scores1 = self.normalize_scores(scores1_list)
        norm_scores2 = self.normalize_scores(scores2_list)

        # Step 4: Create normalized score maps
        norm_scores1_map = {doc_id: score for doc_id, score in zip(all_doc_ids, norm_scores1)}
        norm_scores2_map = {doc_id: score for doc_id, score in zip(all_doc_ids, norm_scores2)}

        # Step 5: Store original results for lookup
        result_lookup: Dict[int, SearchResult] = {}
        for r in results1:
            result_lookup[r.doc_id] = r
        for r in results2:
            if r.doc_id not in result_lookup:
                result_lookup[r.doc_id] = r

        # Step 6: Calculate weighted combined scores
        combined_scores = {}
        for doc_id in all_doc_ids:
            norm1 = norm_scores1_map.get(doc_id, 0)
            norm2 = norm_scores2_map.get(doc_id, 0)
            combined_scores[doc_id] = self.weight1 * norm1 + self.weight2 * norm2

        # Step 7: Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 8: Build final SearchResult objects
        final_results = []
        for doc_id, combined_score in sorted_docs[:top_k]:
            original = result_lookup[doc_id]

            final_results.append(SearchResult(
                doc_id=doc_id,
                url=original.url,
                score=combined_score,
                content_snippet=original.content_snippet,
                metadata={
                    'engine': 'hybrid-weighted',
                    'weight1': self.weight1,
                    'weight2': self.weight2,
                    'score1': scores1.get(doc_id, 0),
                    'score2': scores2.get(doc_id, 0),
                    'norm_score1': norm_scores1_map.get(doc_id, 0),
                    'norm_score2': norm_scores2_map.get(doc_id, 0),
                }
            ))

        return final_results

    def get_name(self) -> str:
        """Return engine name."""
        name1 = self.engine1.get_name()
        name2 = self.engine2.get_name()
        return f"hybrid-weighted-{self.weight1:.1f}-{self.weight2:.1f}-{name1}-{name2}"
