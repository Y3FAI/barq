"""
Hybrid search engine using Reciprocal Rank Fusion (RRF) + score blending.

Stage 1: Run BM25 and Vector engines independently to gather a larger candidate pool.
Stage 2: Combine their rankings with RRF, then blend in normalized scores to
         preserve strong lexical/semantic evidence.

Compared to simple weighted score fusion, RRF makes it easier for a document that
is highly ranked in either engine to surface near the top, which often improves
Hit@K metrics when the engines disagree.
"""

from typing import List, Dict

from search.core.base import BaseSearchEngine, SearchResult
from search.engines.hybrid_weighted import HybridWeightedEngine


class HybridRRFFusionEngine(BaseSearchEngine):
    """
    Hybrid engine that fuses BM25 + Vector rankings via Reciprocal Rank Fusion.

    Each engine contributes 1 / (rrf_k + rank) to the final score, making it easy
    for a top-ranked result in either engine to appear near the top. To retain rich
    signal from the raw scores, we optionally blend in normalized scores from each
    engine.
    """

    def __init__(self,
                 engine1: BaseSearchEngine,
                 engine2: BaseSearchEngine,
                 candidate_pool_size: int = 75,
                 rrf_k: int = 60,
                 score_blend_weight: float = 0.3,
                 weight1: float = 0.5,
                 weight2: float = 0.5):
        """
        Args:
            engine1: First search engine (typically BM25)
            engine2: Second search engine (typically Vector)
            candidate_pool_size: How many docs to take from each engine
            rrf_k: RRF damping factor (higher => less emphasis on top ranks)
            score_blend_weight: Strength of normalized score blending vs RRF
            weight1: Weight for engine1 normalized score inside the blend term
            weight2: Weight for engine2 normalized score inside the blend term
        """
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if not (0.0 <= score_blend_weight <= 1.0):
            raise ValueError("score_blend_weight must be between 0 and 1")
        if not (0 <= weight1 <= 1 and 0 <= weight2 <= 1):
            raise ValueError("Weights must be between 0 and 1")

        self.engine1 = engine1
        self.engine2 = engine2
        self.candidate_pool_size = candidate_pool_size
        self.rrf_k = rrf_k
        self.score_blend_weight = score_blend_weight
        self.weight1 = weight1
        self.weight2 = weight2

    def index(self, db_path: str):
        """Index both engines."""
        print("Indexing engine 1 (RRF hybrid)...")
        self.engine1.index(db_path)
        print("Indexing engine 2 (RRF hybrid)...")
        self.engine2.index(db_path)
        print("✓ Hybrid RRF fusion engine ready")

    def _build_rank_map(self, results: List[SearchResult]) -> Dict[int, int]:
        """Return doc_id -> rank (1-indexed)."""
        return {res.doc_id: idx + 1 for idx, res in enumerate(results)}

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search via RRF fusion + score blending."""
        results1 = self.engine1.search(query, top_k=self.candidate_pool_size)
        results2 = self.engine2.search(query, top_k=self.candidate_pool_size)

        if not results1 and not results2:
            return []

        rank_map1 = self._build_rank_map(results1)
        rank_map2 = self._build_rank_map(results2)

        score_map1 = {r.doc_id: r.score for r in results1}
        score_map2 = {r.doc_id: r.score for r in results2}

        all_doc_ids = list(set(rank_map1.keys()) | set(rank_map2.keys()))

        # Normalize raw scores for optional blending
        scores1_norm = {}
        scores2_norm = {}
        if all_doc_ids:
            arr1 = [score_map1.get(doc_id, 0.0) for doc_id in all_doc_ids]
            arr2 = [score_map2.get(doc_id, 0.0) for doc_id in all_doc_ids]

            norm1 = HybridWeightedEngine.normalize_scores(arr1)
            norm2 = HybridWeightedEngine.normalize_scores(arr2)

            scores1_norm = {doc_id: score for doc_id, score in zip(all_doc_ids, norm1)}
            scores2_norm = {doc_id: score for doc_id, score in zip(all_doc_ids, norm2)}

        # Build doc lookup (prefer engine1 data, fall back to engine2)
        doc_lookup: Dict[int, SearchResult] = {}
        for res in results1 + results2:
            if res.doc_id not in doc_lookup:
                doc_lookup[res.doc_id] = res

        def rrf_contrib(rank: int) -> float:
            return 1.0 / (self.rrf_k + rank - 1)

        combined_scores = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in rank_map1:
                score += rrf_contrib(rank_map1[doc_id])
            if doc_id in rank_map2:
                score += rrf_contrib(rank_map2[doc_id])

            if self.score_blend_weight > 0:
                blended = (
                    self.weight1 * scores1_norm.get(doc_id, 0.0) +
                    self.weight2 * scores2_norm.get(doc_id, 0.0)
                )
                score += self.score_blend_weight * blended

            combined_scores[doc_id] = score

        # Sort and build final SearchResult list
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, fused_score in sorted_docs[:top_k]:
            base_result = doc_lookup[doc_id]
            final_results.append(SearchResult(
                doc_id=doc_id,
                url=base_result.url,
                score=float(fused_score),
                content_snippet=base_result.content_snippet,
                metadata={
                    'engine': 'hybrid-rrf-fusion',
                    'bm25_rank': rank_map1.get(doc_id),
                    'vector_rank': rank_map2.get(doc_id),
                    'bm25_score': score_map1.get(doc_id),
                    'vector_score': score_map2.get(doc_id),
                    'bm25_norm_score': scores1_norm.get(doc_id),
                    'vector_norm_score': scores2_norm.get(doc_id),
                    'rrf_k': self.rrf_k,
                    'score_blend_weight': self.score_blend_weight,
                    'weight1': self.weight1,
                    'weight2': self.weight2
                }
            ))

        return final_results

    def get_name(self) -> str:
        """Return engine name."""
        name1 = self.engine1.get_name()
        name2 = self.engine2.get_name()
        return f"hybrid-rrf-{self.weight1:.1f}-{self.weight2:.1f}-{name1}-{name2}"
