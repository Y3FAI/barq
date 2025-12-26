"""
Hybrid ensemble engine that mixes RRF ranks, normalized scores, and targeted boosts.

Motivation: RRF alone improves mid-ranks (Hit@3/Hit@5) but can hurt Hit@1 when
it demotes very strong BM25 matches. Pure weighted scoring keeps Hit@1 high but
misses cases where vector search alone finds the correct service.

This engine combines the strengths of both:
  - Reciprocal Rank Fusion encourages agreement between BM25 and vectors
  - Normalized score blending preserves magnitude information
  - Max-score & top-rank boosts keep elite BM25 hits near the top
"""

from typing import List, Dict
import numpy as np

from search.core.base import BaseSearchEngine, SearchResult
from search.engines.hybrid_weighted import HybridWeightedEngine


class HybridEnsembleBoostEngine(BaseSearchEngine):
    """Hybrid BM25 + Vector ensemble with RRF + score boosts."""

    def __init__(self,
                 engine1: BaseSearchEngine,
                 engine2: BaseSearchEngine,
                 candidate_pool_size: int = 80,
                 rrf_k: int = 50,
                 rrf_weight: float = 0.5,
                 score_weight: float = 0.35,
                 weight1: float = 0.6,
                 weight2: float = 0.4,
                 max_boost_weight: float = 0.15,
                 bm25_boost_weight: float = 0.1,
                 bm25_boost_rank: int = 2):
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        for name, value in [
            ("rrf_weight", rrf_weight),
            ("score_weight", score_weight),
            ("max_boost_weight", max_boost_weight),
            ("bm25_boost_weight", bm25_boost_weight),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not (0 <= weight1 <= 1 and 0 <= weight2 <= 1):
            raise ValueError("weight1/weight2 must be between 0 and 1")

        self.engine1 = engine1
        self.engine2 = engine2
        self.candidate_pool_size = candidate_pool_size
        self.rrf_k = rrf_k
        self.rrf_weight = rrf_weight
        self.score_weight = score_weight
        self.weight1 = weight1
        self.weight2 = weight2
        self.max_boost_weight = max_boost_weight
        self.bm25_boost_weight = bm25_boost_weight
        self.bm25_boost_rank = bm25_boost_rank

    def index(self, db_path: str):
        print("Indexing engine 1 (ensemble boost)...")
        self.engine1.index(db_path)
        print("Indexing engine 2 (ensemble boost)...")
        self.engine2.index(db_path)
        print("✓ Hybrid ensemble boost engine ready")

    @staticmethod
    def _rank_map(results: List[SearchResult]) -> Dict[int, int]:
        return {res.doc_id: idx + 1 for idx, res in enumerate(results)}

    @staticmethod
    def _normalize(values: List[float]) -> np.ndarray:
        arr = np.array(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        min_val = arr.min()
        max_val = arr.max()
        if abs(max_val - min_val) < 1e-9:
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        results1 = self.engine1.search(query, top_k=self.candidate_pool_size)
        results2 = self.engine2.search(query, top_k=self.candidate_pool_size)

        if not results1 and not results2:
            return []

        rank_map1 = self._rank_map(results1)
        rank_map2 = self._rank_map(results2)
        all_doc_ids = list(set(rank_map1) | set(rank_map2))

        score_map1 = {res.doc_id: res.score for res in results1}
        score_map2 = {res.doc_id: res.score for res in results2}

        norm1 = {}
        norm2 = {}
        if all_doc_ids:
            arr1 = [score_map1.get(doc_id, 0.0) for doc_id in all_doc_ids]
            arr2 = [score_map2.get(doc_id, 0.0) for doc_id in all_doc_ids]
            norm_arr1 = HybridWeightedEngine.normalize_scores(arr1)
            norm_arr2 = HybridWeightedEngine.normalize_scores(arr2)
            norm1 = {doc_id: score for doc_id, score in zip(all_doc_ids, norm_arr1)}
            norm2 = {doc_id: score for doc_id, score in zip(all_doc_ids, norm_arr2)}

        doc_lookup: Dict[int, SearchResult] = {}
        for res in results1 + results2:
            if res.doc_id not in doc_lookup:
                doc_lookup[res.doc_id] = res

        def rrf(rank: int) -> float:
            return 1.0 / (self.rrf_k + rank - 1)

        combined_scores = {}
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            if doc_id in rank_map1:
                rrf_score += rrf(rank_map1[doc_id])
            if doc_id in rank_map2:
                rrf_score += rrf(rank_map2[doc_id])

            score_blend = (
                self.weight1 * norm1.get(doc_id, 0.0) +
                self.weight2 * norm2.get(doc_id, 0.0)
            )
            max_signal = max(norm1.get(doc_id, 0.0), norm2.get(doc_id, 0.0))

            final_score = (
                self.rrf_weight * rrf_score +
                self.score_weight * score_blend +
                self.max_boost_weight * max_signal
            )

            if doc_id in rank_map1 and rank_map1[doc_id] <= self.bm25_boost_rank:
                final_score += self.bm25_boost_weight / rank_map1[doc_id]

            combined_scores[doc_id] = final_score

        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:top_k]:
            base = doc_lookup[doc_id]
            results.append(SearchResult(
                doc_id=doc_id,
                url=base.url,
                score=float(score),
                content_snippet=base.content_snippet,
                metadata={
                    'engine': 'hybrid-ensemble-boost',
                    'bm25_rank': rank_map1.get(doc_id),
                    'vector_rank': rank_map2.get(doc_id),
                    'bm25_score': score_map1.get(doc_id),
                    'vector_score': score_map2.get(doc_id),
                    'bm25_norm': norm1.get(doc_id),
                    'vector_norm': norm2.get(doc_id),
                    'rrf_k': self.rrf_k,
                    'rrf_weight': self.rrf_weight,
                    'score_weight': self.score_weight,
                    'weight1': self.weight1,
                    'weight2': self.weight2,
                    'max_boost_weight': self.max_boost_weight,
                    'bm25_boost_weight': self.bm25_boost_weight
                }
            ))

        return results

    def get_name(self) -> str:
        name1 = self.engine1.get_name()
        name2 = self.engine2.get_name()
        return f"hybrid-ensemble-boost-{name1}-{name2}"
