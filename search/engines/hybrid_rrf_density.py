"""Hybrid RRF + density-aware booster."""

from typing import List, Dict
import math
import numpy as np

from search.core.base import BaseSearchEngine, SearchResult
from search.engines.hybrid_weighted import HybridWeightedEngine


class HybridRRFDensityEngine(BaseSearchEngine):
    def __init__(self,
                 engine1: BaseSearchEngine,
                 engine2: BaseSearchEngine,
                 candidate_pool_size: int = 90,
                 rrf_k: int = 50,
                 rrf_weight: float = 0.55,
                 score_weight: float = 0.25,
                 bm25_weight: float = 0.6,
                 vector_weight: float = 0.4,
                 bm25_boost_weight: float = 0.12,
                 density_boost_weight: float = 0.1,
                 density_window: int = 5,
                 max_position_boost: float = 0.08,
                 combine_exponent: float = 1.2):
        self.engine1 = engine1
        self.engine2 = engine2
        self.candidate_pool_size = candidate_pool_size
        self.rrf_k = rrf_k
        self.rrf_weight = rrf_weight
        self.score_weight = score_weight
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25_boost_weight = bm25_boost_weight
        self.density_boost_weight = density_boost_weight
        self.density_window = density_window
        self.max_position_boost = max_position_boost
        self.combine_exponent = combine_exponent

    def index(self, db_path: str):
        print("Indexing engine 1 (RRFDensity)...")
        self.engine1.index(db_path)
        print("Indexing engine 2 (RRFDensity)...")
        self.engine2.index(db_path)
        print("✓ Hybrid RRFDensity engine ready")

    @staticmethod
    def _rank_map(results: List[SearchResult]) -> Dict[int, int]:
        return {res.doc_id: idx + 1 for idx, res in enumerate(results)}

    def _density_boost(self, rank_map: Dict[int, int], doc_id: int) -> float:
        rank = rank_map.get(doc_id)
        if not rank:
            return 0.0

        boost = math.exp(-(rank - 1) / self.density_window)
        return min(boost, 1.0)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        results1 = self.engine1.search(query, top_k=self.candidate_pool_size)
        results2 = self.engine2.search(query, top_k=self.candidate_pool_size)

        if not results1 and not results2:
            return []

        rank_map1 = self._rank_map(results1)
        rank_map2 = self._rank_map(results2)
        score_map1 = {r.doc_id: r.score for r in results1}
        score_map2 = {r.doc_id: r.score for r in results2}

        all_doc_ids = list(set(rank_map1.keys()) | set(rank_map2.keys()))

        arr1 = [score_map1.get(doc_id, 0.0) for doc_id in all_doc_ids]
        arr2 = [score_map2.get(doc_id, 0.0) for doc_id in all_doc_ids]
        norm1 = HybridWeightedEngine.normalize_scores(arr1)
        norm2 = HybridWeightedEngine.normalize_scores(arr2)
        norm_map1 = {doc_id: score for doc_id, score in zip(all_doc_ids, norm1)}
        norm_map2 = {doc_id: score for doc_id, score in zip(all_doc_ids, norm2)}

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
                self.bm25_weight * norm_map1.get(doc_id, 0.0) +
                self.vector_weight * norm_map2.get(doc_id, 0.0)
            )

            density = max(
                self._density_boost(rank_map1, doc_id),
                self._density_boost(rank_map2, doc_id)
            )

            position_boost = 0.0
            if doc_id in rank_map1 and rank_map1[doc_id] <= 3:
                position_boost = self.max_position_boost / rank_map1[doc_id]

            final_score = (
                (self.rrf_weight * rrf_score) ** self.combine_exponent +
                self.score_weight * score_blend +
                self.density_boost_weight * density +
                position_boost
            )

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
                    'engine': 'hybrid-rrf-density',
                    'bm25_rank': rank_map1.get(doc_id),
                    'vector_rank': rank_map2.get(doc_id),
                    'bm25_norm': norm_map1.get(doc_id),
                    'vector_norm': norm_map2.get(doc_id),
                    'density_window': self.density_window,
                    'combine_exponent': self.combine_exponent
                }
            ))

        return results

    def get_name(self) -> str:
        name1 = self.engine1.get_name()
        name2 = self.engine2.get_name()
        return f"hybrid-rrf-density-{name1}-{name2}"
