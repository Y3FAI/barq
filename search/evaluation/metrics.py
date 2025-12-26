"""Search evaluation metrics"""

import math
from typing import List, Set
from dataclasses import dataclass


@dataclass
class MetricResults:
    """Results from service discovery evaluation metrics"""
    success_at_1: float  # Found correct service as #1 result?
    success_at_3: float  # Found correct service in top 3?
    success_at_5: float  # Found correct service in top 5?
    success_at_10: float  # Found correct service in top 10?
    mrr: float  # How quickly did we find it?
    response_time_ms: float

    def __repr__(self):
        return (f"MetricResults(Hit@1={self.success_at_1:.0%}, "
                f"Hit@3={self.success_at_3:.0%}, "
                f"Hit@5={self.success_at_5:.0%}, "
                f"Hit@10={self.success_at_10:.0%}, "
                f"MRR={self.mrr:.3f}, "
                f"Time={self.response_time_ms:.1f}ms)")


class SearchMetrics:
    """Calculate service discovery metrics"""

    @staticmethod
    def success_at_k(results: List[int], correct_docs: Set[int], k: int) -> float:
        """
        Success at K: Did we find ANY correct service in top K results?

        This is binary: either we found it (1.0) or we didn't (0.0)
        Perfect for service discovery where finding THE service is what matters.

        Args:
            results: List of document IDs returned by search (ordered by rank)
            correct_docs: Set of document IDs that are correct for this query
            k: Number of top results to consider

        Returns:
            1.0 if found, 0.0 if not found
        """
        if k == 0 or len(results) == 0:
            return 0.0

        top_k = results[:k]
        # Binary: did we find it or not?
        found = any(doc_id in correct_docs for doc_id in top_k)
        return 1.0 if found else 0.0

    @staticmethod
    def mrr(results: List[int], correct_docs: Set[int]) -> float:
        """
        Mean Reciprocal Rank: 1 / (position of first correct result)

        Measures UX quality - finding service faster = better experience.

        Args:
            results: List of document IDs returned by search (ordered by rank)
            correct_docs: Set of document IDs that are correct for this query

        Returns:
            MRR score between 0.0 and 1.0
            - 1.0 = first result is correct (perfect!)
            - 0.5 = first correct result at position 2 (good)
            - 0.33 = first correct result at position 3 (okay)
            - 0.0 = no correct results found (failure)
        """
        for position, doc_id in enumerate(results, start=1):
            if doc_id in correct_docs:
                return 1.0 / position
        return 0.0

    @staticmethod
    def evaluate_all(results: List[int], correct_docs: Set[int],
                    response_time_ms: float) -> MetricResults:
        """
        Calculate all service discovery metrics.

        Args:
            results: List of document IDs returned by search (ordered by rank)
            correct_docs: Set of document IDs that are correct for this query
            response_time_ms: Time taken to execute search (milliseconds)

        Returns:
            MetricResults with all metrics computed
        """
        return MetricResults(
            success_at_1=SearchMetrics.success_at_k(results, correct_docs, 1),
            success_at_3=SearchMetrics.success_at_k(results, correct_docs, 3),
            success_at_5=SearchMetrics.success_at_k(results, correct_docs, 5),
            success_at_10=SearchMetrics.success_at_k(results, correct_docs, 10),
            mrr=SearchMetrics.mrr(results, correct_docs),
            response_time_ms=response_time_ms
        )
