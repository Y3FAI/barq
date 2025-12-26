"""Benchmark runner for evaluating search engines"""

import time
from typing import List, Dict
from dataclasses import dataclass, field
from .ground_truth import GroundTruth
from .metrics import SearchMetrics, MetricResults
from ..core.base import BaseSearchEngine
from ..core.orchestrator import SearchOrchestrator


@dataclass
class QueryResult:
    """Results for a single query"""
    query: str
    description: str
    metrics: MetricResults
    found_results: int


@dataclass
class BenchmarkResults:
    """Complete benchmark results for an engine"""
    engine_name: str
    query_results: List[QueryResult] = field(default_factory=list)
    total_time_ms: float = 0.0

    @property
    def avg_hit1(self) -> float:
        """Average Hit@1 (success rate for #1 position)"""
        if not self.query_results:
            return 0.0
        return sum(qr.metrics.success_at_1 for qr in self.query_results) / len(self.query_results)

    @property
    def avg_hit3(self) -> float:
        """Average Hit@3 (success rate in top 3)"""
        if not self.query_results:
            return 0.0
        return sum(qr.metrics.success_at_3 for qr in self.query_results) / len(self.query_results)

    @property
    def avg_hit5(self) -> float:
        """Average Hit@5 (success rate in top 5)"""
        if not self.query_results:
            return 0.0
        return sum(qr.metrics.success_at_5 for qr in self.query_results) / len(self.query_results)

    @property
    def avg_hit10(self) -> float:
        """Average Hit@10 (success rate in top 10)"""
        if not self.query_results:
            return 0.0
        return sum(qr.metrics.success_at_10 for qr in self.query_results) / len(self.query_results)

    @property
    def avg_mrr(self) -> float:
        """Average Mean Reciprocal Rank (quality of position)"""
        if not self.query_results:
            return 0.0
        return sum(qr.metrics.mrr for qr in self.query_results) / len(self.query_results)

    @property
    def avg_time(self) -> float:
        """Average response time (ms)"""
        if not self.query_results:
            return 0.0
        return sum(qr.metrics.response_time_ms for qr in self.query_results) / len(self.query_results)


class Benchmark:
    """
    Benchmark search engines against ground truth.

    Tests all queries in ground truth dataset and computes
    average metrics for comparison.
    """

    def __init__(self, ground_truth: GroundTruth, db_path: str):
        """
        Initialize benchmark.

        Args:
            ground_truth: Ground truth dataset
            db_path: Path to database
        """
        self.ground_truth = ground_truth
        self.db_path = db_path

    def run_single_engine(self, engine: BaseSearchEngine, top_k: int = 10,
                         verbose: bool = False) -> BenchmarkResults:
        """
        Benchmark a single search engine.

        Args:
            engine: Search engine to test
            top_k: Number of results to retrieve per query
            verbose: Print progress

        Returns:
            BenchmarkResults with all metrics
        """
        if verbose:
            print(f"\nBenchmarking {engine.get_name()}...")
            print(f"Testing {self.ground_truth.count()} queries")
            print("-" * 60)

        # Start total timing
        start_total = time.time()

        # Create orchestrator
        orchestrator = SearchOrchestrator(engine, self.db_path)
        orchestrator.initialize()

        # WARMUP: Run first query to load model (excluded from results)
        queries_list = list(self.ground_truth.list_queries())
        if queries_list and verbose:
            warmup_query = queries_list[0]
            print(f"[Warmup] {warmup_query[:30]}...", end=" ")
            warmup_response = orchestrator.search(warmup_query, top_k=top_k)
            print(f"(Excluded from avg, took {warmup_response.timing.search_ms:.1f}ms)")

        results = BenchmarkResults(engine_name=engine.get_name())

        # Test each query
        for i, query in enumerate(queries_list, 1):
            gt = self.ground_truth.get(query)

            if verbose:
                print(f"[{i}/{self.ground_truth.count()}] {query[:30]}...", end=" ")

            # Search
            response = orchestrator.search(query, top_k=top_k)

            # Evaluate
            result_ids = [r.doc_id for r in response.results]
            metrics = SearchMetrics.evaluate_all(
                results=result_ids,
                correct_docs=gt.relevant_docs,
                response_time_ms=response.timing.search_ms
            )

            if verbose:
                hit_icon = "✓" if metrics.success_at_5 > 0 else "✗"
                print(f"{hit_icon} MRR={metrics.mrr:.2f}")

            results.query_results.append(QueryResult(
                query=query,
                description=gt.description,
                metrics=metrics,
                found_results=len(response.results)
            ))

        if verbose:
            print("-" * 60)
            print(f"✓ Complete: Avg Hit@5={results.avg_hit5:.1%}, MRR={results.avg_mrr:.3f}\n")

        return results

    def run_multiple_engines(self, engines: List[BaseSearchEngine],
                           top_k: int = 10, verbose: bool = False) -> Dict[str, BenchmarkResults]:
        """
        Benchmark multiple engines for comparison.

        Args:
            engines: List of search engines to test
            top_k: Number of results per query
            verbose: Print progress

        Returns:
            Dictionary mapping engine name to BenchmarkResults
        """
        results = {}

        for engine in engines:
            results[engine.get_name()] = self.run_single_engine(
                engine, top_k=top_k, verbose=verbose
            )

        return results
