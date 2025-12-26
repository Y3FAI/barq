"""Search orchestrator - manages search execution and timing"""

import time
from typing import List, Dict, Any
from dataclasses import dataclass
from .base import BaseSearchEngine, SearchResult


@dataclass
class SearchTiming:
    """Timing breakdown for search execution"""
    engine_init_ms: float
    index_load_ms: float
    search_ms: float
    total_ms: float

    def __repr__(self):
        return (f"SearchTiming(total={self.total_ms:.1f}ms, "
                f"init={self.engine_init_ms:.1f}ms, "
                f"load={self.index_load_ms:.1f}ms, "
                f"search={self.search_ms:.1f}ms)")


@dataclass
class SearchResponse:
    """Complete search response with results and timing"""
    query: str
    results: List[SearchResult]
    timing: SearchTiming
    engine_name: str
    total_indexed: int

    def __repr__(self):
        return (f"SearchResponse(query='{self.query}', "
                f"found={len(self.results)}, "
                f"{self.timing})")


class SearchOrchestrator:
    """
    Orchestrates search execution across different engines.

    Responsibilities:
    - Initialize and manage search engines
    - Measure end-to-end timing
    - Provide consistent response format
    - Handle errors gracefully

    Future: Will support multiple engines (BM25, Vector, Hybrid)
    """

    def __init__(self, engine: BaseSearchEngine, db_path: str):
        """
        Initialize orchestrator with a search engine.

        Args:
            engine: Search engine instance (must implement BaseSearchEngine)
            db_path: Path to database
        """
        self.engine = engine
        self.db_path = db_path
        self.indexed = False
        self.total_docs = 0

    def initialize(self) -> float:
        """
        Initialize and load search index.

        Returns:
            Time taken to load index (milliseconds)
        """
        start = time.time()
        self.engine.index(self.db_path)
        load_time = (time.time() - start) * 1000

        # Get doc count if available
        if hasattr(self.engine, 'documents'):
            self.total_docs = len(self.engine.documents)

        self.indexed = True
        return load_time

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """
        Execute search and return results with timing.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            SearchResponse with results and timing breakdown
        """
        if not self.indexed:
            raise RuntimeError("Must call initialize() before search()")

        # Start total timing
        start_total = time.time()

        # Execute search
        start_search = time.time()
        results = self.engine.search(query, top_k=top_k)
        search_time = (time.time() - start_search) * 1000

        # Calculate total
        total_time = (time.time() - start_total) * 1000

        # Build timing breakdown
        timing = SearchTiming(
            engine_init_ms=0.0,  # Happens once, not per-query
            index_load_ms=0.0,   # Happens once, not per-query
            search_ms=search_time,
            total_ms=total_time
        )

        return SearchResponse(
            query=query,
            results=results,
            timing=timing,
            engine_name=self.engine.get_name(),
            total_indexed=self.total_docs
        )

    def search_with_full_timing(self, query: str, top_k: int = 10) -> SearchResponse:
        """
        Execute search from cold start (includes initialization).

        This measures the complete end-to-end experience including
        engine initialization and index loading.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            SearchResponse with full timing breakdown
        """
        # Start total timing
        start_total = time.time()

        # Step 1: Engine init (already done in __init__, so ~0ms)
        init_time = 0.0

        # Step 2: Load index
        load_time = self.initialize()

        # Step 3: Search
        start_search = time.time()
        results = self.engine.search(query, top_k=top_k)
        search_time = (time.time() - start_search) * 1000

        # Calculate total
        total_time = (time.time() - start_total) * 1000

        # Build timing breakdown
        timing = SearchTiming(
            engine_init_ms=init_time,
            index_load_ms=load_time,
            search_ms=search_time,
            total_ms=total_time
        )

        return SearchResponse(
            query=query,
            results=results,
            timing=timing,
            engine_name=self.engine.get_name(),
            total_indexed=self.total_docs
        )
