"""
Hybrid search engine using Reciprocal Rank Fusion (RRF).

Combines BM25 keyword search with vector semantic search for better results.
Uses RRF to merge rankings without needing score normalization.
"""

from typing import List, Dict
from search.core.base import BaseSearchEngine, SearchResult


class HybridSearchEngine(BaseSearchEngine):
    """
    Hybrid search using Reciprocal Rank Fusion (RRF).

    Combines two search engines (typically BM25 + Vector) by:
    1. Running both searches
    2. Calculating RRF score: 1/(k+rank1) + 1/(k+rank2)
    3. Re-ranking by combined score

    Documents appearing in both result lists get higher scores.
    """

    def __init__(self, engine1: BaseSearchEngine, engine2: BaseSearchEngine,
                 k: int = 60, candidate_pool_size: int = 50):
        """
        Initialize hybrid search engine.

        Args:
            engine1: First search engine (e.g., BM25)
            engine2: Second search engine (e.g., Vector)
            k: RRF constant (default 60, standard value from research)
            candidate_pool_size: How many results to get from each engine
        """
        self.engine1 = engine1
        self.engine2 = engine2
        self.k = k
        self.candidate_pool_size = candidate_pool_size

    def index(self, db_path: str):
        """Index both engines."""
        print("Indexing engine 1...")
        self.engine1.index(db_path)
        print("Indexing engine 2...")
        self.engine2.index(db_path)
        print("✓ Hybrid engine ready")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search using RRF to combine both engines.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects ranked by RRF score
        """
        # Step 1: Get results from both engines
        results1 = self.engine1.search(query, top_k=self.candidate_pool_size)
        results2 = self.engine2.search(query, top_k=self.candidate_pool_size)

        # Step 2: Build ranking maps (doc_id -> rank position)
        rank_map1 = {r.doc_id: rank + 1 for rank, r in enumerate(results1)}
        rank_map2 = {r.doc_id: rank + 1 for rank, r in enumerate(results2)}

        # Step 3: Store original results for lookup
        result_lookup: Dict[int, SearchResult] = {}
        for r in results1:
            result_lookup[r.doc_id] = r
        for r in results2:
            # If doc appears in both, keep engine1's result (BM25 snippet might be better)
            if r.doc_id not in result_lookup:
                result_lookup[r.doc_id] = r

        # Step 4: Calculate RRF scores for all documents
        all_doc_ids = set(rank_map1.keys()) | set(rank_map2.keys())
        rrf_scores = {}

        for doc_id in all_doc_ids:
            # Get rank from each engine (infinity if not present)
            rank1 = rank_map1.get(doc_id, float('inf'))
            rank2 = rank_map2.get(doc_id, float('inf'))

            # Calculate RRF score
            score1 = 1.0 / (self.k + rank1) if rank1 != float('inf') else 0
            score2 = 1.0 / (self.k + rank2) if rank2 != float('inf') else 0

            rrf_scores[doc_id] = score1 + score2

        # Step 5: Sort by RRF score (highest first)
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 6: Build final SearchResult objects
        final_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            original = result_lookup[doc_id]

            # Create new SearchResult with RRF score but original content
            final_results.append(SearchResult(
                doc_id=doc_id,
                url=original.url,
                score=rrf_score,
                content_snippet=original.content_snippet,
                metadata={
                    'engine': 'hybrid-rrf',
                    'in_engine1': doc_id in rank_map1,
                    'in_engine2': doc_id in rank_map2,
                    'rank1': rank_map1.get(doc_id),
                    'rank2': rank_map2.get(doc_id),
                    'k': self.k
                }
            ))

        return final_results

    def get_name(self) -> str:
        """Return engine name."""
        name1 = self.engine1.get_name()
        name2 = self.engine2.get_name()
        return f"hybrid-rrf-{name1}-{name2}"
