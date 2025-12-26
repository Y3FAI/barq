"""
Two-stage hybrid with DeepInfra Qwen3-Reranker-8B.

Stage 1: Hybrid-weighted fusion (BM25 + Vector) → top-N candidates
Stage 2: DeepInfra Qwen3 Reranker API → precise reranking

Uses cloud-based reranking API to avoid local CPU bottlenecks.
"""

import os
from typing import List
import requests

from search.core.base import BaseSearchEngine, SearchResult
from search.engines.hybrid_weighted import HybridWeightedEngine


class HybridDeepInfraRerankEngine(BaseSearchEngine):
    """
    Two-stage hybrid with DeepInfra Qwen3 Reranker.

    Stage 1: Hybrid-weighted 0.5/0.5 (broad recall)
    Stage 2: DeepInfra API reranking (precise scoring)
    """

    def __init__(self, stage1_engine: HybridWeightedEngine,
                 api_key: str = None,
                 stage2_pool_size: int = 10,
                 api_url: str = "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-8B"):
        """
        Initialize hybrid with DeepInfra reranker.

        Args:
            stage1_engine: Hybrid-weighted engine for Stage 1
            api_key: DeepInfra API key (or set DEEPINFRA_API_KEY env var)
            stage2_pool_size: How many candidates to rerank (default 10)
            api_url: DeepInfra API endpoint
        """
        self.stage1 = stage1_engine
        self.api_key ="QBZUuGBj2yTk642iAS52eWj6f2CTCjJo"
        self.stage2_pool_size = stage2_pool_size
        self.api_url = api_url
        self.documents = {}  # Store full document texts by doc_id

    def index(self, db_path: str):
        """Index Stage 1 engine and load documents."""
        if not self.api_key:
            raise ValueError(
                "DeepInfra API key required. Set DEEPINFRA_API_KEY env var "
                "or pass api_key parameter."
            )

        print("Indexing Stage 1 (Hybrid-weighted)...")
        self.stage1.index(db_path)

        print("Loading documents for Stage 2 (DeepInfra Reranker)...")
        self._load_documents(db_path)

        print("✓ Two-stage hybrid with DeepInfra reranking ready")

    def _load_documents(self, db_path: str):
        """Load full document texts from database."""
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, content_text
            FROM documents
            WHERE crawl_status = 'success'
            ORDER BY id
        """)

        for doc_id, content in cursor.fetchall():
            self.documents[doc_id] = content

        conn.close()
        print(f"✓ Loaded {len(self.documents)} full documents")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Two-stage search with DeepInfra reranking.

        Args:
            query: Search query
            top_k: Number of final results to return

        Returns:
            List of SearchResult objects reranked by DeepInfra
        """
        # Stage 1: Hybrid-weighted fusion
        candidates = self.stage1.search(query, top_k=self.stage2_pool_size)

        if not candidates:
            return []

        # If we have very few candidates, return them as-is
        if len(candidates) <= 1:
            return candidates

        # Stage 2: DeepInfra API reranking
        # Prepare documents for reranking (use full text)
        doc_texts = []
        for c in candidates:
            # Use full document text if available, fallback to snippet
            full_text = self.documents.get(c.doc_id, c.content_snippet)
            # Truncate if too long (API may have limits)
            if len(full_text) > 8000:
                full_text = full_text[:8000]
            doc_texts.append(full_text)

        # Call DeepInfra API
        print(f"Reranking {len(doc_texts)} candidates with DeepInfra Qwen3...")
        try:
            response = requests.post(
                self.api_url,
                headers={
                    'Authorization': f'bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'queries': [query],
                    'documents': doc_texts
                },
                timeout=30  # 30 second timeout
            )
            response.raise_for_status()
            result = response.json()
            scores = result.get('scores', [])

            if not scores:
                print("⚠ No scores returned from API, returning Stage 1 results")
                return candidates[:top_k]

            print(f"✓ Reranking complete (cost: ${result.get('inference_status', {}).get('cost', 0):.6f})")

        except requests.exceptions.RequestException as e:
            print(f"⚠ DeepInfra API error: {e}, returning Stage 1 results")
            return candidates[:top_k]

        # Sort by reranking scores (descending)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Build final results
        results = []
        for new_rank, idx in enumerate(ranked_indices, 1):
            original = candidates[idx]
            rerank_score = float(scores[idx])

            results.append(SearchResult(
                doc_id=original.doc_id,
                url=original.url,
                score=rerank_score,
                content_snippet=original.content_snippet,
                metadata={
                    'engine': 'hybrid-deepinfra-rerank',
                    'stage1_rank': idx + 1,
                    'stage1_weighted_score': original.score,
                    'stage2_rerank_score': rerank_score,
                    'stage2_rank': new_rank,
                    'reranker_model': 'Qwen/Qwen3-Reranker-8B'
                }
            ))

        return results

    def get_name(self) -> str:
        """Return engine name."""
        return "hybrid-deepinfra-rerank"
