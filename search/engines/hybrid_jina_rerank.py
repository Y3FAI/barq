"""
Two-stage hybrid with Jina Reranker v3.

Stage 1: Hybrid-weighted fusion (BM25 + Vector) → broad candidate pool
Stage 2: Jina Reranker v3 (cross-encoder) → rerank with reranking-friendly snippets

Jina Reranker is a cross-encoder that scores query+document pairs directly,
providing more accurate relevance scores than bi-encoder comparisons when fed concise passages.
"""

from typing import List, Tuple
import numpy as np

from search.core.base import BaseSearchEngine, SearchResult
from search.engines.hybrid_weighted import HybridWeightedEngine


class HybridJinaRerankEngine(BaseSearchEngine):
    """
    Two-stage hybrid with Jina Reranker v3.

    Stage 1: Hybrid-weighted 0.5/0.5 (broad recall, larger candidate pool)
    Stage 2: Jina cross-encoder reranking with prompt-aligned snippets
    """

    def __init__(self, stage1_engine: HybridWeightedEngine,
                 model_name: str = 'jinaai/jina-reranker-v2-base-multilingual',
                 stage2_pool_size: int = 30,
                 max_rerank_chars: int = 800,
                 stage1_blend_weight: float = 0.3):
        """
        Initialize hybrid with Jina reranker.

        Args:
            stage1_engine: Hybrid-weighted engine for Stage 1
            model_name: Jina reranker model name
            stage2_pool_size: How many candidates to consider before reranking
            max_rerank_chars: Truncate documents to this length for reranker input
            stage1_blend_weight: Weight assigned to Stage 1 normalized scores when blending
        """
        if not (0.0 <= stage1_blend_weight <= 1.0):
            raise ValueError("stage1_blend_weight must be between 0 and 1")

        self.stage1 = stage1_engine
        self.model_name = model_name
        self.stage2_pool_size = stage2_pool_size
        self.max_rerank_chars = max_rerank_chars
        self.stage1_blend_weight = stage1_blend_weight
        self.reranker = None
        self.documents = {}  # Store full document texts by doc_id

    def index(self, db_path: str):
        """Index Stage 1 engine and load documents."""
        print("Indexing Stage 1 (Hybrid-weighted)...")
        self.stage1.index(db_path)

        print("Loading documents for Stage 2 (Jina Reranker)...")
        self._load_documents(db_path)

        print("✓ Two-stage hybrid with Jina reranking ready")

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

    def _load_reranker(self):
        """Lazy load Jina reranker model."""
        if self.reranker is not None:
            return

        print(f"Loading Jina Reranker: {self.model_name}...")
        from sentence_transformers import CrossEncoder

        self.reranker = CrossEncoder(
            self.model_name,
            automodel_args={"torch_dtype": "auto"},
            trust_remote_code=True,
            device='cpu'
        )
        print(f"✓ Jina Reranker loaded")

    def _prepare_document_text(self, doc_id: int, fallback_snippet: str) -> Tuple[str, bool]:
        """
        Prepare a concise text chunk for reranking.

        Returns:
            (text, was_truncated)
        """
        text = self.documents.get(doc_id) or fallback_snippet or ""
        text = text.strip()

        if not text:
            return "", False

        truncated = False
        if len(text) > self.max_rerank_chars:
            text = text[:self.max_rerank_chars]
            truncated = True

        return text, truncated

    @staticmethod
    def _normalize_array(values: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range using min-max normalization."""
        if values.size == 0:
            return values

        min_val = values.min()
        max_val = values.max()

        if abs(max_val - min_val) < 1e-9:
            return np.ones_like(values)

        return (values - min_val) / (max_val - min_val)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Two-stage search with Jina reranking.

        Args:
            query: Search query
            top_k: Number of final results to return

        Returns:
            List of SearchResult objects reranked by Jina
        """
        # Stage 1: Hybrid-weighted fusion with larger candidate pool
        candidate_pool_size = max(self.stage2_pool_size, top_k)
        candidates = self.stage1.search(query, top_k=candidate_pool_size)

        if not candidates:
            return []

        # If we have fewer candidates than requested, return them
        if len(candidates) <= 1:
            return candidates

        stage1_ranks = {c.doc_id: idx + 1 for idx, c in enumerate(candidates)}

        # Stage 2: Jina cross-encoder reranking
        self._load_reranker()

        doc_texts = []
        candidate_indices = []
        doc_truncated_flags = [False] * len(candidates)

        for idx, candidate in enumerate(candidates):
            text, truncated = self._prepare_document_text(candidate.doc_id, candidate.content_snippet)
            if not text:
                continue
            doc_texts.append(text)
            candidate_indices.append(idx)
            doc_truncated_flags[idx] = truncated

        # If we have no usable documents for reranking, fall back to Stage 1
        if not doc_texts:
            return candidates[:top_k]

        print(f"Reranking {len(doc_texts)} candidates with Jina v2...")
        rankings = self.reranker.rank(
            query,
            doc_texts,
            return_documents=False,
            convert_to_tensor=False,
            top_k=len(doc_texts)
        )
        print(f"✓ Reranking complete")

        stage1_scores = np.array([c.score for c in candidates], dtype=np.float32)
        stage1_norm = self._normalize_array(stage1_scores.copy())

        jina_scores = np.full(len(candidates), np.nan, dtype=np.float32)

        for ranking in rankings:
            corpus_id = ranking['corpus_id']
            candidate_idx = candidate_indices[corpus_id]
            jina_scores[candidate_idx] = float(ranking['score'])

        jina_norm = np.zeros(len(candidates), dtype=np.float32)
        valid_mask = ~np.isnan(jina_scores)
        if valid_mask.any():
            jina_norm[valid_mask] = self._normalize_array(jina_scores[valid_mask])

        final_scores = stage1_norm.copy()
        if valid_mask.any():
            final_scores[valid_mask] = (
                self.stage1_blend_weight * stage1_norm[valid_mask] +
                (1 - self.stage1_blend_weight) * jina_norm[valid_mask]
            )

        sorted_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for new_rank, idx in enumerate(sorted_indices, 1):
            candidate = candidates[idx]
            jina_score = None if np.isnan(jina_scores[idx]) else float(jina_scores[idx])
            jina_norm_score = None if np.isnan(jina_scores[idx]) else float(jina_norm[idx])

            results.append(SearchResult(
                doc_id=candidate.doc_id,
                url=candidate.url,
                score=float(final_scores[idx]),
                content_snippet=candidate.content_snippet,
                metadata={
                    'engine': 'hybrid-jina-rerank',
                    'stage1_rank': stage1_ranks.get(candidate.doc_id, -1),
                    'stage1_weighted_score': candidate.score,
                    'stage1_norm_score': float(stage1_norm[idx]),
                    'stage2_jina_score': jina_score,
                    'stage2_jina_norm_score': jina_norm_score,
                    'stage2_rank': new_rank,
                    'reranker_model': self.model_name,
                    'blend_stage1_weight': self.stage1_blend_weight,
                    'blend_stage2_weight': 1 - self.stage1_blend_weight,
                    'doc_truncated': doc_truncated_flags[idx]
                }
            ))

        return results

    def get_name(self) -> str:
        """Return engine name."""
        return "hybrid-jina-rerank"
