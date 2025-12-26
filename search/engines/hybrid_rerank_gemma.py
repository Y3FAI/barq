"""
Two-stage hybrid with Gemma reranking.

Stage 1: Hybrid-weighted fusion (BM25 + Vector with cached embeddings)
Stage 2: Gemma reranking with "Reranking" prompt for refined scoring
"""

from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from search.core.base import BaseSearchEngine, SearchResult
from search.engines.hybrid_weighted import HybridWeightedEngine
from search.engines.vector_faiss import VectorSearchEngineFAISS


class HybridRerankGemmaEngine(BaseSearchEngine):
    """
    Two-stage hybrid with Gemma reranking.

    Stage 1: Hybrid-weighted 0.5/0.5 (retrieval mode with cache)
    Stage 2: Gemma rerank top-N with "Reranking" prompt
    """

    def __init__(self, stage1_engine: HybridWeightedEngine,
                 reranker: VectorSearchEngineFAISS,
                 stage2_pool_size: int = 50):
        """
        Initialize two-stage hybrid with reranking.

        Args:
            stage1_engine: Hybrid-weighted engine (Stage 1)
            reranker: Vector engine for reranking (Stage 2)
            stage2_pool_size: How many candidates to rerank
        """
        self.stage1 = stage1_engine
        self.reranker = reranker
        self.stage2_pool_size = stage2_pool_size

    def index(self, db_path: str):
        """Index both stages."""
        print("Indexing Stage 1 (Hybrid-weighted)...")
        self.stage1.index(db_path)
        print("Indexing Stage 2 (Reranker - loading documents)...")
        # Reranker needs documents loaded for fresh encoding
        if not self.reranker.documents:
            self.reranker.documents = self.reranker._load_documents(db_path)
        print("✓ Two-stage hybrid with reranking ready")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Two-stage search with Gemma reranking.

        Args:
            query: Search query
            top_k: Number of final results to return

        Returns:
            List of SearchResult objects reranked by Gemma
        """
        # Stage 1: Hybrid-weighted fusion (get more candidates for reranking)
        candidates = self.stage1.search(query, top_k=self.stage2_pool_size)

        if not candidates:
            return []

        stage1_ranks = {c.doc_id: idx + 1 for idx, c in enumerate(candidates)}

        stage1_ranks = {c.doc_id: idx + 1 for idx, c in enumerate(candidates)}

        # Stage 2: Rerank using cached embeddings (fast!)
        # Load model if needed
        if self.reranker.model is None:
            self.reranker._load_model()

        # Encode query with "Retrieval-query" prompt (matching cache)
        encode_kwargs = {'convert_to_numpy': True, 'show_progress_bar': False}
        if self.reranker.config.query_prompt:
            encode_kwargs['prompt_name'] = self.reranker.config.query_prompt

        query_embedding = self.reranker.model.encode([query], **encode_kwargs)

        # Get candidate doc IDs
        candidate_doc_ids = [c.doc_id for c in candidates]

        # Get cached embeddings for candidates (fast - no fresh encoding!)
        candidate_indices = []
        for doc_id in candidate_doc_ids:
            try:
                idx = self.reranker.doc_ids.index(doc_id)
                candidate_indices.append(idx)
            except ValueError:
                continue

        # Get embeddings for candidates from FAISS index
        # Reconstruct only the specific candidates we need
        import numpy as np

        if len(candidate_indices) == 0:
            return candidates[:top_k]  # No valid candidates

        doc_embeddings = np.array([
            self.reranker.faiss_index.reconstruct(int(idx))
            for idx in candidate_indices
        ])

        # Ensure proper shape for cosine_similarity
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if len(doc_embeddings.shape) == 1:
            doc_embeddings = doc_embeddings.reshape(1, -1)

        # Calculate reranking scores (cosine similarity)
        rerank_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Sort by reranking score
        reranked_indices = np.argsort(rerank_scores)[::-1][:top_k]

        # Build final results
        results = []
        for idx in reranked_indices:
            original = candidates[idx]
            rerank_score = float(rerank_scores[idx])

            results.append(SearchResult(
                doc_id=original.doc_id,
                url=original.url,
                score=rerank_score,
                content_snippet=original.content_snippet,
                metadata={
                    'engine': 'hybrid-rerank-gemma',
                    'stage1_rank': idx + 1,
                    'stage1_weighted_score': original.score,
                    'stage2_vector_score': rerank_score,
                    'rerank_method': 'cached_retrieval_embeddings'
                }
            ))

        return results

    def get_name(self) -> str:
        """Return engine name."""
        return "hybrid-rerank-gemma"


class HybridRerankGemmaIndexedEngine(HybridRerankGemmaEngine):
    """Variant that fully indexes the vector reranker to enable Stage 2 scoring."""

    def index(self, db_path: str):
        """Index both stages, ensuring the reranker builds its FAISS index."""
        super().index(db_path)

        print("Indexing Stage 2 (FAISS reranker index)...")
        self.reranker.index(db_path)

        print("✓ Gemma rerank engine fully indexed (Stage 1 + Stage 2)")

    def get_name(self) -> str:
        return "hybrid-rerank-gemma-indexed"

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Two-stage search with Gemma reranking using the dedicated Reranking prompt.

        Stage 1 pulls candidates via the hybrid engine. Stage 2 re-embeds the query
        and candidate documents with prompt_name='Reranking' (per Google guidance)
        and reranks them by cosine similarity.
        """
        candidates = self.stage1.search(query, top_k=self.stage2_pool_size)

        if not candidates:
            return []

        stage1_ranks = {c.doc_id: idx + 1 for idx, c in enumerate(candidates)}

        if self.reranker.model is None:
            self.reranker._load_model()

        rerank_prompt = "Reranking"

        # Encode query with reranking prompt
        query_embedding = self.reranker.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt_name=rerank_prompt
        )

        # Gather candidate texts
        candidate_texts = []
        valid_candidates = []
        for candidate in candidates:
            text = self.reranker.documents.get(candidate.doc_id, candidate.content_snippet)
            if not text:
                continue
            candidate_texts.append(text)
            valid_candidates.append(candidate)

        if not candidate_texts:
            return candidates[:top_k]

        # Encode candidate documents with reranking prompt
        doc_embeddings = self.reranker.model.encode(
            candidate_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=min(len(candidate_texts), self.reranker.config.batch_size),
            prompt_name=rerank_prompt
        )

        # Compute cosine similarity between rerank query embedding and doc embeddings
        rerank_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        # Sort by rerank score
        ranked_indices = np.argsort(rerank_scores)[::-1][:top_k]

        ranked_pairs = list(zip(ranked_indices, rerank_scores[ ranked_indices ]))
        filled_results = []
        used_doc_ids = set()

        for new_rank, (idx, rerank_score) in enumerate(ranked_pairs, 1):
            original = valid_candidates[idx]
            used_doc_ids.add(original.doc_id)

            filled_results.append(SearchResult(
                doc_id=original.doc_id,
                url=original.url,
                score=float(rerank_score),
                content_snippet=original.content_snippet,
                metadata={
                    'engine': 'hybrid-rerank-gemma-indexed',
                    'stage1_rank': stage1_ranks.get(original.doc_id, -1),
                    'stage1_weighted_score': original.score,
                    'stage2_rerank_score': float(rerank_score),
                    'stage2_rank': new_rank,
                    'rerank_prompt': rerank_prompt
                }
            ))

        # Backfill with Stage 1 ordering if we need more results.
        for candidate in candidates:
            if len(filled_results) >= min(top_k, len(candidates)):
                break
            if candidate.doc_id in used_doc_ids:
                continue
            used_doc_ids.add(candidate.doc_id)

            filled_results.append(SearchResult(
                doc_id=candidate.doc_id,
                url=candidate.url,
                score=candidate.score,
                content_snippet=candidate.content_snippet,
                metadata={
                    'engine': 'hybrid-rerank-gemma-indexed',
                    'stage1_rank': stage1_ranks.get(candidate.doc_id, -1),
                    'stage1_weighted_score': candidate.score,
                    'stage2_rerank_score': None,
                    'stage2_rank': None,
                    'rerank_prompt': rerank_prompt,
                    'fallback': True
                }
            ))

        return filled_results[:top_k]
