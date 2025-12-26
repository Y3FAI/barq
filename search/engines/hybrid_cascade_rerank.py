"""
Cascade hybrid engine:

Stage 1: Run BM25 and Vector engines independently, fuse with RRF + score blending
         to build a large candidate pool with strong lexical recall.

Stage 2: Re-embed the top candidates with EmbeddingGemma's "Reranking" prompt and
         combine the cross-encoder signal with the Stage 1 fusion score.

Goal: keep BM25-driven Hit@1 high while letting semantic reranking reorder the top
pool for Hit@3/5 improvements.
"""

from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from search.core.base import BaseSearchEngine, SearchResult
from search.engines.hybrid_weighted import HybridWeightedEngine


class HybridCascadeRerankEngine(BaseSearchEngine):
    def __init__(self,
                 bm25_engine: BaseSearchEngine,
                 vector_engine: BaseSearchEngine,
                 candidate_pool_size: int = 100,
                 stage2_pool_size: int = 40,
                 rrf_k: int = 60,
                 rrf_weight: float = 0.45,
                 score_weight: float = 0.25,
                 bm25_weight: float = 0.6,
                 vector_weight: float = 0.4,
                 bm25_boost_rank: int = 1,
                 bm25_boost_weight: float = 0.12,
                 rerank_weight: float = 0.55,
                 doc_max_chars: int = 800,
                 rerank_batch_size: int = 16):
        if candidate_pool_size <= 0 or stage2_pool_size <= 0:
            raise ValueError("candidate_pool_size and stage2_pool_size must be positive")
        if stage2_pool_size > candidate_pool_size:
            stage2_pool_size = candidate_pool_size
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if not (0 <= bm25_weight <= 1 and 0 <= vector_weight <= 1):
            raise ValueError("bm25/vector weights must be within [0,1]")

        self.bm25_engine = bm25_engine
        self.vector_engine = vector_engine
        self.candidate_pool_size = candidate_pool_size
        self.stage2_pool_size = stage2_pool_size
        self.rrf_k = rrf_k
        self.rrf_weight = rrf_weight
        self.score_weight = score_weight
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25_boost_rank = bm25_boost_rank
        self.bm25_boost_weight = bm25_boost_weight
        self.rerank_weight = rerank_weight
        self.doc_max_chars = doc_max_chars
        self.rerank_batch_size = rerank_batch_size
        self.documents: Dict[int, str] = {}

    def index(self, db_path: str):
        print("Indexing BM25 (cascade stage 1)...")
        self.bm25_engine.index(db_path)
        print("Indexing Vector (cascade stage 1)...")
        self.vector_engine.index(db_path)
        print("Loading documents for reranking...")
        self.documents = self._load_documents(db_path)
        print(f"✓ Cascade hybrid ready with {len(self.documents)} docs")

    def _load_documents(self, db_path: str) -> Dict[int, str]:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, content_text
            FROM documents
            WHERE crawl_status = 'success'
            """
        )
        docs = {doc_id: content or "" for doc_id, content in cursor.fetchall()}
        conn.close()
        return docs

    @staticmethod
    def _rank_map(results: List[SearchResult]) -> Dict[int, int]:
        return {res.doc_id: idx + 1 for idx, res in enumerate(results)}

    def _rrf_fusion(self, results1: List[SearchResult], results2: List[SearchResult]) -> Dict[int, float]:
        rank_map1 = self._rank_map(results1)
        rank_map2 = self._rank_map(results2)

        score_map1 = {r.doc_id: r.score for r in results1}
        score_map2 = {r.doc_id: r.score for r in results2}

        all_doc_ids = list(set(rank_map1.keys()) | set(rank_map2.keys()))
        if not all_doc_ids:
            return {}

        scores1_norm = {}
        scores2_norm = {}
        arr1 = [score_map1.get(doc_id, 0.0) for doc_id in all_doc_ids]
        arr2 = [score_map2.get(doc_id, 0.0) for doc_id in all_doc_ids]
        norm1 = HybridWeightedEngine.normalize_scores(arr1)
        norm2 = HybridWeightedEngine.normalize_scores(arr2)
        scores1_norm = {doc_id: score for doc_id, score in zip(all_doc_ids, norm1)}
        scores2_norm = {doc_id: score for doc_id, score in zip(all_doc_ids, norm2)}

        def rrf(rank: int) -> float:
            return 1.0 / (self.rrf_k + rank - 1)

        fused_scores = {}
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            if doc_id in rank_map1:
                rrf_score += rrf(rank_map1[doc_id])
            if doc_id in rank_map2:
                rrf_score += rrf(rank_map2[doc_id])

            score_blend = (
                self.bm25_weight * scores1_norm.get(doc_id, 0.0) +
                self.vector_weight * scores2_norm.get(doc_id, 0.0)
            )

            final = self.rrf_weight * rrf_score + self.score_weight * score_blend

            if doc_id in rank_map1 and rank_map1[doc_id] <= self.bm25_boost_rank:
                final += self.bm25_boost_weight / rank_map1[doc_id]

            fused_scores[doc_id] = final

        return fused_scores

    def _prepare_text(self, doc_id: int, fallback: str) -> str:
        text = self.documents.get(doc_id) or fallback or ""
        text = text.strip()
        if len(text) > self.doc_max_chars:
            text = text[:self.doc_max_chars]
        return text

    def _rerank_with_gemma(self, query: str, doc_ids: List[int], doc_texts: List[str]) -> np.ndarray:
        if not doc_ids:
            return np.array([])

        if getattr(self.vector_engine, 'model', None) is None:
            self.vector_engine._load_model()  # type: ignore[attr-defined]

        query_emb = self.vector_engine.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt_name="Reranking"
        )

        doc_embeddings = self.vector_engine.model.encode(
            doc_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=min(self.rerank_batch_size, len(doc_texts)),
            prompt_name="Reranking"
        )

        rerank_scores = cosine_similarity(query_emb, doc_embeddings)[0]
        return rerank_scores

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        bm25_results = self.bm25_engine.search(query, top_k=self.candidate_pool_size)
        vector_results = self.vector_engine.search(query, top_k=self.candidate_pool_size)

        if not bm25_results and not vector_results:
            return []

        fused_scores = self._rrf_fusion(bm25_results, vector_results)
        if not fused_scores:
            return bm25_results[:top_k]

        doc_lookup: Dict[int, SearchResult] = {}
        for res in bm25_results + vector_results:
            if res.doc_id not in doc_lookup:
                doc_lookup[res.doc_id] = res

        sorted_candidates = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )

        stage2_candidates = sorted_candidates[:self.stage2_pool_size]
        stage2_doc_ids = [doc_id for doc_id, _ in stage2_candidates]
        text_info = {}
        stage2_texts = []
        for doc_id in stage2_doc_ids:
            full_text = self.documents.get(doc_id) or doc_lookup[doc_id].content_snippet
            full_text = full_text or ""
            prepared = full_text.strip()
            truncated = False
            if len(prepared) > self.doc_max_chars:
                prepared = prepared[:self.doc_max_chars]
                truncated = True
            text_info[doc_id] = (prepared, truncated)
            stage2_texts.append(prepared)

        rerank_scores = self._rerank_with_gemma(query, stage2_doc_ids, stage2_texts)

        fused_arr = np.array([score for _, score in stage2_candidates], dtype=np.float32)
        fused_norm = HybridWeightedEngine.normalize_scores(fused_arr.tolist())
        rerank_norm = HybridWeightedEngine.normalize_scores(rerank_scores.tolist()) if rerank_scores.size else np.zeros_like(fused_arr)

        final_scores = (
            (1 - self.rerank_weight) * fused_norm +
            self.rerank_weight * rerank_norm
        )

        rerank_map = dict(zip(stage2_doc_ids, final_scores))
        combined = {doc_id: rerank_map.get(doc_id, fused_scores[doc_id]) for doc_id in fused_scores}

        final_sorted = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, final_score in final_sorted[:top_k]:
            base = doc_lookup[doc_id]
            prepared_text, truncated_flag = text_info.get(doc_id, (self.documents.get(doc_id, ''), False))

            results.append(SearchResult(
                doc_id=doc_id,
                url=base.url,
                score=float(final_score),
                content_snippet=base.content_snippet,
                metadata={
                    'engine': 'hybrid-cascade-rerank',
                    'bm25_rank': self._rank_map(bm25_results).get(doc_id),
                    'vector_rank': self._rank_map(vector_results).get(doc_id),
                    'fused_score': fused_scores.get(doc_id),
                    'rerank_applied': doc_id in rerank_map,
                    'doc_truncated': truncated_flag,
                    'rerank_input_preview': prepared_text[:120]
                }
            ))

        return results

    def get_name(self) -> str:
        return "hybrid-cascade-rerank"
