"""Search engine factory"""

from .base import BaseSearchEngine
from ..engines.keyword import KeywordSearchEngine
from ..engines.bm25 import BM25SearchEngine
from ..engines.vector_base import VectorSearchEngineBase, VECTOR_MODELS
from ..engines.vector_faiss import VectorSearchEngineFAISS
from ..engines.hybrid import HybridSearchEngine
from ..engines.hybrid_rerank import HybridRerankEngine
from ..engines.hybrid_weighted import HybridWeightedEngine
from ..engines.hybrid_rerank_gemma import HybridRerankGemmaEngine, HybridRerankGemmaIndexedEngine
from ..engines.hybrid_jina_rerank import HybridJinaRerankEngine
from ..engines.hybrid_cosine_rerank import HybridCosineRerankEngine
from ..engines.hybrid_cosine_combined import HybridCosineCombinedEngine
from ..engines.hybrid_deepinfra_rerank import HybridDeepInfraRerankEngine
from ..engines.hybrid_rrf_fusion import HybridRRFFusionEngine
from ..engines.hybrid_ensemble_boost import HybridEnsembleBoostEngine
from ..engines.hybrid_cascade_rerank import HybridCascadeRerankEngine
from ..engines.hybrid_rrf_density import HybridRRFDensityEngine


class EngineFactory:
    """
    Factory for creating search engines.

    Makes it easy to add new engines without changing the orchestrator.
    """

    # Registry of available engines
    ENGINES = {
        'keyword': KeywordSearchEngine,
        'bm25': BM25SearchEngine,
        'vector-gemma': lambda: VectorSearchEngineBase(VECTOR_MODELS['gemma-300M']),
        'vector-gemma-faiss': lambda: VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M']),
        'vector-gemma-int8': lambda: VectorSearchEngineBase(VECTOR_MODELS['gemma-300M-int8']),
        'vector-gemma-int8-faiss': lambda: VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
        'hybrid': lambda: HybridSearchEngine(
            BM25SearchEngine(),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            k=60,
            candidate_pool_size=50
        ),
        'hybrid-weighted': lambda: HybridWeightedEngine(
            BM25SearchEngine(),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            weight1=0.5,
            weight2=0.5,
            candidate_pool_size=50
        ),
        'hybrid-weighted-bm25': lambda: HybridWeightedEngine(
            BM25SearchEngine(),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            weight1=0.7,
            weight2=0.3,
            candidate_pool_size=50
        ),
        'hybrid-weighted-vector': lambda: HybridWeightedEngine(
            BM25SearchEngine(),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            weight1=0.3,
            weight2=0.7,
            candidate_pool_size=50
        ),
        'hybrid-rerank-gemma': lambda: HybridRerankGemmaEngine(
            HybridWeightedEngine(
                BM25SearchEngine(),
                VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
                weight1=0.5,
                weight2=0.5,
                candidate_pool_size=50
            ),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            stage2_pool_size=10  # Rerank top-10 to improve Hit@1 position
        ),
        'hybrid-jina-rerank': lambda: HybridJinaRerankEngine(
            HybridWeightedEngine(
                BM25SearchEngine(),
                VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
                weight1=0.5,
                weight2=0.5,
                candidate_pool_size=50
            ),
            model_name='jinaai/jina-reranker-v2-base-multilingual',
            stage2_pool_size=30  # Rerank a larger pool for better recovery
        ),
        'hybrid-rerank-gemma-indexed': lambda: HybridRerankGemmaIndexedEngine(
            HybridWeightedEngine(
                BM25SearchEngine(),
                VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
                weight1=0.5,
                weight2=0.5,
                candidate_pool_size=50
            ),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            stage2_pool_size=10
        ),
        'hybrid-cosine-rerank': lambda: HybridCosineRerankEngine(
            HybridWeightedEngine(
                BM25SearchEngine(),
                VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
                weight1=0.5,
                weight2=0.5,
                candidate_pool_size=50
            ),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            stage2_pool_size=10  # Rerank top-10 with cosine similarity
        ),
        'hybrid-deepinfra-rerank': lambda: HybridDeepInfraRerankEngine(
            HybridWeightedEngine(
                BM25SearchEngine(),
                VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
                weight1=0.5,
                weight2=0.5,
                candidate_pool_size=50
            ),
            stage2_pool_size=10  # Use DeepInfra to rerank top-10 candidates
        ),
        'hybrid-rrf-fusion': lambda: HybridRRFFusionEngine(
            BM25SearchEngine(),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            candidate_pool_size=75,
            rrf_k=60,
            score_blend_weight=0.35,
            weight1=0.6,
            weight2=0.4
        ),
        'hybrid-ensemble-boost': lambda: HybridEnsembleBoostEngine(
            BM25SearchEngine(),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            candidate_pool_size=80,
            rrf_k=50,
            rrf_weight=0.5,
            score_weight=0.35,
            weight1=0.6,
            weight2=0.4,
            max_boost_weight=0.15,
            bm25_boost_weight=0.12,
            bm25_boost_rank=2
        ),
        'hybrid-rrf-density': lambda: HybridRRFDensityEngine(
            BM25SearchEngine(),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            candidate_pool_size=95,
            rrf_k=50,
            rrf_weight=0.6,
            score_weight=0.25,
            bm25_weight=0.6,
            vector_weight=0.4,
            bm25_boost_weight=0.12,
            density_boost_weight=0.1,
            density_window=6,
            max_position_boost=0.1,
            combine_exponent=1.15
        ),
        'hybrid-cascade-rerank': lambda: HybridCascadeRerankEngine(
            BM25SearchEngine(),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            candidate_pool_size=110,
            stage2_pool_size=45,
            rrf_k=55,
            rrf_weight=0.4,
            score_weight=0.25,
            bm25_weight=0.6,
            vector_weight=0.4,
            bm25_boost_rank=1,
            bm25_boost_weight=0.12,
            rerank_weight=0.55,
            doc_max_chars=900
        ),
        'hybrid-cosine-combined': lambda: HybridCosineCombinedEngine(
            HybridWeightedEngine(
                BM25SearchEngine(),
                VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
                weight1=0.5,
                weight2=0.5,
                candidate_pool_size=50
            ),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            stage2_pool_size=10,  # Rerank top-10 candidates
            stage1_weight=0.5,  # Combine: 50% Stage 1 hybrid + 50% Stage 2 cosine
            stage2_weight=0.5
        ),
        'hybrid-cosine-combined-0.7': lambda: HybridCosineCombinedEngine(
            HybridWeightedEngine(
                BM25SearchEngine(),
                VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
                weight1=0.5,
                weight2=0.5,
                candidate_pool_size=50
            ),
            VectorSearchEngineFAISS(VECTOR_MODELS['gemma-300M-int8']),
            stage2_pool_size=10,
            stage1_weight=0.7,  # Favor Stage 1 hybrid (better Hit@5)
            stage2_weight=0.3   # Light reranking with cosine
        ),
        # Future engines:
        # 'vector-minilm': lambda: VectorSearchEngineBase(VECTOR_MODELS['minilm']),
    }

    @staticmethod
    def create(engine_name: str) -> BaseSearchEngine:
        """
        Create a search engine by name.

        Args:
            engine_name: Name of engine ('keyword', 'bm25', 'vector', etc.)

        Returns:
            Search engine instance

        Raises:
            ValueError: If engine name is not recognized
        """
        if engine_name not in EngineFactory.ENGINES:
            available = ', '.join(EngineFactory.ENGINES.keys())
            raise ValueError(
                f"Unknown engine '{engine_name}'. "
                f"Available engines: {available}"
            )

        engine_class = EngineFactory.ENGINES[engine_name]
        return engine_class()

    @staticmethod
    def list_engines() -> list:
        """
        List all available engine names.

        Returns:
            List of engine names
        """
        return list(EngineFactory.ENGINES.keys())
