"""
Basic vector search engine using sentence transformers for semantic search.

This implementation provides:
- Embedding-based semantic search
- Cosine similarity ranking
- Support for different embedding models via configuration
- GPU acceleration when available
- Disk-based caching for fast loading
"""

import sqlite3
import numpy as np
import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from search.core.base import BaseSearchEngine, SearchResult


@dataclass
class VectorModelConfig:
    """Configuration for a vector search model."""

    model_id: str                          # HuggingFace model ID
    model_name: str                        # Short name for cache files
    embedding_dimension: int               # Expected embedding size
    query_prompt: Optional[str] = None     # Prompt for queries (if model supports)
    doc_prompt: Optional[str] = None       # Prompt for documents (if model supports)
    batch_size: int = 32                   # Batch size for encoding
    device: str = 'cuda'                   # Device: 'cuda', 'cpu', or 'mps'
    trust_remote_code: bool = True         # Required for some models
    local_model_path: Optional[str] = None # Path to local quantized model (if any)

    def __post_init__(self):
        """Validate configuration."""
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


# Pre-defined model configurations
VECTOR_MODELS = {
    'gemma-300M': VectorModelConfig(
        model_id='google/embeddinggemma-300M',
        model_name='gemma-300M',
        embedding_dimension=768,
        query_prompt='Retrieval-query',
        doc_prompt='Retrieval-document',
        batch_size=32,
        device='cpu',  # Use CPU for local testing (change to 'cuda' for GPU)
        trust_remote_code=True
    ),
    'gemma-300M-int8': VectorModelConfig(
        model_id='google/embeddinggemma-300M',  # For metadata only
        model_name='gemma-300M',  # Use same cache as FP32
        embedding_dimension=768,
        query_prompt='Retrieval-query',
        doc_prompt='Retrieval-document',
        batch_size=32,
        device='cpu',  # INT8 quantization requires CPU
        trust_remote_code=True,
        local_model_path='models/gemma-300M-int8-pytorch/quantized_model.pt'
    ),
}


class VectorSearchEngineBase(BaseSearchEngine):
    """
    Basic vector search engine using embeddings and cosine similarity.

    This is the foundation for semantic search. Future variants will add:
    - Re-ranking stages
    - Hybrid search (BM25 + vector)
    - Different embedding models
    """

    def __init__(self, model_config: VectorModelConfig):
        """
        Initialize vector search engine.

        Args:
            model_config: Configuration specifying model and parameters
        """
        self.config = model_config
        self.model: Optional[SentenceTransformer] = None
        self.documents: Dict[int, str] = {}
        self.doc_ids: List[int] = []
        self.doc_embeddings: Optional[np.ndarray] = None

    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is not None:
            return  # Already loaded

        # Check if loading from local quantized model
        if self.config.local_model_path:
            import torch
            from pathlib import Path

            # Resolve path relative to project root
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / self.config.local_model_path

            print(f"Loading quantized model from: {model_path}")

            if not model_path.exists():
                print(f"✗ Error: Model file not found at {model_path}")
                print(f"  Falling back to HuggingFace model")
                self.config.local_model_path = None  # Clear to use HuggingFace
            else:
                # Set quantization engine (required for quantized models)
                torch.backends.quantized.engine = 'qnnpack' if self.config.device == 'cpu' else 'fbgemm'

                self.model = torch.load(model_path, map_location=self.config.device, weights_only=False)
                print(f"✓ Quantized model loaded (INT8, embedding dim: {self.config.embedding_dimension})")
                return  # Exit early

        # Load from HuggingFace (fallback or default)
        print(f"Loading model: {self.config.model_id}")
        self.model = SentenceTransformer(
            self.config.model_id,
            device=self.config.device,
            trust_remote_code=self.config.trust_remote_code
        )
        print(f"✓ Model loaded (embedding dim: {self.config.embedding_dimension})")

    def _load_documents(self, db_path: str) -> Dict[int, str]:
        """Load documents from SQLite database."""
        print(f"Loading documents from {db_path}...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, content_text
            FROM documents
            WHERE crawl_status = 'success' AND content_text IS NOT NULL
            ORDER BY id
        """)

        documents = {}
        for doc_id, content in cursor.fetchall():
            documents[doc_id] = content

        conn.close()
        print(f"✓ Loaded {len(documents)} documents")
        return documents

    def _encode_documents(self, documents: Dict[int, str]) -> tuple[List[int], np.ndarray]:
        """
        Encode all documents to embeddings.

        Args:
            documents: Dict mapping doc_id -> content_text

        Returns:
            Tuple of (doc_ids list, embeddings array)
        """
        doc_ids = list(documents.keys())
        doc_texts = [documents[doc_id] for doc_id in doc_ids]

        print(f"Encoding {len(doc_texts)} documents...")

        # Encode with document-specific prompt if available
        encode_kwargs = {
            'batch_size': self.config.batch_size,
            'show_progress_bar': True,
            'convert_to_numpy': True
        }

        if self.config.doc_prompt:
            encode_kwargs['prompt_name'] = self.config.doc_prompt

        embeddings = self.model.encode(doc_texts, **encode_kwargs)

        print(f"✓ Encoded documents (shape: {embeddings.shape})")
        return doc_ids, embeddings

    def _get_cache_dir(self, db_path: str) -> Path:
        """Get cache directory path (db/cache/)."""
        db_dir = Path(db_path).parent
        cache_dir = db_dir / 'cache'
        return cache_dir

    def _get_cache_paths(self, db_path: str) -> tuple[Path, Path]:
        """
        Get paths for cache files.

        Returns:
            Tuple of (embeddings_path, metadata_path)
        """
        cache_dir = self._get_cache_dir(db_path)
        cache_name = f"vector-{self.config.model_name}"

        embeddings_path = cache_dir / f"{cache_name}.npz"
        metadata_path = cache_dir / f"{cache_name}.meta.json"

        return embeddings_path, metadata_path

    def _load_from_cache(self, db_path: str) -> Optional[tuple[List[int], np.ndarray]]:
        """
        Load embeddings from cache if available and valid.

        Returns:
            Tuple of (doc_ids, embeddings) if cache valid, None otherwise
        """
        embeddings_path, metadata_path = self._get_cache_paths(db_path)

        # Check if cache files exist
        if not embeddings_path.exists() or not metadata_path.exists():
            return None

        print(f"Found cache: {embeddings_path.name}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Validate metadata
        if metadata.get('model_id') != self.config.model_id:
            print(f"⚠ Cache model mismatch: {metadata.get('model_id')} != {self.config.model_id}")
            return None

        # Load embeddings
        print(f"Loading embeddings from cache...")
        cache_data = np.load(embeddings_path)
        embeddings = cache_data['embeddings']
        doc_ids = cache_data['doc_ids'].tolist()

        print(f"✓ Loaded {len(doc_ids)} embeddings from cache (shape: {embeddings.shape})")
        print(f"  Cache created: {metadata.get('created_at')}")
        print(f"  Original encoding: {metadata.get('encoding_time_seconds', 0):.1f}s on {metadata.get('device', 'unknown')}")

        return doc_ids, embeddings

    def index(self, db_path: str):
        """
        Index all documents by loading embeddings from cache or encoding.

        Args:
            db_path: Path to SQLite database
        """
        # Try loading from cache first
        cache_result = self._load_from_cache(db_path)

        if cache_result is not None:
            # Cache hit - use cached embeddings
            self.doc_ids, self.doc_embeddings = cache_result

            # Still need to load documents for content display
            self.documents = self._load_documents(db_path)

            # Validate doc count matches
            if len(self.documents) != len(self.doc_ids):
                print(f"⚠ Warning: Cache has {len(self.doc_ids)} docs but DB has {len(self.documents)} docs")
                print("  Cache may be outdated. Consider regenerating.")

            print(f"✓ Indexing complete: {len(self.doc_ids)} documents ready for search (from cache)")

        else:
            # Cache miss - need to encode
            print("Cache not found. Encoding required.")
            print("⚠ Please generate cache first using Colab (see learn/generate_embedding_cache.ipynb)")

            # Load model
            self._load_model()

            # Load documents
            self.documents = self._load_documents(db_path)

            # Encode all documents
            self.doc_ids, self.doc_embeddings = self._encode_documents(self.documents)

            print(f"✓ Indexing complete: {len(self.doc_ids)} documents ready for search")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for documents using semantic similarity.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of SearchResult objects ranked by cosine similarity
        """
        if self.doc_embeddings is None:
            raise RuntimeError("Engine not indexed. Call index() first.")

        # Load model if not already loaded (happens when using cache)
        if self.model is None:
            self._load_model()

        # Encode query with query-specific prompt if available
        encode_kwargs = {'convert_to_numpy': True}
        if self.config.query_prompt:
            encode_kwargs['prompt_name'] = self.config.query_prompt

        query_embedding = self.model.encode([query], **encode_kwargs)

        # Calculate cosine similarity with all documents
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        # Get top-K results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build SearchResult objects
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = float(similarities[idx])
            content = self.documents[doc_id]

            # Create snippet (first 200 chars)
            snippet = content[:200] + "..." if len(content) > 200 else content

            results.append(SearchResult(
                doc_id=doc_id,
                url=f"doc_{doc_id}",  # URL loaded from DB if needed
                score=score,
                content_snippet=snippet,
                metadata={'model': self.config.model_id}
            ))

        return results

    def get_name(self) -> str:
        """Return engine name for identification."""
        model_short = self.config.model_name
        # Add suffix if using quantized model
        suffix = "-int8" if self.config.local_model_path else ""
        return f"vector-basic-{model_short}{suffix}"
