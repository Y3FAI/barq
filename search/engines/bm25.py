"""BM25 Search Engine - Probabilistic ranking function"""

import sqlite3
import math
from typing import List, Dict, Set
from collections import Counter
from ..core.base import BaseSearchEngine, SearchResult
from ..utils.arabic_normalizer import ArabicNormalizer


class BM25SearchEngine(BaseSearchEngine):
    """
    BM25 (Best Match 25) search engine.

    Improves over keyword search with:
    1. Term frequency saturation (diminishing returns)
    2. Inverse document frequency (rare terms matter more)
    3. Document length normalization (fair comparison)

    Parameters:
        k1: Controls term frequency saturation (default: 1.5)
        b: Controls document length normalization (default: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 engine.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Length normalization parameter (0-1, where 0=none, 1=full)
        """
        self.k1 = k1
        self.b = b

        # Arabic text normalizer
        self.normalizer = ArabicNormalizer()

        # Document storage
        self.documents: Dict[int, dict] = {}

        # Statistics for BM25
        self.doc_count = 0  # Total number of documents
        self.avg_doc_length = 0.0  # Average document length
        self.idf: Dict[str, float] = {}  # IDF score for each term

        self.indexed = False

    def index(self, db_path: str):
        """
        Build BM25 index from database.

        Steps:
        1. Load all documents
        2. Calculate document statistics (count, avg length)
        3. Calculate IDF for all terms
        """
        print(f"[BM25] Indexing documents from {db_path}...")

        # Step 1: Load documents
        self._load_documents(db_path)

        # Step 2: Calculate document statistics
        self._calculate_doc_stats()

        # Step 3: Calculate IDF for all terms
        self._calculate_idf()

        print(f"[BM25] Indexed {self.doc_count} documents")
        print(f"[BM25] Average document length: {self.avg_doc_length:.0f} words")
        print(f"[BM25] Vocabulary size: {len(self.idf)} unique terms")

        self.indexed = True

    def _load_documents(self, db_path: str):
        """Load all documents from database"""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, url, content_text
            FROM documents
            WHERE crawl_status = 'success' AND content_text IS NOT NULL
        """)

        for row in cursor.fetchall():
            doc_id = row['id']
            content = row['content_text']

            # Tokenize with Arabic normalization
            tokens = self.normalizer.tokenize(content)

            # Calculate term frequencies for this document
            term_freq = Counter(tokens)

            self.documents[doc_id] = {
                'id': doc_id,
                'url': row['url'],
                'content': content,
                'tokens': tokens,
                'term_freq': term_freq,
                'length': len(tokens)
            }

        conn.close()

    def _calculate_doc_stats(self):
        """Calculate document count and average length"""
        self.doc_count = len(self.documents)

        total_length = sum(doc['length'] for doc in self.documents.values())
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0

    def _calculate_idf(self):
        """
        Calculate IDF (Inverse Document Frequency) for all terms.

        IDF Formula:
            IDF(term) = log((N - df + 0.5) / (df + 0.5))

        Where:
            N = total number of documents
            df = number of documents containing the term

        High IDF = rare term (appears in few docs) = more valuable
        Low IDF = common term (appears in many docs) = less valuable
        """
        # Count how many documents contain each term
        doc_freq: Dict[str, int] = Counter()

        for doc in self.documents.values():
            # Get unique terms in this document
            unique_terms = set(doc['term_freq'].keys())
            for term in unique_terms:
                doc_freq[term] += 1

        # Calculate IDF for each term
        N = self.doc_count

        for term, df in doc_freq.items():
            # BM25 IDF formula
            idf = math.log((N - df + 0.5) / (df + 0.5))
            self.idf[term] = max(0.0, idf)  # Ensure non-negative

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search using BM25 ranking.

        For each document, calculate:
            score = Σ IDF(term) × [TF × (k1+1)] / [TF + k1 × (1-b + b × doc_len/avg_len)]

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult sorted by BM25 score
        """
        if not self.indexed:
            raise RuntimeError("Index not built. Call index() first.")

        # Tokenize query with Arabic normalization
        query_terms = self.normalizer.tokenize(query)

        # Score each document
        scores = []

        for doc_id, doc in self.documents.items():
            score = self._calculate_bm25_score(query_terms, doc)

            if score > 0:
                scores.append((doc_id, score))

        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Convert top results to SearchResult objects
        results = []
        for doc_id, score in scores[:top_k]:
            doc = self.documents[doc_id]
            snippet = doc['content'][:200]  # First 200 chars

            results.append(SearchResult(
                doc_id=doc_id,
                url=doc['url'],
                score=score,
                content_snippet=snippet
            ))

        return results

    def _calculate_bm25_score(self, query_terms: List[str], doc: dict) -> float:
        """
        Calculate BM25 score for a document given query terms.

        BM25 Formula:
            score = Σ IDF(term) × [TF × (k1+1)] / [TF + k1 × (1-b + b × doc_len/avg_len)]

        Args:
            query_terms: List of query terms
            doc: Document dictionary

        Returns:
            BM25 score (sum over all query terms)
        """
        score = 0.0
        doc_length = doc['length']
        term_freq = doc['term_freq']

        for term in query_terms:
            # Get IDF for this term (0 if term not in vocabulary)
            idf = self.idf.get(term, 0.0)

            if idf == 0:
                continue  # Term not in corpus, skip

            # Get term frequency in this document
            tf = term_freq.get(term, 0)

            if tf == 0:
                continue  # Term not in this document, skip

            # Calculate length normalization
            length_norm = 1 - self.b + self.b * (doc_length / self.avg_doc_length)

            # Calculate BM25 component for this term
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)

            # Add to total score
            score += idf * tf_component

        return score

    def get_name(self) -> str:
        """Return engine name"""
        return "BM25"
