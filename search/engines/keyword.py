"""Simple keyword search engine (baseline)"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import sqlite3
from typing import List, Dict
from search.core.base import BaseSearchEngine, SearchResult


class KeywordSearchEngine(BaseSearchEngine):
    """
    Simple keyword search - baseline for comparison.

    Algorithm:
    1. Tokenize query into words
    2. For each document, count how many query words appear
    3. Rank by count (more matches = higher score)

    This is naive but fast and provides a baseline to beat.
    """

    def __init__(self):
        self.documents: Dict[int, dict] = {}
        self.indexed = False

    def index(self, db_path: str):
        """Load all documents from database into memory"""
        print(f"[KeywordSearch] Indexing documents from {db_path}...")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Load all successful documents
        cursor.execute("""
            SELECT id, url, content_text
            FROM documents
            WHERE crawl_status = 'success' AND content_text IS NOT NULL
        """)

        for row in cursor.fetchall():
            self.documents[row['id']] = {
                'id': row['id'],
                'url': row['url'],
                'content_text': row['content_text']
            }

        conn.close()
        self.indexed = True

        print(f"[KeywordSearch] Indexed {len(self.documents)} documents")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for documents containing query keywords.

        Simple algorithm:
        - Split query into words
        - Count how many times each word appears in each document
        - Rank by total count
        """
        if not self.indexed:
            raise RuntimeError("Must call index() before search()")

        # Tokenize query (simple: lowercase and split)
        query_words = query.lower().split()

        if not query_words:
            return []

        # Score each document
        scores = []

        for doc_id, doc in self.documents.items():
            content = doc['content_text'].lower()

            # Count keyword matches
            score = 0
            for word in query_words:
                score += content.count(word)

            # Only include if there's at least one match
            if score > 0:
                scores.append((doc_id, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Take top K
        top_results = scores[:top_k]

        # Build SearchResult objects
        results = []
        for doc_id, score in top_results:
            doc = self.documents[doc_id]

            # Create snippet (first 200 chars)
            snippet = doc['content_text'][:200].strip()
            if len(doc['content_text']) > 200:
                snippet += "..."

            results.append(SearchResult(
                doc_id=doc_id,
                url=doc['url'],
                score=float(score),
                content_snippet=snippet,
                metadata={'engine': 'keyword', 'keyword_count': score}
            ))

        return results

    def get_name(self) -> str:
        """Return engine name"""
        return "KeywordSearch"


if __name__ == "__main__":
    """Test the keyword search engine"""
    print("=" * 60)
    print("Testing KeywordSearchEngine")
    print("=" * 60)
    print()

    # Initialize engine
    engine = KeywordSearchEngine()

    # Index documents
    db_path = str(Path(__file__).parent.parent.parent / 'db' / 'barq.db')
    engine.index(db_path)

    print()

    # Test queries
    test_queries = [
        "رخصة قيادة",
        "تدريب",
        "صندوق",
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 60)

        results = engine.search(query, top_k=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.0f}")
                print(f"     Doc ID: {result.doc_id}")
                print(f"     URL: {result.url}")
                print(f"     Snippet: {result.content_snippet[:80]}...")
                print()
        else:
            print("  No results found")
            print()

    print("=" * 60)
    print("✓ KeywordSearchEngine test complete")
    print("=" * 60)
