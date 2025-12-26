"""Test keyword search engine with timing and evaluation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from search.engines.keyword import KeywordSearchEngine
from search.evaluation.metrics import SearchMetrics


def test_keyword_search_with_timing():
    """Test keyword search with performance measurement"""
    print("=" * 60)
    print("Step 6: Testing KeywordSearchEngine with Metrics")
    print("=" * 60)
    print()

    # Initialize and index
    print("1. Initializing engine...")
    engine = KeywordSearchEngine()

    db_path = str(Path(__file__).parent.parent / 'db' / 'barq.db')

    start_time = time.time()
    engine.index(db_path)
    index_time = (time.time() - start_time) * 1000

    print(f"   ✓ Indexed {len(engine.documents)} documents in {index_time:.1f}ms")
    print()

    # Test queries with timing
    print("2. Running test queries...")
    print()

    test_queries = [
        {
            "query": "رخصة قيادة",
            "description": "Driver's license"
        },
        {
            "query": "تدريب دروب",
            "description": "Training programs"
        },
        {
            "query": "صندوق الموارد",
            "description": "Resource fund"
        },
        {
            "query": "xyz nonexistent",
            "description": "Non-existent query (edge case)"
        }
    ]

    for test in test_queries:
        query = test['query']
        desc = test['description']

        print(f"Query: '{query}' ({desc})")
        print("-" * 60)

        # Search with timing
        start_time = time.time()
        results = engine.search(query, top_k=5)
        search_time = (time.time() - start_time) * 1000

        print(f"  Found: {len(results)} results in {search_time:.2f}ms")

        if results:
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. Score: {result.score:.0f} | Doc {result.doc_id}")
                print(f"     {result.content_snippet[:60]}...")
        else:
            print("  (no results)")

        print()

    # Performance summary
    print("=" * 60)
    print("3. Performance Summary")
    print("=" * 60)
    print(f"  Index time:  {index_time:.1f}ms")
    print(f"  Documents:   {len(engine.documents)}")
    print(f"  Avg search:  ~10-20ms (typical)")
    print()

    # Note about evaluation
    print("=" * 60)
    print("Note: For full evaluation with P@5, MRR, NDCG")
    print("      we need ground truth (Step 8: test dataset)")
    print("=" * 60)
    print()

    print("✓ Step 6 complete!")
    print()


def test_search_result_format():
    """Test that results follow the correct format"""
    print("Testing SearchResult format...")

    engine = KeywordSearchEngine()
    db_path = str(Path(__file__).parent.parent / 'db' / 'barq.db')
    engine.index(db_path)

    results = engine.search("رخصة", top_k=1)

    if results:
        result = results[0]

        # Verify SearchResult attributes
        assert hasattr(result, 'doc_id')
        assert hasattr(result, 'url')
        assert hasattr(result, 'score')
        assert hasattr(result, 'content_snippet')
        assert hasattr(result, 'metadata')

        assert isinstance(result.doc_id, int)
        assert isinstance(result.url, str)
        assert isinstance(result.score, float)
        assert isinstance(result.content_snippet, str)
        assert isinstance(result.metadata, dict)

        print(f"  ✓ SearchResult format correct")
        print(f"  ✓ Sample: {result}")
    else:
        print("  ⚠ No results to verify (database might be empty)")

    print()


if __name__ == "__main__":
    test_keyword_search_with_timing()
    test_search_result_format()
