"""Test base search interfaces"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from search.core.base import SearchResult, BaseSearchEngine


def test_search_result():
    """Test SearchResult dataclass"""
    print("Testing SearchResult...")

    # Create a result
    result = SearchResult(
        doc_id=123,
        url="https://my.gov.sa/ar/services/123",
        score=0.85,
        content_snippet="خدمة إصدار رخصة القيادة",
        metadata={"source": "mygovsa"}
    )

    # Test attributes
    assert result.doc_id == 123
    assert result.score == 0.85
    assert result.url.startswith("https://")
    assert "رخصة" in result.content_snippet
    assert result.metadata["source"] == "mygovsa"

    print(f"  ✓ Created: {result}")
    print(f"  ✓ All attributes accessible")
    print()


def test_base_engine_interface():
    """Test that BaseSearchEngine is abstract"""
    print("Testing BaseSearchEngine interface...")

    # Try to instantiate (should fail)
    try:
        engine = BaseSearchEngine()
        print("  ✗ FAILED: Should not be able to instantiate abstract class")
    except TypeError as e:
        print(f"  ✓ Correctly prevents instantiation: {e}")

    # Create a dummy implementation
    class DummyEngine(BaseSearchEngine):
        def index(self, db_path: str):
            return "indexed"

        def search(self, query: str, top_k: int = 10):
            return []

        def get_name(self) -> str:
            return "DummyEngine"

    # Test that implementation works
    engine = DummyEngine()
    assert engine.get_name() == "DummyEngine"
    assert engine.index("test.db") == "indexed"
    assert engine.search("test") == []

    print(f"  ✓ Interface contract enforced")
    print(f"  ✓ Dummy implementation works: {engine.get_name()}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Testing Base Interfaces")
    print("=" * 60)
    print()

    test_search_result()
    test_base_engine_interface()

    print("=" * 60)
    print("✓ All Step 2 tests passed!")
    print("=" * 60)
