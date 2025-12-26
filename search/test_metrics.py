"""Test search evaluation metrics"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from search.evaluation.metrics import SearchMetrics, MetricResults


def test_precision_at_k():
    """Test Precision@K metric"""
    print("Testing Precision@K...")

    # Scenario 1: Perfect ranking (all correct at top)
    results = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    correct = {1, 2, 3, 4, 5}

    p5 = SearchMetrics.precision_at_k(results, correct, 5)
    assert p5 == 1.0, f"Expected 1.0, got {p5}"
    print(f"  ✓ Perfect ranking: P@5 = {p5:.2f} (5/5 correct)")

    # Scenario 2: Mixed ranking
    results = [1, 99, 2, 88, 3, 77, 66, 55, 44, 33]
    correct = {1, 2, 3}

    p5 = SearchMetrics.precision_at_k(results, correct, 5)
    assert p5 == 0.6, f"Expected 0.6, got {p5}"
    print(f"  ✓ Mixed ranking: P@5 = {p5:.2f} (3/5 correct)")

    p10 = SearchMetrics.precision_at_k(results, correct, 10)
    assert p10 == 0.3, f"Expected 0.3, got {p10}"
    print(f"  ✓ Mixed ranking: P@10 = {p10:.2f} (3/10 correct)")

    # Scenario 3: No correct results
    results = [99, 88, 77, 66, 55]
    correct = {1, 2, 3}

    p5 = SearchMetrics.precision_at_k(results, correct, 5)
    assert p5 == 0.0, f"Expected 0.0, got {p5}"
    print(f"  ✓ No matches: P@5 = {p5:.2f} (0/5 correct)")

    print()


def test_mrr():
    """Test Mean Reciprocal Rank metric"""
    print("Testing MRR...")

    # Scenario 1: First result is correct
    results = [1, 2, 3, 4, 5]
    correct = {1}

    mrr = SearchMetrics.mrr(results, correct)
    assert mrr == 1.0, f"Expected 1.0, got {mrr}"
    print(f"  ✓ First correct: MRR = {mrr:.2f} (position 1)")

    # Scenario 2: Second result is correct
    results = [99, 1, 2, 3, 4]
    correct = {1}

    mrr = SearchMetrics.mrr(results, correct)
    assert mrr == 0.5, f"Expected 0.5, got {mrr}"
    print(f"  ✓ Second correct: MRR = {mrr:.2f} (position 2)")

    # Scenario 3: Fifth result is correct
    results = [99, 88, 77, 66, 1]
    correct = {1}

    mrr = SearchMetrics.mrr(results, correct)
    assert mrr == 0.2, f"Expected 0.2, got {mrr}"
    print(f"  ✓ Fifth correct: MRR = {mrr:.2f} (position 5)")

    # Scenario 4: No correct results
    results = [99, 88, 77, 66, 55]
    correct = {1}

    mrr = SearchMetrics.mrr(results, correct)
    assert mrr == 0.0, f"Expected 0.0, got {mrr}"
    print(f"  ✓ No matches: MRR = {mrr:.2f}")

    print()


def test_ndcg():
    """Test NDCG metric"""
    print("Testing NDCG@10...")

    # Scenario 1: Perfect ranking
    results = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    correct = {1, 2, 3}

    ndcg = SearchMetrics.ndcg_at_k(results, correct, 10)
    assert ndcg == 1.0, f"Expected 1.0, got {ndcg}"
    print(f"  ✓ Perfect ranking: NDCG = {ndcg:.2f}")

    # Scenario 2: Good ranking (correct at top but not perfect order)
    results = [1, 3, 2, 99, 88, 77, 66, 55, 44, 33]
    correct = {1, 2, 3}

    ndcg = SearchMetrics.ndcg_at_k(results, correct, 10)
    assert ndcg > 0.9, f"Expected > 0.9, got {ndcg}"
    print(f"  ✓ Good ranking: NDCG = {ndcg:.2f}")

    # Scenario 3: Bad ranking (correct at bottom)
    results = [99, 88, 77, 66, 55, 44, 33, 1, 2, 3]
    correct = {1, 2, 3}

    ndcg = SearchMetrics.ndcg_at_k(results, correct, 10)
    assert ndcg < 0.5, f"Expected < 0.5, got {ndcg}"
    print(f"  ✓ Bad ranking: NDCG = {ndcg:.2f}")

    # Scenario 4: No correct results
    results = [99, 88, 77, 66, 55, 44, 33, 22, 11, 10]
    correct = {1, 2, 3}

    ndcg = SearchMetrics.ndcg_at_k(results, correct, 10)
    assert ndcg == 0.0, f"Expected 0.0, got {ndcg}"
    print(f"  ✓ No matches: NDCG = {ndcg:.2f}")

    print()


def test_evaluate_all():
    """Test evaluate_all() function"""
    print("Testing evaluate_all()...")

    results = [1, 99, 2, 88, 3, 77, 66, 55, 44, 33]
    correct = {1, 2, 3}
    response_time = 12.5

    metrics = SearchMetrics.evaluate_all(results, correct, response_time)

    # Verify all metrics are present
    assert hasattr(metrics, 'precision_at_5')
    assert hasattr(metrics, 'precision_at_10')
    assert hasattr(metrics, 'mrr')
    assert hasattr(metrics, 'ndcg_at_10')
    assert hasattr(metrics, 'response_time_ms')

    # Verify values are reasonable
    assert 0.0 <= metrics.precision_at_5 <= 1.0
    assert 0.0 <= metrics.precision_at_10 <= 1.0
    assert 0.0 <= metrics.mrr <= 1.0
    assert 0.0 <= metrics.ndcg_at_10 <= 1.0
    assert metrics.response_time_ms == 12.5

    print(f"  ✓ All metrics calculated: {metrics}")
    print()


def test_real_world_scenario():
    """Test with realistic search scenario"""
    print("Testing real-world scenario...")
    print("  Query: 'رخصة قيادة' (driver's license)")
    print("  Ground truth: docs 18332, 18341 are correct")
    print()

    # Simulate search results from different engines
    scenarios = [
        {
            "name": "Perfect Engine",
            "results": [18332, 18341, 99, 88, 77, 66, 55, 44, 33, 22],
            "time": 10.0
        },
        {
            "name": "Good Engine",
            "results": [18332, 99, 18341, 88, 77, 66, 55, 44, 33, 22],
            "time": 15.0
        },
        {
            "name": "Bad Engine",
            "results": [99, 88, 77, 66, 55, 18332, 44, 33, 22, 18341],
            "time": 5.0
        }
    ]

    correct = {18332, 18341}

    for scenario in scenarios:
        metrics = SearchMetrics.evaluate_all(
            scenario["results"],
            correct,
            scenario["time"]
        )
        print(f"  {scenario['name']:15s} → {metrics}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Step 3: Testing Metrics Module")
    print("=" * 60)
    print()

    test_precision_at_k()
    test_mrr()
    test_ndcg()
    test_evaluate_all()
    test_real_world_scenario()

    print("=" * 60)
    print("✓ All Step 3 tests passed!")
    print("=" * 60)
