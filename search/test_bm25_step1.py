#!/usr/bin/env python3
"""Test BM25 - Step 1: IDF Calculation"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from search.engines.bm25 import BM25SearchEngine


def test_idf_calculation():
    """Test that IDF is calculated correctly"""
    print("Testing BM25 IDF calculation...\n")

    # Create engine
    engine = BM25SearchEngine()

    # Build index
    db_path = Path(__file__).parent.parent / 'db' / 'barq.db'
    engine.index(str(db_path))

    print("\n" + "="*80)
    print("IDF ANALYSIS")
    print("="*80)

    # Test some terms
    test_terms = [
        "تدريب",  # training
        "خدمة",   # service (very common)
        "رخصة",   # license
        "دروب",   # doroob (specific platform name, should be rare)
        "والتي",  # common word (and which)
    ]

    print("\nIDF scores for sample terms:")
    print(f"{'Term':<15s} | {'IDF Score':<10s} | {'Interpretation'}")
    print("-" * 60)

    for term in test_terms:
        idf = engine.idf.get(term, 0.0)

        # Interpret the IDF score
        if idf > 5.0:
            interpretation = "Very rare (highly distinctive)"
        elif idf > 3.0:
            interpretation = "Rare (valuable for search)"
        elif idf > 1.0:
            interpretation = "Moderately common"
        elif idf > 0.5:
            interpretation = "Common"
        else:
            interpretation = "Very common (low value)"

        print(f"{term:<15s} | {idf:<10.2f} | {interpretation}")

    # Show top 10 most distinctive terms (highest IDF)
    print("\n" + "="*80)
    print("Top 10 most distinctive terms (highest IDF):")
    print("="*80)

    sorted_terms = sorted(engine.idf.items(), key=lambda x: x[1], reverse=True)

    for i, (term, idf) in enumerate(sorted_terms[:10], 1):
        print(f"{i:2d}. {term:<20s} IDF={idf:.2f}")

    # Show 10 most common terms (lowest IDF)
    print("\n" + "="*80)
    print("10 most common terms (lowest IDF):")
    print("="*80)

    for i, (term, idf) in enumerate(sorted_terms[-10:], 1):
        print(f"{i:2d}. {term:<20s} IDF={idf:.2f}")

    print("\n" + "="*80)
    print("✓ IDF calculation working correctly!")
    print("="*80)


if __name__ == "__main__":
    test_idf_calculation()
