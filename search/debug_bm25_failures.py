#!/usr/bin/env python3
"""Debug why BM25 fails on certain queries"""

import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from search.engines.bm25 import BM25SearchEngine
from search.engines.keyword import KeywordSearchEngine


def debug_query(query, ground_truth_docs):
    """Debug why BM25 fails on a specific query"""

    print("="*80)
    print(f"DEBUGGING QUERY: '{query}'")
    print("="*80)

    db_path = Path(__file__).parent.parent / 'db' / 'barq.db'

    # Initialize both engines
    bm25 = BM25SearchEngine()
    keyword = KeywordSearchEngine()

    bm25.index(str(db_path))
    keyword.index(str(db_path))

    # Tokenize query
    query_terms = query.lower().split()

    print(f"\nQuery terms: {query_terms}")
    print(f"Ground truth docs: {ground_truth_docs}")

    # Check IDF scores
    print("\n" + "="*80)
    print("QUERY TERM ANALYSIS")
    print("="*80)

    for term in query_terms:
        idf = bm25.idf.get(term, 0.0)

        # Count docs containing term
        docs_with_term = sum(1 for doc in bm25.documents.values() if term in doc['term_freq'])

        print(f"\nTerm: '{term}'")
        print(f"  IDF: {idf:.3f}")
        print(f"  Found in: {docs_with_term}/{bm25.doc_count} docs ({docs_with_term/bm25.doc_count*100:.1f}%)")

        if idf < 1.0:
            print(f"  ⚠️  LOW IDF - Very common term, not distinctive!")

    # Analyze ground truth documents
    print("\n" + "="*80)
    print("GROUND TRUTH DOCUMENT ANALYSIS")
    print("="*80)

    for doc_id in ground_truth_docs:
        if doc_id not in bm25.documents:
            print(f"\n✗ Doc {doc_id}: NOT IN INDEX")
            continue

        doc = bm25.documents[doc_id]

        print(f"\nDoc {doc_id}: {doc['url']}")
        print(f"Length: {doc['length']} words (avg: {bm25.avg_doc_length:.0f})")

        # Check if query terms exist
        print(f"\nQuery term matching:")
        for term in query_terms:
            if term in doc['term_freq']:
                count = doc['term_freq'][term]
                print(f"  ✓ '{term}': appears {count} times")
            else:
                print(f"  ✗ '{term}': NOT FOUND")

                # Check for variations
                print(f"     Looking for variations...")

                # Check if term appears as part of other words
                found_in = []
                for doc_term in doc['term_freq'].keys():
                    if term in doc_term or doc_term in term:
                        found_in.append((doc_term, doc['term_freq'][doc_term]))

                if found_in:
                    print(f"     Found in compound words:")
                    for compound, count in found_in[:5]:
                        print(f"       - '{compound}': {count} times")
                else:
                    print(f"     ❌ No variations found")

        # Calculate BM25 score for this doc
        bm25_score = bm25._calculate_bm25_score(query_terms, doc)

        # Calculate keyword score
        keyword_score = sum(doc['content'].lower().count(term) for term in query_terms)

        print(f"\nScores for this doc:")
        print(f"  BM25:    {bm25_score:.2f}")
        print(f"  Keyword: {keyword_score}")

    # Show what BM25 actually returns
    print("\n" + "="*80)
    print("WHAT BM25 RETURNS (Top 5)")
    print("="*80)

    bm25_results = bm25.search(query, top_k=5)

    for i, result in enumerate(bm25_results, 1):
        doc = bm25.documents[result.doc_id]
        is_correct = "✓" if result.doc_id in ground_truth_docs else "✗"

        print(f"\n[{i}] Doc {result.doc_id} {is_correct}")
        print(f"    Score: {result.score:.2f}")
        print(f"    Length: {doc['length']} words")

        # Show term frequencies
        print(f"    Term frequencies:")
        for term in query_terms:
            tf = doc['term_freq'].get(term, 0)
            print(f"      '{term}': {tf}")

    # Show what Keyword returns
    print("\n" + "="*80)
    print("WHAT KEYWORD RETURNS (Top 5)")
    print("="*80)

    keyword_results = keyword.search(query, top_k=5)

    for i, result in enumerate(keyword_results, 1):
        doc = keyword.documents[result.doc_id]
        is_correct = "✓" if result.doc_id in ground_truth_docs else "✗"

        print(f"\n[{i}] Doc {result.doc_id} {is_correct}")
        print(f"    Score: {result.score:.0f}")
        print(f"    Raw matches of query terms in content")

    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    # Diagnose the issue
    all_terms_common = all(bm25.idf.get(term, 0) < 2.0 for term in query_terms)

    if all_terms_common:
        print("\n⚠️  ISSUE: All query terms are very common (low IDF)")
        print("    BM25 can't distinguish relevant docs because terms appear everywhere")
        print("    Keyword accidentally does better by just counting occurrences")

    # Check if ground truth docs have the exact terms
    missing_terms = False
    for doc_id in ground_truth_docs:
        if doc_id in bm25.documents:
            doc = bm25.documents[doc_id]
            for term in query_terms:
                if term not in doc['term_freq']:
                    missing_terms = True
                    print(f"\n⚠️  ISSUE: Ground truth doc {doc_id} missing term '{term}'")
                    print("    Likely due to word forms (e.g., 'ميلاد' vs 'الميلاد')")


if __name__ == "__main__":
    # Debug the failing queries

    print("\n")
    debug_query("شهادة ميلاد", [1017, 2583])

    print("\n\n")
    input("Press Enter to see next query...")

    debug_query("إصدار سجل تجاري", [39, 71])
