#!/usr/bin/env python3
"""Investigate BM25 - What are we actually indexing?"""

import sys
import sqlite3
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from search.engines.bm25 import BM25SearchEngine


def investigate_indexing():
    """Deep dive into what BM25 is indexing"""

    print("="*80)
    print("INVESTIGATION: What is BM25 actually indexing?")
    print("="*80)

    # First, check the database directly
    db_path = Path(__file__).parent.parent / 'db' / 'barq.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("\n1. DATABASE CONTENT CHECK")
    print("-"*80)

    # Check what content_text contains
    cursor.execute("""
        SELECT id, url,
               length(content_text) as text_length,
               substr(content_text, 1, 300) as sample
        FROM documents
        WHERE crawl_status = 'success' AND content_text IS NOT NULL
        LIMIT 5
    """)

    print("\nFirst 5 documents in database:")
    for row in cursor.fetchall():
        print(f"\nDoc ID: {row['id']}")
        print(f"URL: {row['url']}")
        print(f"Content length: {row['text_length']} chars")
        print(f"Sample content:")
        print(f"  {row['sample'][:200]}...")
        print()

    conn.close()

    # Now let's see what BM25 engine actually indexes
    print("\n" + "="*80)
    print("2. BM25 ENGINE TOKENIZATION")
    print("-"*80)

    engine = BM25SearchEngine()
    engine.index(str(db_path))

    # Pick a specific document and see how it's tokenized
    print("\nDetailed look at a specific document:")
    print("-"*80)

    doc_id = 201  # Training document from our ground truth
    if doc_id in engine.documents:
        doc = engine.documents[doc_id]

        print(f"\nDocument ID: {doc_id}")
        print(f"URL: {doc['url']}")
        print(f"Total tokens: {doc['length']}")
        print(f"\nRaw content (first 500 chars):")
        print(doc['content'][:500])
        print("\n...")

        print(f"\nTokens (first 50):")
        print(doc['tokens'][:50])

        print(f"\nTop 20 most frequent terms in this document:")
        top_terms = doc['term_freq'].most_common(20)
        for term, count in top_terms:
            idf = engine.idf.get(term, 0.0)
            print(f"  {term:<20s} | count={count:3d} | IDF={idf:.2f}")

    # Check query terms
    print("\n" + "="*80)
    print("3. QUERY TERM ANALYSIS")
    print("-"*80)

    test_query = "تدريب دروب"
    query_terms = test_query.lower().split()

    print(f"\nQuery: '{test_query}'")
    print(f"Tokenized to: {query_terms}")

    for term in query_terms:
        idf = engine.idf.get(term, 0.0)

        # Count how many docs contain this term
        docs_with_term = 0
        for doc in engine.documents.values():
            if term in doc['term_freq']:
                docs_with_term += 1

        print(f"\nTerm: '{term}'")
        print(f"  IDF score: {idf:.3f}")
        print(f"  Found in: {docs_with_term}/{engine.doc_count} documents ({docs_with_term/engine.doc_count*100:.1f}%)")

        if docs_with_term > 0:
            # Show sample documents containing this term
            print(f"  Sample doc IDs with this term:", end=" ")
            sample_docs = []
            for doc_id, doc in engine.documents.items():
                if term in doc['term_freq']:
                    sample_docs.append(doc_id)
                    if len(sample_docs) >= 5:
                        break
            print(sample_docs)

    # Check ground truth documents specifically
    print("\n" + "="*80)
    print("4. GROUND TRUTH DOCUMENT ANALYSIS")
    print("-"*80)

    relevant_docs = [8, 9, 198, 201, 221]  # From ground truth for "تدريب دروب"

    print(f"\nGround truth relevant docs for 'تدريب دروب': {relevant_docs}")
    print("\nDo these docs contain the query terms?")

    for doc_id in relevant_docs:
        if doc_id in engine.documents:
            doc = engine.documents[doc_id]
            print(f"\nDoc {doc_id}: {doc['url']}")

            for term in query_terms:
                count = doc['term_freq'].get(term, 0)
                if count > 0:
                    print(f"  ✓ '{term}' appears {count} times")
                else:
                    print(f"  ✗ '{term}' NOT FOUND")

                    # Check if similar terms exist
                    print(f"     Checking for variations...")
                    variations = [term, term.upper(), term.capitalize()]
                    found_variation = False
                    for var in variations:
                        if var in doc['term_freq']:
                            print(f"     Found '{var}' instead: {doc['term_freq'][var]} times")
                            found_variation = True
                            break

                    if not found_variation:
                        # Show what terms are actually in this doc
                        print(f"     Top terms in this doc: {list(doc['term_freq'].most_common(10))}")
        else:
            print(f"\nDoc {doc_id}: NOT IN INDEX!")

    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    investigate_indexing()
