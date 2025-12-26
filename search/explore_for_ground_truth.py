#!/usr/bin/env python3
"""Explore current database to create new ground truth"""

import sys
import sqlite3
from pathlib import Path
from collections import Counter
import re

sys.path.insert(0, str(Path(__file__).parent.parent))


def explore_database():
    """Explore database to find good queries for ground truth"""

    db_path = Path(__file__).parent.parent / 'db' / 'barq.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("="*80)
    print("EXPLORING DATABASE FOR GROUND TRUTH QUERIES")
    print("="*80)

    # Get total count
    cursor.execute("SELECT COUNT(*) as count FROM documents WHERE crawl_status = 'success'")
    total = cursor.fetchone()['count']
    print(f"\nTotal documents: {total}")

    # Get some sample titles to understand content
    print("\n" + "="*80)
    print("SAMPLE DOCUMENT TITLES (to understand content)")
    print("="*80)

    cursor.execute("""
        SELECT id, url, content_text
        FROM documents
        WHERE crawl_status = 'success' AND content_text IS NOT NULL
        ORDER BY id
        LIMIT 100
    """)

    # Extract titles (assuming they're after #)
    titles = []
    for row in cursor.fetchall():
        content = row['content_text']
        # Extract title after #
        match = re.search(r'#\s+(.+?)(?:\n|$)', content)
        if match:
            title = match.group(1).strip()
            titles.append({
                'id': row['id'],
                'title': title,
                'url': row['url']
            })

    # Show diverse sample
    print("\nSample titles from different document ranges:")
    for i in [0, 20, 40, 60, 80]:
        if i < len(titles):
            t = titles[i]
            print(f"\nDoc {t['id']}: {t['title']}")
            print(f"  {t['url']}")

    # Find common topics by looking at frequent words in titles
    print("\n" + "="*80)
    print("ANALYZING COMMON TOPICS")
    print("="*80)

    cursor.execute("""
        SELECT content_text
        FROM documents
        WHERE crawl_status = 'success' AND content_text IS NOT NULL
    """)

    all_titles = []
    for row in cursor.fetchall():
        match = re.search(r'#\s+(.+?)(?:\n|$)', row['content_text'])
        if match:
            all_titles.append(match.group(1).strip())

    # Count words in titles
    word_freq = Counter()
    for title in all_titles:
        words = title.split()
        word_freq.update(words)

    print("\nMost common words in titles:")
    for word, count in word_freq.most_common(30):
        if len(word) > 2:  # Skip very short words
            print(f"  {word:<20s} : {count:4d} times")

    # Now let's search for specific popular queries and find their docs
    print("\n" + "="*80)
    print("SEARCHING FOR POPULAR QUERY PATTERNS")
    print("="*80)

    popular_queries = [
        # Government services
        "رخصة قيادة",      # Driver's license
        "جواز سفر",       # Passport
        "هوية وطنية",     # National ID
        "رخصة بناء",      # Building permit

        # Employment
        "تدريب",          # Training
        "توظيف",          # Employment
        "وظائف",          # Jobs
        "رواتب",          # Salaries

        # Business
        "سجل تجاري",      # Commercial registry
        "رخصة تجارية",    # Commercial license
        "منشأة",          # Establishment

        # Finance
        "ضريبة",          # Tax
        "زكاة",           # Zakat
        "قرض",            # Loan

        # Health
        "صحة",            # Health
        "علاج",           # Treatment
        "مستشفى",         # Hospital

        # Education
        "تعليم",          # Education
        "جامعة",          # University
        "مدرسة",          # School

        # Real estate
        "عقار",           # Real estate
        "سكن",            # Housing
    ]

    print("\nSearching for documents matching popular queries...")

    results_by_query = {}

    for query in popular_queries:
        query_lower = query.lower()

        cursor.execute("""
            SELECT id, content_text
            FROM documents
            WHERE crawl_status = 'success'
            AND content_text IS NOT NULL
            AND LOWER(content_text) LIKE ?
            LIMIT 10
        """, (f'%{query_lower}%',))

        matches = []
        for row in cursor.fetchall():
            # Extract title
            match = re.search(r'#\s+(.+?)(?:\n|$)', row['content_text'])
            title = match.group(1).strip() if match else "No title"

            # Count occurrences
            count = row['content_text'].lower().count(query_lower)

            matches.append({
                'id': row['id'],
                'title': title,
                'count': count
            })

        if matches:
            results_by_query[query] = matches

    # Display results
    print("\n" + "="*80)
    print("SUGGESTED GROUND TRUTH QUERIES")
    print("="*80)

    for query, matches in results_by_query.items():
        if len(matches) >= 3:  # Only include queries with enough docs
            print(f"\n📝 Query: '{query}'")
            print(f"   Found {len(matches)} documents")
            print(f"   Top 5 matches:")

            # Sort by count (most relevant first)
            sorted_matches = sorted(matches, key=lambda x: x['count'], reverse=True)[:5]

            for m in sorted_matches:
                print(f"     - Doc {m['id']}: {m['title'][:60]}... (appears {m['count']}x)")

            print(f"   Suggested doc IDs: {[m['id'] for m in sorted_matches[:5]]}")

    conn.close()

    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    explore_database()
