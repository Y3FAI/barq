#!/usr/bin/env python3
"""Find specific, high-value service queries"""

import sys
import sqlite3
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))


def find_specific_services():
    """Find specific, popular services for ground truth"""

    db_path = Path(__file__).parent.parent / 'db' / 'barq.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("="*80)
    print("FINDING SPECIFIC, POPULAR SERVICES")
    print("="*80)

    # Specific service queries that users would actually search
    specific_queries = [
        # ID/Documents
        ("إصدار رخصة قيادة", "Issue driver's license"),
        ("تجديد رخصة قيادة", "Renew driver's license"),
        ("إصدار جواز سفر", "Issue passport"),
        ("تجديد جواز سفر", "Renew passport"),
        ("إصدار هوية وطنية", "Issue national ID"),

        # Business
        ("إصدار سجل تجاري", "Issue commercial registry"),
        ("تجديد سجل تجاري", "Renew commercial registry"),
        ("إصدار رخصة بناء", "Issue building permit"),
        ("تجديد رخصة بناء", "Renew building permit"),

        # Certificates
        ("شهادة ميلاد", "Birth certificate"),
        ("شهادة وفاة", "Death certificate"),
        ("شهادة تخرج", "Graduation certificate"),

        # Employment
        ("التقديم على وظيفة", "Apply for job"),
        ("وظائف حكومية", "Government jobs"),

        # Health
        ("حجز موعد", "Book appointment"),
        ("وصفة طبية", "Medical prescription"),

        # Utilities
        ("فاتورة كهرباء", "Electricity bill"),
        ("فاتورة ماء", "Water bill"),

        # Education
        ("تسجيل مدرسة", "School registration"),
        ("نقل طالب", "Transfer student"),
    ]

    results = {}

    print("\nSearching for specific service documents...\n")

    for query, description in specific_queries:
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
            results[query] = {
                'description': description,
                'matches': matches
            }

    # Display results
    print("="*80)
    print("SPECIFIC SERVICE QUERIES (Best for Ground Truth)")
    print("="*80)

    good_queries = []

    for query, data in results.items():
        matches = data['matches']
        if len(matches) >= 2:  # At least 2 documents
            good_queries.append((query, data['description'], matches))

    # Sort by number of matches (most to least)
    good_queries.sort(key=lambda x: len(x[2]), reverse=True)

    for query, description, matches in good_queries[:15]:  # Top 15
        print(f"\n📝 Query: '{query}'")
        print(f"   Description: {description}")
        print(f"   Found {len(matches)} documents")

        # Sort by relevance (count)
        sorted_matches = sorted(matches, key=lambda x: x['count'], reverse=True)[:5]

        print(f"   Top matches:")
        for m in sorted_matches:
            print(f"     - Doc {m['id']}: {m['title'][:70]}...")

        doc_ids = [m['id'] for m in sorted_matches]
        print(f"   Suggested IDs: {doc_ids}")

    conn.close()

    print("\n" + "="*80)
    print(f"✓ Found {len(good_queries)} good specific queries")
    print("="*80)


if __name__ == "__main__":
    find_specific_services()
