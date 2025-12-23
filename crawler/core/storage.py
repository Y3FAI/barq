"""Storage layer for crawler - SQLite operations"""

import sqlite3
from datetime import datetime, UTC
from pathlib import Path


class Storage:
    def __init__(self, db_path: str):
        """Initialize storage with database path"""
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def insert_document(self, url: str, content_raw: str = None,
                       content_text: str = None, crawl_status: str = 'pending',
                       http_status: int = None, error_message: str = None):
        """Insert a new document"""
        now = datetime.now(UTC).isoformat()

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO documents
            (url, content_raw, content_text, created_at, updated_at,
             crawled_at, crawl_status, http_status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (url, content_raw, content_text, now, now, now,
              crawl_status, http_status, error_message))

        self.conn.commit()
        return cursor.lastrowid

    def get_document_by_url(self, url: str):
        """Get document by URL"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE url = ?", (url,))
        return cursor.fetchone()

    def get_document_by_id(self, doc_id: int):
        """Get document by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return cursor.fetchone()


if __name__ == "__main__":
    # Quick test
    db_path = Path(__file__).parent.parent.parent / "db" / "barq.db"

    storage = Storage(str(db_path))
    storage.connect()

    # Insert test document
    doc_id = storage.insert_document(
        url="https://example.com/test",
        content_text="Test content",
        crawl_status="success",
        http_status=200
    )

    print(f"Inserted document with ID: {doc_id}")

    # Read it back
    doc = storage.get_document_by_id(doc_id)
    print(f"Retrieved document: {dict(doc)}")

    storage.close()
    print("✓ Storage test passed")
