import sqlite3
import os
from pathlib import Path

# Path to your database (matches what's in crawl.py)
DB_PATH = "./barq.db"

def init_database():
    # 1. Ensure the 'db' folder exists
    db_folder = os.path.dirname(DB_PATH)
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)
        print(f"📁 Created folder: {db_folder}")

    # 2. Connect (this creates the file if missing)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("🔨 Creating table structure...")

    # 3. Create the 'documents' table
    # This matches the columns used in crawler/core/storage.py
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE NOT NULL,
        content_raw TEXT,
        content_text TEXT,
        created_at TEXT,
        updated_at TEXT,
        crawled_at TEXT,
        crawl_status TEXT,
        http_status INTEGER,
        error_message TEXT
    )
    """)
    
    # Optional: Create an index on URL for faster lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_url ON documents(url)")

    conn.commit()
    conn.close()

    print(f"✅ Success! Database initialized at: {os.path.abspath(DB_PATH)}")

if __name__ == "__main__":
    init_database()