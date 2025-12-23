-- Barq Search Engine Database Schema

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    content_raw TEXT,
    content_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    crawled_at TEXT,
    crawl_status TEXT NOT NULL,
    http_status INTEGER,
    error_message TEXT
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_url ON documents(url);
CREATE INDEX IF NOT EXISTS idx_crawl_status ON documents(crawl_status);
