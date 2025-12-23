import sqlite3
import os

# CONFIGURATION
DB_PATH = "./db/barq.db"
OUTPUT_DIR = "./raw_data"

def export():
    # 1. Connect to the database
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 2. Select only successful crawls with content
    print("🔄 extracting documents...")
    cursor.execute("SELECT id, url, content_text FROM documents WHERE crawl_status='success' AND content_text IS NOT NULL")
    rows = cursor.fetchall()

    if not rows:
        print("⚠️ No documents found. Did the crawler save them?")
        return

    # 3. Create output folder
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 4. Save each document as a .txt file
    count = 0
    for row in rows:
        doc_id, url, content = row
        
        # We add the URL at the top so the AI knows the source context
        file_content = f"Source URL: {url}\n\n{content}"
        
        filename = f"doc_{doc_id}.txt"
        with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(file_content)
        
        count += 1

    print(f"✅ Exported {count} documents to '{OUTPUT_DIR}' folder.")
    conn.close()

if __name__ == "__main__":
    export()