"""Brain - The crawler orchestrator"""

from typing import Optional
from datetime import datetime, UTC, timedelta
import time
from .base_bot import BaseBot
from .storage import Storage
from .fetcher import Fetcher
from .preflight import Preflight


class Brain:
    """
    Orchestrates the crawling process.

    Coordinates: Bot → Fetcher → Storage
    Handles: Preflight checks, URL filtering, error handling, progress tracking
    """

    def __init__(self, bot: BaseBot, db_path: str):
        """
        Initialize Brain with a bot and database path.

        Args:
            bot: The crawler bot (must inherit from BaseBot)
            db_path: Path to SQLite database
        """
        self.bot = bot
        self.db_path = db_path
        self.storage = Storage(db_path)
        self.fetcher = Fetcher()

    def crawl(
        self,
        max_pages: Optional[int] = None,
        refresh: bool = False,
        retry_failed: bool = False,
        refresh_days: Optional[int] = None,
        run_preflight: bool = True,
        delay: float = 1.0
    ):
        """
        Start crawling process.

        Args:
            max_pages: Limit number of pages to crawl (None = all)
            refresh: If True, re-crawl ALL pages (ignores existing)
            retry_failed: If True, only re-crawl failed pages
            refresh_days: Re-crawl pages older than X days (None = disabled)
            run_preflight: Run preflight checks before crawling
            delay: Delay in seconds between requests (default: 1.0, prevents IP blocking)

        Refresh modes (mutually exclusive, priority order):
        1. refresh=True: Re-crawl everything
        2. retry_failed=True: Only re-crawl failed
        3. refresh_days=N: Re-crawl pages older than N days
        4. Default: Skip successful pages, crawl new ones
        """
        print("\n" + "=" * 80)
        print(f"Brain: Starting crawl with bot '{self.bot.name}'")
        print("=" * 80)

        # Connect to database
        self.storage.connect()

        try:
            # Step 1: Run preflight checks
            if run_preflight:
                print("\n1. Running preflight checks...")
                preflight = Preflight(self.db_path)
                preflight.run_generic_tests()

                # Run bot-specific preflight
                success, message = self.bot.preflight_test()
                preflight.run_site_test(f"{self.bot.name} preflight", lambda: (success, message))

                if not preflight.result.print_results():
                    print("\n✗ Preflight checks failed. Aborting.")
                    return

            # Step 2: Get URLs from bot
            print(f"\n2. Discovering URLs with {self.bot.name}...")
            all_urls = self.bot.get_urls()
            print(f"✓ Discovered {len(all_urls)} URLs")

            # Step 3: Filter URLs based on refresh mode
            print(f"\n3. Filtering URLs (refresh={refresh}, retry_failed={retry_failed}, refresh_days={refresh_days})...")
            urls_to_crawl = self._filter_urls(all_urls, refresh, retry_failed, refresh_days)

            if max_pages and len(urls_to_crawl) > max_pages:
                urls_to_crawl = urls_to_crawl[:max_pages]
                print(f"✓ Limited to {max_pages} pages")

            print(f"✓ Will crawl {len(urls_to_crawl)} URLs")

            if len(urls_to_crawl) == 0:
                print("\n✓ No URLs to crawl. All done!")
                return

            # Step 4: Crawl loop
            print(f"\n4. Starting crawl (delay={delay}s between requests)...")
            print("-" * 80)
            stats = self._crawl_loop(urls_to_crawl, delay)

            # Step 5: Print stats
            print("\n" + "=" * 80)
            print("Crawl Complete!")
            print("=" * 80)
            print(f"Total URLs processed: {stats['total']}")
            print(f"✓ Success: {stats['success']}")
            print(f"✗ Failed: {stats['failed']}")
            print(f"⊘ Skipped: {stats['skipped']}")
            print("=" * 80 + "\n")

        finally:
            self.storage.close()
            self.fetcher.close()

    def _filter_urls(self, all_urls, refresh, retry_failed, refresh_days):
        """Filter URLs based on refresh mode"""

        # Mode 1: Refresh all - return all URLs
        if refresh:
            print("  Mode: REFRESH ALL (re-crawl everything)")
            return all_urls

        # Mode 2: Retry failed only
        if retry_failed:
            print("  Mode: RETRY FAILED (only re-crawl failed pages)")
            failed_urls = []
            for url in all_urls:
                doc = self.storage.get_document_by_url(url)
                if doc and doc['crawl_status'] == 'failed':
                    failed_urls.append(url)
            print(f"  Found {len(failed_urls)} failed pages to retry")
            return failed_urls

        # Mode 3: Refresh by age
        if refresh_days is not None:
            print(f"  Mode: REFRESH OLD (re-crawl pages older than {refresh_days} days)")
            cutoff_date = datetime.now(UTC) - timedelta(days=refresh_days)
            old_urls = []

            for url in all_urls:
                doc = self.storage.get_document_by_url(url)
                if not doc:
                    # New URL
                    old_urls.append(url)
                else:
                    # Check if old
                    updated_at = datetime.fromisoformat(doc['updated_at'])
                    if updated_at < cutoff_date:
                        old_urls.append(url)

            print(f"  Found {len(old_urls)} pages to crawl (new + old)")
            return old_urls

        # Mode 4: Default - skip successful, crawl new
        print("  Mode: DEFAULT (skip successful, crawl new)")
        new_urls = []
        for url in all_urls:
            doc = self.storage.get_document_by_url(url)
            if not doc or doc['crawl_status'] != 'success':
                new_urls.append(url)

        print(f"  Found {len(new_urls)} new/failed pages to crawl")
        return new_urls

    def _crawl_loop(self, urls, delay):
        """Main crawl loop with rate limiting"""
        stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
        total = len(urls)

        for idx, url in enumerate(urls, 1):
            stats['total'] += 1

            # Progress
            print(f"[{idx}/{total}] {url}")

            try:
                # Fetch
                html, status, error = self.fetcher.fetch(url)

                if status != 200 or not html:
                    # Failed to fetch
                    self._store_failed(url, status, error)
                    stats['failed'] += 1
                    print(f"  ✗ Fetch failed: {status} - {error}")

                    # Delay before next request
                    if idx < total:
                        time.sleep(delay)
                    continue

                # Parse
                content_text = self.bot.parse(html, url)

                if not content_text:
                    # Parsing failed but we have raw HTML
                    self._store_failed(url, status, "Parsing failed: no content extracted", html)
                    stats['failed'] += 1
                    print(f"  ✗ Parse failed: no content extracted")

                    # Delay before next request
                    if idx < total:
                        time.sleep(delay)
                    continue

                # Store success
                self._store_success(url, html, content_text, status)
                stats['success'] += 1
                print(f"  ✓ Success ({len(html)} bytes → {len(content_text)} chars)")

            except Exception as e:
                # Unexpected error
                self._store_failed(url, 0, f"Exception: {e}")
                stats['failed'] += 1
                print(f"  ✗ Exception: {e}")

            # Delay before next request (avoid IP blocking)
            if idx < total:
                time.sleep(delay)

        return stats

    def _store_success(self, url, content_raw, content_text, http_status):
        """Store successful crawl"""
        # Check if exists
        doc = self.storage.get_document_by_url(url)

        if doc:
            # Update existing
            now = datetime.now(UTC).isoformat()
            self.storage.conn.execute("""
                UPDATE documents
                SET content_raw = ?, content_text = ?, updated_at = ?,
                    crawled_at = ?, crawl_status = 'success',
                    http_status = ?, error_message = NULL
                WHERE url = ?
            """, (content_raw, content_text, now, now, http_status, url))
            self.storage.conn.commit()
        else:
            # Insert new
            self.storage.insert_document(
                url=url,
                content_raw=content_raw,
                content_text=content_text,
                crawl_status='success',
                http_status=http_status
            )

    def _store_failed(self, url, http_status, error_message, content_raw=None):
        """Store failed crawl"""
        # Check if exists
        doc = self.storage.get_document_by_url(url)

        if doc:
            # Update existing
            now = datetime.now(UTC).isoformat()
            self.storage.conn.execute("""
                UPDATE documents
                SET content_raw = ?, crawled_at = ?, crawl_status = 'failed',
                    http_status = ?, error_message = ?, updated_at = ?
                WHERE url = ?
            """, (content_raw, now, http_status, error_message, now, url))
            self.storage.conn.commit()
        else:
            # Insert new
            self.storage.insert_document(
                url=url,
                content_raw=content_raw,
                crawl_status='failed',
                http_status=http_status,
                error_message=error_message
            )


if __name__ == "__main__":
    print("Brain is an orchestrator. Use it with a bot:")
    print("""
    from crawler.core.brain import Brain
    from crawler.bots.mygovsa import MyGovSABot

    bot = MyGovSABot()
    brain = Brain(bot, 'path/to/db')
    brain.crawl(max_pages=10)
    """)
