"""Pre-flight checks before crawling"""

from pathlib import Path
from typing import List, Tuple, Callable
from .storage import Storage
from .fetcher import Fetcher


class PreflightResult:
    def __init__(self):
        self.tests: List[Tuple[str, bool, str]] = []  # (name, passed, message)

    def add(self, name: str, passed: bool, message: str = ""):
        self.tests.append((name, passed, message))

    @property
    def all_passed(self) -> bool:
        return all(passed for _, passed, _ in self.tests)

    @property
    def failures(self) -> List[str]:
        return [name for name, passed, _ in self.tests if not passed]

    def print_results(self):
        print("\nRunning pre-flight checks...")
        print("=" * 60)

        for name, passed, message in self.tests:
            status = "✓" if passed else "✗"
            msg = f" - {message}" if message else ""
            print(f"{status} {name}{msg}")

        print("=" * 60)
        if self.all_passed:
            print("All tests passed! Ready to crawl.\n")
        else:
            print(f"Failed: {len(self.failures)} test(s)\n")
            return False
        return True


class Preflight:
    def __init__(self, db_path: str):
        """Initialize preflight checker"""
        self.db_path = db_path
        self.result = PreflightResult()

    def run_generic_tests(self) -> PreflightResult:
        """Run generic tests for all crawlers"""

        # Test 1: Database connectivity
        try:
            storage = Storage(self.db_path)
            storage.connect()
            self.result.add("Database connectivity", True)
            storage.close()
        except Exception as e:
            self.result.add("Database connectivity", False, str(e))
            return self.result

        # Test 2: Database write/read
        try:
            storage = Storage(self.db_path)
            storage.connect()

            # Insert test document
            test_url = "https://preflight-test.example.com/test"
            doc_id = storage.insert_document(
                url=test_url,
                content_text="preflight test",
                crawl_status="success",
                http_status=200
            )

            # Read it back
            doc = storage.get_document_by_id(doc_id)
            if doc and doc['url'] == test_url:
                self.result.add("Database write/read", True)
            else:
                self.result.add("Database write/read", False, "Failed to verify data")

            # Clean up test data
            storage.conn.execute("DELETE FROM documents WHERE url = ?", (test_url,))
            storage.conn.commit()
            storage.close()

        except Exception as e:
            self.result.add("Database write/read", False, str(e))
            return self.result

        # Test 3: Fetcher initialization
        try:
            fetcher = Fetcher()
            self.result.add("Fetcher initialized", True)
            fetcher.close()
        except Exception as e:
            self.result.add("Fetcher initialized", False, str(e))

        return self.result

    def run_site_test(self, name: str, test_func: Callable) -> PreflightResult:
        """Run a site-specific test"""
        try:
            passed, message = test_func()
            self.result.add(name, passed, message)
        except Exception as e:
            self.result.add(name, False, f"Exception: {e}")

        return self.result


if __name__ == "__main__":
    # Test the preflight system
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from crawler.core.preflight import Preflight

    db_path = Path(__file__).parent.parent.parent / "db" / "barq.db"

    preflight = Preflight(str(db_path))
    preflight.run_generic_tests()
    success = preflight.result.print_results()

    exit(0 if success else 1)
