"""
Example: How bots use the preflight system

This shows how a bot can add site-specific tests before crawling.
"""

from preflight import Preflight
from fetcher import Fetcher
import time


def run_preflight_for_mygovsa(db_path: str) -> bool:
    """Run all preflight checks for my.gov.sa crawler"""

    preflight = Preflight(db_path)

    # 1. Run generic tests (database, fetcher)
    preflight.run_generic_tests()

    # 2. Add site-specific test
    def test_mygovsa_accessible():
        """Test if my.gov.sa is accessible"""
        fetcher = Fetcher()
        start = time.time()

        html, status, error = fetcher.fetch("https://my.gov.sa")
        elapsed = time.time() - start
        fetcher.close()

        if status == 200 and html:
            return (True, f"Status: {status}, Time: {elapsed:.1f}s")
        else:
            return (False, f"Status: {status}, Error: {error}")

    preflight.run_site_test("my.gov.sa accessible", test_mygovsa_accessible)

    # 3. Print results and return success/failure
    return preflight.result.print_results()


# Usage in crawler:
# if __name__ == "__main__":
#     if not run_preflight_for_mygovsa("/path/to/db"):
#         print("Pre-flight checks failed. Aborting.")
#         exit(1)
#
#     # Start crawling...
