"""HTTP fetcher with CloudFlare bypass for crawling"""

import cloudscraper
from typing import Optional, Tuple


class Fetcher:
    def __init__(self, timeout: int = 30):
        """Initialize fetcher with cloudscraper for CloudFlare bypass"""
        self.timeout = timeout
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'darwin',
                'desktop': True
            }
        )

    def fetch(self, url: str) -> Tuple[Optional[str], int, Optional[str]]:
        """
        Fetch a URL and return (html, status_code, error_message)

        Returns:
            tuple: (html_content, http_status, error_message)
                   html_content is None if failed
        """
        try:
            response = self.scraper.get(url, timeout=self.timeout)
            response.raise_for_status()  # Raise exception for 4xx/5xx

            return (response.text, response.status_code, None)

        except cloudscraper.exceptions.CloudflareChallengeError as e:
            return (None, 0, f"CloudFlare challenge failed: {e}")

        except Exception as e:
            # Handle timeout, HTTP errors, etc
            status = getattr(e, 'response', None)
            status_code = status.status_code if status and hasattr(status, 'status_code') else 0
            return (None, status_code, f"Request failed: {type(e).__name__}: {e}")

    def close(self):
        """Close scraper session"""
        self.scraper.close()


if __name__ == "__main__":
    # Quick test with my.gov.sa
    fetcher = Fetcher()

    print("Testing fetcher with my.gov.sa service page...")
    html, status, error = fetcher.fetch("https://my.gov.sa/ar/services/684127")

    if html:
        print(f"✓ Success! Status: {status}")
        print(f"✓ Received {len(html)} bytes")
        has_arabic = any(char in html for char in ['خ', 'د', 'م', 'ة'])
        print(f"✓ Contains Arabic: {has_arabic}")
        print(f"✓ First 200 chars: {html[:200]}")
    else:
        print(f"✗ Failed! Status: {status}, Error: {error}")

    fetcher.close()
    print("✓ Fetcher test complete")
