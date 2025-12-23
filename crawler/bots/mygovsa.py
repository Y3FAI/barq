"""my.gov.sa crawler bot"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crawler.core.base_bot import BaseBot
from crawler.core.fetcher import Fetcher
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET
import time


class MyGovSABot(BaseBot):
    """Bot for crawling my.gov.sa service pages"""

    def __init__(self):
        super().__init__(name="MyGovSA")
        self.base_url = "https://my.gov.sa"
        self.sitemap_url = "https://my.gov.sa/sitemap.xml"

    def get_urls(self) -> List[str]:
        """
        Parse sitemap to discover service page URLs.

        Returns:
            List of URLs to crawl (only /ar/services/* pages)
        """
        fetcher = Fetcher()

        try:
            # Fetch sitemap
            html, status, error = fetcher.fetch(self.sitemap_url)

            if status != 200 or not html:
                print(f"Failed to fetch sitemap: {status} - {error}")
                return []

            # Parse sitemap XML
            root = ET.fromstring(html)

            # sitemap.xml might be a sitemap index, pointing to other sitemaps
            # Let's check for both cases
            urls = []

            # Namespace handling for sitemap
            ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            # Check if this is a sitemap index
            sitemaps = root.findall('sm:sitemap', ns)
            if sitemaps:
                # This is a sitemap index, fetch individual sitemaps
                print(f"Found {len(sitemaps)} sitemaps in index")
                for sitemap in sitemaps:
                    loc = sitemap.find('sm:loc', ns)
                    if loc is not None and loc.text:
                        print(f"Fetching subsitemap: {loc.text}")
                        sub_urls = self._parse_sitemap_urls(loc.text, fetcher)
                        urls.extend(sub_urls)
            else:
                # This is a direct sitemap with URLs
                urls = self._parse_sitemap_urls(self.sitemap_url, fetcher)

            # Filter only service pages
            service_urls = [url for url in urls if '/ar/services/' in url]

            print(f"Found {len(service_urls)} service pages")
            return service_urls

        except Exception as e:
            print(f"Error parsing sitemap: {e}")
            return []
        finally:
            fetcher.close()

    def _parse_sitemap_urls(self, sitemap_url: str, fetcher: Fetcher) -> List[str]:
        """Parse URLs from a single sitemap"""
        html, status, error = fetcher.fetch(sitemap_url)

        if status != 200 or not html:
            return []

        try:
            root = ET.fromstring(html)
            ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            urls = []
            for url_elem in root.findall('sm:url', ns):
                loc = url_elem.find('sm:loc', ns)
                if loc is not None and loc.text:
                    urls.append(loc.text)

            return urls
        except Exception as e:
            print(f"Error parsing {sitemap_url}: {e}")
            return []

    def parse(self, html: str, url: str) -> Optional[str]:
        """
        Extract valuable content from service page HTML.

        Uses markdownify to convert HTML to markdown, then removes
        header/navigation by cutting everything before the service breadcrumb.
        """
        if not html:
            return None

        try:
            from bs4 import BeautifulSoup
            from markdownify import markdownify

            soup = BeautifulSoup(html, 'html.parser')

            # Remove unwanted elements
            for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
                tag.decompose()

            # Get body
            body = soup.find('body')
            if not body:
                return None

            # Convert to markdown
            markdown = markdownify(str(body), heading_style="ATX")

            # Remove everything before the service breadcrumb
            # This cuts out all header/navigation noise
            service_marker = "[الخدمات](/ar/services)"
            if service_marker in markdown:
                # Find the marker and keep everything after it
                marker_pos = markdown.find(service_marker)
                markdown = markdown[marker_pos:]

            # Clean up: remove excessive blank lines
            lines = [line for line in markdown.split('\n') if line.strip()]
            markdown = '\n'.join(lines)

            return markdown.strip() if markdown.strip() else None

        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return None

    def preflight_test(self) -> Tuple[bool, str]:
        """
        Test if my.gov.sa is accessible and parsing works.

        Tests:
        1. Main site accessibility
        2. Sample service page parsing (checks for breadcrumb marker)
        """
        fetcher = Fetcher()

        try:
            # Test 1: Main site accessibility
            start = time.time()
            html, status, error = fetcher.fetch(self.base_url)
            elapsed = time.time() - start

            if status != 200 or not html:
                return (False, f"Main site failed: {status}, {error}")

            # Test 2: Sample service page parsing
            sample_url = "https://my.gov.sa/ar/services/112124"
            html, status, error = fetcher.fetch(sample_url)

            if status != 200 or not html:
                return (False, f"Sample page fetch failed: {status}, {error}")

            # Parse and check for breadcrumb marker
            parsed = self.parse(html, sample_url)
            marker = "[الخدمات](/ar/services)"

            if not parsed:
                return (False, "Parsing failed: no content extracted")

            if marker not in parsed:
                return (False, f"Breadcrumb marker '{marker}' not found in parsed content")

            return (True, f"Main site OK ({elapsed:.1f}s), Parsing OK, Marker found")

        finally:
            fetcher.close()


if __name__ == "__main__":
    print("Testing MyGovSA Bot")
    print("=" * 60)

    bot = MyGovSABot()

    # Test get_urls
    print("\n1. Testing get_urls() - Discovering service pages...")
    print("-" * 60)
    urls = bot.get_urls()

    print(f"\nTotal service pages found: {len(urls)}")
    print("\nFirst 10 URLs:")
    for url in urls[:10]:
        print(f"  - {url}")

    # Test preflight
    print("\n2. Testing preflight_test()...")
    print("-" * 60)
    success, message = bot.preflight_test()
    print(f"Success: {success}")
    print(f"Message: {message}")

    print("\n" + "=" * 60)
    print("✓ Bot test complete")
