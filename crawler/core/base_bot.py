"""Base bot interface - all crawler bots must implement this"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseBot(ABC):
    """
    Base class for all crawler bots.

    Each bot must implement:
    1. get_urls() - How to discover URLs to crawl
    2. parse() - How to extract clean text from HTML
    3. preflight_test() - Site-specific pre-flight check
    """

    def __init__(self, name: str):
        """Initialize bot with a name"""
        self.name = name

    @abstractmethod
    def get_urls(self) -> List[str]:
        """
        Discover URLs to crawl.

        Returns:
            List of URLs to crawl

        Example implementations:
        - Parse sitemap.xml
        - Scrape pagination links
        - Call an API
        - Read from a file
        """
        pass

    @abstractmethod
    def parse(self, html: str, url: str) -> Optional[str]:
        """
        Extract clean text from HTML.

        Args:
            html: Raw HTML content
            url: URL of the page (for context)

        Returns:
            Cleaned text content, or None if parsing failed

        Example implementations:
        - BeautifulSoup to extract specific elements
        - Regex to clean HTML
        - Custom parsing logic
        """
        pass

    @abstractmethod
    def preflight_test(self) -> Tuple[bool, str]:
        """
        Site-specific pre-flight check.

        Returns:
            Tuple of (success: bool, message: str)

        Example:
            Test if main page is accessible,
            check if sitemap exists, etc.
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"


if __name__ == "__main__":
    print("BaseBot is an abstract class. Create a concrete bot to test.")
    print("\nExample bot structure:")
    print("""
    class MyBot(BaseBot):
        def get_urls(self):
            return ["https://example.com/page1", ...]

        def parse(self, html, url):
            return "cleaned text..."

        def preflight_test(self):
            return (True, "Site accessible")
    """)
