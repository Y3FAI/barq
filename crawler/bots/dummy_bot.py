"""Dummy bot for testing the BaseBot interface"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crawler.core.base_bot import BaseBot
from typing import List, Optional, Tuple


class DummyBot(BaseBot):
    """A simple dummy bot for testing the interface"""

    def __init__(self):
        super().__init__(name="DummyBot")

    def get_urls(self) -> List[str]:
        """Return a few test URLs"""
        return [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ]

    def parse(self, html: str, url: str) -> Optional[str]:
        """Extract text from HTML (dummy implementation)"""
        # Just return first 100 chars for testing
        if html:
            cleaned = html.replace("<", "").replace(">", "")
            return f"Parsed from {url}: {cleaned[:100]}"
        return None

    def preflight_test(self) -> Tuple[bool, str]:
        """Always passes for dummy bot"""
        return (True, "Dummy bot always ready")


if __name__ == "__main__":
    # Test the dummy bot
    print("Testing DummyBot interface...")
    print("=" * 60)

    bot = DummyBot()
    print(f"Bot: {bot}")
    print(f"Name: {bot.name}")

    # Test get_urls
    print("\n1. Testing get_urls():")
    urls = bot.get_urls()
    print(f"   Found {len(urls)} URLs:")
    for url in urls:
        print(f"   - {url}")

    # Test parse
    print("\n2. Testing parse():")
    sample_html = "<html><body><h1>Test Page</h1><p>Some content here</p></body></html>"
    parsed = bot.parse(sample_html, "https://example.com/test")
    print(f"   Result: {parsed}")

    # Test preflight
    print("\n3. Testing preflight_test():")
    success, message = bot.preflight_test()
    print(f"   Success: {success}")
    print(f"   Message: {message}")

    print("\n" + "=" * 60)
    print("✓ All interface methods work correctly!")
