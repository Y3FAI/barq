#!/usr/bin/env python3
"""
Barq Crawler - Command Line Interface

Usage:
    python crawl.py --bot mygovsa --max-pages 100
    python crawl.py --bot mygovsa --refresh
    python crawl.py --bot mygovsa --retry-failed
    python crawl.py --bot mygovsa --refresh-days 30
    python crawl.py --bot mygovsa --delay 2.0
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from crawler.core.brain import Brain
from crawler.bots.mygovsa import MyGovSABot


def get_bot(bot_name: str):
    """Get bot instance by name"""
    bots = {
        'mygovsa': MyGovSABot,
    }

    if bot_name not in bots:
        print(f"Error: Unknown bot '{bot_name}'")
        print(f"Available bots: {', '.join(bots.keys())}")
        sys.exit(1)

    return bots[bot_name]()


def main():
    parser = argparse.ArgumentParser(
        description='Barq Crawler - Crawl and index Arabic web content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl first 10 pages
  python crawl.py --bot mygovsa --max-pages 10

  # Crawl all pages
  python crawl.py --bot mygovsa

  # Force refresh all content
  python crawl.py --bot mygovsa --refresh

  # Retry failed pages only
  python crawl.py --bot mygovsa --retry-failed

  # Update pages older than 30 days
  python crawl.py --bot mygovsa --refresh-days 30

  # Skip preflight checks (faster start)
  python crawl.py --bot mygovsa --max-pages 100 --no-preflight

  # Custom delay to avoid rate limiting (2 seconds between requests)
  python crawl.py --bot mygovsa --delay 2.0

  # Fast crawl with minimal delay (use cautiously)
  python crawl.py --bot mygovsa --delay 0.5
        """
    )

    # Required arguments
    parser.add_argument(
        '--bot',
        required=True,
        choices=['mygovsa'],
        help='Bot to use for crawling'
    )

    # Optional arguments
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Maximum number of pages to crawl (default: all)'
    )

    parser.add_argument(
        '--db',
        type=str,
        default=str(Path(__file__).parent / 'db' / 'barq.db'),
        help='Path to SQLite database (default: project/db/barq.db)'
    )

    # Refresh modes (mutually exclusive)
    refresh_group = parser.add_mutually_exclusive_group()

    refresh_group.add_argument(
        '--refresh',
        action='store_true',
        help='Force refresh: re-crawl ALL pages (ignores existing)'
    )

    refresh_group.add_argument(
        '--retry-failed',
        action='store_true',
        help='Retry failed: only re-crawl pages that failed previously'
    )

    refresh_group.add_argument(
        '--refresh-days',
        type=int,
        default=None,
        metavar='DAYS',
        help='Refresh old: re-crawl pages not updated in DAYS days'
    )

    # Other options
    parser.add_argument(
        '--no-preflight',
        action='store_true',
        help='Skip preflight checks (start crawling immediately)'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        metavar='SECONDS',
        help='Delay between requests in seconds (default: 1.0, prevents IP blocking)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Barq Crawler v0.1.0'
    )

    args = parser.parse_args()

    # Print header
    print()
    print("=" * 80)
    print("Barq Crawler v0.1.0")
    print("=" * 80)

    # Initialize bot
    try:
        bot = get_bot(args.bot)
        print(f"Bot: {bot.name}")
    except Exception as e:
        print(f"Error initializing bot: {e}")
        sys.exit(1)

    # Initialize brain
    try:
        brain = Brain(bot, args.db)
        print(f"Database: {args.db}")
    except Exception as e:
        print(f"Error initializing brain: {e}")
        sys.exit(1)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Max pages: {args.max_pages if args.max_pages else 'all'}")
    print(f"  Refresh mode: ", end="")
    if args.refresh:
        print("REFRESH ALL")
    elif args.retry_failed:
        print("RETRY FAILED")
    elif args.refresh_days:
        print(f"REFRESH OLD ({args.refresh_days} days)")
    else:
        print("DEFAULT (skip successful)")
    print(f"  Preflight checks: {'disabled' if args.no_preflight else 'enabled'}")
    print(f"  Delay between requests: {args.delay}s")

    # Start crawling
    try:
        brain.crawl(
            max_pages=args.max_pages,
            refresh=args.refresh,
            retry_failed=args.retry_failed,
            refresh_days=args.refresh_days,
            run_preflight=not args.no_preflight,
            delay=args.delay
        )
    except KeyboardInterrupt:
        print("\n\nCrawl interrupted by user.")
        print("Progress has been saved to database.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during crawl: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
