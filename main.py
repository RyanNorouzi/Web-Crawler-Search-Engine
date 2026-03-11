"""
main.py - Entry Point & CLI
=============================
Ties the whole search engine together.

Usage:
    # Step 1 — crawl a website and build the index
    python main.py crawl --url https://en.wikipedia.org/wiki/Python_(programming_language) --pages 30 --depth 2

    # Step 2 — search interactively
    python main.py search

    # Or run a one-shot query
    python main.py search --query "list comprehension"

    # Reset the database and start fresh
    python main.py reset
"""

import argparse
import logging
import sys
from pathlib import Path

from crawler  import Crawler
from indexer  import Indexer, DEFAULT_DB_PATH
from search   import interactive_search, single_query

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # Only show warnings and errors by default
    format="%(levelname)s │ %(name)s │ %(message)s",
)


def cmd_crawl(args: argparse.Namespace) -> None:
    """
    Crawl a website starting from a seed URL, then index the result.
    Pass --fresh to wipe the database first.
    """
    db_path = Path(args.db)
    indexer = Indexer(db_path)

    if args.fresh:
        indexer.clear()

    crawler = Crawler(
        seed_url  = args.url,
        max_pages = args.pages,
        max_depth = args.depth,
        delay     = args.delay,
    )
    pages = crawler.crawl()

    if not pages:
        print("⚠️  No pages were crawled. Check the URL and try again.")
        sys.exit(1)

    indexer.index_pages(pages)
    print(f"✅  Done! Index stored at: {db_path}")
    print(f"   Run searches with:  python main.py search\n")


def cmd_search(args: argparse.Namespace) -> None:
    """
    Run a search — either interactive REPL or a single query.
    """
    db_path = Path(args.db)

    if args.query:
        single_query(args.query, db_path=db_path, top_k=args.top)
    else:
        interactive_search(db_path=db_path)


def cmd_reset(args: argparse.Namespace) -> None:
    """Delete all indexed data."""
    db_path = Path(args.db)
    indexer = Indexer(db_path)
    indexer.clear()
    print(f"✅  Database cleared: {db_path}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Print index statistics without opening a search prompt."""
    db_path = Path(args.db)
    indexer = Indexer(db_path)
    doc_count = indexer.document_count()
    index = indexer.get_index()
    vocab = len(index)
    avg  = sum(len(v) for v in index.values()) / max(vocab, 1)

    print(f"\n  📊 Index Statistics — {db_path}")
    print(f"  ├─ Documents  : {doc_count:,}")
    print(f"  ├─ Vocabulary : {vocab:,} unique terms")
    print(f"  └─ Avg postings per term: {avg:.1f}\n")


# ── Argument parser ────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-search-engine",
        description="🔍 Mini Search Engine — crawl, index, and search the web.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl Wikipedia's Python page (30 pages, depth 2)
  python main.py crawl --url https://en.wikipedia.org/wiki/Python_(programming_language) --pages 30

  # Crawl a blog and build a fresh index
  python main.py crawl --url https://realpython.com --pages 50 --depth 3 --fresh

  # Open the interactive search REPL
  python main.py search

  # One-shot search
  python main.py search --query "decorators in python"

  # Show index statistics
  python main.py stats
        """,
    )

    # Global option — custom database path
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        metavar="PATH",
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── crawl ──────────────────────────────────────────────────────────
    crawl_p = subparsers.add_parser("crawl", help="Crawl a website and build the search index")
    crawl_p.add_argument(
        "--url", required=True, metavar="URL",
        help="Seed URL to start crawling from"
    )
    crawl_p.add_argument(
        "--pages", type=int, default=50, metavar="N",
        help="Maximum pages to crawl (default: 50)"
    )
    crawl_p.add_argument(
        "--depth", type=int, default=2, metavar="D",
        help="Maximum link depth from seed (default: 2)"
    )
    crawl_p.add_argument(
        "--delay", type=float, default=0.5, metavar="SEC",
        help="Seconds between requests — be polite! (default: 0.5)"
    )
    crawl_p.add_argument(
        "--fresh", action="store_true",
        help="Clear existing index before crawling"
    )
    crawl_p.set_defaults(func=cmd_crawl)

    # ── search ─────────────────────────────────────────────────────────
    search_p = subparsers.add_parser("search", help="Search the index")
    search_p.add_argument(
        "--query", "-q", metavar="QUERY",
        help="Run a single query (omit for interactive mode)"
    )
    search_p.add_argument(
        "--top", "-k", type=int, default=10, metavar="K",
        help="Number of results to return (default: 10)"
    )
    search_p.set_defaults(func=cmd_search)

    # ── reset ──────────────────────────────────────────────────────────
    reset_p = subparsers.add_parser("reset", help="Clear the search index database")
    reset_p.set_defaults(func=cmd_reset)

    # ── stats ──────────────────────────────────────────────────────────
    stats_p = subparsers.add_parser("stats", help="Print index statistics")
    stats_p.set_defaults(func=cmd_stats)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
