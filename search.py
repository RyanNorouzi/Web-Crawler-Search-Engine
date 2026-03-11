"""
search.py - Interactive Search Interface
==========================================
The user-facing command-line interface for querying the search engine.

This module loads the pre-built index from SQLite and provides:
  - An interactive REPL for exploring search results
  - A single-query mode for scripting / testing
  - A debug mode to explain how documents were scored
"""

import sys
import textwrap
import logging
from pathlib import Path

from indexer import Indexer, DEFAULT_DB_PATH
from ranker import Ranker

logger = logging.getLogger(__name__)


# ANSI color codes — makes CLI output easier to scan
class Color:
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

def c(code: str, text: str) -> str:
    """Wrap text in an ANSI color code (skipped if not a TTY)."""
    if sys.stdout.isatty():
        return f"{code}{text}{Color.RESET}"
    return text


def print_results(results: list[dict], query: str) -> None:
    """Pretty-print search results to the terminal."""
    if not results:
        print(c(Color.YELLOW, f'\n  No results found for "{query}".\n'))
        print("  Suggestions:")
        print("  • Try different keywords")
        print("  • Check that the site has been crawled (run: python main.py crawl)")
        print()
        return

    print(c(Color.BOLD, f'\n  Top {len(results)} results for "{query}":\n'))
    print("  " + "─" * 70)

    for result in results:
        rank    = result["rank"]
        title   = result["title"] or "Untitled"
        url     = result["url"]
        snippet = result["snippet"]
        score   = result["score"]

        # Rank badge
        print(f"\n  {c(Color.CYAN, f'[{rank}]')}  {c(Color.BOLD, title)}")
        print(f"       {c(Color.GREEN, url)}")
        print(f"       {c(Color.DIM, 'Score:')} {c(Color.MAGENTA, str(score))}")

        if snippet:
            # Word-wrap the snippet to 65 chars, indent nicely
            wrapped = textwrap.fill(snippet[:200], width=65)
            indented = textwrap.indent(wrapped, prefix="       ")
            print(c(Color.DIM, indented))

    print("\n  " + "─" * 70 + "\n")


def interactive_search(db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Launch a REPL (Read-Eval-Print Loop) search session.
    
    Special commands:
        :quit / :exit  → exit the program
        :explain <url> → show TF-IDF breakdown for last query + given URL
        :stats         → show index statistics
        :help          → show this help
    """
    indexer = Indexer(db_path)
    doc_count = indexer.document_count()

    if doc_count == 0:
        print(c(Color.YELLOW, "\n⚠️  The search index is empty."))
        print("   Run the crawler first:  python main.py crawl\n")
        return

    index     = indexer.get_index()
    documents = indexer.get_documents()
    ranker    = Ranker(index, documents)

    _print_welcome(doc_count, len(index))

    last_query = ""

    while True:
        try:
            raw = input(c(Color.CYAN, "  🔍 Search › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(c(Color.DIM, "\n\n  Goodbye!\n"))
            break

        if not raw:
            continue

        # ── Special commands ───────────────────────────────────────────
        if raw.lower() in (":quit", ":exit", "q", "quit", "exit"):
            print(c(Color.DIM, "\n  Goodbye!\n"))
            break

        if raw.lower() == ":help":
            _print_help()
            continue

        if raw.lower() == ":stats":
            _print_stats(doc_count, index, documents)
            continue

        if raw.lower().startswith(":explain"):
            parts = raw.split(maxsplit=1)
            if len(parts) < 2:
                print(c(Color.YELLOW, "  Usage: :explain <url>\n"))
                continue
            _print_explain(ranker, last_query, parts[1])
            continue

        # ── Normal search ──────────────────────────────────────────────
        last_query = raw
        results = ranker.rank(raw, top_k=10)
        print_results(results, raw)


def single_query(query: str, db_path: Path = DEFAULT_DB_PATH, top_k: int = 10) -> list[dict]:
    """
    Run a single search query (non-interactive, good for scripts/tests).
    
    Returns the result list and also prints to stdout.
    """
    indexer   = Indexer(db_path)
    index     = indexer.get_index()
    documents = indexer.get_documents()
    ranker    = Ranker(index, documents)

    results = ranker.rank(query, top_k=top_k)
    print_results(results, query)
    return results


# ------------------------------------------------------------------ #
#  Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _print_welcome(doc_count: int, vocab_size: int) -> None:
    print()
    print(c(Color.BOLD, "  ╔══════════════════════════════════════════╗"))
    print(c(Color.BOLD, "  ║       🔎  Mini Search Engine  🔍          ║"))
    print(c(Color.BOLD, "  ╚══════════════════════════════════════════╝"))
    print(f"  {c(Color.DIM, f'Index: {doc_count:,} documents  ·  {vocab_size:,} unique terms')}")
    print(f"  {c(Color.DIM, 'Type :help for commands  ·  :quit to exit')}")
    print()


def _print_help() -> None:
    print(c(Color.BOLD, "\n  Commands:"))
    print("    <query>          → search the index")
    print("    :explain <url>   → show TF-IDF score breakdown for last query")
    print("    :stats           → index statistics")
    print("    :quit            → exit\n")


def _print_stats(doc_count: int, index: dict, documents: dict) -> None:
    vocab_size  = len(index)
    avg_postings = sum(len(v) for v in index.values()) / max(vocab_size, 1)
    print(c(Color.BOLD, "\n  Index statistics:"))
    print(f"    Documents indexed : {doc_count:,}")
    print(f"    Vocabulary size   : {vocab_size:,} unique terms")
    print(f"    Avg docs per term : {avg_postings:.1f}\n")


def _print_explain(ranker: Ranker, query: str, url: str) -> None:
    if not query:
        print(c(Color.YELLOW, "  Run a search query first, then use :explain <url>\n"))
        return
    info = ranker.explain(query, url)
    if "error" in info:
        print(c(Color.YELLOW, f"  {info['error']}\n"))
        return

    print(c(Color.BOLD, f"\n  Score explanation for: {url}"))
    print(f"  Query: {info['query']}  |  Total score: {info['total_score']}\n")
    print(f"  {'Term':<20} {'TF':>8} {'IDF':>8} {'TF-IDF':>8}")
    print("  " + "─" * 48)
    for term, vals in info["breakdown"].items():
        note = f"  ← {vals.get('note', '')}" if vals.get("note") else ""
        print(f"  {term:<20} {vals['tf']:>8.5f} {vals['idf']:>8.5f} {vals['tfidf']:>8.5f}{note}")
    print()
