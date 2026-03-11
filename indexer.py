"""
indexer.py - Inverted Index Builder & Storage
===============================================
An inverted index is the core data structure of every search engine.

Traditional index:  document → [list of words it contains]
Inverted index:     word     → [list of documents containing it]

This "inversion" makes search fast: given a query word, we instantly look up
which documents contain it instead of scanning every document.

Example:
    Documents:
        doc1: "python is great"
        doc2: "python and java are popular"
        doc3: "great coffee shops in java"

    Inverted index:
        "python" → {doc1: 0.33, doc2: 0.25}
        "great"  → {doc1: 0.33, doc3: 0.25}
        "java"   → {doc2: 0.25, doc3: 0.25}

Storage uses SQLite so the index persists between runs without loading
everything into memory at once.
"""

import json
import logging
import sqlite3
from pathlib import Path

from utils import tokenize, compute_tf

logger = logging.getLogger(__name__)

# Default location for the SQLite database
DEFAULT_DB_PATH = Path("search_index.db")


class Indexer:
    """
    Builds and stores an inverted index from crawled pages.
    
    Schema (SQLite):
        documents(id, url, title, snippet)
        index(word, doc_id, tf)   ← the inverted index itself
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def index_pages(self, pages: list[dict]) -> None:
        """
        Process a list of crawled page dicts and write them into the index.
        
        Args:
            pages: Output from Crawler.crawl() — each dict has url, title, text.
        """
        print(f"📚 Indexing {len(pages)} pages...")
        for i, page in enumerate(pages, 1):
            self._index_single_page(page)
            print(f"  [{i:>3}/{len(pages)}] indexed: {page['url'][:70]}")
        print(f"\n✅  Indexing complete. Database saved to: {self.db_path}\n")

    def get_index(self) -> dict[str, dict[int, float]]:
        """
        Load the full inverted index from SQLite into memory.
        
        Returns:
            {word: {doc_id: tf_score, ...}, ...}
        
        Note: Fine for small indexes. For millions of docs, you'd use
              on-demand lookups instead of loading everything at once.
        """
        index: dict[str, dict[int, float]] = {}
        with self._connect() as conn:
            for row in conn.execute("SELECT word, doc_id, tf FROM inverted_index"):
                word, doc_id, tf = row
                index.setdefault(word, {})[doc_id] = tf
        return index

    def get_documents(self) -> dict[int, dict]:
        """
        Load all document metadata from SQLite.
        
        Returns:
            {doc_id: {url, title, snippet}, ...}
        """
        docs = {}
        with self._connect() as conn:
            for row in conn.execute("SELECT id, url, title, snippet FROM documents"):
                doc_id, url, title, snippet = row
                docs[doc_id] = {"url": url, "title": title, "snippet": snippet}
        return docs

    def document_count(self) -> int:
        """Return how many documents are stored in the index."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def clear(self) -> None:
        """Wipe the database (useful when re-crawling from scratch)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM inverted_index")
        print("🗑️  Index cleared.")

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Create the SQLite schema if it doesn't exist yet."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    url     TEXT UNIQUE NOT NULL,
                    title   TEXT,
                    snippet TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inverted_index (
                    word   TEXT NOT NULL,
                    doc_id INTEGER NOT NULL,
                    tf     REAL NOT NULL,
                    PRIMARY KEY (word, doc_id),
                    FOREIGN KEY (doc_id) REFERENCES documents(id)
                )
            """)
            # Speed up word lookups with an index on the word column
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_word ON inverted_index(word)
            """)

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL mode for better concurrency."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _index_single_page(self, page: dict) -> None:
        """
        Tokenize one page and write it into the database.
        
        Steps:
            1. Insert document metadata (url, title, snippet)
            2. Tokenize the body text
            3. Compute TF (term frequency) for each token
            4. Write (word, doc_id, tf) rows into inverted_index
        """
        url = page.get("url", "")
        title = page.get("title", "Untitled")
        text = page.get("text", "")
        snippet = text[:200].strip()  # Short preview for search results

        tokens = tokenize(text)
        if not tokens:
            return  # Skip empty pages

        tf_scores = compute_tf(tokens)  # {word: tf_score}

        with self._connect() as conn:
            # INSERT OR IGNORE skips if we've already indexed this URL
            cursor = conn.execute(
                "INSERT OR IGNORE INTO documents (url, title, snippet) VALUES (?, ?, ?)",
                (url, title, snippet),
            )

            # Fetch the doc_id (works whether we just inserted or it existed)
            doc_id = conn.execute(
                "SELECT id FROM documents WHERE url = ?", (url,)
            ).fetchone()[0]

            # Bulk-insert TF scores for all tokens in this document
            rows = [(word, doc_id, tf) for word, tf in tf_scores.items()]
            conn.executemany(
                "INSERT OR REPLACE INTO inverted_index (word, doc_id, tf) VALUES (?, ?, ?)",
                rows,
            )
