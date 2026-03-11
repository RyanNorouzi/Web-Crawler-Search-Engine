"""
crawler.py - Web Crawler
=========================
A focused web crawler that starts from a seed URL and systematically
visits pages within the same domain, collecting text content for indexing.

Key concepts used here:
- BFS (Breadth-First Search): We explore pages level by level using a queue
- Visited set: Prevents revisiting the same URL (deduplication)
- Domain scoping: Only follows links that stay on the original host
"""

import time
import logging
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class Crawler:
    """
    A polite, domain-scoped web crawler.
    
    Usage:
        crawler = Crawler(seed_url="https://example.com", max_pages=50, max_depth=3)
        pages = crawler.crawl()
        # pages = [{"url": ..., "title": ..., "text": ..., "links": [...]}, ...]
    """

    # Be a good citizen: identify ourselves in the User-Agent header
    HEADERS = {
        "User-Agent": "MiniSearchEngine/1.0 (educational crawler; github.com/you/mini-search-engine)"
    }

    def __init__(
        self,
        seed_url: str,
        max_pages: int = 50,
        max_depth: int = 3,
        delay: float = 0.5,
        timeout: int = 10,
    ):
        """
        Args:
            seed_url:  The starting URL for the crawl.
            max_pages: Hard cap on total pages to fetch (keeps runs manageable).
            max_depth: Maximum link depth from the seed (1 = seed only).
            delay:     Seconds to wait between requests (respect server load).
            timeout:   HTTP request timeout in seconds.
        """
        self.seed_url = seed_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.timeout = timeout

        # Extract the domain we're allowed to crawl (e.g. "en.wikipedia.org")
        parsed = urlparse(seed_url)
        self.allowed_domain = parsed.netloc

    def crawl(self) -> list[dict]:
        """
        Execute the BFS crawl starting from seed_url.
        
        Returns:
            A list of page dicts, each containing:
              - url   (str)
              - title (str)
              - text  (str)  ← visible body text, cleaned
              - links (list) ← absolute URLs found on the page
        """
        visited: set[str] = set()
        results: list[dict] = []

        # BFS queue stores (url, current_depth) tuples
        queue: deque[tuple[str, int]] = deque()
        queue.append((self.seed_url, 0))

        print(f"\n🕷️  Starting crawl from: {self.seed_url}")
        print(f"   Domain: {self.allowed_domain} | Max pages: {self.max_pages} | Max depth: {self.max_depth}\n")

        while queue and len(results) < self.max_pages:
            url, depth = queue.popleft()

            # Skip already-visited URLs
            if url in visited:
                continue

            # Enforce depth limit
            if depth > self.max_depth:
                continue

            visited.add(url)

            page_data = self._fetch_page(url)
            if page_data is None:
                continue

            results.append(page_data)
            print(f"  [{len(results):>3}/{self.max_pages}] depth={depth}  {url[:80]}")

            # Enqueue outgoing links for the next depth level
            for link in page_data["links"]:
                if link not in visited:
                    queue.append((link, depth + 1))

            # Polite crawl delay — don't hammer the server
            time.sleep(self.delay)

        print(f"\n✅  Crawl complete. Collected {len(results)} pages.\n")
        return results

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _fetch_page(self, url: str) -> dict | None:
        """
        Fetch a single URL and parse its content.
        Returns None if the request fails or the page is not HTML.
        """
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=self.timeout)
            response.raise_for_status()

            # Only process HTML content (skip PDFs, images, etc.)
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None

            soup = BeautifulSoup(response.text, "html.parser")

            return {
                "url": url,
                "title": self._extract_title(soup),
                "text": self._extract_text(soup),
                "links": self._extract_links(soup, url),
            }

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Return the page's <title> text, or 'Untitled' if missing."""
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else "Untitled"

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract all human-readable text from the page.
        
        We remove <script>, <style>, and navigation elements first,
        then grab the remaining visible text — the stuff a user would read.
        """
        # Remove elements that contain non-content text
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # get_text with separator=" " prevents words from two tags merging
        raw = soup.get_text(separator=" ")

        # Collapse excessive whitespace
        import re
        text = re.sub(r"\s+", " ", raw).strip()
        return text

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """
        Find all <a href="..."> links on the page, resolve them to absolute
        URLs, and filter to only same-domain links.
        """
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()

            # Skip fragment-only links (#section) and javascript: links
            if href.startswith("#") or href.startswith("javascript:"):
                continue

            # Resolve relative URLs → absolute (e.g. "/about" → "https://example.com/about")
            absolute_url = urljoin(base_url, href)
            parsed = urlparse(absolute_url)

            # Only follow links that stay on the same domain and use http(s)
            if parsed.netloc == self.allowed_domain and parsed.scheme in ("http", "https"):
                # Normalize: drop URL fragment to avoid crawling the same page twice
                clean_url = absolute_url.split("#")[0]
                links.append(clean_url)

        # Deduplicate while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        return unique_links
