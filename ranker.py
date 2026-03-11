"""
ranker.py - TF-IDF Ranking Engine
===================================
Given a search query, this module scores every candidate document and
returns them sorted by relevance.

How TF-IDF Ranking Works (plain English):
------------------------------------------
Imagine you search for "python tutorial".

For each document in the index, we ask two questions per query word:
  1. TF  (Term Frequency):   How often does "python" appear in *this* doc?
  2. IDF (Inverse Doc Freq): How rare is "python" across *all* docs?

TF-IDF score = TF × IDF

A document that mentions "python" 20 times gets a high TF.
If "python" only appears in 5 out of 1000 docs, it gets a high IDF.
The product rewards documents that discuss the query topic in depth and
aren't just generic pages where the word happens to drift in.

Multi-word queries: we sum the TF-IDF scores across all query words.
The document that best covers *all* query words rises to the top.

Formula recap:
    TF(t, d)  = count(t in d) / total_words(d)
    IDF(t)    = log(N / (1 + df(t)))          where df(t) = #docs containing t
    Score(d)  = Σ TF(t, d) × IDF(t)           summed over all query terms t
"""

import math
import logging
from utils import tokenize, compute_idf, compute_tfidf

logger = logging.getLogger(__name__)


class Ranker:
    """
    Ranks documents against a query using TF-IDF scoring.
    
    Usage:
        ranker = Ranker(index, documents)
        results = ranker.rank("python web scraping", top_k=10)
    """

    def __init__(self, index: dict[str, dict[int, float]], documents: dict[int, dict]):
        """
        Args:
            index:     Inverted index from Indexer.get_index()
                       Format: {word: {doc_id: tf_score, ...}}
            documents: Document metadata from Indexer.get_documents()
                       Format: {doc_id: {url, title, snippet}}
        """
        self.index = index
        self.documents = documents

    def rank(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Score all matching documents for a query and return the top results.
        
        Args:
            query:  Raw query string from the user (e.g. "machine learning tutorial")
            top_k:  Maximum number of results to return.
        
        Returns:
            List of result dicts, sorted by relevance (highest first):
            [
                {
                    "rank":    1,
                    "url":     "https://...",
                    "title":   "...",
                    "snippet": "...",
                    "score":   0.427,
                },
                ...
            ]
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Accumulate TF-IDF scores across all query tokens
        # scores[doc_id] = running total relevance score
        scores: dict[int, float] = {}

        for token in query_tokens:
            if token not in self.index:
                # Word not in any indexed document → contributes 0 to all scores
                continue

            idf = compute_idf(token, self.index)

            # For each document that contains this token, add its TF-IDF contribution
            for doc_id, tf in self.index[token].items():
                tfidf = compute_tfidf(tf, idf)
                scores[doc_id] = scores.get(doc_id, 0.0) + tfidf

        if not scores:
            return []

        # Sort documents by total score, highest first
        ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)

        # Build the result list, capped at top_k
        results = []
        for rank, doc_id in enumerate(ranked_ids[:top_k], start=1):
            doc = self.documents.get(doc_id, {})
            results.append({
                "rank":    rank,
                "url":     doc.get("url", "N/A"),
                "title":   doc.get("title", "Untitled"),
                "snippet": doc.get("snippet", ""),
                "score":   round(scores[doc_id], 4),
            })

        return results

    def explain(self, query: str, url: str) -> dict:
        """
        Debug helper: show exactly how a specific document was scored.
        
        Useful for understanding why a page ranked high or low.
        
        Returns a breakdown of TF, IDF, and TF-IDF per query term.
        """
        # Find doc_id for this URL
        doc_id = next(
            (did for did, d in self.documents.items() if d["url"] == url),
            None
        )
        if doc_id is None:
            return {"error": f"URL not in index: {url}"}

        query_tokens = tokenize(query)
        breakdown = {}
        total_score = 0.0

        for token in query_tokens:
            if token not in self.index or doc_id not in self.index[token]:
                breakdown[token] = {"tf": 0.0, "idf": 0.0, "tfidf": 0.0, "note": "word absent"}
                continue

            tf = self.index[token][doc_id]
            idf = compute_idf(token, self.index)
            tfidf = compute_tfidf(tf, idf)
            total_score += tfidf

            breakdown[token] = {
                "tf":    round(tf, 5),
                "idf":   round(idf, 5),
                "tfidf": round(tfidf, 5),
            }

        return {
            "url":        url,
            "query":      query,
            "total_score": round(total_score, 5),
            "breakdown":  breakdown,
        }
