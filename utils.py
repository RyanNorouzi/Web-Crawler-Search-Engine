"""
utils.py - Text Processing Utilities
=====================================
This module handles all text cleaning, tokenization, and normalization.
Think of this as the "text preprocessing pipeline" — raw HTML goes in,
clean searchable tokens come out.
"""

import re
import math
from collections import Counter


# Common English stopwords — words so frequent they add no search value.
# Removing them reduces index size and improves ranking accuracy.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those", "i",
    "me", "my", "we", "our", "you", "your", "he", "his", "she", "her",
    "they", "their", "what", "which", "who", "whom", "not", "no", "so",
    "if", "then", "than", "too", "very", "just", "also", "as", "into",
    "about", "up", "out", "more", "some", "when", "there", "all", "any",
    "how", "each", "both", "few", "other", "such", "only", "own", "same",
}


def clean_text(text: str) -> str:
    """
    Normalize raw text by lowercasing and removing non-alphabetic characters.
    
    Example:
        "Hello, World! 123" → "hello world"
    """
    text = text.lower()
    # Keep only letters and spaces; strip punctuation and numbers
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse multiple whitespace into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """
    Split cleaned text into individual word tokens.
    Filters out stopwords and very short words (< 2 chars).
    
    Example:
        "the quick brown fox" → ["quick", "brown", "fox"]
    """
    tokens = clean_text(text).split()
    return [word for word in tokens if word not in STOPWORDS and len(word) > 2]


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    Compute Term Frequency (TF) for a list of tokens.
    
    TF measures how often a word appears in a document.
    Formula: TF(word) = count(word) / total_words
    
    Normalizing by document length prevents long documents from dominating
    just because they have more words in total.
    
    Returns:
        dict mapping word → TF score (float between 0 and 1)
    """
    if not tokens:
        return {}
    
    word_counts = Counter(tokens)
    total_words = len(tokens)
    
    return {word: count / total_words for word, count in word_counts.items()}


def compute_idf(word: str, index: dict) -> float:
    """
    Compute Inverse Document Frequency (IDF) for a word.
    
    IDF measures how *rare* a word is across all documents.
    Formula: IDF(word) = log(total_docs / docs_containing_word)
    
    Intuition:
    - A word in every document (like "website") carries low signal → low IDF
    - A word in only 2 docs (like "photosynthesis") is very distinctive → high IDF
    
    Args:
        word:  The search term
        index: The inverted index {word: {doc_id: tf_score, ...}}
    
    Returns:
        IDF score (float). Returns 0.0 if word not in index.
    """
    total_docs = len({doc for postings in index.values() for doc in postings})
    if total_docs == 0 or word not in index:
        return 0.0
    
    docs_with_word = len(index[word])
    # Add 1 to avoid division-by-zero for words appearing in every document
    return math.log(total_docs / (1 + docs_with_word))


def compute_tfidf(tf: float, idf: float) -> float:
    """
    Combine TF and IDF into a single relevance score.
    
    TF-IDF = TF × IDF
    
    A word scores highly when it:
      1. Appears frequently in a specific document (high TF), AND
      2. Is rare across other documents (high IDF)
    
    This is the core of most classical search engines.
    """
    return tf * idf


def truncate(text: str, max_length: int = 120) -> str:
    """Helper to create a short snippet from a longer text block."""
    return text[:max_length].rstrip() + "..." if len(text) > max_length else text
