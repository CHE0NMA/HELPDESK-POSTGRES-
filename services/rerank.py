"""
services/rerank.py — generic rarity-weighted reranking for full-text
search results.

WHY THIS EXISTS
----------------
SQL "NATURAL LANGUAGE MODE" full-text scores are dragged around by whatever
words are most frequent across the corpus. For a student handbook, that's
usually the institution's own name ("Philippine Advent College", "PAC",
"Student") — appearing in nearly every chunk, contributing large but
totally non-discriminating relevance mass. A specific question like "what
was the original name of the school" gets outscored by a chunk that just
happens to repeat the institution's name a lot, instead of the chunk that
actually contains the answer.

This module is corpus-agnostic: it computes IDF (inverse document
frequency) from whatever chunks are actually loaded, at runtime, so it
adapts automatically to a different handbook with different repeated
boilerplate — nothing here is tuned to any specific school's vocabulary.

USAGE
-----
    idf = get_idf_index(all_chunks_fn=load_all_chunk_texts)   # cached
    reranked = rarity_rerank(question, scored_chunks, idf)
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter
from typing import Callable, Dict, List, Tuple

_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'are', 'was',
    'were', 'for', 'on', 'at', 'by', 'with', 'as', 'that', 'this', 'it',
    'be', 'shall', 'must', 'may', 'can', 'will', 'not', 'their', 'they',
    'what', 'when', 'where', 'who', 'how', 'do', 'does', 'did',
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']{3,}", (text or '').lower())


class _IDFCache:
    """Rebuilds the IDF table from the corpus periodically instead of on
    every single request (full-text corpora don't change every second),
    but stays correct after re-ingestion because of the TTL."""

    def __init__(self, ttl_seconds: int = 600):
        self.ttl = ttl_seconds
        self._idf: Dict[str, float] = {}
        self._doc_freq_ubiquity: Dict[str, float] = {}
        self._built_at: float = 0.0
        self._lock = threading.Lock()

    def get(self, loader: Callable[[], List[str]]) -> Dict[str, float]:
        with self._lock:
            if self._idf and (time.time() - self._built_at) < self.ttl:
                return self._idf
            docs = loader() or []
            self._idf, self._doc_freq_ubiquity = _build_idf(docs)
            self._built_at = time.time()
            return self._idf

    def invalidate(self):
        with self._lock:
            self._built_at = 0.0


_cache = _IDFCache()


def _build_idf(documents: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    n_docs = max(1, len(documents))
    df = Counter()
    for doc in documents:
        for w in set(tokenize(doc)):
            df[w] += 1
    idf: Dict[str, float] = {}
    doc_freq_ratio: Dict[str, float] = {}
    for w, d in df.items():
        idf[w] = math.log((n_docs + 1) / (d + 1)) + 1.0
        doc_freq_ratio[w] = d / n_docs
    return idf, doc_freq_ratio


def get_idf_index(loader: Callable[[], List[str]]) -> Dict[str, float]:
    """`loader` returns a list of full document texts (title + content is
    fine) for the ENTIRE active corpus — not just current search results —
    so ubiquitous terms get correctly identified regardless of what query
    triggered this call."""
    return _cache.get(loader)


def invalidate_idf_cache():
    """Call this right after re-ingestion so the next query rebuilds IDF
    against the new corpus instead of waiting out the TTL."""
    _cache.invalidate()


def rarity_rerank(
    question: str,
    scored_items: List[dict],
    idf: Dict[str, float],
    text_fn: Callable[[dict], str],
    score_key: str = 'score',
    blend: float = 0.45,
    ubiquity_dampener: float = 0.2,
) -> List[dict]:
    """Blend an existing relevance score with a rarity-weighted term-overlap
    score between the question and each item's text.

    `ubiquity_dampener`: words appearing in >60% of the corpus (i.e. very
    low idf, near the floor of ~1.0) get their contribution multiplied by
    this factor even after IDF weighting — belt-and-suspenders against
    boilerplate phrases that repeat in almost every chunk.
    """
    q_words = [w for w in tokenize(question) if w not in _STOPWORDS]
    if not q_words or not scored_items:
        return scored_items

    max_idf = max(idf.values()) if idf else 1.0

    for item in scored_items:
        blob_words = set(tokenize(text_fn(item)))
        rarity_score = 0.0
        for w in q_words:
            if w not in blob_words:
                continue
            weight = idf.get(w, max_idf)
            if weight <= 1.15:  # appears in most documents -> near-zero signal
                weight *= ubiquity_dampener
            rarity_score += weight
        normalized_rarity = min(rarity_score / max(1.0, len(q_words) * max_idf), 1.0)

        base_score = float(item.get(score_key, 0.0) or 0.0)
        item[score_key] = base_score * (1 - blend) + normalized_rarity * blend
        item['_rarity_score'] = normalized_rarity

    scored_items.sort(key=lambda x: x.get(score_key, 0.0), reverse=True)
    return scored_items