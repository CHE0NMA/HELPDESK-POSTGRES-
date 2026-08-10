"""
services/grounding.py — post-generation check that an LLM answer is
actually supported by the source excerpts it was given.

Prompts already say "answer ONLY using the excerpts below" and "if you
can't, say CANNOT_ANSWER" — but smaller/faster models (which is the whole
point of using Groq) don't reliably obey that instruction under the hood;
they paraphrase-and-drift instead of cleanly refusing. This is a cheap,
model-agnostic backstop: after generation, check whether the distinctive
content words in the answer actually appear somewhere in the source text.
If they don't, treat it exactly like an explicit CANNOT_ANSWER.

This is intentionally corpus-agnostic — no keyword lists specific to any
one handbook. It just measures word overlap.
"""
import re
from typing import Iterable

_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'are', 'was',
    'were', 'for', 'on', 'at', 'by', 'with', 'as', 'that', 'this', 'it',
    'be', 'shall', 'must', 'may', 'can', 'will', 'not', 'their', 'they',
    'them', 'been', 'being', 'you', 'your', 'we', 'our', 'i', 'he', 'she',
    'his', 'her', 'its', 'if', 'then', 'so', 'also', 'according',
}


def _content_words(text: str) -> set:
    words = re.findall(r"[a-z0-9']{3,}", (text or '').lower())
    return {w for w in words if w not in _STOPWORDS}


def answer_is_grounded(answer: str, source_text: str, min_overlap: float = 0.4,
                        min_words_checked: int = 4) -> bool:
    """Returns True if the answer looks like it's actually drawn from
    source_text, False if it looks like drift/fabrication.

    - min_overlap: fraction of the answer's distinctive words that must
      appear somewhere in the source text.
    - min_words_checked: if the answer has fewer distinctive words than
      this (e.g. a one-line "Yes." answer), skip the check — there's not
      enough signal either way, and rejecting short factual answers would
      cause more harm than it prevents.
    """
    if not answer or not answer.strip():
        return False
    ans_words = _content_words(answer)
    if len(ans_words) < min_words_checked:
        return True
    src_words = _content_words(source_text)
    if not src_words:
        return False
    overlap = len(ans_words & src_words) / len(ans_words)
    return overlap >= min_overlap


def grounding_ratio(answer: str, source_text: str) -> float:
    """Raw ratio, for logging/debugging rather than a hard gate."""
    ans_words = _content_words(answer)
    if not ans_words:
        return 1.0
    src_words = _content_words(source_text)
    if not src_words:
        return 0.0
    return len(ans_words & src_words) / len(ans_words)


def combined_source_text(chunks: Iterable[dict], content_key: str = 'content',
                          title_key: str = 'section_title') -> str:
    parts = []
    for c in chunks:
        parts.append(c.get(title_key, '') or '')
        parts.append(c.get(content_key, '') or '')
    return " ".join(parts)