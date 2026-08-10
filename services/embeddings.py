"""
services/embeddings.py — semantic (meaning-based) similarity for FAQ
matching, to catch paraphrased questions that share no words with the
FAQ's own phrasing (e.g. "pwede pa ba ako mag-enroll late" vs "is late
enrollment allowed" — same meaning, almost no shared tokens, so
fuzzywuzzy scores them low even though the FAQ has the answer).

WHY GEMINI'S EMBEDDING API INSTEAD OF A LOCAL MODEL
-----------------------------------------------------
sentence-transformers (the usual local option) pulls in torch + a
multi-hundred-MB model download — rough on a free-tier host with limited
RAM/disk. This calls Gemini's `embedContent` endpoint instead: one small
HTTP request, no new heavy dependency, no model to ship. The trade-off is
it needs GEMINI_API_KEY set (Google AI Studio has a free tier) — this is
independent of AI_PROVIDER, so it works even if your chat responses are
running on Groq.

If GEMINI_API_KEY isn't set, every function here degrades to a no-op
(returns None / does nothing) rather than raising — FAQ matching falls
back to exactly the old fuzzy-only behavior. Nothing breaks if you don't
want to use this.

HOW IT'S WIRED IN
------------------
- faq.py embeds the incoming user question once per request (1 API call)
  and passes that vector into answer_engine.get_top_faq_candidates().
- Each FAQ row's OWN embedding is precomputed once (at add/update time —
  see routes/admin.py's add_faq/update_faq — and via the one-time
  backfill script for FAQs that existed before this feature) and cached
  in faq.embedding (JSONB), so matching a question against N FAQs costs
  1 API call total, not N+1.
- answer_engine.py blends cosine similarity into the existing fuzzy
  `combined` score rather than replacing it — see SEMANTIC_BLEND below.
"""
import logging
from typing import List, Optional

import requests

from config import GEMINI_API_KEY

logger = logging.getLogger("pac_helpdesk.embeddings")

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/{EMBEDDING_MODEL}:embedContent"
)
EMBEDDING_TIMEOUT_SECONDS = 8


def is_configured() -> bool:
    return bool(GEMINI_API_KEY)


def get_embedding(text: str, cache=None) -> Optional[List[float]]:
    """Returns an embedding vector for `text`, or None if unconfigured/
    failed — callers must treat None as "skip semantic scoring", never as
    an error to surface to the student.

    `cache`: optional object with .get(key)/.set(key, value) — pass in
    ai_providers.ai_cache to avoid re-embedding identical repeated
    questions (very common for a school FAQ bot — "what time is
    enrollment" gets asked verbatim a lot).
    """
    if not text or not text.strip():
        return None
    if not is_configured():
        return None

    cache_key = f"embed_{hash(text.strip().lower())}"
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return cached

    try:
        resp = requests.post(
            EMBEDDING_URL,
            params={'key': GEMINI_API_KEY},
            json={'content': {'parts': [{'text': text}]}},
            timeout=EMBEDDING_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            logger.warning(f"Embedding API returned {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        values = data.get('embedding', {}).get('values')
        if not values:
            return None
        if cache:
            cache.set(cache_key, values)
        return values
    except Exception as e:
        logger.error(f"get_embedding error: {e}")
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embedding_text_for_faq(question: str, keywords: str = "") -> str:
    """What actually gets embedded for a FAQ row. Deliberately question +
    keywords, NOT the answer — we're matching against what a STUDENT
    would type, and answers are often phrased very differently (longer,
    more formal) than how someone would ask the question."""
    parts = [question or ""]
    if keywords:
        parts.append(keywords)
    return ". ".join(p.strip() for p in parts if p and p.strip())


def store_embedding_for_faq(get_db_fn, question: str, keywords: str = "",
                             faq_id: int = None, public_uuid: str = None) -> bool:
    """Best-effort: compute + store an embedding for one FAQ row. Returns
    False (never raises) on any failure — callers should treat this as
    "nice to have", not block saving the FAQ on it succeeding.
    """
    if faq_id is None and public_uuid is None:
        return False
    text = embedding_text_for_faq(question, keywords)
    vector = get_embedding(text)
    if vector is None:
        return False

    conn = get_db_fn()
    cur = conn.cursor()
    try:
        if faq_id is not None:
            cur.execute("UPDATE faq SET embedding = %s::jsonb WHERE id = %s",
                        (list(vector), faq_id))
        else:
            cur.execute("UPDATE faq SET embedding = %s::jsonb WHERE public_uuid = %s",
                        (list(vector), public_uuid))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"store_embedding_for_faq error: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()