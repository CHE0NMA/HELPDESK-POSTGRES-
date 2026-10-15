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
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, List, Optional

import requests

from ai_providers.providers import (
    GEMINI_API_KEYS,
    get_next_gemini_key,
    mark_gemini_auth_failed,
    mark_gemini_rate_limited,
    mark_gemini_success,
)

logger = logging.getLogger("pac_helpdesk.embeddings")

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/{EMBEDDING_MODEL}:embedContent"
)
EMBEDDING_TIMEOUT_SECONDS = 8
_embedding_metrics_lock = Lock()
_embedding_metrics = {
    "api_calls": 0,
    "successful_calls": 0,
    "failed_calls": 0,
    "cache_hits": 0,
    "last_success_at": None,
    "last_failure_reason": None,
}


def is_configured() -> bool:
    return bool(GEMINI_API_KEYS)


def _record_embedding_api_call() -> None:
    with _embedding_metrics_lock:
        _embedding_metrics["api_calls"] += 1


def _record_embedding_result(success: bool, failure_reason: Optional[str] = None) -> None:
    with _embedding_metrics_lock:
        if success:
            _embedding_metrics["successful_calls"] += 1
            _embedding_metrics["last_success_at"] = datetime.now(timezone.utc).isoformat()
        else:
            _embedding_metrics["failed_calls"] += 1
            _embedding_metrics["last_failure_reason"] = failure_reason or "unknown"


def get_embedding_status() -> dict:
    with _embedding_metrics_lock:
        metrics = dict(_embedding_metrics)
    return {"configured": is_configured(), **metrics}


def get_embedding_coverage(get_db_fn) -> dict:
    """Report stored vector coverage for the corpora used by semantic search."""
    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 'faq' AS corpus, COUNT(*) AS total, COUNT(embedding) AS embedded
            FROM faq
            UNION ALL
            SELECT 'handbook_chunks' AS corpus, COUNT(*) AS total,
                   COUNT(embedding) AS embedded
            FROM handbook_chunks
            WHERE is_active = 1
        """)
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    corpora = {}
    for name, total, embedded in rows:
        total = int(total or 0)
        embedded = int(embedded or 0)
        corpora[name] = {
            "total": total,
            "embedded": embedded,
            "coverage_percent": round(embedded * 100 / total, 1) if total else 0.0,
            "semantic_ready": is_configured() and embedded > 0,
        }
    return corpora


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
        _record_embedding_result(False, "not_configured")
        return None

    cache_key = f"embed_{hash(text.strip().lower())}"
    if cache:
        cached = cache.get(cache_key)
        if cached:
            with _embedding_metrics_lock:
                _embedding_metrics["cache_hits"] += 1
            return cached

    for _ in range(len(GEMINI_API_KEYS)):
        api_key = get_next_gemini_key()
        if not api_key:
            break
        try:
            _record_embedding_api_call()
            resp = requests.post(
                EMBEDDING_URL,
                params={'key': api_key},
                json={'content': {'parts': [{'text': text}]}},
                timeout=EMBEDDING_TIMEOUT_SECONDS,
            )
            error_text = (resp.text or "").lower()
            auth_failure = (
                resp.status_code in (401, 403)
                or "invalid_api_key" in error_text
                or "invalid api key" in error_text
                or "api key not valid" in error_text
            )
            if auth_failure:
                mark_gemini_auth_failed(api_key)
                continue
            if resp.status_code == 429:
                mark_gemini_rate_limited(api_key)
                continue
            if resp.status_code != 200:
                logger.warning(f"Embedding API returned {resp.status_code}: {resp.text[:200]}")
                _record_embedding_result(False, f"http_{resp.status_code}")
                return None
            data = resp.json()
            values = data.get('embedding', {}).get('values')
            if not values:
                _record_embedding_result(False, "empty_response")
                return None
            mark_gemini_success(api_key)
            if cache:
                cache.set(cache_key, values)
            _record_embedding_result(True)
            return values
        except Exception as e:
            logger.error(f"get_embedding error: {e}")
            _record_embedding_result(False, type(e).__name__)
            return None
    logger.warning("Embedding API unavailable: all configured Gemini keys are rate-limited")
    _record_embedding_result(False, "no_usable_keys")
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
                             faq_id: Optional[int] = None,
                             public_uuid: Optional[str] = None) -> bool:
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


def embedding_text_for_chunk(section_title: str, content: str, keywords: str = "") -> str:
    """What gets embedded for a handbook chunk. Title + keywords + a
    generous slice of content -- long enough to capture the section's
    actual meaning (unlike the FAQ case, a handbook section is often a
    paragraph or more, so a title alone underrepresents it), capped so
    one section doesn't blow the embedding API's input size."""
    parts = [section_title or ""]
    if keywords:
        parts.append(keywords)
    if content:
        parts.append(content[:2000])
    return ". ".join(p.strip() for p in parts if p and p.strip())


def store_embedding_for_chunk(get_db_fn, section_title: str, content: str,
                               keywords: str = "",
                               chunk_id: Optional[int] = None) -> bool:
    """Best-effort: compute + store an embedding for one handbook_chunks
    row. Same "nice to have, never blocks the caller" contract as
    store_embedding_for_faq above."""
    if chunk_id is None:
        return False
    text = embedding_text_for_chunk(section_title, content, keywords)
    vector = get_embedding(text)
    if vector is None:
        return False

    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE handbook_chunks SET embedding = %s::jsonb WHERE id = %s",
                    (list(vector), chunk_id))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"store_embedding_for_chunk error: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def backfill_faq_embeddings(get_db_fn: Callable[[], Any], batch_size: int = 200,
                           limit: Optional[int] = None) -> int:
    """One-shot (or periodic) pass over every FAQ row that doesn't have an
    embedding yet and computes+stores one. This is the "let the AI
    process it instead of hand-tuning per FAQ" path for bulk FAQ imports
    (e.g. adding 500+ FAQs at once) -- run this once after a bulk import
    (or wire it into the import itself, see handbook_ingest.mass_insert_faqs)
    and every one of those FAQs immediately gets semantic-similarity
    matching with zero manual alias work. Safe to re-run any time; only
    touches rows where embedding IS NULL. Returns count actually updated.
    """
    if not is_configured():
        logger.warning("backfill_faq_embeddings: GEMINI_API_KEY not set, nothing to do")
        return 0
    conn = get_db_fn()
    cur = conn.cursor()
    try:
        q = "SELECT id, question, keywords FROM faq WHERE embedding IS NULL"
        if limit:
            q += f" LIMIT {int(limit)}"
        cur.execute(q)
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    updated = 0
    for i in range(0, len(rows), batch_size):
        for faq_id, question, keywords in rows[i:i + batch_size]:
            if store_embedding_for_faq(get_db_fn, question or "", keywords or "", faq_id=faq_id):
                updated += 1
    logger.info(f"backfill_faq_embeddings: updated {updated}/{len(rows)} FAQ rows")
    return updated


def backfill_chunk_embeddings(get_db_fn: Callable[[], Any], batch_size: int = 200,
                             limit: Optional[int] = None) -> int:
    """Same idea as backfill_faq_embeddings, for handbook_chunks. Run this
    once after any handbook re-ingest/import (or wire it into
    handbook_ingest.import_chunks_json / ingest_handbook directly) so
    every section immediately participates in semantic matching -- this
    is what lets hypothetical/paraphrased questions about a section find
    it by meaning even when the section's own wording shares no keywords
    with how a student happened to phrase the question."""
    if not is_configured():
        logger.warning("backfill_chunk_embeddings: GEMINI_API_KEY not set, nothing to do")
        return 0
    conn = get_db_fn()
    cur = conn.cursor()
    try:
        q = ("SELECT id, section_title, content, keywords FROM handbook_chunks "
             "WHERE is_active = 1 AND embedding IS NULL")
        if limit:
            q += f" LIMIT {int(limit)}"
        cur.execute(q)
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    updated = 0
    for i in range(0, len(rows), batch_size):
        for chunk_id, title, content, keywords in rows[i:i + batch_size]:
            if store_embedding_for_chunk(get_db_fn, title or "", content or "",
                                          keywords or "", chunk_id=chunk_id):
                updated += 1
    logger.info(f"backfill_chunk_embeddings: updated {updated}/{len(rows)} handbook chunks")
    return updated


def backfill_correction_embeddings(get_db_fn: Callable[[], Any], batch_size: int = 200,
                                  limit: Optional[int] = None) -> int:
    """Same idea again, for answer_corrections. Needed because corrections
    saved before services/corrections.py's paraphrase-matching feature
    existed have no embedding, and without one, find_similar_correction()
    can only paraphrase-match those older corrections by fuzzy string
    similarity (near-identical wording only) instead of by meaning. Run
    once after upgrading; every new correction gets its embedding
    automatically at save time (see corrections.save_correction) from
    then on."""
    if not is_configured():
        logger.warning("backfill_correction_embeddings: GEMINI_API_KEY not set, nothing to do")
        return 0
    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, question FROM answer_corrections WHERE embedding IS NULL")
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()
    if limit:
        rows = rows[:limit]

    from services.language import normalize_vernacular_spelling

    updated = 0
    for i in range(0, len(rows), batch_size):
        for correction_id, question in rows[i:i + batch_size]:
            vector = get_embedding(normalize_vernacular_spelling((question or "").lower()))
            if vector is None:
                continue
            conn2 = get_db_fn()
            cur2 = conn2.cursor()
            try:
                cur2.execute(
                    "UPDATE answer_corrections SET embedding = %s::jsonb WHERE id = %s",
                    (list(vector), correction_id),
                )
                conn2.commit()
                updated += 1
            except Exception as e:
                logger.error(f"backfill_correction_embeddings row {correction_id} failed: {e}")
                conn2.rollback()
            finally:
                cur2.close()
                conn2.close()
    logger.info(f"backfill_correction_embeddings: updated {updated}/{len(rows)} corrections")
    return updated