"""services/corrections.py — admin-verified answer overrides.

This is the "fix it directly" mechanism: when a student reports a wrong
answer, an admin can attach the *correct* answer to that exact question
right from the dashboard. From then on, that question — asked by anyone,
any time — is answered from this table first, before FAQ matching,
handbook retrieval, date logic, or the answer_success_cache. Unlike
answer_success_cache (thumbs-up driven, TTL-free but can be silently
evicted by a thumbs-down on the same question, and was never meant to be
authoritative), a correction here is permanent, admin-authored ground
truth — it never expires and is never auto-deleted by student feedback.

The stored text is treated exactly like a trusted FAQ answer: find_answer()
still runs it through the same extraction+grounding+language pipeline
(services/answer_engine._extract_from_answer) so the reply is reworded
naturally per question phrasing/language, but every fact in it must trace
back to the admin's own corrected text — the LLM cannot add anything the
correction didn't say.

Table is created lazily on first use (same pattern as
services/conversation.py's conversation_state table) so no separate
schema-migration wiring is required at app startup.

PARAPHRASE/LANGUAGE MATCHING
------------------------------
get_correction() used to only match the EXACT normalized question text —
an admin fixing "what would happen if I bully someone" did NOT
automatically fix "ano ang mangyayari kung ako mang-bully" or "unsay
silot sa bully", even though they mean the same thing, because the
literal strings differ. That meant every paraphrase/language variant of
a wrong question had to be individually reported and individually fixed
before the system actually "learned" from it -- the opposite of what an
admin doing this work would expect. Fixed by falling back to fuzzy +
semantic (embedding) matching against every stored correction's question
when there's no exact hit, with a deliberately HIGH bar (corrections are
authoritative and skip every other check, so a wrong match here is worse
than a wrong FAQ match) -- see find_similar_correction() below.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, List, Optional

from services.language import expand_query, normalize_vernacular_spelling
from utils.text import fuzzy_score

logger = logging.getLogger("pac_helpdesk.corrections")

_TABLE_READY = False

FUZZY_MATCH_THRESHOLD = 0.90
SEMANTIC_MATCH_THRESHOLD = 0.90


def normalize_question(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", "", q)
    return q[:300]


def _ensure_table(get_db_fn: Callable[[], Any]) -> bool:
    global _TABLE_READY
    if _TABLE_READY:
        return True
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS answer_corrections (
                    id SERIAL PRIMARY KEY,
                    question_norm TEXT NOT NULL UNIQUE,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    faq_id INTEGER,
                    created_by VARCHAR(100),
                    source_report_id VARCHAR(40),
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
            # Best-effort -- older deployments already have this table
            # without the column. Same "nice to have, never block" pattern
            # as services/embeddings.py.
            cur.execute("ALTER TABLE answer_corrections ADD COLUMN IF NOT EXISTS embedding JSONB")
            conn.commit()
            _TABLE_READY = True
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"answer_corrections ensure_table skipped: {e}")
    return _TABLE_READY


def _bump_hit(get_db_fn: Callable[[], Any], correction_id: int) -> None:
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE answer_corrections SET hit_count = hit_count + 1 WHERE id = %s",
                (correction_id,),
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"_bump_hit failed: {e}")


def find_similar_correction(get_db_fn: Callable[[], Any], question: str) -> Optional[dict]:
    """Fuzzy + semantic fallback used when no EXACT normalized match
    exists (see get_correction below). Scans every stored correction --
    fine at the scale this table actually runs at (admin-authored fixes,
    not a bulk FAQ table) -- and returns the best match only if it clears
    FUZZY_MATCH_THRESHOLD or SEMANTIC_MATCH_THRESHOLD. Question text is
    run through the same vernacular-spelling-fix pipeline used everywhere
    else (services.language.expand_query) before comparing, so a
    misspelled or hypothetically-framed repeat of an already-corrected
    question still finds it.
    """
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, question, answer, faq_id, embedding FROM answer_corrections")
            rows = cur.fetchall()
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"find_similar_correction scan failed: {e}")
        return None

    if not rows:
        return None

    q_norm = normalize_vernacular_spelling((question or '').lower())
    q_expanded = expand_query(question or '')

    q_embedding = None
    try:
        from services.embeddings import get_embedding, cosine_similarity
        q_embedding = get_embedding(q_norm)
    except Exception:
        cosine_similarity = None  # noqa: F841

    best_row = None
    best_score = 0.0
    for row in rows:
        cid, stored_q, answer, faq_id, stored_embedding = row
        stored_norm = normalize_vernacular_spelling((stored_q or '').lower())

        fscore = max(
            fuzzy_score(q_norm, stored_norm),
            fuzzy_score(q_expanded, stored_norm),
        )
        sscore = 0.0
        if q_embedding and stored_embedding and cosine_similarity is not None:
            try:
                sscore = max(0.0, cosine_similarity(q_embedding, stored_embedding))
            except Exception:
                sscore = 0.0

        score = max(fscore, sscore)
        if (fscore >= FUZZY_MATCH_THRESHOLD or sscore >= SEMANTIC_MATCH_THRESHOLD) and score > best_score:
            best_score = score
            best_row = (cid, stored_q, answer, faq_id)

    if not best_row:
        return None
    logger.info(
        f"find_similar_correction: paraphrase match (score={best_score:.2f}) "
        f"q={question[:80]!r} -> stored={best_row[1][:80]!r}"
    )
    _bump_hit(get_db_fn, best_row[0])
    return {"id": best_row[0], "question": best_row[1], "answer": best_row[2], "faq_id": best_row[3]}


def get_correction(get_db_fn: Callable[[], Any], question: str) -> Optional[dict]:
    """Return {'id', 'question', 'answer', 'faq_id'} for an admin-verified
    correction matching this question, or None. Tries an exact
    (normalized) match first -- fast, unambiguous -- then falls back to
    find_similar_correction() for paraphrases/misspellings/hypothetical
    framing/other languages of an already-corrected question."""
    if not question or not _ensure_table(get_db_fn):
        return None
    qn = normalize_question(question)
    if not qn:
        return None
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, question, answer, faq_id FROM answer_corrections
                WHERE question_norm = %s
                LIMIT 1
            """, (qn,))
            row = cur.fetchone()
        finally:
            cur.close()
            conn.close()
        if row:
            _bump_hit(get_db_fn, row[0])
            return {"id": row[0], "question": row[1], "answer": row[2], "faq_id": row[3]}
    except Exception as e:
        logger.debug(f"get_correction failed: {e}")
        return None

    return find_similar_correction(get_db_fn, question)


def get_exact_correction(
    get_db_fn: Callable[[], Any],
    question: str,
) -> Optional[dict]:
    """Look up an approved correction without invoking fuzzy/semantic search."""
    if not question or not _ensure_table(get_db_fn):
        return None
    normalized_full = re.sub(r"\s+", " ", question.strip().lower())
    normalized_full = re.sub(r"[^\w\s]", "", normalized_full)
    if len(normalized_full) > 300:
        return None
    qn = normalize_question(question)
    if not qn:
        return None
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT id, question, answer, faq_id FROM answer_corrections
                WHERE question_norm = %s
                LIMIT 1
                """,
                (qn,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
            conn.close()
        if not row:
            return None
        _bump_hit(get_db_fn, row[0])
        return {
            "id": row[0],
            "question": row[1],
            "answer": row[2],
            "faq_id": row[3],
        }
    except Exception as e:
        logger.error(f"get_exact_correction failed: {e}")
        return None


def save_correction(get_db_fn: Callable[[], Any], question: str, answer: str,
                    created_by: Optional[str] = None, faq_id: Optional[int] = None,
                    source_report_id: Optional[str] = None) -> bool:
    """Create or overwrite the correction for a question. Overwriting is
    intentional — if an admin corrects the same question twice, the
    newer correction wins."""
    if not question or not answer:
        return False
    if not _ensure_table(get_db_fn):
        return False
    qn = normalize_question(question)
    if not qn:
        return False

    embedding = None
    try:
        from services.embeddings import get_embedding
        embedding = get_embedding(normalize_vernacular_spelling(question.lower()))
    except Exception as e:
        logger.debug(f"save_correction: embedding skipped: {e}")

    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO answer_corrections
                    (question_norm, question, answer, faq_id, created_by, source_report_id,
                     embedding, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
                ON CONFLICT (question_norm) DO UPDATE SET
                    question = EXCLUDED.question,
                    answer = EXCLUDED.answer,
                    faq_id = EXCLUDED.faq_id,
                    created_by = EXCLUDED.created_by,
                    source_report_id = EXCLUDED.source_report_id,
                    embedding = COALESCE(EXCLUDED.embedding, answer_corrections.embedding),
                    updated_at = NOW()
            """, (qn, question[:500], answer[:4000], faq_id, created_by, source_report_id,
                  json.dumps(list(embedding)) if embedding else None))
            conn.commit()
            return True
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.error(f"save_correction failed: {e}")
        return False


def list_corrections(get_db_fn: Callable[[], Any]) -> List[dict]:
    if not _ensure_table(get_db_fn):
        return []
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, question, answer, faq_id, created_by, hit_count, created_at, updated_at
                FROM answer_corrections ORDER BY updated_at DESC
            """)
            rows = cur.fetchall()
            return [{
                "id": r[0], "question": r[1], "answer": r[2], "faq_id": r[3],
                "created_by": r[4], "hit_count": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
                "updated_at": r[7].isoformat() if r[7] else None,
            } for r in rows]
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"list_corrections failed: {e}")
        return []


def delete_correction(get_db_fn: Callable[[], Any], correction_id: int) -> bool:
    if not _ensure_table(get_db_fn):
        return False
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM answer_corrections WHERE id = %s", (correction_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.error(f"delete_correction failed: {e}")
        return False