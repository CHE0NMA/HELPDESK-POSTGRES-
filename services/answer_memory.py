"""
services/answer_memory.py — verified-answer cache + feedback records.

Student ratings are feedback signals, not proof that an answer is correct.
Cached answers are returned only when explicitly marked verified; reviewed
corrections remain the normal way to teach a durable answer without code edits.

Feedback lands in answer_feedback for analysis, and negative feedback can
remove a verified cached answer for the same question and language.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger("pac_helpdesk.answer_memory")


def normalize_question(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", "", q)
    return q[:300]


def get_cached_answer(get_db_fn, question: str, detected_lang: str = "english") -> Optional[dict]:
    """Return a previously verified exact-question answer, or None.

    Unverified legacy rows are deliberately ignored; a thumbs-up alone is
    not enough to make an answer authoritative.
    """
    if not question:
        return None
    qn = normalize_question(question)
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT answer_text, id, faq_id FROM answer_success_cache
                WHERE question_norm = %s AND detected_lang = %s
                  AND is_verified = TRUE
                LIMIT 1
                """,
                (qn, detected_lang),
            )
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                """
                UPDATE answer_success_cache
                SET hit_count = hit_count + 1, last_hit_at = NOW()
                WHERE id = %s
                """,
                (row[1],),
            )
            conn.commit()
            return {"answer": row[0], "faq_id": row[2]}
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"get_cached_answer: {e}")
        return None


def store_successful_answer(
    get_db_fn,
    question: str,
    answer_text: str,
    detected_lang: str = "english",
    source_type: str = "handbook",
    chunk_ids: Optional[List[int]] = None,
    faq_id: Optional[int] = None,
    is_verified: bool = False,
) -> bool:
    if not question or not answer_text or len(answer_text) < 20:
        return False
    qn = normalize_question(question)
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO answer_success_cache
                    (question_norm, detected_lang, source_type, chunk_ids, faq_id,
                     is_verified, answer_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (question_norm, detected_lang) DO UPDATE SET
                    answer_text = EXCLUDED.answer_text,
                    source_type = EXCLUDED.source_type,
                    chunk_ids = EXCLUDED.chunk_ids,
                    faq_id = EXCLUDED.faq_id,
                    is_verified = EXCLUDED.is_verified,
                    hit_count = answer_success_cache.hit_count + 1,
                    last_hit_at = NOW()
                """,
                (
                    qn,
                    detected_lang,
                    source_type,
                    chunk_ids or [],
                    faq_id,
                    is_verified,
                    answer_text[:4000],
                ),
            )
            conn.commit()
            return True
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"store_successful_answer: {e}")
        return False


def record_feedback(
    get_db_fn,
    question: str,
    rating: str,
    correction: str = "",
    source_type: Optional[str] = None,
    chunk_ids: Optional[List[int]] = None,
    faq_id: Optional[int] = None,
    session_id: Optional[str] = None,
    detected_lang: str = "english",
) -> bool:
    if not question or rating not in ("thumbs_up", "thumbs_down"):
        return False
    qn = normalize_question(question)
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO answer_feedback
                    (question, question_norm, rating, correction, source_type,
                     chunk_ids, faq_id, detected_language, session_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    question[:500],
                    qn,
                    rating,
                    (correction or "")[:2000] or None,
                    source_type,
                    chunk_ids or [],
                    faq_id,
                    detected_lang,
                    session_id,
                ),
            )

            if rating == "thumbs_down":
                cur.execute(
                    "DELETE FROM answer_success_cache "
                    "WHERE question_norm = %s AND detected_lang = %s",
                    (qn, detected_lang),
                )
            conn.commit()
            return True
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.error(f"record_feedback: {e}")
        return False
