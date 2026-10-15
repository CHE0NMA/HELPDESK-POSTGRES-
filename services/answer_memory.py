"""
services/answer_memory.py — successful-answer cache + feedback loop.

When a grounded answer is thumbs-up'd (or repeatedly served without
complaint), we store question_norm → answer so identical / near-identical
asks skip LLM extraction next time.

Feedback (thumbs down + optional correction) lands in answer_feedback for
admin review and for suggest_aliases_from_unanswered-style jobs.
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
    """Return {'answer': str, 'faq_id': int|None} for a previously
    successful grounded answer, or None.

    BUG FIX: used to return a bare answer string, so callers had no way
    to know the cached answer's original faq_id -- services/faq.py's
    find_answer() was hardcoding faq_id=None on every cache hit, which
    silently starved promote_high_frequency_faqs() of usage data for
    exactly the FAQs popular enough to be getting cached. Now returns
    faq_id alongside the text so it can keep flowing into chat_history.
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
                    (question_norm, detected_lang, source_type, chunk_ids, faq_id, answer_text)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (question_norm, detected_lang) DO UPDATE SET
                    answer_text = EXCLUDED.answer_text,
                    source_type = EXCLUDED.source_type,
                    chunk_ids = EXCLUDED.chunk_ids,
                    faq_id = EXCLUDED.faq_id,
                    hit_count = answer_success_cache.hit_count + 1,
                    last_hit_at = NOW()
                """,
                (
                    qn,
                    detected_lang,
                    source_type,
                    chunk_ids or [],
                    faq_id,
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
                     chunk_ids, faq_id, session_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    question[:500],
                    qn,
                    rating,
                    (correction or "")[:2000] or None,
                    source_type,
                    chunk_ids or [],
                    faq_id,
                    session_id,
                ),
            )

            if rating == "thumbs_down":
                cur.execute(
                    "DELETE FROM answer_success_cache WHERE question_norm = %s",
                    (qn,),
                )
            conn.commit()
            return True
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.error(f"record_feedback: {e}")
        return False


def promote_from_thumbs_up(
    get_db_fn,
    question: str,
    answer_text: str,
    detected_lang: str = "english",
    source_type: str = "handbook",
) -> None:
    """Call when the student gives thumbs_up — lock in the answer."""
    store_successful_answer(
        get_db_fn, question, answer_text, detected_lang=detected_lang, source_type=source_type
    )
    record_feedback(get_db_fn, question, "thumbs_up", source_type=source_type)