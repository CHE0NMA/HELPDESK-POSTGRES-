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
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger("pac_helpdesk.corrections")

_TABLE_READY = False


def normalize_question(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", "", q)
    return q[:300]


def _ensure_table(get_db_fn) -> bool:
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
            conn.commit()
            _TABLE_READY = True
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"answer_corrections ensure_table skipped: {e}")
    return _TABLE_READY


def get_correction(get_db_fn, question: str) -> Optional[dict]:
    """Return {'id', 'question', 'answer', 'faq_id'} for an admin-verified
    correction matching this exact (normalized) question, or None."""
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
            if not row:
                return None
            cur.execute(
                "UPDATE answer_corrections SET hit_count = hit_count + 1 WHERE id = %s",
                (row[0],),
            )
            conn.commit()
            return {"id": row[0], "question": row[1], "answer": row[2], "faq_id": row[3]}
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"get_correction failed: {e}")
        return None


def save_correction(get_db_fn, question: str, answer: str, created_by: str = None,
                     faq_id: int = None, source_report_id: str = None) -> bool:
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
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO answer_corrections
                    (question_norm, question, answer, faq_id, created_by, source_report_id, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (question_norm) DO UPDATE SET
                    question = EXCLUDED.question,
                    answer = EXCLUDED.answer,
                    faq_id = EXCLUDED.faq_id,
                    created_by = EXCLUDED.created_by,
                    source_report_id = EXCLUDED.source_report_id,
                    updated_at = NOW()
            """, (qn, question[:500], answer[:4000], faq_id, created_by, source_report_id))
            conn.commit()
            return True
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.error(f"save_correction failed: {e}")
        return False


def list_corrections(get_db_fn) -> List[dict]:
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


def delete_correction(get_db_fn, correction_id: int) -> bool:
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