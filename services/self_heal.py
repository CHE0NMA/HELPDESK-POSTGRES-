"""
services/self_heal.py — turn repeated misses into ready-to-approve drafts.

WHY
---
services/alerts.py's batch_process_unanswered() and
services/topic_aliases.py's suggest_aliases_from_unanswered() only notice
that a miss is FREQUENT (asked >= N times) -- neither actually tries to
answer it. A question asked 10 times just sits as a "someone should look
at this" alert until a human manually finds the right section/FAQ and
writes the fix by hand -- the exact manual-maintenance loop this whole
change set is trying to get you out of.

This module closes that loop: for each frequently-missed question, it
runs the SAME retrieval + AI-router + grounding + self-check pipeline
already used for live traffic (services.handbook.answer_handbook_question,
services.answer_engine's FAQ path) against it. Because that pipeline now
generalizes to new phrasings on its own (see services/ai_retrieval.py),
a lot of old misses will simply succeed on retry -- especially anything
that was failing because of the routing/grounding bugs fixed in this
change set, or because content was added/ingested after the question was
first logged as unanswered. When the pipeline produces a genuinely
grounded, self-check-passed answer, it's saved as a DRAFT.

Drafts are NEVER auto-published. Promoting a draft into
services.corrections (which is treated as unverified admin ground truth
-- see that module's docstring) stays a deliberate one-click admin
action via approve_draft(). This module only removes the "find it and
write it by hand" step, not the "a human signs off before it's
authoritative" step.

Wire run_self_heal_pass() into your existing scheduler (app.py already
has an APScheduler block for batch_process_unanswered/check_stale_faqs/
etc. -- add this alongside them) or trigger it on-demand from the admin
dashboard next to the existing "Suggest aliases" action.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("pac_helpdesk.self_heal")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pending_correction_drafts (
    id              SERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    question_norm   TEXT NOT NULL,
    detected_language TEXT NOT NULL DEFAULT 'english',
    draft_answer    TEXT NOT NULL,
    source_type     TEXT NOT NULL,        -- 'handbook' | 'faq'
    source_ref      TEXT,                 -- section title / faq id, for admin context
    times_asked     INT NOT NULL DEFAULT 1,
    student_suggestions TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | dismissed
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at     TIMESTAMPTZ,
    reviewed_by     TEXT
);
ALTER TABLE pending_correction_drafts
    ADD COLUMN IF NOT EXISTS detected_language TEXT NOT NULL DEFAULT 'english';
ALTER TABLE pending_correction_drafts
    DROP CONSTRAINT IF EXISTS pending_correction_drafts_q_key;
CREATE UNIQUE INDEX IF NOT EXISTS idx_pcd_question_language
    ON pending_correction_drafts (question_norm, detected_language);
ALTER TABLE pending_correction_drafts
    ADD COLUMN IF NOT EXISTS student_suggestions TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];
CREATE INDEX IF NOT EXISTS idx_pcd_status ON pending_correction_drafts (status, created_at DESC);
"""

_TABLE_READY = False


def ensure_self_heal_schema(get_db_fn) -> bool:
    global _TABLE_READY
    if _TABLE_READY:
        return True
    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute(SCHEMA_SQL)
        conn.commit()
        _TABLE_READY = True
    except Exception as e:
        conn.rollback()
        logger.error(f"ensure_self_heal_schema: {e}")
    finally:
        cur.close()
        conn.close()
    return _TABLE_READY


def _normalize(q: str) -> str:
    from services.corrections import normalize_question
    return normalize_question(q)


def _save_draft(get_db_fn, question: str, draft_answer: str, source_type: str,
                 source_ref: Optional[str], times_asked: int,
                 detected_language: str = "english") -> bool:
    qn = _normalize(question)
    if not qn or not draft_answer or len(draft_answer) < 20:
        return False
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO pending_correction_drafts
                    (question, question_norm, detected_language, draft_answer,
                     source_type, source_ref, times_asked)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (question_norm, detected_language) DO UPDATE SET
                    draft_answer = EXCLUDED.draft_answer,
                    source_type = EXCLUDED.source_type,
                    source_ref = EXCLUDED.source_ref,
                    times_asked = GREATEST(pending_correction_drafts.times_asked, EXCLUDED.times_asked)
                WHERE pending_correction_drafts.status = 'pending'
                """,
                (
                    question[:500],
                    qn,
                    detected_language,
                    draft_answer[:4000],
                    source_type,
                    source_ref,
                    times_asked,
                ),
            )
            conn.commit()
            return True
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.error(f"_save_draft failed: {e}")
        return False


def save_feedback_draft(
    get_db_fn,
    question: str,
    suggested_answer: str,
    source_ref: Optional[str] = None,
    detected_language: str = "english",
) -> bool:
    """Queue a student's suggested correction for an admin to verify.

    Feedback never overwrites an existing draft or publishes an answer.
    """
    qn = _normalize(question or "")
    answer = (suggested_answer or "").strip()
    if not qn or len(answer) < 20 or not ensure_self_heal_schema(get_db_fn):
        return False

    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO pending_correction_drafts
                (question, question_norm, detected_language, draft_answer,
                 source_type, source_ref, times_asked, student_suggestions)
            VALUES (%s, %s, %s, %s, 'student_feedback', %s, 1, ARRAY[]::TEXT[])
            ON CONFLICT (question_norm, detected_language) DO UPDATE SET
                times_asked = pending_correction_drafts.times_asked + 1,
                student_suggestions = CASE
                    WHEN EXCLUDED.draft_answer = ANY(pending_correction_drafts.student_suggestions)
                        THEN pending_correction_drafts.student_suggestions
                    WHEN COALESCE(array_length(pending_correction_drafts.student_suggestions, 1), 0) < 5
                        THEN array_append(pending_correction_drafts.student_suggestions, EXCLUDED.draft_answer)
                    ELSE pending_correction_drafts.student_suggestions
                END
            WHERE pending_correction_drafts.status = 'pending'
            """,
            (
                question[:500],
                qn,
                detected_language,
                answer[:4000],
                f"Student suggestion - verify before approving"
                f"{': ' + source_ref if source_ref else ''}"[:500],
            ),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception as e:
        conn.rollback()
        logger.error(f"save_feedback_draft failed: {e}")
        return False
    finally:
        cur.close()
        conn.close()


def run_self_heal_pass(get_db_fn, min_count: int = 2, limit: int = 50) -> int:
    """Scan frequently-missed questions (unanswered_questions, grouped
    like alerts.batch_process_unanswered already does) and try the live
    answer pipeline against each. Saves a draft for any that now get a
    real, grounded, self-check-passed answer. Returns drafts created/
    updated. Safe to re-run repeatedly -- skips anything that already has
    an admin correction, and upserts on question so a re-run just
    refreshes times_asked/draft text rather than duplicating rows.
    """
    if not ensure_self_heal_schema(get_db_fn):
        return 0

    from services.corrections import get_correction
    from services import handbook as handbook_service

    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT LOWER(TRIM(question)) AS q,
                   COALESCE(detected_language, 'english') AS detected_language,
                   COUNT(*) AS cnt,
                   MAX(question) AS sample
            FROM unanswered_questions
            WHERE question IS NOT NULL AND LENGTH(TRIM(question)) > 5
            GROUP BY LOWER(TRIM(question)), COALESCE(detected_language, 'english')
            HAVING COUNT(*) >= %s
            ORDER BY cnt DESC
            LIMIT %s
            """,
            (min_count, limit),
        )
        rows = cur.fetchall()
    except Exception as e:
        logger.error(f"run_self_heal_pass query failed: {e}")
        return 0
    finally:
        cur.close()
        conn.close()

    created = 0
    for _, detected_language, count, sample in rows:
        try:
            if get_correction(get_db_fn, sample):
                continue  # already has an admin correction -- nothing to draft
        except Exception:
            pass

        try:
            answer = handbook_service.answer_handbook_question(
                sample, detected_lang=detected_language
            )
            if answer and 'CANNOT_ANSWER' not in answer.upper():
                chunks = handbook_service.get_top_handbook_chunks(
                    sample, max_chunks=1, detected_lang=detected_language
                )
                source_ref = chunks[0].get('section_title') if chunks else None
                if _save_draft(
                    get_db_fn,
                    sample,
                    answer,
                    'handbook',
                    source_ref,
                    count,
                    detected_language=detected_language,
                ):
                    created += 1
                continue
        except Exception as e:
            logger.warning(f"self_heal handbook attempt failed for {sample[:60]!r}: {e}")

        try:
            faq_answer, faq_ref = _try_faq_answer(sample, detected_language)
            if faq_answer and _save_draft(
                get_db_fn,
                sample,
                faq_answer,
                'faq',
                faq_ref,
                count,
                detected_language=detected_language,
            ):
                created += 1
        except Exception as e:
            logger.warning(f"self_heal faq attempt failed for {sample[:60]!r}: {e}")

    logger.info(f"run_self_heal_pass: {created} draft(s) created/updated from {len(rows)} candidate(s)")
    return created


def _try_faq_answer(question: str, detected_lang: str = "english"):
    """Best-effort FAQ attempt using answer_engine directly (not
    services.faq's find_answer, which has session/chat-history side
    effects that don't belong in an offline batch job). Mirrors the same
    fuzzy+embedding+AI-router+grounding pipeline live traffic uses.

    Reuses services.faq's own _get_all_faqs() (confirmed against the real
    schema: `faq` has no `is_active` column at all -- every row is
    considered live, which is also why _get_all_faqs() itself has no
    WHERE clause) instead of duplicating that query here, so this can
    never drift out of sync with the real table again."""
    from services.embeddings import get_embedding
    from services.faq import _get_all_faqs, _get_faq_candidates

    try:
        all_faqs = _get_all_faqs()
    except Exception as e:
        logger.debug(f"_try_faq_answer: could not load FAQ table ({e})")
        return None, None

    if not all_faqs:
        return None, None

    q_embedding = get_embedding(question)
    candidates = _get_faq_candidates(
        question,
        detected_lang,
        all_faqs,
        query_embedding=q_embedding,
    )
    from services import answer_engine as engine
    if not candidates or candidates[0]['combined'] < engine.FUZZY_MIN_SCORE:
        return None, None
    if engine.faq_should_disambiguate(candidates, user_question=question):
        return None, None  # genuinely ambiguous -- not a safe auto-draft

    from ai_providers import ai_cache, llm_chat
    ai_ans, matched = engine.smart_extract_answer(
        user_question=question, candidates=candidates, detected_lang=detected_lang,
        llm_chat_fn=llm_chat, ai_cache=ai_cache,
    )
    if not ai_ans:
        return None, None
    best_row = (matched or candidates[0])['faq']
    return ai_ans, best_row[1]  # (answer text, matched FAQ's own question, for admin context)


def list_pending_drafts(get_db_fn) -> List[dict]:
    if not ensure_self_heal_schema(get_db_fn):
        return []
    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
                 SELECT id, question, draft_answer, source_type, source_ref, times_asked,
                     created_at, student_suggestions, detected_language
            FROM pending_correction_drafts
            WHERE status = 'pending'
            ORDER BY times_asked DESC, created_at DESC
            """
        )
        return [{
            "id": r[0], "question": r[1], "draft_answer": r[2], "source_type": r[3],
            "source_ref": r[4], "times_asked": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
            "student_suggestions": list(r[7] or []),
            "detected_language": r[8] or "english",
        } for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"list_pending_drafts failed: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def approve_draft(get_db_fn, draft_id: int, reviewed_by: Optional[str] = None,
                  approved_answer: Optional[str] = None) -> bool:
    """Promote a draft into services.corrections' authoritative table.
    This is the deliberate "trust this now" step -- it always stays a
    human action, never automatic."""
    from services.corrections import save_correction

    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT question, draft_answer FROM pending_correction_drafts "
            "WHERE id = %s AND status = 'pending'",
            (draft_id,),
        )
        row = cur.fetchone()
        if not row:
            return False
        question, draft_answer = row
    except Exception as e:
        logger.error(f"approve_draft lookup failed: {e}")
        return False
    finally:
        cur.close()
        conn.close()

    if approved_answer is not None:
        draft_answer = approved_answer.strip()
        if len(draft_answer) < 20:
            return False

    ok = save_correction(get_db_fn, question, draft_answer, created_by=reviewed_by or 'self_heal')
    if not ok:
        return False

    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE pending_correction_drafts SET status = 'approved', "
            "reviewed_at = NOW(), reviewed_by = %s WHERE id = %s",
            (reviewed_by, draft_id),
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"approve_draft status update failed: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()


def dismiss_draft(get_db_fn, draft_id: int, reviewed_by: Optional[str] = None) -> bool:
    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE pending_correction_drafts SET status = 'dismissed', "
            "reviewed_at = NOW(), reviewed_by = %s WHERE id = %s",
            (reviewed_by, draft_id),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception as e:
        logger.error(f"dismiss_draft failed: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()