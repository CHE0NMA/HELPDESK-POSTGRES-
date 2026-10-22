"""Popular prompts backed by an actual FAQ or handbook answer."""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

_CONTEXT_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "please", "the",
    "there", "this", "to", "us", "was", "we", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
}

_FOLLOW_UP_START = re.compile(
    r"^\s*(and|also|what about|how about|then|why|what if|"
    r"how much more|what else|that one|those|them)\b",
    re.IGNORECASE,
)

_GENERIC_PROMPTS = {
    "hi", "hello", "hey", "yes", "no", "ok", "okay", "thanks",
    "thank you", "kumusta", "kamusta", "musta",
}


def has_question_context(question: str) -> bool:
    text = re.sub(r"\s+", " ", (question or "")).strip()
    if len(text) < 12 or text.casefold().strip(" ?!.") in _GENERIC_PROMPTS:
        return False
    if _FOLLOW_UP_START.search(text):
        return False
    content_words = {
        word.casefold()
        for word in re.findall(r"[A-Za-z0-9]+", text)
        if word.casefold() not in _CONTEXT_WORDS
    }
    return len(content_words) >= 2


def get_popular_knowledge_questions(
    get_db_fn: Callable[[], Any],
    limit: int = 10,
    days: Optional[int] = 30,
    connection: Optional[Any] = None,
) -> list[dict]:
    """Return only contextual questions answered from a FAQ or handbook.

    FAQ prompts use the canonical FAQ wording. Handbook prompts retain the
    student's wording, but contextless follow-ups and conversational turns
    are excluded.
    """
    safe_limit = max(1, min(int(limit), 100))
    safe_days = int(days) if days is not None else None
    if safe_days is not None:
        safe_days = max(1, min(safe_days, 3650))

    conn = connection or get_db_fn()
    owns_connection = connection is None
    cur = conn.cursor()
    try:
        time_filter = (
            "AND ch.created_at >= NOW() - (%s * INTERVAL '1 day')"
            if safe_days is not None
            else ""
        )
        query = f"""
            SELECT
                CASE
                    WHEN f.id IS NOT NULL THEN f.question
                    ELSE TRIM(ch.user_message)
                END AS question,
                COUNT(*) AS frequency,
                CASE
                    WHEN f.id IS NOT NULL THEN COALESCE(NULLIF(TRIM(f.topic), ''), 'FAQ')
                    ELSE 'Handbook'
                END AS category
            FROM chat_history ch
            LEFT JOIN faq f ON f.id = ch.faq_id
            WHERE (f.id IS NOT NULL OR LOWER(COALESCE(ch.category, '')) = 'handbook_query')
              AND ch.user_message IS NOT NULL
              AND LENGTH(TRIM(ch.user_message)) > 8
              {time_filter}
            GROUP BY
                CASE
                    WHEN f.id IS NOT NULL THEN f.question
                    ELSE TRIM(ch.user_message)
                END,
                CASE
                    WHEN f.id IS NOT NULL THEN COALESCE(NULLIF(TRIM(f.topic), ''), 'FAQ')
                    ELSE 'Handbook'
                END
            ORDER BY frequency DESC
            LIMIT %s
        """
        params = (safe_days, safe_limit * 5) if safe_days is not None else (safe_limit * 5,)
        cur.execute(query, params)
        items = []
        for question, frequency, category in cur.fetchall():
            if not has_question_context(question):
                continue
            items.append({
                "question": question.strip(),
                "user_message": question.strip(),
                "frequency": frequency,
                "count": frequency,
                "category": category or "FAQ",
            })
            if len(items) >= safe_limit:
                break
        return items
    finally:
        cur.close()
        if owns_connection:
            conn.close()
