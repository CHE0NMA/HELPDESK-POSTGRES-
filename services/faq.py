import logging
import re
import time
from typing import List, Optional, Tuple

from ai_providers import ai_cache, circuit_breaker, faq_cache, llm_chat
from database import get_db
from services import answer_engine as engine
from services import dates as date_service
from services import handbook as handbook_service
from services import language as lang_service
from services import nlg
from services.conversation import (
    ConversationContext,
    get_conversation_context,
    resolve_disambiguation_choice,
)
from services.embeddings import get_embedding

from utils.text import normalize_input

logger = logging.getLogger("pac_helpdesk.faq")

# Split multi-intent questions so each clause can be answered.
# Handles: "What is X and where is Y?", "A? B?", "X, and who founded it?"
COMPOUND_SPLIT_RE = re.compile(
    r'(?:'
    r'\s+(?:and|at saka|saka|tas|ug|dayon|unya|,\s*and)\s+'
    r'(?=what|how|when|where|who|which|is|are|can|could|do|does|will|what\'s|whats|'
    r'paano|ano|kailan|saan|sino|magkano|ilan|'
    r'unsa|kanus-a|asa|kinsa|pila)\b'
    r'|\?\s+(?=what|how|when|where|who|which|is|are|can|do|does|will|'
    r'paano|ano|kailan|saan|sino)'
    r')',
    re.IGNORECASE
)

CROSS_CHECK_HANDBOOK_BAR = 0.34
CROSS_CHECK_FAQ_BAR = engine.FUZZY_MIN_SCORE + 0.05

# Phrases that mean the follow-up LLM is hedging / admitting it doesn't
# actually know, rather than giving a real answer. When we see one of
# these we should NOT treat it as a final answer — we should fall
# through to the real handbook/FAQ lookup instead, since the model is
# only reasoning over the last few chat turns and has no idea whether
# the handbook actually covers the topic.
_NON_ANSWER_RE = re.compile(
    r"\b(don'?t have (?:the |that |this |complete |full |specific |confirmed )*"
    r"(?:information|details|data|specifics|answer)|"
    r"didn'?t have (?:the |complete |full )*(?:information|details|data)|"
    r"haven'?t (discussed|covered|talked about)|"
    r"didn'?t (cover|discuss|talk about)|"
    r"wasn'?t covered|(not|isn'?t) something we'?ve covered|i don'?t know|"
    r"not sure (what|about)|no information (on|about)|not readily available|"
    r"i('| a)m not able to find|i can'?t find (that|this)|"
    r"i (only|just) (mentioned|found|got to|had)\b|"
    r"let me try to find|let me look that up|i'?d be happy to (try and )?find|"
    r"i'?d need to check (the handbook|that|this)|need to check the handbook|"
    r"give me a (moment|sec|second|minute)|"
    r"check(ing)? page \d+ (of|for) the relevant section|"
    r"couldn'?t find any specific information)\b",
    re.IGNORECASE,
)


def _looks_like_non_answer(text: str) -> bool:
    return bool(_NON_ANSWER_RE.search(text or ''))


def split_compound_question(question: str) -> List[str]:
    """Split multi-part questions into standalone clauses.

    Handbook topics are included — previously we skipped compound handling
    when is_handbook_question was true, so "full name and campuses" only
    answered one half (often the wrong half).
    """
    q = (question or '').strip()
    if len(q.split()) < 6:
        return [q]
    parts = COMPOUND_SPLIT_RE.split(q)
    parts = [p.strip(' ?.,') for p in parts if p and p.strip(' ?.,')]
    # Keep only parts that still look like a real question fragment
    cleaned = []
    for p in parts:
        if len(p.split()) < 2:
            continue
        cleaned.append(p if p.endswith('?') else p)
    return cleaned if len(cleaned) > 1 else [q]


def _answer_followup_via_search(question: str, ctx, detected_lang: str):
    """A follow-up like "how many demerits is a first degree offense" after
    talking about third-degree offenses isn't answerable from chat history
    alone — the bot needs to actually go look up "first degree offense" in
    the handbook, the same as it would for a brand new question. This
    resolves the (possibly pronoun-y / elliptical) follow-up into a
    standalone question using the conversation so far, then runs THAT
    through the normal handbook/FAQ search — not just a second glance at
    whatever chunks the previous turn already used.

    Returns (answer, topic, intent) — all None if nothing was found, in
    which case the caller should fall back to answer_followup_with_ai.
    """
    from services.conversation import build_context_snapshot

    context_snapshot = build_context_snapshot(ctx)
    resolved_q = question
    if context_snapshot and not circuit_breaker.is_open():
        resolved_q = _resolve_standalone_question(question, context_snapshot)

    # Check both the resolved and original phrasing — the rewrite usually
    # helps ("it" -> "a second-degree offense"), but shouldn't be trusted
    # blindly if it drifted off-topic.
    looks_handbooky = (
        handbook_service.is_handbook_question(resolved_q)
        or handbook_service.is_handbook_question(question)
    )
    if looks_handbooky:
        hb_ans = handbook_service.answer_handbook_question(resolved_q, detected_lang, ctx=ctx)
        if hb_ans:
            return hb_ans, 'Handbook', 'handbook_query'

    faq_result = _lookup_faq_answer(resolved_q, detected_lang)
    if faq_result and faq_result[0]:
        answer, topic, intent = faq_result
        return answer, (topic or 'General'), intent

    return None, None, None


def answer_followup_with_ai(user_question: str, context_summary: str = "",
                             conversation_history: List[dict] = None,
                             detected_lang: str = 'english',
                             last_faq_answer: str = None,
                             last_faq_question: str = None) -> Optional[str]:
    if circuit_breaker.is_open():
        return None

    lang_config = lang_service.LANGUAGE_CONFIG.get(
        detected_lang, lang_service.LANGUAGE_CONFIG['english']
    )

    history_text = ""
    if conversation_history:
        lines = []
        for ex in conversation_history[-3:]:
            lines.append(f"Student: {ex.get('user', '')}")
            lines.append(f"Assistant: {ex.get('bot', '')}")
        history_text = "\n".join(lines)

    prompt = f"""{nlg.now_context()}

{lang_config['system_prompt']}

{lang_config['instruction']}

{context_summary}

{history_text}

{f"Previous FAQ asked: {last_faq_question}" if last_faq_question else ""}
{f"Previous FAQ answer: {last_faq_answer[:300]}" if last_faq_answer else ""}

NOW the student asks: "{user_question}"

YOUR TASK: Answer this follow-up question naturally.
- Use the conversation history above for context
- If it's a follow-up to a previous answer, reference that information
- Do not invent facts that weren't in the conversation history above
- Be conversational and helpful
- If you don't know, politely say so
- Keep answer concise (2-3 sentences max)

Answer:"""

    try:
        response = llm_chat(prompt, max_tokens=250)
        if response:
            response = response.strip()
            response = re.sub(
                r'^(?:answer|sagot|tubag|assistant)[:\s-]*',
                '', response, flags=re.IGNORECASE
            ).strip()
            if response and response[-1] not in '.!?':
                response += '.'
            return response
    except Exception as e:
        logger.error(f"Follow-up AI error: {e}")
    return None


def _lookup_faq_answer(question: str, detected_lang: str) -> Optional[Tuple[str, str, str]]:
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT answer, question, id, topic, intent, keywords,
                   related_questions, priority_score, image_path, embedding
            FROM faq
        """)
        all_faqs = cur.fetchall()
        search_q = question.lower()
        expanded_q = lang_service.expand_abbreviations(search_q)
        query_embedding = get_embedding(search_q, cache=ai_cache)
        candidates = engine.get_top_faq_candidates(
            search_q, expanded_q, all_faqs, detected_lang, query_embedding=query_embedding
        )
        if not candidates:
            return None
        best = candidates[0]
        if best['combined'] < engine.FUZZY_MIN_SCORE:
            if best['fuzzy'] >= 0.60:
                row = best['faq']
                return (row[0] or '', row[3], row[4])
            return None
        if circuit_breaker.is_open():
            return None
        ai_ans, matched = engine.smart_extract_answer(
            user_question=question,
            candidates=candidates,
            detected_lang=detected_lang,
            llm_chat_fn=llm_chat,
            ai_cache=ai_cache,
        )
        if not ai_ans:
            return None
        best_row = (matched or best)['faq']
        return (ai_ans, best_row[3], best_row[4])
    finally:
        cur.close()
        conn.close()


def _log_unanswered(cur, conn, question: str, ctx=None):
    from services.conversation import build_context_snapshot
    try:
        context_snapshot = build_context_snapshot(ctx) if ctx else ''
        resolved_question = None
        if context_snapshot and not circuit_breaker.is_open():
            resolved_question = _resolve_standalone_question(question, context_snapshot)
            if (resolved_question
                    and resolved_question.strip().lower() == question.strip().lower()):
                resolved_question = None
        cur.execute("""
            INSERT INTO unanswered_questions
                (question, resolved_question, context, created_at)
            VALUES (%s, %s, %s, NOW())
        """, (
            question[:500],
            resolved_question,
            (context_snapshot[:2000] if context_snapshot else None),
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"_log_unanswered error: {e}")


def _resolve_standalone_question(question: str, context_snapshot: str) -> str:
    if not context_snapshot or circuit_breaker.is_open():
        return question
    prompt = f"""Conversation so far:
{context_snapshot}

The student's question was: "{question}"

Rewrite it as a single, fully self-contained question that spells out
whatever it depends on from the conversation above. If it's already
self-contained, output it unchanged. Output ONLY the rewritten question,
nothing else."""
    try:
        rewritten = llm_chat(prompt, max_tokens=80)
        rewritten = (rewritten or '').strip().strip('"').strip()
        if (rewritten
                and 3 < len(rewritten) < 400
                and rewritten.lower() != question.strip().lower()):
            return rewritten
    except Exception as e:
        logger.error(f"_resolve_standalone_question error: {e}")
    return question


def log_unanswered_standalone(question: str, session_id: str = None):
    conn = get_db()
    cur = conn.cursor()
    try:
        ctx = get_conversation_context(session_id) if session_id else None
        _log_unanswered(cur, conn, question, ctx)
    finally:
        cur.close()
        conn.close()


def _try_resolve_pending_disambiguation(question: str, detected_lang: str, ctx):
    disamb = ctx.last_disambiguation
    if not disamb:
        return None

    choice = resolve_disambiguation_choice(question, disamb)
    ctx.clear_disambiguation()
    if not choice:
        return None

    original_q = disamb.get('original_question', question)
    lang = disamb.get('detected_lang', detected_lang)

    if disamb['type'] == 'handbook':
        answer = handbook_service.answer_from_selected_chunk(original_q, choice['chunk'], lang)
        if not answer:
            return None
        ctx.update(user_msg=question, bot_response=answer, faq_answer=answer, faq_question=original_q)
        return (answer, 'Handbook', None, 'handbook_query', lang)

    if disamb['type'] == 'faq':
        answer = engine.answer_from_selected_faq(original_q, choice['faq'], llm_chat_fn=llm_chat,
                                                  detected_lang=lang)
        if not answer:
            return None
        row = choice['faq']
        topic = row[3] if len(row) > 3 else 'General'
        intent = row[4] if len(row) > 4 else None
        img = row[8] if len(row) > 8 else None
        answer_text = answer + (f"\n\n[Image: {img}]" if img else "")
        ctx.update(topic=topic, intent=intent, user_msg=question, bot_response=answer_text,
                    faq_answer=answer_text, faq_question=original_q)
        return (answer_text, topic, row[2] if len(row) > 2 else None, intent, lang)

    if disamb['type'] == 'date':
        dept = choice.get('dept')
        answer = date_service.answer_for_selected_department(original_q, dept, lang, ctx=ctx)
        if not answer:
            return None
        ctx.update(user_msg=question, bot_response=answer)
        return (answer, 'Schedule', None, 'date_query', lang)

    return None


def _handbook_best_score(question: str) -> Tuple[float, list]:
    chunks = handbook_service.get_top_handbook_chunks(question)
    score = chunks[0]['score'] if chunks else 0.0
    return score, chunks


def _faq_candidates_and_score(question: str, detected_lang: str) -> Tuple[list, float]:
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT answer, question, id, topic, intent, keywords,
                   related_questions, priority_score, image_path
            FROM faq
        """)
        all_faqs = cur.fetchall()
    finally:
        cur.close()
        conn.close()
    search_q = question.lower()
    expanded_q = lang_service.expand_abbreviations(search_q)
    candidates = engine.get_top_faq_candidates(search_q, expanded_q, all_faqs, detected_lang)
    best_score = candidates[0]['combined'] if candidates else 0.0
    return candidates, best_score


def _cross_check_sources(question: str, detected_lang: str, ctx=None) -> dict:
    hb_score, hb_chunks = _handbook_best_score(question)
    faq_candidates, faq_score = _faq_candidates_and_score(question, detected_lang)
    is_date_q = date_service.is_date_question(question, ctx=ctx)
    return {
        'hb_score': hb_score,
        'hb_chunks': hb_chunks,
        'faq_candidates': faq_candidates,
        'faq_score': faq_score,
        'is_date_q': is_date_q,
    }


_NOTE_HEADERS = {
    'handbook': {
        'english': "From the handbook, for the general policy:",
        'tagalog': "Mula naman sa handbook, para sa pangkalahatang patakaran:",
        'bisaya': "Gikan sa handbook, para sa kinatibuk-ang palisiya:",
    },
    'faq': {
        'english': "More specifically, from our help desk notes:",
        'tagalog': "Mas partikular, mula sa help desk notes:",
        'bisaya': "Mas piho, gikan sa help desk notes:",
    },
    'date': {
        'english': "Also, regarding the schedule:",
        'tagalog': "Tungkol din sa iskedyul:",
        'bisaya': "Bahin pud sa iskedyul:",
    },
}


def _note_header(kind: str, lang: str) -> str:
    d = _NOTE_HEADERS.get(kind, {})
    return d.get(lang, d.get('english', ''))


def _merge_with_handbook(base_answer: str, question: str, hb_chunks: list,
                          detected_lang: str) -> str:
    hb_ans = handbook_service.answer_from_selected_chunk(question, hb_chunks[0], detected_lang)
    if not hb_ans:
        return base_answer
    return f"{base_answer}\n\n{_note_header('handbook', detected_lang)} {hb_ans}"


def _merge_with_faq(base_answer: str, question: str, faq_candidates: list,
                     detected_lang: str) -> str:
    faq_ans, _matched = engine.smart_extract_answer(
        user_question=question,
        candidates=faq_candidates,
        detected_lang=detected_lang,
        llm_chat_fn=llm_chat,
        ai_cache=ai_cache,
    )
    if not faq_ans:
        return base_answer
    return f"{base_answer}\n\n{_note_header('faq', detected_lang)} {faq_ans}"


def _merge_with_date(base_answer: str, date_ans: str, detected_lang: str) -> str:
    if not date_ans:
        return base_answer
    return f"{base_answer}\n\n{_note_header('date', detected_lang)} {date_ans}"


def _combine_handbook_and_faq(question: str, candidates: list, hb_chunks: list,
                               detected_lang: str) -> Optional[str]:
    """FAQ often holds the clearer student-facing answer; handbook holds
    the official wording. Prefer FAQ first, handbook as supporting policy.
    Either alone is fine when the other is missing.
    """
    faq_ans, _matched = engine.smart_extract_answer(
        user_question=question,
        candidates=candidates,
        detected_lang=detected_lang,
        llm_chat_fn=llm_chat,
        ai_cache=ai_cache,
    )
    hb_ans = None
    if hb_chunks:
        hb_ans = handbook_service.answer_from_selected_chunk(
            question, hb_chunks[0], detected_lang
        )

    if faq_ans and hb_ans:
        # Avoid near-duplicate doubles
        if faq_ans.strip().lower()[:120] == hb_ans.strip().lower()[:120]:
            return faq_ans
        return (
            f"{faq_ans}\n\n"
            f"{_note_header('handbook', detected_lang)} {hb_ans}"
        )
    return faq_ans or hb_ans


def find_answer(question: str, return_topic: bool = False, session_id: str = None):
    start = time.time()
    ctx = get_conversation_context(session_id or 'default')
    detected_lang = lang_service.detect_language(question, prior_lang=ctx.last_detected_lang)
    ctx.last_detected_lang = detected_lang

    if ctx.last_disambiguation:
        resolved = _try_resolve_pending_disambiguation(question, detected_lang, ctx)
        if resolved:
            return resolved if return_topic else resolved[0]

    # Successful-answer memory: identical questions that previously earned
    # a grounded, thumbs-up-worthy reply skip re-extraction.
    try:
        from services.answer_memory import get_cached_answer
        from database import get_db as _gdb
        cached_ans = get_cached_answer(_gdb, question, detected_lang)
        if cached_ans:
            ctx.update(user_msg=question, bot_response=cached_ans)
            result = (cached_ans, 'Cached', None, None, detected_lang)
            return result if return_topic else result[0]
    except Exception:
        pass

    classification = lang_service.classify_question_ai(question)

    if classification['is_gibberish']:
        result = (
            lang_service.get_unclear_input_response(detected_lang),
            'Unknown', None, None, detected_lang,
        )
        return result if return_topic else result[0]

    if not classification['is_school_related']:
        result = (
            lang_service.get_non_school_response(detected_lang),
            'Unknown', None, None, detected_lang,
        )
        return result if return_topic else result[0]

    # ---- Multi-source gate (cheap retrieval, no triple LLM) ----
    # Never lock onto the schedule path solely because a keyword matched.
    # Pull handbook/FAQ relevance first; only answer as Schedule when the
    # date intent is clear *and* other sources are not clearly stronger.
    _looks_date = date_service.is_date_question(question, ctx=ctx)
    _looks_hb = handbook_service.is_handbook_question(question)
    _hb_score_early, _hb_chunks_early = (0.0, [])
    _faq_score_early = 0.0
    if _looks_date or _looks_hb:
        try:
            _hb_score_early, _hb_chunks_early = _handbook_best_score(question)
        except Exception:
            pass
        try:
            _faq_cands_early, _faq_score_early = _faq_candidates_and_score(
                question, detected_lang
            )
        except Exception:
            _faq_cands_early = []
            _faq_score_early = 0.0

    # Handbook wins when topic detector fires OR retrieval score is
    # decisively strong.
    #
    # BUG FIX: the score-only branch used to fire at hb_score >= 0.36 —
    # almost identical to CROSS_CHECK_HANDBOOK_BAR (0.34), which is only
    # meant to be the bar for tacking on a handbook *footnote*, not for
    # completely overriding the schedule engine. In practice the
    # handbook's attendance-policy chunk mentions "special holidays"
    # (for religious-observance excused absences), so a plainly
    # date-shaped question like "do we have holidays this month?" was
    # scoring high enough on pure full-text overlap to hijack routing
    # away from date_service.answer_date_question() entirely — the
    # student got a random handbook paragraph instead of real
    # schedule/holiday data, and this repeated for "do we have class
    # today" etc.
    #
    # A bare retrieval score is real signal when nothing else claims the
    # question, but it must never preempt a genuinely date-shaped query
    # (_looks_date) on score alone — only actual topical confirmation
    # (is_handbook_question) is allowed to do that. The date engine
    # already gets a handbook footnote merged in afterwards (see
    # CROSS_CHECK_HANDBOOK_BAR below) when the handbook score is solid,
    # so nothing is lost by letting the date engine go first.
    _hb_clearly_better = (
        _looks_hb and _hb_score_early >= 0.12
    ) or (
        not _looks_date
        and _hb_score_early >= 0.36
        and _hb_score_early > (_faq_score_early + 0.05)
    )

    if _looks_date and not _hb_clearly_better:
        dept_dis = date_service.check_department_disambiguation(
            question, detected_lang, ctx=ctx
        )
        if dept_dis:
            message, options = dept_dis
            ctx.set_disambiguation(
                'date', options,
                original_question=question, detected_lang=detected_lang,
            )
            result = (message, 'Disambiguation', None, None, detected_lang)
            return result if return_topic else result[0]

        date_ans = date_service.answer_date_question(
            question, detected_lang, ctx=ctx
        )
        if date_ans:
            # Still allow a handbook/FAQ footnote when those scores are solid
            signals = {
                'hb_score': _hb_score_early,
                'hb_chunks': _hb_chunks_early,
                'faq_candidates': locals().get('_faq_cands_early') or [],
                'faq_score': _faq_score_early,
            }
            final_ans = date_ans
            if (signals['hb_score'] >= CROSS_CHECK_HANDBOOK_BAR
                    and signals['hb_chunks']
                    and not circuit_breaker.is_open()):
                final_ans = _merge_with_handbook(
                    final_ans, question, signals['hb_chunks'], detected_lang
                )
            elif (signals['faq_score'] >= CROSS_CHECK_FAQ_BAR
                    and signals['faq_candidates']
                    and not circuit_breaker.is_open()):
                final_ans = _merge_with_faq(
                    final_ans, question, signals['faq_candidates'], detected_lang
                )
            result = (final_ans, 'Schedule', None, 'date_query', detected_lang)
            return result if return_topic else result[0]

    dept_followup = date_service.answer_schedule_department_followup(question, ctx)
    if dept_followup:
        result = (dept_followup, 'Schedule', None, 'date_query', detected_lang)
        return result if return_topic else result[0]

    if session_id and ctx.conversation_history and not circuit_breaker.is_open():
        last_ex = ctx.conversation_history[-1]
        is_followup = nlg.classify_followup(
            question, last_ex.get('user', ''), last_ex.get('bot', '')
        )
        word_count = len(question.split())
        # Only fall back to this word-count guess when the short message
        # doesn't already look like a fresh, specific handbook topic on
        # its own (e.g. "dresscode", "logo or motto") — those should go
        # straight to the real lookup below, not be guessed at from
        # chat history alone.
        if (not is_followup and word_count <= 5 and last_ex.get('bot')
                and not handbook_service.is_handbook_question(question)):
            is_followup = True

        if is_followup:
            search_ans, search_topic, search_intent = _answer_followup_via_search(
                question, ctx, detected_lang
            )
            if search_ans:
                ctx.update(
                    topic=search_topic, intent=search_intent,
                    user_msg=question, bot_response=search_ans,
                    faq_answer=search_ans, faq_question=question,
                )
                result = (search_ans, search_topic or 'Handbook', None, search_intent, detected_lang)
                return result if return_topic else result[0]

            ai_followup = answer_followup_with_ai(
                user_question=question,
                context_summary=ctx.get_context_summary(),
                conversation_history=ctx.get_history(3),
                detected_lang=detected_lang,
                last_faq_answer=ctx.last_faq_answer,
                last_faq_question=ctx.last_faq_question,
            )
            # A hedging "I don't have that info" / "we haven't covered
            # that" response is the model guessing blind from chat
            # history — it has no way to know whether the handbook/FAQ
            # actually has the answer. Treat that as no answer and fall
            # through to the real lookup instead of returning a
            # confabulated-sounding non-answer to the student.
            if ai_followup and not _looks_like_non_answer(ai_followup):
                ctx.update(user_msg=question, bot_response=ai_followup)
                result = (ai_followup, 'Follow-up', None, None, detected_lang)
                return result if return_topic else result[0]


    is_hb = handbook_service.is_handbook_question(question)
    parts = split_compound_question(question)
    if len(parts) > 1:
        sub_answers = []
        last_topic, last_intent = None, None
        for part in parts:
            part_class = lang_service.classify_question_ai(part)
            if part_class.get('is_gibberish') or not part_class.get('is_school_related', True):
                continue
            # Prefer handbook when the clause is handbook-shaped
            part_ans = None
            if handbook_service.is_handbook_question(part) or is_hb:
                part_ans = handbook_service.answer_handbook_question(
                    part, detected_lang, min_score_to_answer=0.12, ctx=None,
                )
                if part_ans:
                    sub_answers.append(part_ans)
                    last_topic = 'Handbook'
                    last_intent = 'handbook_query'
                    continue
            r = _lookup_faq_answer(part, detected_lang)
            if r and r[0]:
                sub_answers.append(r[0])
                last_topic = r[1] or last_topic
                last_intent = r[2] or last_intent
        if len(sub_answers) >= 2:
            combined = "\n\n".join(sub_answers)
            ctx.update(
                topic=last_topic,
                intent=last_intent,
                user_msg=question,
                bot_response=combined,
            )
            result = (
                combined,
                last_topic or 'General',
                None,
                last_intent,
                detected_lang,
            )
            return result if return_topic else result[0]
        # One part only succeeded — still better than a single wrong chunk
        if len(sub_answers) == 1 and len(parts) >= 2:
            # Fall through so full question also tries normal path;
            # but prefer the good partial if later path fails.
            pass


    norm_q = normalize_input(question)
    cache_key = f"{norm_q}_{detected_lang}"
    if not is_hb:
        cached = faq_cache.get(cache_key)
        if cached:
            return (
                cached if return_topic
                else (cached[0] if cached else lang_service.get_fallback_response(detected_lang))
            )

    hb_score, hb_chunks = _handbook_best_score(question)

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT answer, question, id, topic, intent, keywords,
                   related_questions, priority_score, image_path, embedding
            FROM faq
        """)
        all_faqs = cur.fetchall()
        search_q = question.lower()
        expanded_q = lang_service.expand_abbreviations(search_q)
        query_embedding = get_embedding(search_q, cache=ai_cache)
        candidates = engine.get_top_faq_candidates(
            search_q, expanded_q, all_faqs, detected_lang, query_embedding=query_embedding
        )
        faq_best_score = candidates[0]['combined'] if candidates else 0.0

        word_count = len(question.split())

        if is_hb:
            hb_threshold = 0.12
        elif word_count <= 8:
            hb_threshold = 0.28
        else:
            hb_threshold = 0.35

        # FAQ can lead when it clearly outscores a weak handbook hit.
        # Handbook text is often broad; curated FAQs fill specific gaps
        # (dress details, scholarships, strands, appeal windows, etc.).
        # FAQ must beat handbook by a clear margin. For handbook-shaped
        # questions (dress, demerits, strands, RA laws), require an even
        # larger gap so a loosely matched FAQ (e.g. office hours, scholarships)
        # cannot steal the answer.
        # Filter FAQ candidates: reject if the FAQ *question text* does not
        # share distinctive words with the user question. Stops office-hours /
        # scholarship / enrollment FAQs from matching dress / vision / demerits.
        def _faq_text_relevant(user_q: str, cand: dict) -> bool:
            try:
                row = cand.get('faq') or cand.get('row')
                faq_q = ''
                if row is not None:
                    # faq tuple: answer, question, id, ...
                    faq_q = (row[1] if len(row) > 1 else '') or ''
                faq_q = faq_q or cand.get('best_q_text') or cand.get('question') or ''
                u = set(w for w in re.findall(r"[a-z0-9']{4,}", user_q.lower())
                        if w not in ('this', 'that', 'what', 'when', 'where',
                                     'does', 'have', 'with', 'from', 'about',
                                     'school', 'college', 'student', 'students'))
                f = set(re.findall(r"[a-z0-9']{4,}", faq_q.lower()))
                if not u:
                    return True
                overlap = len(u & f)
                # Handbook-shaped questions: require stronger topical overlap
                # so office-hours / enrollment / scholarship FAQs cannot steal
                # dress / demerit / vision / dorm / strands answers.
                if is_hb:
                    topical = (
                        'dress', 'demerit', 'offense', 'vision', 'mission',
                        'core', 'value', 'dorm', 'bullying', 'spaces',
                        'strand', 'senior', 'appeal', 'fine', 'chaperone',
                        'hike', 'cook', 'cafeteria', 'jewelry', 'philosophy',
                        'logo', 'hymn', 'founded', 'history',
                    )
                    faq_has_topic = any(t in faq_q.lower() for t in topical)
                    return overlap >= 2 or (overlap >= 1 and faq_has_topic)
                return overlap >= 1
            except Exception:
                return True

        if candidates:
            filtered = [c for c in candidates if _faq_text_relevant(question, c)]
            if filtered:
                candidates = filtered
                faq_best_score = candidates[0].get('combined', 0.0)
            else:
                # No FAQ is actually about this question
                candidates = []
                faq_best_score = 0.0

        # Handbook-shaped questions: handbook leads unless FAQ is a clear
        # topical match (after the filter above) with a strong score gap.
        faq_margin = 0.28 if is_hb else 0.10
        faq_clearly_better = (
            (not is_hb)
            and candidates
            and faq_best_score >= engine.FUZZY_MIN_SCORE + 0.12
            and faq_best_score >= (hb_score + faq_margin)
        )
        # For is_hb: never let FAQ "clearly better" — try handbook first.
        if is_hb:
            faq_clearly_better = False

        handbook_should_lead = (
            hb_chunks
            and not faq_clearly_better
            and (
                (is_hb and hb_score >= 0.08)  # low bar when detector fired
                or (hb_score >= hb_threshold and hb_score >= faq_best_score)
            )
        )

        pure_handbook_topics = bool(re.search(
            r'\b(vision|mission|philosophy|core values|logo|hymn|founded|'
            r'original name|brief history|want to become|envisions|future of|'
            r'grounds? for (suspension|expulsion)|behavioral code|'
            r'(first|second|third)[ -]?degree|fraternity|sorority|hazing|'
            r'demerit|merit points?|max(imum)?\s*fine|appeal|probation|'
            r'unchaperoned|chaperone|hike|cook(ing)? (in|inside)|visit other dorms|'
            r'anti.?bullying|safe spaces|ra.?11?3?13|ra.?10627|11313|10627|'
            r'senior high strands?|heaviest demerit|'
            r'food .{0,20}(cafeteria|not allowed)|'
            r'how bad is a (second|third)|what.?s a (second|third)[ -]?degree|'
            r'jewelry|necklace|dress code|slippers|crop tops|short pants|'
            r'pda|public display|curfew|worship absences|miss worship|'
            r'macias|founders?|something about 1975|campuses|main campus|'
            r'incomplete grades?|residency|drug test|sibling|family scholar|'
            r'dormitory fees?|tesda|computer science|information technology|'
            r'business administration|week of spiritual|morphe|2025 edition|'
            r'social media|ellen white)\b',
            question, re.I,
        ))
        # Pure handbook topics must never be answered from FAQ/schedule —
        # that was the source of wrong "cashier / late enrollment /
        # scholarships" answers on dress/dorm/offense/vision questions.
        # Data-driven handbook-only topics (admin aliases) expand the pure set
        try:
            from services.topic_aliases import is_handbook_only
            if is_handbook_only(question):
                pure_handbook_topics = True
        except Exception:
            try:
                from topic_aliases import is_handbook_only
                if is_handbook_only(question):
                    pure_handbook_topics = True
            except Exception:
                pass

        if pure_handbook_topics:
            is_hb = True
            faq_clearly_better = False
            handbook_should_lead = bool(hb_chunks) or True

        combine_gap = 0.22
        # Never force-combine a weak/unrelated FAQ into a handbook answer
        both_solid = (
            (not is_hb)
            and (not pure_handbook_topics)
            and hb_chunks and hb_score >= 0.28
            and candidates and faq_best_score >= engine.FUZZY_MIN_SCORE
            and abs(hb_score - faq_best_score) < combine_gap
        )
        if both_solid and not circuit_breaker.is_open():
            combined = _combine_handbook_and_faq(
                question, candidates, hb_chunks, detected_lang
            )
            if combined and date_service.is_date_question(question, ctx=ctx):
                date_ans = date_service.answer_date_question(
                    question, detected_lang, ctx=ctx
                )
                combined = _merge_with_date(combined, date_ans, detected_lang)
            if combined:
                ctx.update(
                    topic='General', intent=None,
                    user_msg=question, bot_response=combined,
                    faq_answer=combined, faq_question=question,
                )
                result_data = (
                    (combined, 'General', None, None, detected_lang)
                    if return_topic else combined
                )
                faq_cache.set(cache_key, result_data)
                return result_data

        if handbook_should_lead or is_hb or pure_handbook_topics:
            hb_ans = handbook_service.answer_handbook_question(
                question, detected_lang,
                min_score_to_answer=(None if (is_hb or pure_handbook_topics) else hb_threshold),
                ctx=ctx,
            )
            if hb_ans:
                # Never merge FAQ into pure handbook topic answers
                if (not is_hb and not pure_handbook_topics
                        and faq_best_score >= CROSS_CHECK_FAQ_BAR
                        and faq_best_score >= hb_score - 0.05
                        and candidates and not circuit_breaker.is_open()):
                    hb_ans = _merge_with_faq(
                        hb_ans, question, candidates, detected_lang
                    )
                is_disambiguation = ctx.last_disambiguation is not None
                if not is_disambiguation:
                    ctx.update(
                        user_msg=question,
                        bot_response=hb_ans,
                        faq_answer=hb_ans,
                        faq_question=question,
                    )
                result = (hb_ans, 'Handbook', None, 'handbook_query', detected_lang)
                return result if return_topic else result[0]
            # pure handbook topic with no handbook hit → do NOT fall through
            # to unrelated FAQ (cashier/enrollment/etc.). Log + fallback.
            if pure_handbook_topics:
                _log_unanswered(cur, conn, question, ctx)
                result = (
                    lang_service.get_logged_fallback_response(detected_lang),
                    'Unknown', None, None, detected_lang,
                )
                return result if return_topic else result[0]

        if not candidates:
            hb_ans = handbook_service.answer_handbook_question(question, detected_lang, ctx=ctx)
            if hb_ans:
                if ctx.last_disambiguation is None:
                    ctx.update(
                        user_msg=question,
                        bot_response=hb_ans,
                        faq_answer=hb_ans,
                        faq_question=question,
                    )
                result = (hb_ans, 'Handbook', None, 'handbook_query', detected_lang)
                return result if return_topic else result[0]
            _log_unanswered(cur, conn, question, ctx)
            result = (
                lang_service.get_logged_fallback_response(detected_lang),
                'Unknown', None, None, detected_lang,
            )
            return result if return_topic else result[0]

        if engine.faq_should_disambiguate(candidates):
            dis = engine.build_disambiguation(candidates, detected_lang)
            ctx.set_disambiguation(
                'faq',
                engine.build_disambiguation_options(candidates),
                original_question=question,
                detected_lang=detected_lang,
            )
            result = (dis, 'Disambiguation', None, None, detected_lang)
            return result if return_topic else result[0]

        best = candidates[0]

        if best['combined'] >= engine.FUZZY_MIN_SCORE and not circuit_breaker.is_open():
            ai_ans, matched_cand = engine.smart_extract_answer(
                user_question=question,
                candidates=candidates,
                detected_lang=detected_lang,
                llm_chat_fn=llm_chat,
                ai_cache=ai_cache,
                context_faq_answer=ctx.last_faq_answer,
                context_faq_question=ctx.last_faq_question,
            )
            if ai_ans:
                best_row = (matched_cand or best)['faq']
                answer, question_text, fid, topic, intent, kw, related, score_val = best_row[:8]
                img = best_row[8] if len(best_row) > 8 else None
                if (hb_score >= CROSS_CHECK_HANDBOOK_BAR
                        and hb_chunks and not circuit_breaker.is_open()):
                    ai_ans = _merge_with_handbook(ai_ans, question, hb_chunks, detected_lang)
                ctx.update(
                    topic=topic,
                    intent=intent,
                    user_msg=question,
                    bot_response=ai_ans,
                    faq_answer=answer,
                    faq_question=question_text,
                )
                result_answer = ai_ans + (f"\n\n[Image: {img}]" if img else "")
                result_data = (
                    (result_answer, topic, fid, intent, detected_lang)
                    if return_topic else result_answer
                )
                faq_cache.set(cache_key, result_data)
                return result_data

        if best['combined'] >= engine.FUZZY_MIN_SCORE or best['fuzzy'] >= 0.60:
            row = best['faq']
            answer, question_text, fid, topic, intent, kw, related, score_val = row[:8]
            img = row[8] if len(row) > 8 else None
            answer_text = (answer or '') + (f"\n\n[Image: {img}]" if img else "")
            ctx.update(
                topic=topic,
                intent=intent,
                faq_answer=answer_text,
                faq_question=question_text,
                user_msg=question,
                bot_response=answer_text,
            )
            result_data = (
                (answer_text, topic, fid, intent, detected_lang)
                if return_topic else answer_text
            )
            faq_cache.set(cache_key, result_data)
            return result_data

        hb_ans = handbook_service.answer_handbook_question(question, detected_lang, ctx=ctx)
        if hb_ans:
            if ctx.last_disambiguation is None:
                ctx.update(
                    user_msg=question,
                    bot_response=hb_ans,
                    faq_answer=hb_ans,
                    faq_question=question,
                )
            result = (hb_ans, 'Handbook', None, 'handbook_query', detected_lang)
            return result if return_topic else result[0]

        _log_unanswered(cur, conn, question, ctx)
        result = (
            lang_service.get_logged_fallback_response(detected_lang),
            'Unknown', None, None, detected_lang,
        )
        return result if return_topic else result[0]

    except Exception as e:
        logger.error(f"Error in find_answer: {e}", exc_info=True)
        result = (
            lang_service.get_fallback_response('english'),
            'Unknown', None, None, 'english',
        )
        return result if return_topic else result[0]
    finally:
        cur.close()
        conn.close()