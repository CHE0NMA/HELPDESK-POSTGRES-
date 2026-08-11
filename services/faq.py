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

COMPOUND_SPLIT_RE = re.compile(
    r'\s+(?:and|at saka|saka|tas|ug|dayon|unya)\s+'
    r'(?=what|how|when|where|who|which|is|are|can|could|do|does|will'
    r'|paano|ano|kailan|saan|sino|magkano|ilan'
    r'|unsa|kanus-a|asa|kinsa|pila)\b',
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
    r"wasn'?t covered|(not|isn'?t) something we'?ve covered|i don'?t know|"
    r"not sure (what|about)|no information (on|about)|not readily available|"
    r"i('| a)m not able to find|i can'?t find (that|this)|"
    r"i (only|just) (mentioned|found|got to|had)\b|"
    r"let me try to find|i'?d be happy to (try and )?find|"
    r"check(ing)? page \d+ (of|for) the relevant section|"
    r"couldn'?t find any specific information)\b",
    re.IGNORECASE,
)


def _looks_like_non_answer(text: str) -> bool:
    return bool(_NON_ANSWER_RE.search(text or ''))


def split_compound_question(question: str) -> List[str]:
    if len(question.split()) < 6:
        return [question]
    parts = COMPOUND_SPLIT_RE.split(question)
    parts = [p.strip(' ?.,') for p in parts if p and p.strip(' ?.,')]
    return parts if len(parts) > 1 else [question]


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
        'english': "Related to that, here's what we also have on file:",
        'tagalog': "Kaugnay nito, ito rin ang nasa records namin:",
        'bisaya': "Kalabot niini, kini pud ang naa sa among mga record:",
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
    faq_ans, _matched = engine.smart_extract_answer(
        user_question=question,
        candidates=candidates,
        detected_lang=detected_lang,
        llm_chat_fn=llm_chat,
        ai_cache=ai_cache,
    )
    hb_ans = handbook_service.answer_from_selected_chunk(question, hb_chunks[0], detected_lang)

    if faq_ans and hb_ans:
        return f"{faq_ans}\n\n{_note_header('handbook', detected_lang)} {hb_ans}"
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

    if date_service.is_date_question(question, ctx=ctx):
        dept_dis = date_service.check_department_disambiguation(question, detected_lang, ctx=ctx)
        if dept_dis:
            message, options = dept_dis
            ctx.set_disambiguation('date', options, original_question=question, detected_lang=detected_lang)
            result = (message, 'Disambiguation', None, None, detected_lang)
            return result if return_topic else result[0]

        date_ans = date_service.answer_date_question(
            question, detected_lang, ctx=ctx
        )
        if date_ans:
            signals = _cross_check_sources(question, detected_lang, ctx=ctx)
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
    if len(parts) > 1 and not is_hb:
        sub_answers, last_topic, last_intent = [], None, None
        for part in parts:
            part_class = lang_service.classify_question_ai(part)
            if part_class['is_gibberish'] or not part_class['is_school_related']:
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

        handbook_should_lead = hb_chunks and (
            (is_hb and hb_score >= hb_threshold)
            or (hb_score >= hb_threshold and hb_score >= faq_best_score)
        )

        pure_handbook_topics = bool(re.search(
            r'\b(vision|mission|philosophy|core values|logo|hymn|founded|'
            r'original name|brief history|want to become|envisions)\b',
            question, re.I,
        ))
        combine_gap = 0.18
        both_solid = (
            not pure_handbook_topics
            and hb_chunks and hb_score >= 0.32
            and candidates and faq_best_score >= engine.FUZZY_MIN_SCORE
            and abs(hb_score - faq_best_score) < combine_gap
        )
        if both_solid and not circuit_breaker.is_open():
            combined = _combine_handbook_and_faq(question, candidates, hb_chunks, detected_lang)
            if combined and date_service.is_date_question(question, ctx=ctx):
                date_ans = date_service.answer_date_question(question, detected_lang, ctx=ctx)
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

        if handbook_should_lead:
            hb_ans = handbook_service.answer_handbook_question(
                question, detected_lang,
                min_score_to_answer=(None if is_hb else hb_threshold),
                ctx=ctx,
            )
            if hb_ans:
                if (not pure_handbook_topics
                        and faq_best_score >= CROSS_CHECK_FAQ_BAR
                        and candidates and not circuit_breaker.is_open()):
                    hb_ans = _merge_with_faq(hb_ans, question, candidates, detected_lang)
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

        if is_hb and pure_handbook_topics and (
            not candidates or faq_best_score < engine.FUZZY_MIN_SCORE + 0.10
        ):
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