import logging
import re
import time
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai_providers import (
    ai_cache,
    circuit_breaker,
    faq_cache,
    llm_chat,
    provider_for_language,
)
from ai_providers.cache import stable_digest
from database import get_db
from services import answer_engine as engine
from services import dates as date_service
from services import handbook as handbook_service
from services import hybrid_search
from services import language as lang_service
from services import nlg
from services.conversation import (
    get_conversation_context,
    resolve_disambiguation_choice,
)
from services.disambiguation import FAQ_POLICY
from services.embeddings import get_embedding
from services.ai_retrieval import (
    QuestionUnderstanding,
    llm_expand_faq_query,
    llm_self_check,
    question_understanding_prompt,
    understand_question,
)
from services.personnel_titles import personnel_titles
from utils.text import normalize_input

logger = logging.getLogger("pac_helpdesk.faq")


_SCORING_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="faq_scoring")

_FAQ_ROWS_CACHE = None
_FAQ_ROWS_TS = 0.0
_FAQ_CACHE_TTL = 45.0
FAQ_IMAGE_MIN_SCORE = 0.85

_TUITION_QUERY_RE = re.compile(
    r'\b(tuition|tuition fee|school fees?|matrikula|bayranan|bayad sa school)\b',
    re.I,
)
_TUITION_PROGRAMS = (
    (re.compile(r'\b(?:bsit|information technology)\b', re.I),
     ('BSIT', 'Information Technology')),
    (re.compile(r'\b(?:bsn|nursing)\b', re.I),
     ('BSN', 'Nursing')),
    (re.compile(r'\b(?:bsba|business administration)\b', re.I),
     ('BSBA', 'Business Administration')),
    (re.compile(r'\b(?:bpa|public administration)\b', re.I),
     ('BPA', 'Public Administration')),
    (re.compile(r'\b(?:bscs|computer science)\b', re.I),
     ('BSCS', 'Computer Science')),
    (re.compile(r'\b(?:beed|elementary education)\b', re.I),
     ('BEED', 'Elementary Education')),
    (re.compile(r'\b(?:bsed|secondary education)\b', re.I),
     ('BSED', 'Secondary Education')),
    (re.compile(r'\b(?:beced|early childhood education)\b', re.I),
     ('BECED', 'Early Childhood Education')),
    (re.compile(r'\b(?:bped|physical education)\b', re.I),
     ('BPED', 'Physical Education')),
)
_TUITION_YEAR = re.compile(
    r'\b(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+year\b|'
    r'\byear\s*(?:1|2|3|4)\b',
    re.I,
)
_TUITION_SEMESTER = re.compile(
    r'\b(?:first|second|1st|2nd)\s+sem(?:ester)?\b|'
    r'\bsem(?:ester)?\s*(?:1|2)\b',
    re.I,
)


def _image_for_faq_candidate(
    candidate: Optional[Dict],
    candidates: Optional[Sequence[Dict]] = None,
) -> Optional[str]:
    """Return an image only for a high-confidence FAQ match."""
    if not candidate or candidate.get('combined', 0.0) < FAQ_IMAGE_MIN_SCORE:
        return None
    row = candidate.get('faq') or ()
    image_path = row[8] if len(row) > 8 else None
    if not image_path:
        return None

    score = candidate['combined']
    for other in candidates or ():
        if other is candidate:
            continue
        other_row = other.get('faq') or ()
        other_image = other_row[8] if len(other_row) > 8 else None
        other_score = other.get('combined', 0.0)
        if (other_image and other_image != image_path
                and other_score >= FAQ_IMAGE_MIN_SCORE
                and abs(score - other_score) <= FAQ_POLICY.max_gap):
            return None
    return image_path


def _tuition_requirements(question: str) -> List[Tuple[str, ...]]:
    if not _TUITION_QUERY_RE.search(question or ''):
        return []
    requirements: List[Tuple[str, ...]] = [
        aliases for pattern, aliases in _TUITION_PROGRAMS
        if pattern.search(question or '')
    ]
    year = _TUITION_YEAR.search(question or '')
    if year:
        year_number = re.search(r'first|1st|year\s*1', year.group(0), re.I)
        if year_number:
            requirements.append(('first year', '1st year', 'year 1', 'year one'))
        elif re.search(r'second|2nd|year\s*2', year.group(0), re.I):
            requirements.append(('second year', '2nd year', 'year 2', 'year two'))
        elif re.search(r'third|3rd|year\s*3', year.group(0), re.I):
            requirements.append(('third year', '3rd year', 'year 3', 'year three'))
        else:
            requirements.append(('fourth year', '4th year', 'year 4', 'year four'))
    semester = _TUITION_SEMESTER.search(question or '')
    if semester:
        if re.search(r'first|1st|sem(?:ester)?\s*1', semester.group(0), re.I):
            requirements.append(('first semester', '1st semester', 'first sem', '1st sem', 'semester 1', 'sem 1'))
        else:
            requirements.append(('second semester', '2nd semester', 'second sem', '2nd sem', 'semester 2', 'sem 2'))
    return requirements


def _matches_tuition_requirements(text: str, requirements: Sequence[Tuple[str, ...]]) -> bool:
    normalized_text = re.sub(r'\s+', ' ', (text or '').lower())
    return all(
        any(re.search(rf'\b{re.escape(alias.lower())}\b', normalized_text) for alias in aliases)
        for aliases in requirements
    )


def _generic_tuition_faq(all_faqs: Sequence[Sequence[Any]]) -> Optional[Sequence[Any]]:
    for row in all_faqs:
        if (
            len(row) > 1
            and row[1]
            and re.search(r'\bhow much is the tuition fee\b', row[1], re.I)
        ):
            return row
    return None


def _return_generic_tuition_answer(question: str, detected_lang: str, ctx,
                                   all_faqs: Sequence[Sequence[Any]],
                                   return_topic: bool):
    row = _generic_tuition_faq(all_faqs)
    if not row or not row[0]:
        return None
    answer = _validated_faq_source_answer(question, row, detected_lang)
    if not answer:
        return None
    topic = row[3] or 'Tuition'
    intent = row[4] if len(row) > 4 else None
    faq_id = row[2] if len(row) > 2 else None
    ctx.update(
        topic=topic,
        intent=intent,
        user_msg=question,
        bot_response=answer,
        faq_answer=answer,
        faq_question=row[1],
        clear_schedule_context=True,
    )
    result = (answer, topic, faq_id, intent, detected_lang)
    return result if return_topic else result[0]


def _get_all_faqs():
    global _FAQ_ROWS_CACHE, _FAQ_ROWS_TS
    now = time.time()
    if _FAQ_ROWS_CACHE is not None and (now - _FAQ_ROWS_TS) < _FAQ_CACHE_TTL:
        return _FAQ_ROWS_CACHE
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT answer, question, id, topic, intent, keywords,
                   related_questions, priority_score, image_path, embedding
            FROM faq
        """)
        rows = cur.fetchall()
        _FAQ_ROWS_CACHE = rows
        _FAQ_ROWS_TS = now
        return rows
    finally:
        cur.close()
        conn.close()


def _find_exact_faq(question: str) -> Optional[Sequence[Any]]:
    """Find an FAQ whose complete question matches, without fuzzy ranking."""
    question_key = engine.normalize_text(question)
    if not question_key:
        return None

    exact_rows = [
        row for row in _get_all_faqs()
        if len(row) > 1
        and row[1]
        and engine.normalize_text(row[1]) == question_key
        and row[0]
    ]
    if not exact_rows:
        return None
    if len({str(row[0]).strip() for row in exact_rows}) > 1:
        logger.warning(
            "Exact FAQ question maps to multiple different answers; using normal retrieval"
        )
        return None
    best = max(
        exact_rows,
        key=lambda row: (
            float(row[7] or 0) if len(row) > 7 else 0,
            int(row[2] or 0) if len(row) > 2 else 0,
        ),
    )
    image_paths = {
        row[8] if len(row) > 8 else None
        for row in exact_rows
    }
    if len(image_paths) > 1:
        logger.warning(
            "Exact FAQ question maps to multiple images; suppressing the image"
        )
        return (*best[:8], None, *best[9:])
    return best


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

CROSS_CHECK_HANDBOOK_BAR = 0.58
CROSS_CHECK_FAQ_BAR = engine.FUZZY_MIN_SCORE + 0.18

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


class _QueryEmbeddingUnset:
    pass


_QUERY_EMBEDDING_UNSET = _QueryEmbeddingUnset()


def _get_faq_candidates(
    question: str,
    detected_lang: str,
    all_faqs: Sequence[Sequence[Any]],
    query_embedding: Optional[List[float]] | _QueryEmbeddingUnset = _QUERY_EMBEDDING_UNSET,
    query_variants: Optional[List[str]] = None,
) -> List[Dict]:
    if not all_faqs:
        return []
    if isinstance(query_embedding, _QueryEmbeddingUnset):
        query_embedding = _get_query_embedding(question)
    search_q = lang_service.normalize_vernacular_spelling(question.lower())
    expanded_q = lang_service.expand_query(search_q)
    alternate_questions = [
        lang_service.expand_query(
            lang_service.normalize_vernacular_spelling(variant.lower())
        )
        for variant in (query_variants or [])
        if variant and variant.strip()
    ]
    if detected_lang != "english" and not query_variants:
        alternate_questions.extend([
            lang_service.expand_query(
                lang_service.normalize_vernacular_spelling(variant.lower())
            )
            for variant in llm_expand_faq_query(question, detected_lang)
        ])
    candidate_ids = hybrid_search.search_faq_candidate_ids(
        get_db,
        query_embedding,
        [question, search_q, expanded_q, *alternate_questions],
    )
    if candidate_ids:
        candidate_rows = [
            row for row in all_faqs
            if len(row) > 2 and int(row[2]) in candidate_ids
        ]
        if candidate_rows:
            all_faqs = candidate_rows

    candidates = engine.get_top_faq_candidates(
        search_q,
        expanded_q,
        all_faqs,
        detected_lang,
        query_embedding=query_embedding,
        alternate_questions=alternate_questions,
    )
    logger.debug(
        "FAQ retrieval query_id=%s language=%s candidate_pool=%d top=%s",
        stable_digest(question)[:12],
        detected_lang,
        len(all_faqs),
        [
            {
                "faq_id": candidate["faq"][2],
                "fuzzy": round(candidate["fuzzy"], 3),
                "semantic": round(candidate.get("semantic_score", 0.0), 3),
                "combined": round(candidate["combined"], 3),
            }
            for candidate in candidates[:5]
        ],
    )
    return candidates


def split_compound_question(question: str) -> List[str]:
    q = (question or '').strip()
    if len(q.split()) < 6:
        return [q]
    parts = COMPOUND_SPLIT_RE.split(q)
    parts = [p.strip(' ?.,') for p in parts if p and p.strip(' ?.,')]
    cleaned = []
    for p in parts:
        if len(p.split()) < 2:
            continue
        cleaned.append(p if p.endswith('?') else p)
    return cleaned if len(cleaned) > 1 else [q]


def _answer_followup_via_search(
    question: str,
    ctx,
    detected_lang: str,
    understanding: Optional[QuestionUnderstanding] = None,
):
    from services.conversation import build_context_snapshot

    context_snapshot = build_context_snapshot(ctx)
    resolved_q = question
    if context_snapshot and not circuit_breaker.is_open():
        resolved_q = _resolve_standalone_question(question, context_snapshot)

    looks_handbooky = (
        handbook_service.is_handbook_question(resolved_q, detected_lang)
        or handbook_service.is_handbook_question(question, detected_lang)
    )
    if looks_handbooky:
        hb_ans = handbook_service.answer_handbook_question(
            resolved_q,
            detected_lang,
            ctx=ctx,
            understanding=understanding,
        )
        if hb_ans:
            return hb_ans, 'Handbook', 'handbook_query'

    faq_result = _lookup_faq_answer(resolved_q, detected_lang, understanding)
    if faq_result and faq_result[0]:
        answer, topic, intent = faq_result
        return answer, (topic or 'General'), intent

    return None, None, None


def answer_followup_with_ai(user_question: str, context_summary: str = "",
                             conversation_history: Optional[List[Dict[str, str]]] = None,
                             detected_lang: str = 'english',
                             last_faq_answer: Optional[str] = None,
                             last_faq_question: Optional[str] = None,
                             understanding: Optional[QuestionUnderstanding] = None
                             ) -> Optional[str]:
    if circuit_breaker.is_open():
        return _faq_refusal(detected_lang)

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
    interpretation = question_understanding_prompt(understanding)

    prompt = f"""{nlg.now_context()}

{lang_config['system_prompt']}

{lang_config['instruction']}

{context_summary}

{history_text}

{f"Previous FAQ asked: {last_faq_question}" if last_faq_question else ""}
{f"Previous FAQ answer: {last_faq_answer[:300]}" if last_faq_answer else ""}

NOW the student asks: "{user_question}"
{interpretation}

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
            if response and not lang_service.verify_output_language(response, detected_lang):
                return _faq_refusal(detected_lang)
            if not llm_self_check(
                user_question,
                response,
                "\n".join(
                    item for item in (
                        context_summary,
                        history_text,
                        last_faq_answer or "",
                    ) if item
                ),
                understanding,
            ):
                return _faq_refusal(detected_lang)
            return response
    except Exception as e:
        logger.error(f"Follow-up AI error: {e}")
    return _faq_refusal(detected_lang)


def _lookup_faq_answer(
    question: str,
    detected_lang: str,
    understanding: Optional[QuestionUnderstanding] = None,
) -> Optional[Tuple[str, str, str]]:
    all_faqs = _get_all_faqs()
    candidates = _get_faq_candidates(
        question,
        detected_lang,
        all_faqs,
        query_variants=understanding.search_queries if understanding else None,
    )
    if not candidates:
        return None

    best = candidates[0]
    if best['combined'] < engine.FUZZY_MIN_SCORE:
        refusal = None if understanding else _faq_refusal(detected_lang)
        if refusal:
            row = best['faq']
            return refusal, row[3], row[4]
        return None
    if circuit_breaker.is_open():
        refusal = None if understanding else _faq_refusal(detected_lang)
        if refusal:
            row = best['faq']
            return refusal, row[3], row[4]
        return None
    ai_ans, matched = engine.smart_extract_answer(
        user_question=question,
        candidates=candidates,
        detected_lang=detected_lang,
        llm_chat_fn=llm_chat,
        ai_cache=ai_cache,
        understanding=understanding,
    )
    if not ai_ans:
        refusal = None if understanding else _faq_refusal(detected_lang)
        if refusal:
            row = best['faq']
            return refusal, row[3], row[4]
        return None
    best_row = (matched or best)['faq']
    return (ai_ans, best_row[3], best_row[4])


def _source_answer_in_language(
    question: str,
    source_question: Optional[str],
    source_answer: Optional[str],
    detected_lang: str,
    translate: bool = False,
) -> str:
    """Return a source answer only if it already matches the requested language.

    A failed translation must not silently expose the English source answer
    to someone who asked in Tagalog or another supported language.
    """
    answer = (source_answer or '').strip()
    if not answer or detected_lang == 'english':
        return answer
    if lang_service.verify_output_language(answer, detected_lang):
        return answer
    if translate and not circuit_breaker.is_open():
        translated = engine._extract_from_answer(
            question,
            source_question or '',
            answer,
            llm_chat,
            detected_lang=detected_lang,
        )
        if translated:
            return translated
    return lang_service.get_translation_unavailable_response(detected_lang)


def _faq_refusal(detected_lang: str) -> Optional[str]:
    if detected_lang == 'english':
        return None
    return lang_service.get_translation_unavailable_response(detected_lang)


def _validated_faq_source_answer(
    question: str,
    row: Sequence[Any],
    detected_lang: str,
    understanding: Optional[QuestionUnderstanding] = None,
    translate: bool = False,
) -> Optional[str]:
    if len(row) < 2 or not row[0]:
        return None
    answer = _source_answer_in_language(
        question, str(row[1] or ''), str(row[0]), detected_lang,
        translate=translate,
    )
    if (
        not answer
        or answer == lang_service.get_translation_unavailable_response(detected_lang)
        or not lang_service.verify_output_language(answer, detected_lang)
    ):
        return None

    source_excerpt = (
        f"FAQ question: {row[1] or ''}\nFAQ answer: {row[0]}"
    )
    if not llm_self_check(question, answer, source_excerpt, understanding):
        logger.info("FAQ source answer rejected by intent/grounding check")
        return None
    return answer


def _answer_personnel_faq(
    question: str,
    detected_lang: str,
    ctx,
    understanding: Optional[QuestionUnderstanding] = None,
):
    """Prefer a FAQ for an explicitly named staff role over generic handbook hits."""
    roles = personnel_titles(question)
    if not roles:
        return None

    candidates = _get_faq_candidates(
        question,
        detected_lang,
        _get_all_faqs(),
        query_variants=understanding.search_queries if understanding else None,
    )
    candidate = next(
        (item for item in candidates if item.get('personnel_title_match')),
        None,
    )
    if not candidate:
        return None

    row = candidate['faq']
    answer = (row[0] or '').strip()
    if not answer:
        return None

    response = None
    if not circuit_breaker.is_open():
        response, _ = engine.smart_extract_answer(
            user_question=question,
            candidates=[candidate],
            detected_lang=detected_lang,
            llm_chat_fn=llm_chat,
            ai_cache=ai_cache,
            understanding=understanding,
        )
    if not response:
        response = _validated_faq_source_answer(
            question, row, detected_lang, understanding
        )
        if not response:
            refusal = _faq_refusal(detected_lang)
            if not refusal:
                return None
            ctx.update(
                user_msg=question,
                bot_response=refusal,
                detected_lang=detected_lang,
                clear_schedule_context=True,
            )
            return (
                refusal,
                row[3] or 'Personnel',
                row[2] if len(row) > 2 else None,
                row[4] if len(row) > 4 else None,
                detected_lang,
            )

    image = row[8] if len(row) > 8 else None
    if image:
        response = f"{response}\n\n[Image: {image}]"
    topic = row[3] or 'Personnel'
    intent = row[4] if len(row) > 4 else None
    faq_id = row[2] if len(row) > 2 else None
    ctx.update(
        topic=topic,
        intent=intent,
        user_msg=question,
        bot_response=response,
        faq_answer=answer,
        faq_question=row[1],
        faq_image=image,
        detected_lang=detected_lang,
        clear_schedule_context=True,
    )
    return response, topic, faq_id, intent, detected_lang


def _log_unanswered(
    question: str,
    ctx=None,
    detected_lang: Optional[str] = None,
    detected_topic: Optional[str] = None,
):
    """PERF: opens its own short-lived connection right when it's actually
    needed, instead of requiring find_answer() to check a connection out
    of the (small, 3-on-Vercel) pool at the very start of a request and
    hold it, unused, through several seconds of sequential AI calls, on
    the off chance this ends up being an unanswered-question miss at the
    very end. Mirrors log_unanswered_standalone()'s existing pattern."""
    from services.conversation import build_context_snapshot
    conn = get_db()
    cur = conn.cursor()
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
                (question, resolved_question, context, detected_language,
                 detected_topic, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
        """, (
            question[:500],
            resolved_question,
            (context_snapshot[:2000] if context_snapshot else None),
            detected_lang or lang_service.detect_language(question),
            detected_topic or lang_service.classify_topic(question),
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"_log_unanswered error: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


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


def log_unanswered_standalone(question: str, session_id: Optional[str] = None):
    ctx = get_conversation_context(session_id) if session_id else None
    _log_unanswered(
        question,
        ctx,
        detected_lang=lang_service.detect_language(
            question, prior_lang=getattr(ctx, "last_detected_lang", None)
        ),
        detected_topic=lang_service.classify_topic(question),
    )


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
        ctx.update(user_msg=question, bot_response=answer, faq_answer=answer, faq_question=original_q,
                    clear_schedule_context=True)
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
                    faq_answer=answer_text, faq_question=original_q,
                    faq_image=img, clear_schedule_context=True)
        return (answer_text, topic, row[2] if len(row) > 2 else None, intent, lang)

    if disamb['type'] == 'date':
        dept = choice.get('dept')
        if not isinstance(dept, str):
            return None
        answer = date_service.answer_for_selected_department(original_q, dept, lang, ctx=ctx)
        if not answer:
            return None
        ctx.update(user_msg=question, bot_response=answer)
        return (answer, 'Schedule', None, 'date_query', lang)

    return None


def _get_query_embedding(question: str):
    """Compute (or fetch from cache) ONE embedding for this question,
    shared between FAQ scoring and handbook-chunk scoring so a single
    message costs at most one embeddings API call, not one per lookup
    site. Embed the original wording, especially for vernacular queries.
    get_embedding() caches by stable digest (ai_cache),
    so calling this more than once with the same question is free after
    the first real call -- but computing it once up front and threading
    it through makes that explicit rather than relying on cache timing.
    Returns None (never raises) if embeddings aren't configured
    (GEMINI_API_KEY unset) -- every caller already treats None as "skip
    semantic scoring, fuzzy/keyword-only like before this feature"."""
    search_q = question or ""
    try:
        return get_embedding(
            search_q, cache=ai_cache, task_type="RETRIEVAL_QUERY"
        )
    except Exception:
        return None


def _handbook_best_score(
    question: str,
    query_embedding=None,
    detected_lang: str = "english",
    understanding: Optional[QuestionUnderstanding] = None,
) -> Tuple[float, list]:
    chunks = handbook_service.get_top_handbook_chunks(
        question,
        query_embedding=query_embedding,
        detected_lang=detected_lang,
        query_variants=understanding.search_queries if understanding else None,
    )
    score = chunks[0]['score'] if chunks else 0.0
    return score, chunks


def _faq_candidates_and_score(
    question: str,
    detected_lang: str,
    query_embedding=None,
    understanding: Optional[QuestionUnderstanding] = None,
) -> Tuple[list, float]:
    all_faqs = _get_all_faqs()
    candidates = _get_faq_candidates(
        question,
        detected_lang,
        all_faqs,
        query_embedding=query_embedding,
        query_variants=understanding.search_queries if understanding else None,
    )
    best_score = candidates[0]['combined'] if candidates else 0.0
    return candidates, best_score


def _score_sources_parallel(
    question: str,
    detected_lang: str,
    query_embedding=None,
    understanding: Optional[QuestionUnderstanding] = None,
) -> dict:
    """PERF: run handbook scoring and FAQ scoring concurrently instead of
    sequentially -- they're independent (separate DB queries + in-process
    scoring, no shared mutable state), so there's no reason to pay their
    latencies back-to-back. Returns the same shape callers were already
    unpacking from the two calls combined:
    {'hb_score', 'hb_chunks', 'faq_candidates', 'faq_score'}.
    Falls back to sequential execution if the pool submission itself fails
    for any reason (never raises past this function).
    """
    try:
        hb_context = copy_context()
        faq_context = copy_context()
        hb_future = _SCORING_EXECUTOR.submit(
            hb_context.run,
            _handbook_best_score, question, query_embedding, detected_lang,
            understanding,
        )
        faq_future = _SCORING_EXECUTOR.submit(
            faq_context.run,
            _faq_candidates_and_score, question, detected_lang, query_embedding,
            understanding,
        )
        hb_score, hb_chunks = hb_future.result()
        faq_candidates, faq_score = faq_future.result()
    except Exception as e:
        logger.debug(f"_score_sources_parallel: pool failed, falling back to sequential: {e}")
        hb_score, hb_chunks = _handbook_best_score(
            question,
            query_embedding=query_embedding,
            detected_lang=detected_lang,
            understanding=understanding,
        )
        faq_candidates, faq_score = _faq_candidates_and_score(
            question,
            detected_lang,
            query_embedding=query_embedding,
            understanding=understanding,
        )
    return {
        'hb_score': hb_score, 'hb_chunks': hb_chunks,
        'faq_candidates': faq_candidates, 'faq_score': faq_score,
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

_NOTE_HEADER_LANG_FALLBACK = {
    'mixed_te': 'tagalog', 'mixed_be': 'bisaya', 'mixed_tb': 'tagalog',
}


def _note_header(kind: str, lang: str) -> str:
    d = _NOTE_HEADERS.get(kind, {})
    lookup_lang = _NOTE_HEADER_LANG_FALLBACK.get(lang, lang)
    return d.get(lookup_lang, d.get('english', ''))


def _try_soft_clarification(question: str, candidates: list, hb_chunks: list,
                             detected_lang: str, ctx) -> Optional[str]:
    """Last resort before the canned "I don't have a confirmed answer"
    fallback. If there are two or more plausible-but-not-confident-enough
    candidates (FAQ or handbook), ask the student to pick instead of
    flatly giving up -- the same way a person skimming 2-3 similarly
    themed entries would say "did you mean X or Y?" rather than shrug.

    Uses the SAME disambiguation machinery as a normal confident-tie
    picker (ctx.set_disambiguation / _try_resolve_pending_disambiguation
    resolves it exactly the same way on the student's next message) but
    at the much looser services.disambiguation.FAQ_SOFT_POLICY /
    HANDBOOK_SOFT_POLICY thresholds -- reserved specifically for this
    "about to say I don't know" moment. A normal, confident answer never
    reaches this function; it only ever runs once every earlier, higher-
    confidence path has already failed.

    Returns None (never raises) if fewer than 2 genuine contenders exist,
    in which case the caller should fall back to the plain "I don't know"
    message as before.
    """
    from services.disambiguation import FAQ_SOFT_POLICY, HANDBOOK_SOFT_POLICY, select_contenders

    try:
        if candidates:
            scores = [c.get('combined', 0.0) for c in candidates]
            contenders = select_contenders(candidates, scores, FAQ_SOFT_POLICY)
            if len(contenders) >= 2:
                dis = engine.build_disambiguation(contenders, detected_lang)
                if dis:
                    ctx.set_disambiguation(
                        'faq', engine.build_disambiguation_options(contenders),
                        original_question=question, detected_lang=detected_lang,
                    )
                    return dis
    except Exception as e:
        logger.debug(f"_try_soft_clarification (faq) skipped: {e}")

    try:
        if hb_chunks:
            scores = [c.get('score', 0.0) for c in hb_chunks]
            contenders = select_contenders(hb_chunks, scores, HANDBOOK_SOFT_POLICY)
            if len(contenders) >= 2:
                dis = handbook_service.build_handbook_disambiguation(contenders, detected_lang)
                if dis:
                    ctx.set_disambiguation(
                        'handbook', handbook_service.build_handbook_disambiguation_options(contenders),
                        original_question=question, detected_lang=detected_lang,
                    )
                    return dis
    except Exception as e:
        logger.debug(f"_try_soft_clarification (handbook) skipped: {e}")

    return None


def _give_up(question: str, candidates: list, hb_chunks: list, detected_lang: str,
             ctx, return_topic: bool):
    """Shared final fallback: try a soft clarification first (see above);
    only fall through to the flat canned "I don't have a confirmed
    answer" message, and only THEN log the miss to unanswered_questions,
    if clarification didn't find anything worth offering either. Logging
    is deliberately skipped when a clarification question is returned --
    that's not an unanswered miss, it's still an active exchange."""
    soft = _try_soft_clarification(question, candidates, hb_chunks, detected_lang, ctx)
    if soft:
        result = (soft, 'Disambiguation', None, None, detected_lang)
        return result if return_topic else result[0]

    _log_unanswered(
        question,
        ctx,
        detected_lang=detected_lang,
        detected_topic=lang_service.classify_topic(question),
    )
    result = (
        lang_service.get_logged_fallback_response(detected_lang),
        'Unknown', None, None, detected_lang,
    )
    return result if return_topic else result[0]


def _chunk_is_schedule_related(chunk: dict) -> bool:
    title = (chunk.get('section_title') or '').lower()
    content = (chunk.get('content') or '')[:600].lower()
    blob = f"{title} {content}"
    return bool(re.search(
        r'\b(holiday|holidays|class suspension|school suspension|'
        r'no class|walang pasok|walay klase|office hours|schedule of|'
        r'attendance|excused absence|special holiday|'
        r'calendar|academic calendar|class days?)\b',
        blob, re.I,
    ))


def _ai_pick_best_source(question: str, hb_chunks: list, faq_candidates: list,
                         llm_chat_fn, date_excerpt: Optional[str] = None) -> str:
    if llm_chat_fn is None:
        return 'date' if date_excerpt else 'both'

    handbook_excerpts = []
    for chunk in hb_chunks[:3]:
        title = (chunk.get('section_title') or '').strip()
        body = (chunk.get('content') or '')[:700].strip()
        if body:
            handbook_excerpts.append(f"{title}: {body}" if title else body)

    faq_excerpts = []
    for candidate in faq_candidates[:3]:
        row = candidate.get('faq') or ()
        faq_question = row[1] if len(row) > 1 else ''
        faq_answer = (row[0] if row else '') or ''
        if faq_question or faq_answer:
            faq_excerpts.append(
                f"Q: {faq_question}\nA: {faq_answer[:500]}"
            )

    available_sources = []
    source_blocks = []
    if handbook_excerpts:
        available_sources.append('HANDBOOK')
        source_blocks.append(
            "--- HANDBOOK EXCERPTS ---\n" + "\n\n".join(handbook_excerpts)
        )
    if faq_excerpts:
        available_sources.append('FAQ')
        source_blocks.append("--- FAQ EXCERPTS ---\n" + "\n\n".join(faq_excerpts))
    if date_excerpt:
        available_sources.append('DATES')
        source_blocks.append(f"--- DYNAMIC DATE/SCHEDULE ANSWER ---\n{date_excerpt[:900]}")

    if not available_sources:
        return 'neither'
    if available_sources == ['faq']:
        return 'faq'
    if available_sources == ['handbook']:
        return 'handbook'
    if available_sources == ['DATES']:
        return 'date'

    choices = "\n".join(
        f"{source} - this source directly answers the question"
        for source in available_sources
    )
    choices += "\nBOTH - two or more sources directly answer different parts"
    choices += "\nNEITHER - none of the excerpts actually answers the question"

    prompt = f"""A student asked: "{question}"

Compare all available sources below. Some excerpts may share words with the
question without actually answering it. Choose a source only when it states
the requested fact, policy, or detail. Do not infer missing course, year,
semester, date, or policy details from a loosely related excerpt.

{chr(10).join(source_blocks)}

Reply with EXACTLY ONE label, nothing else:
{choices}

Answer:"""

    try:
        raw = (llm_chat_fn(prompt, max_tokens=5) or '').strip().upper()
    except Exception as e:
        logger.debug(f"_ai_pick_best_source call failed: {e}")
        return 'date' if date_excerpt else 'both'

    if raw.startswith('HANDBOOK'):
        return 'handbook'
    if raw.startswith('FAQ'):
        return 'faq'
    if raw.startswith('DATE'):
        return 'date'
    if raw.startswith('NEITHER'):
        return 'neither'
    if raw.startswith('BOTH'):
        return 'both'
    logger.warning("Source judge returned an invalid verdict; abstaining")
    return 'neither'


def _merge_with_handbook(base_answer: str, question: str, hb_chunks: list,
                          detected_lang: str,
                          understanding: Optional[QuestionUnderstanding] = None) -> str:
    hb_ans = handbook_service.answer_from_selected_chunk(
        question, hb_chunks[0], detected_lang, understanding=understanding
    )
    if not hb_ans:
        return base_answer
    source_text = f"{hb_chunks[0].get('section_title','')} {hb_chunks[0].get('content','')}"
    if not llm_self_check(question, hb_ans, source_text, understanding):
        logger.info(f"_merge_with_handbook: self-check rejected supplementary note for q={question[:80]!r}")
        return base_answer
    return f"{base_answer}\n\n{_note_header('handbook', detected_lang)} {hb_ans}"


def _merge_with_faq(base_answer: str, question: str, faq_candidates: list,
                     detected_lang: str,
                     understanding: Optional[QuestionUnderstanding] = None) -> str:
    faq_ans, matched = engine.smart_extract_answer(
        user_question=question,
        candidates=faq_candidates,
        detected_lang=detected_lang,
        llm_chat_fn=llm_chat,
        ai_cache=ai_cache,
        understanding=understanding,
    )
    if not faq_ans:
        return base_answer

    matched_row = (matched or faq_candidates[0]).get('faq') if faq_candidates else None
    source_text = (matched_row[0] if matched_row else '') or ''
    if not llm_self_check(question, faq_ans, source_text, understanding):
        logger.info(f"_merge_with_faq: self-check rejected supplementary note for q={question[:80]!r}")
        return base_answer
    return f"{base_answer}\n\n{_note_header('faq', detected_lang)} {faq_ans}"


def _merge_with_date(base_answer: str, date_ans: Optional[str], detected_lang: str) -> str:
    if not date_ans:
        return base_answer
    return f"{base_answer}\n\n{_note_header('date', detected_lang)} {date_ans}"


def _combine_handbook_and_faq(question: str, candidates: list, hb_chunks: list,
                               detected_lang: str,
                               understanding: Optional[QuestionUnderstanding] = None
                               ) -> Optional[str]:
    faq_ans, _ = engine.smart_extract_answer(
        user_question=question,
        candidates=candidates,
        detected_lang=detected_lang,
        llm_chat_fn=llm_chat,
        ai_cache=ai_cache,
        understanding=understanding,
    )
    hb_ans = None
    if hb_chunks:
        hb_ans = handbook_service.answer_from_selected_chunk(
            question, hb_chunks[0], detected_lang, understanding=understanding
        )

        if hb_ans:
            source_text = f"{hb_chunks[0].get('section_title','')} {hb_chunks[0].get('content','')}"
            if not llm_self_check(question, hb_ans, source_text, understanding):
                logger.info(f"_combine_handbook_and_faq: self-check rejected handbook half for q={question[:80]!r}")
                hb_ans = None

    if faq_ans and hb_ans:
        if faq_ans.strip().lower()[:120] == hb_ans.strip().lower()[:120]:
            return faq_ans
        return (
            f"{faq_ans}\n\n"
            f"{_note_header('handbook', detected_lang)} {hb_ans}"
        )
    return faq_ans or hb_ans


def _is_translate_request(question: str) -> bool:
    q = (question or '').lower().strip()
    return bool(re.search(
        r'\b(paki\s*translate|i-?translate|translate\s+(to|into|sa|in)|'
        r'sa\s+bisaya|in\s+bisaya|sa\s+tagalog|in\s+tagalog|'
        r'tagalog\s*(version|naman|please)|bisaya\s*(version|naman|please)|'
        r'cebuano|i-bisaya|i-tagalog|translate\s*(this|that|previous)|'
        r'pwede\s*(ba\s*)?(i-)?translate)\b',
        q, re.I,
    ))


def _translate_previous_answer(question: str, ctx, detected_lang: str) -> Optional[str]:
    prev = (getattr(ctx, 'last_answer', None) or '').strip()
    if not prev or len(prev) < 20:
        return None

    q = question.lower()
    if re.search(r'\b(bisaya|cebuano|i-bisaya|sa\s+bisaya)\b', q):
        target = 'bisaya'
    elif re.search(r'\b(tagalog|filipino|i-tagalog|sa\s+tagalog)\b', q):
        target = 'tagalog'
    else:
        target = detected_lang if detected_lang != 'english' else 'bisaya'

    if circuit_breaker.is_open():
        return prev

    from services.config import bot_identity_line
    from services.grounding import answer_is_grounded

    prompt = f"""{nlg.now_context()}

{bot_identity_line()}

A student previously received this exact answer:
\"\"\"{prev[:2500]}\"\"\"

Now they asked: "{question}"

Rephrase the answer above into natural {nlg.lang_name(target)}. Keep every fact identical — do not add, remove, or change any rule, number, name, or policy. Only change the language and make it sound natural and friendly. {nlg.STYLE_GUIDE}

Translated answer:"""
    try:
        response = llm_chat(prompt, max_tokens=500)
        if not response or not response.strip():
            return prev
        response = response.strip()
        if not answer_is_grounded(response, prev, min_overlap=0.30):
            logger.info("translate_previous: grounding failed")
            return prev
        if not lang_service.verify_output_language(response, target):
            return prev
        return response
    except Exception as e:
        logger.error(f"_translate_previous_answer error: {e}")
        return prev


def _find_answer(question: str, return_topic: bool = False,
                 session_id: Optional[str] = None):
    ctx = get_conversation_context(session_id or 'default')

    is_translation_followup = (
        _is_translate_request(question) and bool(getattr(ctx, 'last_answer', None))
    )
    if not ctx.last_disambiguation and not is_translation_followup:
        exact_language = lang_service.detect_language(
            question, prior_lang=ctx.last_detected_lang
        )
        exact_answer = None
        exact_image = None
        exact_source_question = question
        try:
            from services.corrections import get_exact_correction
            exact_correction = get_exact_correction(get_db, question)
            if exact_correction:
                exact_answer = (
                    exact_correction['answer'],
                    'Corrected',
                    exact_correction.get('faq_id'),
                    None,
                )
                exact_source_question = exact_correction.get('question') or question
            else:
                exact_faq = _find_exact_faq(question)
                if exact_faq:
                    answer, _, faq_id, topic, intent = exact_faq[:5]
                    exact_source_question = exact_faq[1] or question
                    image = exact_faq[8] if len(exact_faq) > 8 else None
                    exact_image = image
                    exact_answer = (answer, topic, faq_id, intent)
        except Exception as e:
            logger.warning("Exact knowledge lookup failed; using normal retrieval: %s", e)

        if exact_answer:
            answer, topic, faq_id, intent = exact_answer
            exact_understanding = understand_question(
                question,
                exact_language,
                context_hint=(
                    ctx.get_context_summary()
                    if ctx and ctx.conversation_history else None
                ),
            )
            answer = _validated_faq_source_answer(
                question,
                (answer, exact_source_question),
                exact_language,
                exact_understanding,
                translate=True,
            )
            if answer:
                if exact_image:
                    answer = f"{answer}\n\n[Image: {exact_image}]"
                ctx.update(
                    topic=topic or 'FAQ',
                    intent=intent,
                    user_msg=question,
                    bot_response=answer,
                    faq_answer=answer,
                    faq_question=question,
                    faq_image=exact_image,
                    detected_lang=exact_language,
                    clear_schedule_context=True,
                )
                result = (
                    answer,
                    topic or 'FAQ',
                    faq_id,
                    intent,
                    exact_language,
                )
                return result if return_topic else result[0]

    _lang_and_class = lang_service.classify_and_detect_language(
        question, prior_lang=ctx.last_detected_lang
    )
    detected_lang = _lang_and_class['language']
    ctx.last_detected_lang = detected_lang

    if ctx.last_disambiguation:
        resolved = _try_resolve_pending_disambiguation(question, detected_lang, ctx)
        if resolved:
            return resolved if return_topic else resolved[0]

    if _is_translate_request(question) and getattr(ctx, 'last_answer', None):
        translated = _translate_previous_answer(question, ctx, detected_lang)
        if translated:
            ctx.update(user_msg=question, bot_response=translated)
            result = (translated, 'Translation', None, None, detected_lang)
            return result if return_topic else result[0]

    if ctx and getattr(ctx, 'last_schedule_query', None):
        follow = date_service.answer_schedule_department_followup(question, ctx)
        if follow:
            ctx.update(user_msg=question, bot_response=follow)
            result = (follow, 'Schedule', None, 'date_query', detected_lang)
            return result if return_topic else result[0]

    try:
        from services.corrections import get_correction
        from database import get_db as _gdb
        correction = get_correction(_gdb, question)
        if correction:
            from services.answer_engine import _extract_from_answer
            src = (correction.get('answer') or '').strip()
            src_for_extract = src if len(src) <= 1200 else (src[:1200].rsplit(' ', 1)[0] + '…')
            correction_understanding = understand_question(
                question,
                detected_lang,
                context_hint=ctx.get_context_summary()
                if ctx and ctx.conversation_history else None,
            )
            reworded = _extract_from_answer(
                question, correction['question'], src_for_extract,
                llm_chat, detected_lang=detected_lang,
                understanding=correction_understanding,
            )
            if reworded:
                final_answer = reworded
            else:
                final_answer = lang_service.get_fallback_response(detected_lang)
            ctx.update(user_msg=question, bot_response=final_answer, clear_schedule_context=True)
            result = (final_answer, 'Corrected', correction.get('faq_id'), None, detected_lang)
            return result if return_topic else result[0]
    except Exception as e:
        logger.debug(f"corrections lookup failed: {e}")

    try:
        from services.answer_memory import get_cached_answer
        from database import get_db as _gdb
        cached = get_cached_answer(_gdb, question, detected_lang)
        if cached:
            cached_answer = cached['answer']
            if (detected_lang == 'english'
                    or lang_service.verify_output_language(cached_answer, detected_lang)):
                ctx.update(user_msg=question, bot_response=cached_answer, clear_schedule_context=True)
                result = (cached_answer, 'Cached', cached.get('faq_id'), None, detected_lang)
                return result if return_topic else result[0]
    except Exception:
        pass

    classification = _lang_and_class

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

    understanding = understand_question(
        question,
        detected_lang,
        context_hint=ctx.get_context_summary() if ctx and ctx.conversation_history else None,
    )

    personnel_result = _answer_personnel_faq(
        question, detected_lang, ctx, understanding=understanding
    )
    if personnel_result:
        return personnel_result if return_topic else personnel_result[0]

    _looks_date = date_service.is_date_question(question, ctx=ctx)
    _looks_hb = handbook_service.is_handbook_question(question, detected_lang)
    _hb_score_early, _hb_chunks_early = (0.0, [])
    _faq_score_early = 0.0
    _faq_cands_early = []
    _early_scoring_done = False
    if _looks_date or _looks_hb:
        try:
            _q_embedding = _get_query_embedding(question)
            scored = _score_sources_parallel(
                question,
                detected_lang,
                query_embedding=_q_embedding,
                understanding=understanding,
            )
            _hb_score_early, _hb_chunks_early = scored['hb_score'], scored['hb_chunks']
            _faq_cands_early, _faq_score_early = scored['faq_candidates'], scored['faq_score']
        except Exception:
            pass
      
        _early_scoring_done = True

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
            signals = {
                'hb_score': _hb_score_early,
                'hb_chunks': _hb_chunks_early,
                'faq_candidates': locals().get('_faq_cands_early') or [],
                'faq_score': _faq_score_early,
            }
            source_pick = _ai_pick_best_source(
                question,
                signals['hb_chunks'],
                signals['faq_candidates'],
                llm_chat,
                date_excerpt=date_ans,
            )
            if source_pick == 'neither':
                return _give_up(
                    question,
                    signals['faq_candidates'],
                    signals['hb_chunks'],
                    detected_lang,
                    ctx,
                    return_topic,
                )
            if source_pick == 'faq' and signals['faq_candidates']:
                candidate = signals['faq_candidates'][0]
                row = candidate['faq']
                if understanding and not circuit_breaker.is_open():
                    answer, _ = engine.smart_extract_answer(
                        user_question=question,
                        candidates=[candidate],
                        detected_lang=detected_lang,
                        llm_chat_fn=llm_chat,
                        ai_cache=ai_cache,
                        understanding=understanding,
                    )
                else:
                    answer = _validated_faq_source_answer(
                        question, row, detected_lang, understanding
                    )
                if answer:
                    ctx.update(
                        topic=row[3] or 'FAQ',
                        intent=row[4] if len(row) > 4 else None,
                        user_msg=question,
                        bot_response=answer,
                        faq_answer=answer,
                        faq_question=row[1],
                        clear_schedule_context=True,
                    )
                    result = (
                        answer, row[3] or 'FAQ', row[2],
                        row[4] if len(row) > 4 else None, detected_lang,
                    )
                    return result if return_topic else result[0]
                return _give_up(
                    question,
                    signals['faq_candidates'],
                    signals['hb_chunks'],
                    detected_lang,
                    ctx,
                    return_topic,
                )
            if source_pick == 'handbook' and signals['hb_chunks']:
                handbook_answer = handbook_service.answer_handbook_question(
                    question, detected_lang, ctx=ctx,
                    understanding=understanding,
                )
                if handbook_answer:
                    result = (
                        handbook_answer, 'Handbook', None, 'handbook_query',
                        detected_lang,
                    )
                    return result if return_topic else result[0]
                return _give_up(
                    question,
                    signals['faq_candidates'],
                    signals['hb_chunks'],
                    detected_lang,
                    ctx,
                    return_topic,
                )

            final_ans = date_ans
            hb_ok = (
                source_pick == 'both'
                and
                signals['hb_score'] >= 0.70
                and signals['hb_chunks']
                and _chunk_is_schedule_related(signals['hb_chunks'][0])
                and not circuit_breaker.is_open()
            )
            if hb_ok:
                final_ans = _merge_with_handbook(
                    final_ans,
                    question,
                    signals['hb_chunks'],
                    detected_lang,
                    understanding=understanding,
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

        if is_followup:
            search_ans, search_topic, search_intent = _answer_followup_via_search(
                question, ctx, detected_lang, understanding=understanding
            )
            if search_ans:
                ctx.update(
                    topic=search_topic, intent=search_intent,
                    user_msg=question, bot_response=search_ans,
                    faq_answer=search_ans, faq_question=question,
                    clear_schedule_context=True,
                )
                result = (search_ans, search_topic or 'Handbook', None, search_intent, detected_lang)
                return result if return_topic else result[0]

            prev_src = ctx.last_faq_answer or ctx.last_answer
            if isinstance(prev_src, str) and prev_src and word_count <= 12:
                from services.answer_engine import _extract_from_answer
                try:
                    re_ans = _extract_from_answer(
                        question,
                        ctx.last_faq_question or last_ex.get('user', ''),
                        prev_src,
                        llm_chat,
                        detected_lang=detected_lang,
                        understanding=understanding,
                    )
                    if re_ans and not _looks_like_non_answer(re_ans):
                        ctx.update(user_msg=question, bot_response=re_ans, clear_schedule_context=True)
                        result = (re_ans, 'Follow-up', None, None, detected_lang)
                        return result if return_topic else result[0]
                except Exception as e:
                    logger.debug(f"same-topic follow-up extract failed: {e}")

            ai_followup = answer_followup_with_ai(
                user_question=question,
                context_summary=ctx.get_context_summary(),
                conversation_history=ctx.get_history(3),
                detected_lang=detected_lang,
                last_faq_answer=ctx.last_faq_answer,
                last_faq_question=ctx.last_faq_question,
                understanding=understanding,
            )

            if ai_followup and not _looks_like_non_answer(ai_followup):
                if not lang_service.verify_output_language(ai_followup, detected_lang):
                    ai_followup = lang_service.get_translation_unavailable_response(detected_lang)
                ctx.update(user_msg=question, bot_response=ai_followup, clear_schedule_context=True)
                result = (ai_followup, 'Follow-up', None, None, detected_lang)
                return result if return_topic else result[0]

    is_hb = handbook_service.is_handbook_question(question, detected_lang)
    parts = split_compound_question(question)
    if len(parts) > 1:
        sub_answers = []
        last_topic, last_intent = None, None
        for part in parts:
            part_class = lang_service.classify_question_ai(part)
            if part_class.get('is_gibberish') or not part_class.get('is_school_related', True):
                continue

            part_understanding = understand_question(
                part,
                detected_lang,
                context_hint=(
                    ctx.get_context_summary()
                    if ctx and ctx.conversation_history else None
                ),
            )
            if not part_understanding:
                continue

            part_ans = None
            if handbook_service.is_handbook_question(part, detected_lang) or is_hb:
                part_ans = handbook_service.answer_handbook_question(
                    part,
                    detected_lang,
                    min_score_to_answer=0.12,
                    ctx=None,
                    understanding=part_understanding,
                )
                if part_ans:
                    sub_answers.append(part_ans)
                    last_topic = 'Handbook'
                    last_intent = 'handbook_query'
                    continue
            r = _lookup_faq_answer(part, detected_lang, part_understanding)
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
                clear_schedule_context=True,
            )
            result = (
                combined,
                last_topic or 'General',
                None,
                last_intent,
                detected_lang,
            )
            return result if return_topic else result[0]

        if len(sub_answers) == 1 and len(parts) >= 2:
            pass

    norm_q = normalize_input(question)
    cache_key = f"{norm_q}_{detected_lang}"
    if not is_hb and not understanding:
        cached = faq_cache.get(cache_key)
        if cached:
            return (
                cached if return_topic
                else (cached[0] if cached else lang_service.get_fallback_response(detected_lang))
            )

    if _early_scoring_done:
        hb_score, hb_chunks = _hb_score_early, _hb_chunks_early
        candidates, faq_best_score = _faq_cands_early, _faq_score_early
    else:
        query_embedding = _get_query_embedding(question)
        hb_score, hb_chunks = _handbook_best_score(
            question,
            query_embedding=query_embedding,
            detected_lang=detected_lang,
            understanding=understanding,
        )
        candidates = _get_faq_candidates(
            question,
            detected_lang,
            _get_all_faqs(),
            query_embedding=query_embedding,
            query_variants=understanding.search_queries if understanding else None,
        )
        faq_best_score = candidates[0]['combined'] if candidates else 0.0

    try:
        word_count = len(question.split())

        if is_hb:
            hb_threshold = 0.12
        elif word_count <= 8:
            hb_threshold = 0.28
        else:
            hb_threshold = 0.35

        tuition_requirements = _tuition_requirements(question)
        if tuition_requirements:
            candidates = [
                candidate for candidate in candidates
                if _matches_tuition_requirements(
                    " ".join(
                        str(candidate['faq'][index] or "")
                        for index in range(min(len(candidate['faq']), 7))
                    ),
                    tuition_requirements,
                )
            ]
            faq_best_score = candidates[0]['combined'] if candidates else 0.0
            hb_chunks = [
                chunk for chunk in hb_chunks
                if _matches_tuition_requirements(
                    f"{chunk.get('section_title', '')} {chunk.get('content', '')}",
                    tuition_requirements,
                )
            ]
            hb_score = hb_chunks[0].get('score', 0.0) if hb_chunks else 0.0
            if not candidates and not hb_chunks:
                generic_answer = _return_generic_tuition_answer(
                    question,
                    detected_lang,
                    ctx,
                    _get_all_faqs(),
                    return_topic,
                )
                if generic_answer:
                    return generic_answer

        faq_margin = 0.28 if is_hb else 0.10
        faq_clearly_better = (
            (not is_hb)
            and candidates
            and faq_best_score >= engine.FUZZY_MIN_SCORE + 0.12
            and faq_best_score >= (hb_score + faq_margin)
        )

        if is_hb:
            faq_clearly_better = False

        handbook_should_lead = (
            hb_chunks
            and not faq_clearly_better
            and (
                (is_hb and hb_score >= 0.08)
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
            r'dormitory fees?|tesda|week of spiritual|morphe|2025 edition|'
            r'social media|ellen white|'
            r'unsay (dapat )?suoton|unsaon pagbisti|pananamit|bisti)\b',
            question, re.I,
        ))

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

        both_solid = False
        if (
            not pure_handbook_topics
            and hb_chunks
            and hb_score >= 0.35
            and candidates
            and faq_best_score >= engine.FUZZY_MIN_SCORE
        ):
            source_pick = _ai_pick_best_source(
                question, hb_chunks, candidates, llm_chat
            ) if not circuit_breaker.is_open() else 'both'
            if source_pick == 'handbook':
                handbook_should_lead = True
            elif source_pick == 'faq':
                is_hb = False
                handbook_should_lead = False
                faq_clearly_better = True
            elif source_pick == 'neither':
                candidates = []
                hb_chunks = []
                faq_best_score = 0.0
                hb_score = 0.0
                handbook_should_lead = False
                faq_clearly_better = False
            elif source_pick == 'both':
                both_solid = True

        if both_solid and not circuit_breaker.is_open():
            combined = _combine_handbook_and_faq(
                question, candidates, hb_chunks, detected_lang,
                understanding=understanding,
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

        selected_source = (
            'handbook' if (handbook_should_lead or is_hb or pure_handbook_topics)
            else 'faq' if candidates
            else 'none'
        )
        logger.debug(
            "Retrieval ranking: language=%s source=%s handbook_score=%.3f "
            "faq_score=%.3f handbook_candidates=%d faq_candidates=%d",
            detected_lang,
            selected_source,
            hb_score,
            faq_best_score,
            len(hb_chunks),
            len(candidates),
        )

        if handbook_should_lead or is_hb or pure_handbook_topics:
            hb_ans = handbook_service.answer_handbook_question(
                question, detected_lang,
                min_score_to_answer=(None if (is_hb or pure_handbook_topics) else hb_threshold),
                ctx=ctx,
                understanding=understanding,
            )
            if hb_ans:
                if (not is_hb and not pure_handbook_topics
                        and faq_best_score >= CROSS_CHECK_FAQ_BAR + 0.08
                        and faq_best_score >= hb_score
                        and candidates and not circuit_breaker.is_open()):
                    hb_ans = _merge_with_faq(
                        hb_ans, question, candidates, detected_lang,
                        understanding=understanding,
                    )
                is_disambiguation = ctx.last_disambiguation is not None
                if not is_disambiguation:
                    ctx.update(
                        user_msg=question,
                        bot_response=hb_ans,
                        faq_answer=hb_ans,
                        faq_question=question,
                        clear_schedule_context=True,
                    )
                result = (hb_ans, 'Handbook', None, 'handbook_query', detected_lang)
                return result if return_topic else result[0]

            if pure_handbook_topics:
                return _give_up(question, candidates, hb_chunks, detected_lang, ctx, return_topic)

        if not candidates:
            hb_ans = handbook_service.answer_handbook_question(
                question, detected_lang, ctx=ctx, understanding=understanding
            )
            if hb_ans:
                if ctx.last_disambiguation is None:
                    ctx.update(
                        user_msg=question,
                        bot_response=hb_ans,
                        faq_answer=hb_ans,
                        faq_question=question,
                        clear_schedule_context=True,
                    )
                result = (hb_ans, 'Handbook', None, 'handbook_query', detected_lang)
                return result if return_topic else result[0]
            return _give_up(question, candidates, hb_chunks, detected_lang, ctx, return_topic)

        if engine.faq_should_disambiguate(candidates, user_question=question):
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
                understanding=understanding,
            )
            if ai_ans:
                best_row = (matched_cand or best)['faq']
                answer, question_text, fid, topic, intent, _, _, _ = best_row[:8]
                img = _image_for_faq_candidate(matched_cand or best, candidates)
                if (hb_score >= CROSS_CHECK_HANDBOOK_BAR + 0.05
                        and hb_score >= best['combined']
                        and hb_chunks and not circuit_breaker.is_open()):
                    ai_ans = _merge_with_handbook(
                        ai_ans, question, hb_chunks, detected_lang,
                        understanding=understanding,
                    )
                ctx.update(
                    topic=topic,
                    intent=intent,
                    user_msg=question,
                    bot_response=ai_ans,
                    faq_answer=answer,
                    faq_question=question_text,
                    faq_image=img,
                    clear_schedule_context=True,
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
            answer, question_text, fid, topic, intent, _, _, _ = row[:8]
            answer = _validated_faq_source_answer(
                question, row, detected_lang, understanding
            )
            if answer:
                img = _image_for_faq_candidate(best, candidates)
                answer_text = answer + (f"\n\n[Image: {img}]" if img else "")
                ctx.update(
                    topic=topic,
                    intent=intent,
                    faq_answer=answer_text,
                    faq_question=question_text,
                    faq_image=img,
                    user_msg=question,
                    bot_response=answer_text,
                    clear_schedule_context=True,
                )
                result_data = (
                    (answer_text, topic, fid, intent, detected_lang)
                    if return_topic else answer_text
                )
                faq_cache.set(cache_key, result_data)
                return result_data

        hb_ans = handbook_service.answer_handbook_question(
            question, detected_lang, ctx=ctx, understanding=understanding
        )
        if hb_ans:
            if ctx.last_disambiguation is None:
                ctx.update(
                    user_msg=question,
                    bot_response=hb_ans,
                    faq_answer=hb_ans,
                    faq_question=question,
                    clear_schedule_context=True,
                )
            result = (hb_ans, 'Handbook', None, 'handbook_query', detected_lang)
            return result if return_topic else result[0]

        return _give_up(question, candidates, hb_chunks, detected_lang, ctx, return_topic)

    except Exception as e:
        logger.error(f"Error in find_answer: {e}", exc_info=True)
        fallback_lang = locals().get('detected_lang', 'english')
        result = (
            lang_service.get_fallback_response(fallback_lang),
            'Unknown', None, None, fallback_lang,
        )
        return result if return_topic else result[0]


def find_answer(question: str, return_topic: bool = False,
                session_id: Optional[str] = None):
    """Answer a question, routing all Bisaya/Cebuano AI work to Gemini only."""
    context = get_conversation_context(session_id or 'default')
    guessed_lang, _ = lang_service._heuristic_language(question)
    if (
        guessed_lang in {'bisaya', 'mixed_be', 'mixed_tb'}
        or (
            len((question or '').split()) <= 2
            and context.last_detected_lang in {'bisaya', 'mixed_be', 'mixed_tb'}
        )
    ):
        with provider_for_language('bisaya'):
            return _find_answer(question, return_topic, session_id)
    return _find_answer(question, return_topic, session_id)