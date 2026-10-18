"""FAQ fuzzy-candidate ranking + AI extraction/disambiguation — v2.

Changes from v1:
- Ollama removed; uses `llm_chat` (Groq -> Gemini).
- No hardcoded bot/institution name — pulled from services.config, so this
  works unmodified for any handbook/FAQ set.
- AI-extracted answers are checked against the source FAQ answer with
  services.grounding before being trusted (same rationale as handbook.py:
  small/fast models drift instead of cleanly saying CANNOT_ANSWER).
- FUZZY_MIN_SCORE / DISAMBIGUATION_GAP replaced by the shared
  services.disambiguation policy (FAQ_POLICY) — disambiguation now only
  fires when there are genuinely two-or-more plausible, similarly-confident
  candidates, not on any close-ish gap between two mediocre matches.
- get_top_faq_candidates() now optionally blends in semantic similarity
  (services.embeddings) on top of the existing fuzzy string scoring — see
  SEMANTIC_BLEND below. Purely additive: pass no query_embedding and this
  behaves exactly as before.
"""
import logging
import re
from typing import Any, List, Dict, Optional, Sequence, Tuple
from fuzzywuzzy import fuzz

from services import nlg
from services.ai_retrieval import llm_pick_faq, llm_self_check
from services.config import bot_identity_line
from services.disambiguation import FAQ_POLICY, select_contenders, should_disambiguate
from services.embeddings import cosine_similarity
from services.grounding import answer_is_grounded
from services.language import verify_output_language

logger = logging.getLogger("pac_helpdesk.answer_engine")


FUZZY_MIN_SCORE = 0.60

SEMANTIC_BLEND = 0.35


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip().lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    return text


def get_top_faq_candidates(user_question: str, expanded_question: str,
                            all_faqs: Sequence[Sequence[Any]],
                            detected_lang: str = 'english', max_candidates: int = 5,
                            query_embedding: Optional[List[float]] = None) -> List[Dict]:
    """`query_embedding`: optional, from services.embeddings.get_embedding()
    on the raw user question. When provided, each FAQ row's OWN cached
    embedding (expected as a 10th tuple element — see faq.py's SELECT,
    which now includes the `embedding` column) is compared via cosine
    similarity and blended into `combined`. Rows with no cached embedding
    yet (not yet backfilled, or GEMINI_API_KEY unset when they were
    saved) just don't get the semantic boost — they still get scored
    fuzzy-only, same as before this feature existed.
    """
    candidates = []
    user_norm = normalize_text(user_question)
    expanded_norm = normalize_text(expanded_question)

    for faq in all_faqs:
        if len(faq) >= 9:
            _, question, _, _, _, kw, _, score_val, _ = faq[:9]
        else:
            _, question, _, _, _, kw, _, score_val, _ = (
                faq[0], faq[1], faq[2], faq[3], faq[4], faq[5], faq[6], faq[7],
                faq[8] if len(faq) > 8 else None)
        faq_embedding = faq[9] if len(faq) >= 10 else None

        if not question:
            continue

        q_norm = normalize_text(question)
        token_ratio = fuzz.token_set_ratio(user_norm, q_norm) / 100.0
        partial_ratio = fuzz.partial_ratio(user_norm, q_norm) / 100.0
        full_ratio = fuzz.ratio(user_norm, q_norm) / 100.0
        expanded_ratio = fuzz.token_set_ratio(expanded_norm, q_norm) / 100.0

        combined = max(
            token_ratio * 0.7 + full_ratio * 0.3,
            partial_ratio * 0.8 + full_ratio * 0.2,
            expanded_ratio * 0.9 + token_ratio * 0.1,
        )

        if kw and user_norm:
            kw_list = kw.lower().split(',')
            keyword_boost = sum(1 for k in kw_list if k.strip() in user_norm) / max(1, len(kw_list))
            combined = min(1.0, combined + keyword_boost * 0.1)

        priority_boost = min(0.05, max(0.0, float(score_val or 0)) / 2000.0)
        combined = min(1.0, combined + priority_boost)

        semantic_score = 0.0
        if query_embedding and faq_embedding:
            semantic_score = max(0.0, cosine_similarity(query_embedding, faq_embedding))
            combined = min(1.0, combined * (1 - SEMANTIC_BLEND) + semantic_score * SEMANTIC_BLEND)

        candidates.append({
            'faq': faq, 'q_text': question, 'q_norm': q_norm,
            'token_ratio': token_ratio, 'partial_ratio': partial_ratio,
            'full_ratio': full_ratio, 'expanded_ratio': expanded_ratio,
            'semantic_score': semantic_score,
            'combined': combined, 'fuzzy': max(token_ratio, partial_ratio, full_ratio),
            'best_q_text': question,
        })

    candidates.sort(key=lambda x: x['combined'], reverse=True)
    return candidates[:max_candidates]


def faq_should_disambiguate(candidates: List[Dict], user_question: Optional[str] = None) -> bool:
    """Before falling back to the disambiguation picker, give the AI
    router (services.ai_retrieval.llm_pick_faq) a chance to break the tie
    on its own -- it reads the actual candidate question texts against
    the student's question, the way a person skimming the shortlist
    would, rather than relying purely on the fuzzy/embedding score gap.
    This is what keeps disambiguation rare even as more FAQs get added
    over time (more FAQs -> more near-score-ties -> more disambiguation
    prompts under the old scores-only check), with nothing to configure
    or hand-tune when new FAQs are added.

    On a confident single pick, mutates `candidates` in place (moves the
    picked FAQ to index 0, the position every caller already reads from)
    and returns False so the caller answers directly instead of asking.
    `user_question` is optional for backward compatibility with existing
    call sites that don't pass it -- without it, this behaves exactly as
    before (score-gap check only).
    """
    scores = [c['combined'] for c in candidates]
    if not should_disambiguate(scores, FAQ_POLICY):
        return False

    if not user_question or not candidates:
        return True

    try:
        contenders = select_contenders(candidates, scores, FAQ_POLICY)
        texts = [c.get('best_q_text') or c.get('q_text') or '' for c in contenders]
        picks = llm_pick_faq(user_question, texts, max_picks=1)
    except Exception as e:
        logger.warning(f"faq_should_disambiguate AI tie-break skipped: {e}")
        return True

    if len(picks) != 1:
        return True  # AI itself is unsure (or picked >1) -- ask the student, same as before

    winner = contenders[picks[0]]
    if winner is not candidates[0]:
        candidates.remove(winner)
        candidates.insert(0, winner)
    candidates[0]['combined'] = max(candidates[0]['combined'], FAQ_POLICY.decisive_score)
    return False


_STOP = {
    'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'are', 'was', 'were',
    'for', 'on', 'at', 'by', 'with', 'as', 'that', 'this', 'it', 'be', 'what',
    'when', 'where', 'who', 'how', 'does', 'did', 'do', 'its', 'their',
    'student', 'students', 'school', 'college', 'handbook', 'manual',
}


def _content_words(text: str) -> set:
    words = re.findall(r"[a-z0-9']{3,}", (text or '').lower())
    return {w for w in words if w not in _STOP}


def _answer_relevant_to_question(question: str, answer: str, min_overlap: int = 1) -> bool:
    """Reject answers that don't touch the question's distinctive words."""
    q_words = _content_words(question)
    a_words = _content_words(answer)
    if not q_words or not a_words:
        return True
    return len(q_words & a_words) >= min_overlap


def _extract_from_answer(user_question: str, best_text: str, answer: str,
                          llm_chat_fn, detected_lang: str = 'english') -> Optional[str]:
    """Shared extract + grounding + relevance check.

    BUG FIX: this prompt previously had no language instruction at all —
    detected_lang was accepted by smart_extract_answer() but silently
    dropped before it ever reached the actual prompt, so FAQ answers had
    no explicit instruction on what language to reply in and relied
    entirely on the model inferring it from user_question on its own.
    Added the same explicit instruction + post-generation language check
    (with one retry) that services/nlg.py's speak_naturally() already
    does for date answers, so FAQ answers get the same treatment.
    """

    lang_instruction = (
        f"You must reply ENTIRELY in {nlg.lang_name(detected_lang)} — not English, not a "
        f"mix — unless a proper noun has no natural translation."
        if detected_lang != 'english' else ""
    )
    few_shot = nlg.few_shot_example(detected_lang) if detected_lang != 'english' else ""
    casual_note = nlg.casual_tone_note(user_question)
    prompt = f"""{nlg.now_context()}

    {bot_identity_line()}

    A student asked: "{user_question}"

    Here is the most relevant FAQ entry:
    Question: {best_text}
    Answer: {answer}

        Instructions:
    - Answer using ONLY the information in the FAQ/source text above. Do not invent names, dates, amounts, or requirements that aren't stated.
    - Answer ONLY what the student asked. Do NOT paste or restate the entire source text.
    - For when/what-year/started/founded questions: give the year/date in 1–2 short sentences. Skip long history unless they asked for the full story.
    - For who/founder questions: name the people in 1–2 short sentences.
    - For yes/no (okay? allowed? can I?): your FIRST WORD must be Yes or No, then a short reason.
    - If the student named a SPECIFIC item and the source only covers a DIFFERENT, similar
      item without actually naming the one asked about, don't answer as if it was covered —
      say plainly it isn't specifically addressed, or reply CANNOT_ANSWER if nothing else applies.
    - Never start with "According to…".
    - If asking for a full list of items, enumerate the relevant items — still keep it concise.
    - {nlg.STYLE_GUIDE}
    - {lang_instruction}
    {few_shot}{casual_note}
    - If the source text doesn't actually contain the information needed to answer the question, respond with exactly: CANNOT_ANSWER

    Answer:"""

    try:
        response = llm_chat_fn(prompt, max_tokens=220)
        if not response or "CANNOT_ANSWER" in response:
            return None
        response = response.strip()

        if not answer_is_grounded(response, answer):
            return None

        if not verify_output_language(response, detected_lang):
            logger.info(f"_extract_from_answer: drifted language (wanted {detected_lang}), "
                        f"retrying once for q={user_question[:80]!r}")
            retry_prompt = prompt + (
                f"\n\nYour previous attempt was not in {nlg.lang_name(detected_lang)} — "
                f"try again, and make sure the reply is actually in "
                f"{nlg.lang_name(detected_lang)} this time, not English."
            )
            try:
                retry_response = llm_chat_fn(retry_prompt, max_tokens=250)
            except Exception:
                retry_response = None
            if (retry_response and retry_response.strip()
                    and "CANNOT_ANSWER" not in retry_response
                    and answer_is_grounded(retry_response.strip(), answer)
                    and verify_output_language(retry_response.strip(), detected_lang)):
                response = retry_response.strip()
            else:
                logger.info("_extract_from_answer: retry still didn't land in the right "
                            "language; returning None so the caller falls back "
                            "(e.g. to the raw FAQ answer text).")
                return None

        if not _answer_relevant_to_question(user_question, response, min_overlap=1):
            logger.info(f"answer-relevance REJECTED for q={user_question[:80]!r}")
            return None


        if not llm_self_check(user_question, response, answer):
            logger.info(f"answer self-check REJECTED for q={user_question[:80]!r}")
            return None

        return response
    except Exception as e:
        print(f"_extract_from_answer error: {e}")
        return None

def smart_extract_answer(user_question: str, candidates: List[Dict], detected_lang: str = 'english',
                          llm_chat_fn=None, ai_cache=None
                          ) -> Tuple[Optional[str], Optional[Dict]]:
    if not candidates or llm_chat_fn is None:
        return None, None

    cache_key = f"extract_{hash(user_question)}_{detected_lang}"
    if ai_cache:
        cached = ai_cache.get(cache_key)
        if cached:
            if verify_output_language(cached, detected_lang):
                return cached, None

    best = candidates[0]
    best_text = best['best_q_text']
    faq = best['faq']
    answer = faq[0]

    if not answer:
        return None, None

    response = _extract_from_answer(user_question, best_text, answer, llm_chat_fn,
                                     detected_lang=detected_lang)
    if not response:
        return None, None
    if ai_cache:
        ai_cache.set(cache_key, response)
    return response, best


def answer_from_selected_faq(original_question: str, faq_row: tuple,
                              llm_chat_fn=None, detected_lang: str = 'english') -> Optional[str]:
    """Answer using exactly the FAQ row the student picked out of a
    disambiguation list, grounded in `original_question` (the question
    that triggered the disambiguation) rather than whatever short text
    they typed to make their selection."""
    if llm_chat_fn is None:
        return None
    answer = faq_row[0]
    best_text = faq_row[1]
    if not answer:
        return None
    return _extract_from_answer(original_question, best_text, answer, llm_chat_fn,
                                 detected_lang=detected_lang)


def build_disambiguation(candidates: List[Dict], detected_lang: str = 'english') -> str:
    if not candidates:
        return "I'm not sure what you're asking. Could you please rephrase your question?"
    headers = {
        'english': "I found a couple of possible matches. Which one did you mean?",
        'tagalog': "Mayroon akong ilang posibleng tugma. Alin ang ibig mong sabihin?",
        'bisaya': "Naa koy daghang posible nga tugma. Asa ang imong gipasabot?",
        'mixed_te': "I found a couple of possible matches. Alin po ang ibig niyong sabihin?",
        'mixed_be': "Naa koy daghang posible. Asa ang inyong gipasabot?",
        'mixed_tb': "Naa koy pipila ka posibleng tugma. Asa niini ang imong gipasabot?",
    }
    header = headers.get(detected_lang, headers['english'])
    scores = [c['combined'] for c in candidates]
    contenders = select_contenders(candidates, scores, FAQ_POLICY)
    options = [f"{i}. {c['best_q_text'][:80]}" for i, c in enumerate(contenders, 1)]
    return f"{header}\n\n" + "\n".join(options)


def build_disambiguation_options(candidates: List[Dict]) -> List[Dict]:
    """[{'label': ..., 'faq': faq_row}, ...] for ConversationContext.set_disambiguation,
    kept in sync (same contenders, same labels) with build_disambiguation."""
    scores = [c['combined'] for c in candidates]
    contenders = select_contenders(candidates, scores, FAQ_POLICY)
    return [
        {'label': c['best_q_text'][:80], 'faq': c['faq']}
        for c in contenders
    ]