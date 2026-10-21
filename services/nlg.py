"""Shared natural-language helpers used by dates.py, faq.py, and
handbook.py so all three "sound like the same person" and share the same
anti-hallucination safety net.

THE CORE PATTERN (used everywhere in this file):
    canonical fact-based text  -->  AI rephrases it warmly  -->  AI checks
    its own rephrasing didn't change any fact  -->  if it did, throw the
    rephrasing away and use the canonical text as-is.

This is stricter than "compose an answer from JSON facts", which is what
the previous version of dates.py did and why it could still hallucinate
("college has no class Monday" out of nowhere) — the model was
*composing*, which leaves room to invent. Here it's only *paraphrasing* a
sentence that's already 100% correct, which leaves much less room to
drift, and if it does drift anyway we just don't use it.
"""
import logging
import re
from datetime import datetime
from typing import Optional

from ai_providers import ai_cache, circuit_breaker

from ai_providers.providers import llm_chat_fast
from services.grounding import answer_is_grounded
from services.language import (
    get_translation_unavailable_response,
    verify_output_language,
)

logger = logging.getLogger("pac_helpdesk.nlg")

_LANG_NAMES = {
    'tagalog': 'Tagalog', 'mixed_te': 'a natural mix of Tagalog and English',
    'bisaya': 'Bisaya/Cebuano', 'mixed_be': 'a natural mix of Bisaya and English',
    'mixed_tb': 'a natural mix of Tagalog and Bisaya', 'english': 'English',
}

_CASUAL_ADDRESS_TERMS = [
    'bai', 'bro', 'sis', 'pre', 'day', 'uy', 'dawg', 'cuz', 'te', 'kuya', 'ate', 'jowa', 'mars',
]


def detect_casual_address(text: str) -> Optional[str]:
    """Returns the first casual address term the student used (e.g. 'bai',
    'bro'), or None if the message reads as neutral/formal."""
    if not text:
        return None
    t = text.lower()
    for term in _CASUAL_ADDRESS_TERMS:
        if re.search(rf'\b{re.escape(term)}\b', t):
            return term
    return None


def casual_tone_note(question: str) -> str:
    """A short prompt addendum telling the model it's fine (encouraged,
    even) to mirror a casual address term the student already used —
    never invented if they didn't use one."""
    term = detect_casual_address(question)
    if not term:
        return ""
    return (
        f'\nThe student greeted you casually with "{term}" — it\'s good to mirror '
        f'that back naturally (e.g. start your reply with "{term}" too), the way a '
        f'friend texting back would. Keep it playful and warm, but every fact in '
        f'the reply must still be exactly correct.'
    )


STYLE_GUIDE = (
    "Reply like a warm, switched-on staff member texting a student back — "
    "never like a document. Sound like you're glad to help, not reciting a manual. "
    "Never start with 'According to the section…' or formal citations. "
    "HARD RULE for yes/no-shaped questions (starts with Is/Are/Can/Do/Does/Will/May/"
    "Allowed/Pwede/Okay lang ba): your very first word must be 'Yes' or 'No' "
    "(or 'Yes,'/'No,' immediately followed by the reason) — never open with "
    "background, a related rule, or a rephrase of the question instead of "
    "actually answering it first. "
    "Do not answer about something adjacent to what was asked just because it's "
    "similar or nearby in the source — if the SPECIFIC thing the student named "
    "isn't explicitly covered, say so plainly rather than quietly substituting "
    "a related rule as if it answers the question. "
    "Use light structure on ALL answers when it helps readability: "
    "**bold** for key terms or short labels, line breaks between points, "
    "and blank lines between sections (rules, verses, steps). "
    "For lists use one item per line starting with - . "
    "Keep facts exact. Short answers can be 1–3 sentences; longer structure is fine "
    "when listing rules, lyrics, or multi-part info."
)

_FEW_SHOT_EXAMPLES = {
    'tagalog': (
        'Halimbawa:\n'
        'Tanong: "May pasok ba bukas?"\n'
        'Sagot: "Wala pong pasok bukas — Foundation Day po kasi, kaya sarado rin '
        'ang mga opisina. Sa Lunes na po ulit ang klase!"'
    ),
    'bisaya': (
        'Pananglitan:\n'
        'Pangutana: "Bai, naa bay klase ugma?"\n'
        'Tubag: "Bai, wala nay klase ugma — Foundation Day man gud, mao nga sirado pud '
        'ang mga opisina. Sunod Lunes na sad ang klase!"'
    ),
    'mixed_te': (
        'Halimbawa:\n'
        'Tanong: "May klase ba bukas?"\n'
        'Sagot: "Wala pong klase bukas kasi Foundation Day — pero yung Registrar\'s '
        'Office, closed din po. Monday na po ulit ang classes!"'
    ),
    'mixed_be': (
        'Pananglitan:\n'
        'Pangutana: "Bai, naa bay office ugma?"\n'
        'Tubag: "Bai, wala nay office ugma kay Foundation Day man — closed pud ang '
        'Registrar. Monday na sad!"'
    ),
    'mixed_tb': (
        'Halimbawa:\n'
        'Tanong: "Pwede pa ba mag-enroll?"\n'
        'Sagot: "Pwede pa po/pa — until end of the month pa ang late enrollment, '
        'pero naa nay additional fee unya."'
    ),
}


def few_shot_example(detected_lang: str) -> str:
    return _FEW_SHOT_EXAMPLES.get(detected_lang, "")


def now_context() -> str:
    """A single line stating the real current date/time, meant to be
    prepended to every AI prompt in the app so the model never has to
    guess (or hallucinate) what 'today', 'this Thursday', etc. actually
    resolve to."""
    now = datetime.now()
    return f"Current real date and time: {now.strftime('%A, %B %d, %Y, %I:%M %p')}."


def lang_name(detected_lang: str) -> str:
    return _LANG_NAMES.get(detected_lang, 'English')


def _rephrase_is_grounded(
    response: str,
    canonical_text: str,
    question: str,
    detected_lang: str,
) -> bool:
    """Use semantic checking for translations, where English word overlap fails."""
    if detected_lang == 'english':
        return answer_is_grounded(response, canonical_text, min_overlap=0.4)

    from services.ai_retrieval import llm_self_check

    return llm_self_check(question, response, canonical_text)


def speak_naturally(canonical_text: str, question: str, detected_lang: str = 'english',
                     extra_note: str = "") -> str:
    """Paraphrase `canonical_text` (which is already 100% factually
    correct — built deterministically, no AI involved) into something
    that sounds like a person, without letting the model add, remove, or
    change any fact in it.

    Always returns a usable string — falls back to `canonical_text`
    itself if the AI is unavailable or its paraphrase doesn't check out.
    """
    if not canonical_text:
        return canonical_text

    fallback_text = (
        canonical_text if detected_lang == 'english'
        else get_translation_unavailable_response(detected_lang)
    )
    if circuit_breaker.is_open():
        return fallback_text

    cache_key = f"speak_{hash(question)}_{hash(canonical_text)}_{detected_lang}"
    cached = ai_cache.get(cache_key)
    if cached and verify_output_language(cached, detected_lang):
        return cached


    from services.config import bot_identity_line

    lang_directive = (
        f"You must reply ENTIRELY in {lang_name(detected_lang)} — not English, "
        f"not a mix — unless a proper noun has no natural translation.\n"
        if detected_lang != 'english' else ""
    )
    prompt = f"""{now_context()}

    {bot_identity_line()}
    A student asked: "{question}"

    Here is the exact, factually-correct answer:
    "{canonical_text}"

    {extra_note}{casual_tone_note(question)}
    {lang_directive}
    Rephrase the answer above so it reads naturally, in {lang_name(detected_lang)}. {STYLE_GUIDE}
    {("\n" + few_shot_example(detected_lang) + "\n") if detected_lang != 'english' else ""}
    STRICT RULE: Do not add, remove, or change any fact — no dates, times, names, amounts,
    department names, or open/closed statuses beyond exactly what's in the answer above.
    You are only allowed to change wording and tone, never content. If you're not confident
    you can do that safely, just return the answer above unchanged.
    When the student asked in Bisaya or Tagalog, reply fully in that language (not English).

    Rephrased answer:"""

    try:

        response = llm_chat_fast(prompt, max_tokens=200)
        if not response:
            return fallback_text
        response = response.strip()

        if not _rephrase_is_grounded(
            response, canonical_text, question, detected_lang
        ):
            logger.info(f"speak_naturally: rephrase failed local fact-check, using canonical text. "
                        f"canonical={canonical_text!r} rephrase={response!r}")
            return fallback_text

        if not verify_output_language(response, detected_lang):
            logger.info(f"speak_naturally: rephrase drifted language (wanted {detected_lang}), "
                        f"retrying once. rephrase={response!r}")
            retry_prompt = prompt + (
                f"\n\nYour previous attempt was not in {lang_name(detected_lang)} — "
                f"try again, and make sure the reply is actually in {lang_name(detected_lang)} "
                f"this time, not English."
            )
            try:
                retry_response = llm_chat_fast(retry_prompt, max_tokens=200)
            except Exception:
                retry_response = None
            if (retry_response and retry_response.strip()
                    and _rephrase_is_grounded(
                        retry_response.strip(),
                        canonical_text,
                        question,
                        detected_lang,
                    )
                    and verify_output_language(retry_response.strip(), detected_lang)):
                response = retry_response.strip()
            else:
                logger.info("speak_naturally: retry still didn't land in the right language, "
                            "using localized translation-unavailable response.")
                return fallback_text

        ai_cache.set(cache_key, response)
        return response
    except Exception as e:
        logger.error(f"speak_naturally error: {e}")
        return fallback_text


def classify_followup(question: str, last_question: str, last_answer: str) -> bool:
    """Ask the model whether `question` is a follow-up continuing the
    previous exchange, rather than a fresh, unrelated question. Falls
    back to a light heuristic if the AI is unavailable.

    This replaces the old fixed phrase-list heuristic
    (ConversationContext.is_followup), which couldn't tell "how about
    thursday" (a follow-up) apart from "do you know superman" (not one)
    if neither happened to contain one of its hardcoded trigger phrases —
    or, worse, could misfire on ordinary short new questions.
    """
    if not last_question:
        return False

    q = question.strip()
    if not q:
        return False

    words = q.lower().split()
    if len(words) <= 2:
        return True
    followup_phrases = [
        'and the', 'and if', 'also', 'how about', 'what about', 'then what',
        'after that', 'what if', 'so that means', 'so do i', 'so can i',
        'so will i', 'do i also', 'is it required', 'do i need to',
        'ano naman', 'paano naman', 'tapos', 'so ibig sabihin',
        'so meaning', 'meaning', 'ibig sabihin', 'so', 'pano naman',
        'ug ang', 'unya', 'then', 'ok so', 'about that', 'regarding',
        'paano', 'unsaon'
    ]
    ql = q.lower()
    if any(ql.startswith(p) or f' {p} ' in ql for p in followup_phrases):
        return True

    if not circuit_breaker.is_open():
        prompt = f"""Previous exchange:
Student asked: "{last_question}"
Assistant answered: "{(last_answer or '')[:300]}"

New message from the student: "{q}"

Is this new message a FOLLOW-UP that continues/depends on the previous exchange
(e.g. asking about a different day/department for the same topic, saying "how about X",
"what about that", a short reply that only makes sense with the previous context)?
Or is it a NEW, unrelated question/topic on its own?

Reply with exactly one word: FOLLOWUP or NEW."""
        try:
            verdict = llm_chat_fast(prompt, max_tokens=5)
            if verdict:
                v = verdict.strip().upper()
                if v.startswith('FOLLOWUP'):
                    return True
                if v.startswith('NEW'):
                    return False
        except Exception as e:
            logger.error(f"classify_followup AI error: {e}")

    return False