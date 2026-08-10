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
from datetime import datetime

from ai_providers import ai_cache, circuit_breaker, llm_chat
from services.grounding import answer_is_grounded
from services.language import verify_output_language

logger = logging.getLogger("pac_helpdesk.nlg")

_LANG_NAMES = {
    'tagalog': 'Tagalog', 'mixed_te': 'a natural mix of Tagalog and English',
    'bisaya': 'Bisaya/Cebuano', 'mixed_be': 'a natural mix of Bisaya and English',
    'mixed_tb': 'a natural mix of Tagalog and Bisaya', 'english': 'English',
}

# Short, real examples of the tone/register wanted per language — shown
# to the model as few-shot examples rather than just described in an
# instruction. Small/fast models follow a shown example far more
# reliably than a described rule, especially for Bisaya, which is the
# most under-represented of the three in most training data. These are
# generic (not tied to any specific fact), just demonstrating register.
_FEW_SHOT_EXAMPLES = {
    'tagalog': (
        'Halimbawa:\n'
        'Tanong: "May pasok ba bukas?"\n'
        'Sagot: "Wala pong pasok bukas — Foundation Day po kasi, kaya sarado rin '
        'ang mga opisina. Sa Lunes na po ulit ang klase!"'
    ),
    'bisaya': (
        'Pananglitan:\n'
        'Pangutana: "Naa bay klase ugma?"\n'
        'Tubag: "Wala nay klase ugma — Foundation Day man gud, mao nga sirado pud '
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
        'Pangutana: "Naa bay office ugma?"\n'
        'Tubag: "Wala nay office ugma kay Foundation Day man — closed pud ang '
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


STYLE_GUIDE = (
    "Reply the way a helpful person would actually text a friend back — natural, warm, "
    "a little conversational. NOT a formatted announcement, NOT a bulleted status report, "
    "no markdown headers. 1-4 short sentences unless the student is asking for a list."
)


def now_context() -> str:
    """A single line stating the real current date/time, meant to be
    prepended to every AI prompt in the app so the model never has to
    guess (or hallucinate) what 'today', 'this Thursday', etc. actually
    resolve to."""
    now = datetime.now()
    return f"Current real date and time: {now.strftime('%A, %B %d, %Y, %I:%M %p')}."


def lang_name(detected_lang: str) -> str:
    return _LANG_NAMES.get(detected_lang, 'English')


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

    if circuit_breaker.is_open():
        return canonical_text

    cache_key = f"speak_{hash(question)}_{hash(canonical_text)}_{detected_lang}"
    cached = ai_cache.get(cache_key)
    if cached:
        return cached

    prompt = f"""{now_context()}

You are PAC Bot, the student helpdesk assistant for Philippine Advent College (PAC).
A student asked: "{question}"

Here is the exact, factually-correct answer:
"{canonical_text}"

{extra_note}

Rephrase the answer above so it reads naturally, in {lang_name(detected_lang)}. {STYLE_GUIDE}
{("\n" + few_shot_example(detected_lang) + "\n") if detected_lang != 'english' else ""}
STRICT RULE: Do not add, remove, or change any fact — no dates, times, names, amounts,
department names, or open/closed statuses beyond exactly what's in the answer above.
You are only allowed to change wording and tone, never content. If you're not confident
you can do that safely, just return the answer above unchanged.

Rephrased answer:"""

    try:
        response = llm_chat(prompt, max_tokens=200)
        if not response:
            return canonical_text
        response = response.strip()

        if not answer_is_grounded(response, canonical_text, min_overlap=0.4):
            logger.info(f"speak_naturally: rephrase failed local fact-check, using canonical text. "
                        f"canonical={canonical_text!r} rephrase={response!r}")
            return canonical_text

        if not verify_output_language(response, detected_lang):
            logger.info(f"speak_naturally: rephrase drifted language (wanted {detected_lang}), "
                        f"retrying once. rephrase={response!r}")
            retry_prompt = prompt + (
                f"\n\nYour previous attempt was not in {lang_name(detected_lang)} — "
                f"try again, and make sure the reply is actually in {lang_name(detected_lang)} "
                f"this time, not English."
            )
            try:
                retry_response = llm_chat(retry_prompt, max_tokens=200)
            except Exception:
                retry_response = None
            if (retry_response and retry_response.strip()
                    and answer_is_grounded(retry_response.strip(), canonical_text, min_overlap=0.4)
                    and verify_output_language(retry_response.strip(), detected_lang)):
                response = retry_response.strip()
            else:
                logger.info("speak_naturally: retry still didn't land in the right language, "
                            "using canonical text.")
                return canonical_text

        ai_cache.set(cache_key, response)
        return response
    except Exception as e:
        logger.error(f"speak_naturally error: {e}")
        return canonical_text


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
            verdict = llm_chat(prompt, max_tokens=5)
            if verdict:
                v = verdict.strip().upper()
                if v.startswith('FOLLOWUP'):
                    return True
                if v.startswith('NEW'):
                    return False
        except Exception as e:
            logger.error(f"classify_followup AI error: {e}")

    return False