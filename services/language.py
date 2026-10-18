"""Language detection, topic classification, abbreviation expansion, and
admin-managed greeting/fallback/non-school responses.
"""
import hashlib
import logging
import re
from threading import Lock
from typing import Dict, Optional, Tuple

from ai_providers import ai_cache, circuit_breaker
# PERF: cheap classification calls (language id, gibberish/on-topic check)
# route through the small/fast model instead of the main answer-generation
# model -- see ai_providers/providers.py's llm_chat_fast() docstring.
from ai_providers.providers import llm_chat_fast
from database import get_db
from services.config import bot_identity_line

logger = logging.getLogger("pac_helpdesk.language")

LANGUAGE_CONFIG = {
    'english': {
        'name': 'English',
        'system_prompt': (
            "You are PAC Bot, a friendly student helpdesk assistant "
            "at Philippine Advent College."
        ),
        'instruction': (
            "Answer in natural, conversational English. Be warm and helpful. "
            "When the student asks for a policy or definition, give the full clear answer."
        ),
        'greeting': (
            "Hello! I'm PAC Bot, your student helpdesk assistant. "
            "How can I help you today?"
        ),
        'fallback': (
            "I'm not sure about that yet. You can visit the Registrar's "
            "Office or ask your academic adviser for more information."
        ),
        'non_school': (
            "I'm the PAC student helpdesk — I can help with schedules, "
            "handbook rules, enrollment, tuition, clearance, and other "
            "school services. What would you like to know about PAC?"
        ),
    },
    'tagalog': {
        'name': 'Tagalog',
        'system_prompt': (
            "Ikaw ay PAC Bot, isang magiliw na student helpdesk assistant "
            "sa Philippine Advent College."
        ),
        'instruction': (
            "Sumagot sa TAGALOG nang natural at palakaibigan. "
            "Maging mainit at matulungin. Kung patakaran o definition ang "
            "tinatanong, ibigay ang buong malinaw na sagot."
        ),
        'greeting': (
            "Kumusta! Ako si PAC Bot, ang inyong student helpdesk assistant. "
            "Paano kita matutulungan ngayon?"
        ),
        'fallback': (
            "Hindi pa ako sigurado doon. Maaari kang bumisita sa "
            "Registrar's Office o tanungin ang iyong akademikong tagapayo "
            "para sa karagdagang impormasyon."
        ),
        'non_school': (
            "Ako ang PAC student helpdesk. Matutulungan kita sa schedule, "
            "handbook, enrollment, tuition, clearance, at iba pang school "
            "concerns. Ano ang itatanong mo tungkol sa PAC?"
        ),
    },
    'bisaya': {
        'name': 'Bisaya',
        'system_prompt': (
            "Ikaw usa ka PAC Bot, usa ka mahigalaon nga student helpdesk "
            "assistant sa Philippine Advent College."
        ),
        'instruction': (
            "Tubaga sa BISAYA/CEBUANO nga natural ug mainit. "
            "Pagkamahigalaon ug matabangon. Kung policy o definition ang "
            "pangutana, ihatag ang kompleto ug klaro nga tubag."
        ),
        'greeting': (
            "Kumusta! Ako si PAC Bot, ang imong student helpdesk assistant. "
            "Unsaon nako ikaw tabangan karon?"
        ),
        'fallback': (
            "Wala pa ko kabalo niana. Mahimo kang moadto sa Registrar's "
            "Office o pangutan-on ang imong akademikong magtatambag alang "
            "sa dugang impormasyon."
        ),
        'non_school': (
            "Ako ang PAC student helpdesk. Makatabang ko nimo sa schedule, "
            "handbook, enrollment, tuition, clearance, ug uban pang school "
            "concerns. Unsa may imong pangutana bahin sa PAC?"
        ),
    },
    'mixed_te': {
        'name': 'Tagalog-English',
        'system_prompt': (
            "Ikaw ay PAC Bot, isang friendly student helpdesk assistant "
            "sa Philippine Advent College."
        ),
        'instruction': (
            "Answer naturally mixing Tagalog and English. "
            "Be warm and conversational."
        ),
        'greeting': "Hello! Ako si PAC Bot. Paano kita matutulungan today?",
        'fallback': (
            "Hindi pa ako sure doon. Visit the Registrar's Office or ask "
            "your academic adviser."
        ),
        'non_school': (
            "Sorry po, PAC-related questions lang po ang kaya kong sagutin "
            "— like enrollment, tuition, grades, scholarships, and student "
            "handbook."
        ),
    },
    'mixed_be': {
        'name': 'Bisaya-English',
        'system_prompt': (
            "Ikaw usa ka PAC Bot, usa ka friendly student helpdesk "
            "assistant sa Philippine Advent College."
        ),
        'instruction': (
            "Answer naturally mixing Bisaya and English. "
            "Be warm and conversational."
        ),
        'greeting': "Hello! Ako si PAC Bot. Unsaon nako ikaw tabangan today?",
        'fallback': (
            "Wala pa ko kabalo niana. Visit the Registrar's Office or ask "
            "your academic adviser."
        ),
        'non_school': (
            "Sorry, PAC-related questions lang ang akong matubag — like "
            "enrollment, tuition, grades, scholarships, and student handbook."
        ),
    },
    'mixed_tb': {
        'name': 'Tagalog-Bisaya',
        'system_prompt': (
            "Ikaw ay PAC Bot, isang magiliw na student helpdesk assistant "
            "sa Philippine Advent College."
        ),
        'instruction': (
            "Answer naturally mixing Tagalog and Bisaya. "
            "Be warm and conversational."
        ),
        'greeting': (
            "Kumusta! Ako si PAC Bot. "
            "Paano/Unsaon nako ikaw matutulungan/tabangan?"
        ),
        'fallback': (
            "Hindi pa ako sure/Wala pa ko kabalo niana. "
            "Visit the Registrar's Office or ask your academic adviser."
        ),

        'non_school': (
            "Ako ang PAC student helpdesk. Matutulungan/Makatabang ko nimo sa "
            "schedule, handbook, enrollment, tuition, clearance, ug uban pang "
            "school concerns. Ano/Unsa ang itatanong mo bahin sa PAC?"
        ),
    },
}

_LANG_CACHE: Dict[str, str] = {}
_LANG_CACHE_LOCK = Lock()
_LANG_CACHE_MAX = 2000


def _lang_cache_key(text: str) -> str:
    return hashlib.md5(text[:120].lower().encode()).hexdigest()


def _cache_lang(key: str, value: str) -> None:
    with _LANG_CACHE_LOCK:
        if len(_LANG_CACHE) >= _LANG_CACHE_MAX:
            for k in list(_LANG_CACHE.keys())[:200]:
                del _LANG_CACHE[k]
        _LANG_CACHE[key] = value


def _heuristic_language(text: str) -> Tuple[str, bool]:
    """Regex-based language guess. Returns (label, confident).

    `confident=True` means the heuristic result is trustworthy enough to
    skip an AI call entirely — reserved for the common case of plain
    English or text with no recognizable dialect markers at all. Anything
    that looks like it contains Tagalog/Bisaya (where getting the exact
    mixed_xx label right actually matters for template selection) is left
    for the caller to optionally confirm with AI.

    Word lists expanded from the original ~10 words each — those covered
    only the most obvious markers (paano/bakit/unsa/ngano) and missed a
    lot of everyday phrasing (wala, gusto, pwede, naa, dinhi, lagi, etc.),
    so short casual messages using only common words like those were
    falling through to 'english' with confident=True and never reaching
    the AI fallback at all.

    Note: a handful of words (wala, oo, sige, gusto, pwede, salamat) show
    up in both lists on purpose — they're genuinely used in both Tagalog
    and Bisaya casual speech (shared vocabulary / Spanish loanwords), not
    a copy-paste mistake. A message using only one of these gets flagged
    as low-confidence 'mixed_tb' and goes to the AI classifier for a real
    answer when one is available (see detect_language() below) — this
    heuristic only has to be a reasonable guess for the case where the AI
    is unreachable, not the final word.
    """
    t = text.lower()
    has_tagalog = bool(re.search(
        r'\b(po|opo|paano|bakit|magkano|kumusta|kamusta|salamat|ano|saan|kailan'
        r'|pwede|puwede|gusto|ayaw|wala|meron|mayroon|yung|yun|ito|iyon|naman'
        r'|talaga|lang|din|rin|ba|ko|mo|niya|natin|nila|siya|sila|kayo|tayo'
        r'|hindi|oo|sige|kasi|kailangan|dapat|maaari|tulungan|tulong'
        r'|mag[a-z]{2,}|nag[a-z]{2,})\b',
        t,
    ))
    has_bisaya = bool(re.search(
        r'\b(unsa|unsay|ngano|giunsa|jud|gyud|pud|usab|kaayo|mao|amo|kinsa|asa|kanus-a'
        r'|naa|wala|nay|naay|dili|oo|sige|lagi|bitaw|diha|dinhi|adto|karon'
        r'|buhat|himoa|palihug|salamat|maayo|unta|gusto|ganahan|pwede'
        r'|mo[a-z]{3,})\b',
        t,
    ))
    has_english = bool(re.search(
        r'\b(what|how|why|when|where|the|is|are|can|will|would|please|thank)\b',
        t,
    ))

    has_strong_bisaya = bool(re.search(
        r'\b(unsay|gikan|hangtod|hangtud|kutob|sulod|og|bai|bako|suoton)\b', t
    )) or bool(
        re.search(r'\bpila\b', t)
        and re.search(r'\b(tuition|fee|bayad|gasto|cost|price)\b', t)
    )
    has_strong_tagalog = bool(re.search(
        r'\b(ako|magsuot|mag[a-z]{2,}|nag[a-z]{2,}|magkano|paano|bakit|'
        r'naman|talaga|hindi|yung|iyon|opo)\b', t
    ))
    if has_strong_bisaya:
        if has_strong_tagalog:
            return 'mixed_tb', False
        if has_english:
            return 'mixed_be', False
        return 'bisaya', True
    if has_strong_tagalog:
        return 'tagalog', True

    if has_tagalog and has_bisaya:
        return 'mixed_tb', False
    if has_tagalog and has_english:
        return 'mixed_te', False
    if has_bisaya and has_english:
        return 'mixed_be', False
    if has_tagalog:
        return 'tagalog', False
    if has_bisaya:
        return 'bisaya', False

    return 'english', True


def verify_output_language(text: str, expected_lang: str) -> bool:
    """Cheap post-generation check: did the model actually reply in the
    language it was asked to? Same pattern as services/grounding.py's
    fact-check, just checking language instead of content — smaller/
    faster models (the whole reason to be on Groq) are noticeably more
    likely to drift into English mid-reply when asked for Bisaya
    specifically, since it's the most under-represented of the three in
    most training data.

    Reuses _heuristic_language() rather than another AI call — this only
    needs to catch obvious drift, not make a fine judgment call, and
    burning a second AI call to check the first AI call's language would
    double latency for not much extra accuracy.
    """
    if not text or not text.strip():
        return False
    if expected_lang == 'english':
        return True  # nothing to verify against
    if expected_lang not in LANGUAGE_CONFIG:
        return True  # unknown label, don't block on it

    guess, _ = _heuristic_language(text)
    if guess == expected_lang:
        return True

    mixed_components = {
        'mixed_te': {'tagalog', 'mixed_te'},
        'mixed_be': {'bisaya', 'mixed_be'},
        'mixed_tb': {'tagalog', 'bisaya', 'mixed_tb'},
    }
    return guess in mixed_components.get(expected_lang, set())


def detect_language(text: str, prior_lang: Optional[str] = None) -> str:
    """`prior_lang`: the conversation's most recently established
    language (ConversationContext.last_detected_lang), if any.

    Very short/ambiguous messages ("opo", "3", "sige", a disambiguation
    pick) carry almost no signal on their own — the heuristic can't tell,
    and an AI classification call has little more to work with either.
    Rather than guess cold, trust the conversation's already-established
    language for these. Deliberately NOT run through the global
    text->language cache below (unlike every other branch here) — the
    right answer for a 1-2 word reply depends on which conversation it's
    in, not just the words themselves, so caching it globally would leak
    one student's context into another's.
    """
    if not text or not isinstance(text, str):
        return prior_lang or 'english'

    guess, confident = _heuristic_language(text)

    word_count = len(text.split())
    if not confident and word_count <= 2 and prior_lang:
        return prior_lang

    key = _lang_cache_key(text)
    with _LANG_CACHE_LOCK:
        if key in _LANG_CACHE:
            return _LANG_CACHE[key]

    if confident or circuit_breaker.is_open():
        _cache_lang(key, guess)
        return guess

    prompt = (
        "Classify the language of this text. Reply with ONLY one label: "
        "english | tagalog | bisaya | mixed_te | mixed_be | mixed_tb\n\n"
        f"Text: {text[:200]}\n\nLabel:"
    )
    try:
        # PERF: was llm_chat() (main/slow model) for a single-word label --
        # llm_chat_fast() gives the same answer at a fraction of the latency.
        raw = llm_chat_fast(prompt, max_tokens=8).strip().lower()
        label_map = {
            'english': 'english', 'tagalog': 'tagalog',
            'bisaya': 'bisaya', 'cebuano': 'bisaya',
            'mixed_te': 'mixed_te', 'mixed_tb': 'mixed_tb',
            'mixed_be': 'mixed_be', 'filipino': 'tagalog',
        }
        for token, label in label_map.items():
            if token in raw:
                _cache_lang(key, label)
                logger.debug(f"detect_language: AI-classified {text[:40]!r} -> {label}")
                return label
    except Exception:
        pass

    _cache_lang(key, guess)
    logger.debug(f"detect_language: heuristic fallback {text[:40]!r} -> {guess}")
    return guess


def classify_topic(question: str) -> str:
    topics = {
        'enrollment': [
            'enroll', 'registration', 'admission', 'enrolment', 'inscription',
        ],
        'tuition': ['tuition', 'fee', 'payment', 'cost', 'price', 'bayad'],
        'scholarship': [
            'scholarship', 'scholar', 'financial aid', 'assistance', 'iskolar',
        ],
        'grades': ['grades', 'grade', 'gpa', 'gwa', 'rating', 'mark'],
        'clearance': ['clearance', 'clear'],
        'schedule': ['schedule', 'class', 'time', 'calendar', 'iskedyul'],
        'handbook': [
            'handbook', 'rule', 'rules', 'regulation', 'regulations', 'policy',
            'policies', 'code of conduct', 'discipline', 'violation',
            'dress code', 'uniform', 'patakaran', 'bawal',
        ],
    }
    q_lower = question.lower()
    for topic, keywords in topics.items():
        if any(kw in q_lower for kw in keywords):
            return topic
    return 'general'


_DB_RESPONSE_CACHE: Dict[str, str] = {}
_DB_RESPONSE_LOCK = Lock()
_DB_RESPONSE_TABLE_MAP = {
    'greetings': ('greetings', 'greeting_text'),
    'fallbacks': ('fallback_responses', 'response_text'),
    'non_school': ('non_school_responses', 'response_text'),
}


def get_db_response(table_key: str, language: str) -> str:
    """Admin-managed response text, falling back to LANGUAGE_CONFIG."""
    lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG['english'])
    default = {
        'greetings': lang_config['greeting'],
        'fallbacks': lang_config['fallback'],
        'non_school': lang_config['non_school'],
    }.get(table_key, lang_config['fallback'])

    table_column = _DB_RESPONSE_TABLE_MAP.get(table_key)
    if not table_column:
        return default
    table, column = table_column

    cache_key = f"{table_key}:{language}"
    with _DB_RESPONSE_LOCK:
        if cache_key in _DB_RESPONSE_CACHE:
            return _DB_RESPONSE_CACHE[cache_key]

    result = default
    try:
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute(
                f"SELECT {column} FROM {table} "
                f"WHERE language=%s AND is_active=1",
                (language,),
            )
            row = cur.fetchone()
            if row and row[0]:
                result = row[0]
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        print(
            f"get_db_response DB lookup failed for "
            f"{table_key}/{language}, using default: {e}"
        )

    with _DB_RESPONSE_LOCK:
        _DB_RESPONSE_CACHE[cache_key] = result
    return result


def get_greeting(language: str) -> str:
    return get_db_response('greetings', language)


def get_fallback_response(language: str) -> str:
    return get_db_response('fallbacks', language)


def get_non_school_response(language: str) -> str:
    return get_db_response('non_school', language)


LOGGED_FALLBACK_MESSAGES = {
    'english': (
        "Hmm, I couldn't find a solid answer for that one — I've saved your "
        "question so our staff can add it soon. In the meantime, feel free to "
        "try rephrasing it, or ask me something else!"
    ),
    'tagalog': (
        "Hmm, wala akong makitang malinaw na sagot dyan — na-save ko na ang "
        "tanong mo para madagdag ito ng aming staff. Habang hinihintay, "
        "subukan mo rin siyang i-rephrase, o magtanong ka na lang ng iba!"
    ),
    'bisaya': (
        "Hmm, wala koy nakit-an nga klaro nga tubag ana — gi-save nako ang "
        "imong pangutana aron madugang sa among staff. Samtang naghulat, "
        "sulayi og lahi nga pagkasulti, o pangutana lang og lain!"
    ),
    'mixed_te': (
        "Hmm, wala akong nakitang malinaw na sagot dyan — but I've saved "
        "your question so it can be added soon. Try rephrasing it, or "
        "ask me something else in the meantime!"
    ),
    'mixed_be': (
        "Hmm, wala koy nakit-an nga klaro nga tubag ana — but gi-save nako "
        "ang imong pangutana para madugang unya. Sulayi og lahi nga "
        "pagkasulti, o pangutana og lain samtang naghulat!"
    ),
    'mixed_tb': (
        "Hmm, wala koy/akong nakit-an nga klaro nga tubag/sagot ana — "
        "Gi-save/Na-save na nako/ko kini para madugang/madagdag unya. "
        "Sulayi og/ng laing paagi sa pagsulti, o pangutana og/ng lain "
        "samtang naghulat!"
    ),
}


def get_logged_fallback_response(language: str) -> str:
    return LOGGED_FALLBACK_MESSAGES.get(
        language, LOGGED_FALLBACK_MESSAGES['english']
    )


_TRANSLATION_UNAVAILABLE_MESSAGES = {
    'english': (
        "I found relevant information, but I can't reliably translate it right now. "
        "Please try again later."
    ),
    'tagalog': (
        "May nahanap akong kaugnay na impormasyon, pero hindi ko ito maisalin nang "
        "maaasahan sa ngayon. Pakisubukan muli mamaya."
    ),
    'bisaya': (
        "Nakakita ko og may kalabotang impormasyon, apan dili pa ko kasaligan nga "
        "makahubad niini karon. Palihog sulayi pag-usab unya."
    ),
    'mixed_te': (
        "May nakita akong related information, pero hindi ko ito ma-translate "
        "nang maaasahan ngayon. Please try again later."
    ),
    'mixed_be': (
        "Nakita nako ang related information, pero dili pa ko kasaligan nga "
        "makatranslate niini karon. Please try again later."
    ),
    'mixed_tb': (
        "Nakita nako ang impormasyon, pero dili pa ko kasaligan nga mahubad "
        "kini karon. Pakisulay pag-usab unya."
    ),
}


def get_translation_unavailable_response(language: str) -> str:
    return _TRANSLATION_UNAVAILABLE_MESSAGES.get(
        language, _TRANSLATION_UNAVAILABLE_MESSAGES['english']
    )


UNCLEAR_INPUT_MESSAGES = {
    'english': (
        "Hmm, I couldn't quite understand that — could you try rephrasing "
        "your question?"
    ),
    'tagalog': (
        "Hindi ko masyadong naintindihan iyan — pwede mo bang i-rephrase "
        "ang tanong mo?"
    ),
    'bisaya': (
        "Wala nako kasabot ana — pwede ba nimo i-rephrase ang imong pangutana?"
    ),
    'mixed_te': "Sorry, hindi ko gets — could you rephrase your question?",
    'mixed_be': "Sorry, wala ko kasabot — pwede ba nimo i-rephrase?",
    'mixed_tb': (
        "Wala nako/ako kasabot/naintindihan — pwede ba/mo i-rephrase?"
    ),
}


def get_unclear_input_response(language: str) -> str:
    return UNCLEAR_INPUT_MESSAGES.get(
        language, UNCLEAR_INPUT_MESSAGES['english']
    )


_VOWEL_RE = re.compile(r'[aeiou]', re.IGNORECASE)


def _obviously_not_gibberish_and_school_related(question: str) -> bool:
    """Fast, non-AI pre-check for the overwhelming majority of real
    messages: real words (has vowels, reasonable length, not a single
    mashed token) that also contain a recognizable school-topic keyword.
    Only questions that DON'T pass this get the (slower) AI classifier —
    genuine edge cases (off-topic questions, actual gibberish) are rare
    enough that this doesn't hurt accuracy in practice.
    """
    q = (question or '').strip()
    if len(q) < 2:
        return False
    words = re.findall(r"[a-zA-Z']+", q)
    if not words:
        return False
    if not any(_VOWEL_RE.search(w) for w in words):
        return False
    topical = (
        'enroll', 'admission', 'tuition', 'fee', 'payment', 'scholar',
        'grade', 'gwa', 'gpa', 'clearance', 'schedule', 'class', 'exam',
        'requirement', 'registrar', 'transcript', 'tor', 'id', 'uniform',
        'dress code', 'handbook', 'rule', 'policy', 'office', 'campus',
        'holiday', 'vacation', 'break', 'calendar', 'deadline', 'dorm',
        'hostel', 'library', 'subject', 'professor', 'teacher', 'faculty',
        'section', 'dean', 'department', 'graduation', 'diploma',
        'hi', 'hello', 'kumusta', 'kamusta', 'salamat', 'thank',
        'paano', 'unsa', 'ano', 'kailan', 'magkano', 'saan', 'asa',
    )
    ql = q.lower()
    return any(t in ql for t in topical)


def classify_question_ai(question: str) -> dict:
    """
    Returns {'is_gibberish': bool, 'is_school_related': bool}
    When AI providers fail, default to treating the question as valid + school-related.

    Fast paths (no AI call, no network round trip):
      1. Obviously real + on-topic message (has a recognizable school
         keyword) -> {False, True} immediately.
      2. Result cache -- this is a pure function of the question text, so
         repeated/common questions ("hi", "schedule?", etc.) across ANY
         session are answered from cache after the first time.
    Only genuinely ambiguous messages pay for an AI round trip.
    """
    q = (question or '').strip()
    if _obviously_not_gibberish_and_school_related(q):
        return {'is_gibberish': False, 'is_school_related': True}

    cache_key = f"classify_q_{hash(q.lower())}"
    cached = ai_cache.get(cache_key)
    if cached:
        return cached

    if not circuit_breaker.is_open():
        prompt = f"""A student typed this message to a school helpdesk chatbot.
{bot_identity_line()} The student may write in English, Tagalog, Bisaya/Cebuano,
or a mix of these.

Message: "{question}"

Answer two questions about it:
1. GIBBERISH: Is this random keyboard-mashing / not real words in ANY language
   (English, Tagalog, or Bisaya)? A short or informal real message (even just one
   real word, a greeting, or slang) is NOT gibberish.
2. SCHOOL: Is this message plausibly related to the school in some way (enrollment,
   tuition, grades, clearance, scholarships, class schedules, office hours, campus
   rules/handbook, or is it a greeting/small talk directed at the school assistant)?
   If unsure, prefer SCHOOL=yes over no — only mark it no if it is CLEARLY about
   something unrelated (e.g. celebrities, general trivia, other companies, math
   homework unrelated to school policy).

Reply with EXACTLY this format, nothing else:
GIBBERISH: yes/no
SCHOOL: yes/no"""
        try:
            # PERF: small yes/no classification -- fast model, not the main one.
            response = llm_chat_fast(prompt, max_tokens=20)
            if response:
                r = response.strip().lower()
                gibberish_match = re.search(r'gibberish:\s*(yes|no)', r)
                school_match = re.search(r'school:\s*(yes|no)', r)
                if gibberish_match and school_match:
                    result = {
                        'is_gibberish': gibberish_match.group(1) == 'yes',
                        'is_school_related': school_match.group(1) == 'yes',
                    }
                    ai_cache.set(cache_key, result)
                    return result
        except Exception as e:
            print(f"classify_question_ai error, falling back to safe defaults: {e}")

    return {
        'is_gibberish': False,
        'is_school_related': True,
    }


def classify_and_detect_language(question: str, prior_lang: Optional[str] = None) -> dict:
    """PERF: combined replacement for detect_language() + classify_question_ai()
    for callers (see services/faq.py's find_answer()) that need BOTH results
    up front. Both functions already have fast paths (heuristic language
    guess, obvious-keyword topic check, per-text caches) that cost zero AI
    calls for the common case -- this only changes what happens when BOTH
    fast paths miss and BOTH would otherwise fire their own separate AI
    round-trip. In that case (genuinely ambiguous language on a genuinely
    ambiguous topic -- exactly the worst-case, highest-latency messages)
    this merges them into ONE prompt/ONE call instead of two sequential
    ones. When only one side needs AI, this just delegates to that single
    function, so behavior/accuracy is identical to calling them separately
    -- only the double-AI-call case changes.

    Returns {'language': str, 'is_gibberish': bool, 'is_school_related': bool}.
    """
    q = (question or '').strip()
    if not q:
        return {'language': prior_lang or 'english', 'is_gibberish': False, 'is_school_related': True}

    lang_guess, lang_confident = _heuristic_language(q)
    word_count = len(q.split())
    lang_result, lang_needs_ai = None, False
    if not lang_confident and word_count <= 2 and prior_lang:
        lang_result = prior_lang
    else:
        lang_key = _lang_cache_key(q)
        with _LANG_CACHE_LOCK:
            cached_lang = _LANG_CACHE.get(lang_key)
        if cached_lang:
            lang_result = cached_lang
        elif lang_confident or circuit_breaker.is_open():
            lang_result = lang_guess
            _cache_lang(lang_key, lang_guess)
        else:
            lang_needs_ai = True

    class_result: Optional[Dict[str, bool]] = None
    class_needs_ai = False
    if _obviously_not_gibberish_and_school_related(q):
        class_result = {'is_gibberish': False, 'is_school_related': True}
    else:
        class_cache_key = f"classify_q_{hash(q.lower())}"
        cached_class = ai_cache.get(class_cache_key)
        if cached_class:
            class_result = cached_class
        elif circuit_breaker.is_open():
            class_result = {'is_gibberish': False, 'is_school_related': True}
        else:
            class_needs_ai = True

    if not lang_needs_ai and not class_needs_ai:
        assert lang_result is not None and class_result is not None
        return {'language': lang_result, **class_result}

    # Only one side needs AI -- delegate to the existing single-purpose
    # function rather than duplicating its prompt/parsing here.
    if lang_needs_ai and not class_needs_ai:
        assert class_result is not None
        return {'language': detect_language(q, prior_lang=prior_lang), **class_result}
    if class_needs_ai and not lang_needs_ai:
        assert lang_result is not None
        return {'language': lang_result, **classify_question_ai(q)}

    prompt = f"""A student typed this message to a school helpdesk chatbot.
{bot_identity_line()} The student may write in English, Tagalog, Bisaya/Cebuano,
or a mix of these.

Message: "{q}"

Answer three questions about it. Reply with EXACTLY this format, nothing else:
LANGUAGE: english/tagalog/bisaya/mixed_te/mixed_be/mixed_tb
GIBBERISH: yes/no  (yes only if random keyboard-mashing / not real words in ANY language;
  a short or informal real message, even one real word or a greeting, is NOT gibberish)
SCHOOL: yes/no  (plausibly related to the school -- enrollment, tuition, grades, clearance,
  scholarships, class schedules, office hours, campus rules/handbook, or greeting/small talk
  directed at the school assistant; if unsure, prefer yes -- only say no if CLEARLY unrelated,
  e.g. celebrities, general trivia, other companies, unrelated math homework)"""

    try:
        response = llm_chat_fast(prompt, max_tokens=30)
        if response:
            r = response.strip().lower()
            lang_match = re.search(r'language:\s*(\w+)', r)
            gibberish_match = re.search(r'gibberish:\s*(yes|no)', r)
            school_match = re.search(r'school:\s*(yes|no)', r)
            label_map = {
                'english': 'english', 'tagalog': 'tagalog', 'bisaya': 'bisaya',
                'cebuano': 'bisaya', 'mixed_te': 'mixed_te', 'mixed_tb': 'mixed_tb',
                'mixed_be': 'mixed_be', 'filipino': 'tagalog',
            }
            resolved_lang = lang_guess
            if lang_match:
                resolved_lang = label_map.get(lang_match.group(1), lang_guess)
            if gibberish_match and school_match:
                resolved_class = {
                    'is_gibberish': gibberish_match.group(1) == 'yes',
                    'is_school_related': school_match.group(1) == 'yes',
                }
                _cache_lang(_lang_cache_key(q), resolved_lang)
                ai_cache.set(f"classify_q_{hash(q.lower())}", resolved_class)
                return {'language': resolved_lang, **resolved_class}
    except Exception as e:
        logger.warning(f"classify_and_detect_language merged call failed, "
                        f"falling back to heuristics: {e}")

    _cache_lang(_lang_cache_key(q), lang_guess)
    return {'language': lang_guess, 'is_gibberish': False, 'is_school_related': True}


_ABBR_CACHE: Optional[Dict[str, str]] = None
_ABBR_CACHE_LOCK = Lock()
_HARDCODED_ABBREVIATIONS = {
    'tor': 'transcript of records',
    'gpa': 'grade point average',
    'gwa': 'general weighted average',
    'id': 'identification',
    'soa': 'statement of account',
    'tes': 'tuition fee',
    'shs': 'senior high school',
    'jhs': 'junior high school',
}


def _load_abbreviations() -> Dict[str, str]:
    global _ABBR_CACHE
    with _ABBR_CACHE_LOCK:
        if _ABBR_CACHE is not None:
            return _ABBR_CACHE

    merged = dict(_HARDCODED_ABBREVIATIONS)
    try:
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT abbreviation, expansion FROM abbreviations "
                "WHERE is_active=1"
            )
            for abbr, expansion in cur.fetchall():
                if abbr and expansion:
                    merged[abbr.lower()] = expansion
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        print(
            f"_load_abbreviations DB lookup failed, "
            f"using hardcoded set only: {e}"
        )

    with _ABBR_CACHE_LOCK:
        _ABBR_CACHE = merged
    return merged


def expand_abbreviations(text: str) -> str:
    abbreviations = _load_abbreviations()
    result = text.lower()
    for abbr, expansion in abbreviations.items():
        result = re.sub(
            r'\b' + re.escape(abbr) + r'\b',
            expansion,
            result,
            flags=re.IGNORECASE,
        )
    return result


_VERNACULAR_VARIANTS: Dict[str, str] = {

    'bulana': 'buwan', 'bulanan': 'buwan', 'bulan': 'buwan',
    'buwana': 'buwan', 'buwanan': 'buwan',

    'simana': 'semana', 'simanaha': 'semana', 'semanaha': 'semana',
    'simanahan': 'semana', 'semanahan': 'semana',

    'karung': 'karon', 'karun': 'karon',

    'haxing': 'hazing', 'hazeing': 'hazing', 'hasing': 'hazing',
    'buly': 'bully', 'bulli': 'bully', 'bullyng': 'bullying',
    'bulying': 'bullying', 'nambubully': 'bully', 'nanbubully': 'bully',
    'pambully': 'bully',

    'isul-ob': 'isuot uniform', 'isulob': 'isuot uniform',
    'isuutin': 'isuot uniform', 'suoton': 'isuot uniform',
    'sul-ob': 'isuot uniform', 'ipasul-ob': 'isuot uniform',
    'sout': 'suot',
    'sinina': 'uniform', 'bisti': 'uniform', 'panapton': 'uniform',
}

_VERNACULAR_WORD_RE = re.compile(r"[a-zA-Z']+(?:-[a-zA-Z']+)?")


def normalize_vernacular_spelling(text: str) -> str:
    """Best-effort phonetic/spelling canonicalization for FAQ/handbook
    matching. See _VERNACULAR_VARIANTS above for what/why. Always safe to
    call -- words not in the table pass through unchanged."""
    if not text:
        return text

    def _sub(m):
        w = m.group(0)
        repl = _VERNACULAR_VARIANTS.get(w.lower())
        return repl if repl else w

    return _VERNACULAR_WORD_RE.sub(_sub, text)


_HYPOTHETICAL_SUBJECT = (
    r'(?:i|we|you|he|she|they|(?:(?:a|the) )?students?|someone|somebody|'
    r'anyone|anybody|a person|my friend|another student)'
)
_HYPOTHETICAL_PATTERNS = [
    # English
    rf'what (?:would|will|could|might) happen(?:s)? if {_HYPOTHETICAL_SUBJECT}\s+',
    rf'what happens if {_HYPOTHETICAL_SUBJECT}\s+',
    rf'what if {_HYPOTHETICAL_SUBJECT}\s*',
    rf"what'?s the (?:penalty|punishment|consequence)s? (?:if|for) {_HYPOTHETICAL_SUBJECT}\s*",
    rf'(?:suppose|assuming|imagine)(?: that)? {_HYPOTHETICAL_SUBJECT}\s+',
    rf'if {_HYPOTHETICAL_SUBJECT}\s+',

    r'ano (?:ang |po ang )?mangyayari (?:kung|kapag) ako(?:y|\s+ay)?\s*',
    r'paano kung ako\s*',
    r'kung ako(?:y|\s+ay)?\s+',

    r'unsa(?:y)? (?:ang )?mahitabo kung ako\s*',
    r'unsaon (?:kung|kon) ako\s*',
    r'kon ako\s+',
]
_HYPOTHETICAL_RE = re.compile('|'.join(f'(?:{p})' for p in _HYPOTHETICAL_PATTERNS), re.I)


def expand_hypothetical(text: str) -> str:
    """If `text` is phrased as a hypothetical ("what would happen if I
    bully someone"), append the direct-question form ("...policy penalty
    violation offense for bully someone") so FAQ/handbook keyword and
    topic-alias matching -- which was built around direct phrasing --
    still fires. Never removes anything from the original text, so
    nothing is lost if the pattern doesn't match."""
    if not text:
        return text
    q = text.strip()
    m = _HYPOTHETICAL_RE.search(q)
    if not m:
        return q
    remainder = q[m.end():].strip(' ?.!')
    if not remainder:
        return q
    return f"{q} policy penalty violation offense consequence for {remainder}"


def expand_query(text: str) -> str:
    """Full search-time normalization pipeline used before FAQ/handbook
    matching: fix common vernacular misspellings, unwrap hypothetical
    ('what if I...'/'kung ako...') framing into direct-question keywords,
    then expand known abbreviations. Only for building the text that gets
    matched -- never for what's shown back to the student or used for
    language detection."""
    normalized = normalize_vernacular_spelling(text)
    normalized = expand_hypothetical(normalized)
    normalized = expand_abbreviations(normalized)

    aliases = []
    if re.search(r'\bpila\b', normalized, re.I) and re.search(
        r'\b(tuition|fee|bayad|gasto|cost|price)\b', normalized, re.I
    ):
        aliases.append('how much tuition fee cost')
    if re.search(r'\bgikan\b', normalized, re.I):
        aliases.append('from')
    if re.search(r'\b(hangtod|hangtud|kutob)\b', normalized, re.I):
        aliases.append('until')

    return f"{normalized} {' '.join(aliases)}".strip()


def invalidate_response_tables():
    global _ABBR_CACHE
    with _ABBR_CACHE_LOCK:
        _ABBR_CACHE = None
    with _DB_RESPONSE_LOCK:
        _DB_RESPONSE_CACHE.clear()


def invalidate_lang_cache():
    with _LANG_CACHE_LOCK:
        _LANG_CACHE.clear()


def warm_response_tables():
    pass