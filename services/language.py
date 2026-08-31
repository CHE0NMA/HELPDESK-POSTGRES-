"""Language detection, topic classification, abbreviation expansion, and
admin-managed greeting/fallback/non-school responses.
"""
import hashlib
import logging
import re
from threading import Lock
from typing import Dict, Optional, Tuple

from ai_providers import ai_cache, circuit_breaker, llm_chat
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
        # Was missing this top-level key before (only nested inside
        # unused 'english'/'tagalog'/'bisaya' sub-dicts below, which
        # nothing else in the codebase reads) — get_db_response() does
        # `lang_config['non_school']` for every language including this
        # one, so a message classified as mixed_tb would raise KeyError
        # the moment it hit get_non_school_response(). Fixed by adding
        # the key every other language entry has.
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
        r'\b(unsa|ngano|giunsa|jud|gyud|pud|usab|kaayo|mao|amo|kinsa|asa|kanus-a'
        r'|naa|wala|nay|naay|dili|oo|sige|lagi|bitaw|diha|dinhi|adto|karon'
        r'|buhat|himoa|palihug|salamat|maayo|unta|gusto|ganahan|pwede'
        r'|mag[a-z]{2,}|nag[a-z]{2,}|mo[a-z]{3,})\b',
        t,
    ))
    has_english = bool(re.search(
        r'\b(what|how|why|when|where|the|is|are|can|will|would|please|thank)\b',
        t,
    ))

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

    guess, _confident = _heuristic_language(text)
    if guess == expected_lang:
        return True

    # A mixed_xx reply that comes back as one of its pure components is
    # close enough -- not the "drifted into a totally different
    # language" failure this check exists to catch.
    mixed_components = {
        'mixed_te': {'tagalog', 'english'},
        'mixed_be': {'bisaya', 'english'},
        'mixed_tb': {'tagalog', 'bisaya'},
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
        raw = llm_chat(prompt, max_tokens=8).strip().lower()
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
            if not row and language != 'english':
                cur.execute(
                    f"SELECT {column} FROM {table} "
                    f"WHERE language='english' AND is_active=1"
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


_GIBBERISH_RE = re.compile(r'^[a-z]{1,2}$|^(.)\1{3,}$', re.IGNORECASE)
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
    # Looks like real language: has vowels, not one long mashed string
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
            response = llm_chat(prompt, max_tokens=20)
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


# ---------------------------------------------------------------------------
# Vernacular spelling/vocabulary normalization + hypothetical framing
# ---------------------------------------------------------------------------
# Students type Bisaya/Tagalog phonetically and inconsistently -- "bulan"
# vs "buwan" vs "bulana" for "month", "semana" vs "simana" for "week",
# "haxing" for "hazing", clothing words (damit/sinina/bisti/suoton) that
# never share a token with the handbook's own English "uniform"/"dress"
# wording, etc. Every one of these silently tanks fuzzy-match/regex
# scoring even though the FAQ or handbook chunk that answers the question
# exists.
#
# This is SEARCH-TIME-ONLY normalization: the output is used to build the
# text that gets matched against FAQ/handbook keywords, never the raw
# text shown back to the student or used for language detection --
# rewriting "bulan" as "buwan" is fine for matching purposes but would be
# wrong to actually say back to someone who just spoke to you in Bisaya.
_VERNACULAR_VARIANTS: Dict[str, str] = {
    # month -- canonicalize every Bisaya/Tagalog spelling variant to
    # "buwan" so the existing "this month"/"next month" patterns (which
    # were only ever written expecting "buwan") also fire for "bulan".
    'bulana': 'buwan', 'bulanan': 'buwan', 'bulan': 'buwan',
    'buwana': 'buwan', 'buwanan': 'buwan',
    # week
    'simana': 'semana', 'simanaha': 'semana', 'semanaha': 'semana',
    'simanahan': 'semana', 'semanahan': 'semana',
    # today/this (common typo forms not already covered by fuzzy TODAY_WORDS)
    'karung': 'karon', 'karun': 'karon',
    # hazing / bullying misspellings students actually type
    'haxing': 'hazing', 'hazeing': 'hazing', 'hasing': 'hazing',
    'buly': 'bully', 'bulli': 'bully', 'bullyng': 'bullying',
    'bulying': 'bullying', 'nambubully': 'bully', 'nanbubully': 'bully',
    'pambully': 'bully',
    # uniform / what-to-wear vocabulary -- map to words that DO appear in
    # the handbook's/FAQ's own "Dress and Appearance"/"School Uniform"
    # phrasing, so "unsa dapat suoton sa school" scores against an FAQ
    # literally titled "unsay uniform sa skwelahan" the way it should --
    # they mean the exact same thing but shared almost no tokens before.
    'isul-ob': 'isuot uniform', 'isulob': 'isuot uniform',
    'isuutin': 'isuot uniform', 'suoton': 'isuot uniform',
    'sul-ob': 'isuot uniform', 'ipasul-ob': 'isuot uniform',
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


# "What would happen if I ___" style hypothetical framing. A student
# asking this almost always wants the SAME answer as if they'd asked
# directly ("is there a policy on ___", "what's the penalty for ___") --
# but none of the direct-question phrasing patterns fire on "what if"
# framing, so questions like "what would happen if I bully someone" were
# falling through even though "is there a policy for bullying" worked.
# Deliberately topic-agnostic: covers bullying, hazing, curfew, or any
# other rule someone hypothetically asks about, not just one topic.
_HYPOTHETICAL_PATTERNS = [
    # English
    r'what (?:would|will) happen(?:s)? if i\s+',
    r'what happens if i\s+',
    r'what if i\s+',
    r"what'?s the (?:penalty|punishment|consequence)s? (?:if|for) i\s*",
    # Tagalog
    r'ano (?:ang |po ang )?mangyayari (?:kung|kapag) ako(?:y|\s+ay)?\s*',
    r'paano kung ako\s*',
    r'kung ako(?:y|\s+ay)?\s+',
    # Bisaya/Cebuano
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
    return expand_abbreviations(normalized)


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