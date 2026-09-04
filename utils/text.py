"""Text normalization, validation, and fuzzy-matching helpers shared by
faq.py, dates.py, and handbook.py. Nothing here touches the DB or Flask.
"""
import re
from fuzzywuzzy import fuzz

FILLER_PHRASES = [
    r'\bplease\b', r'how can i\b', r'can you tell me\b', r'i want to know\b',
    r'i would like to know\b', r'i need to know\b', r'could you tell me\b',
    r'i was wondering\b', r'do you know\b', r'what is the\b', r'what are the\b',
    r'can i ask\b', r'may i ask\b', r'i just want to ask\b', r'i want to ask\b',
    r'pwede mo bang i-explain\b', r'pwede mo bang sabihin\b',
    r'paki-explain\b', r'pakisabi\b',
]

CONFIRMATION_WORDS = [
    'yes', 'yeah', 'yep', 'yup', 'oo', 'uo', 'sige', 'pwede', 'go', 'sure',
    'amo', 'mao', 'okay', 'ok', 'yes please', 'cge', 'ge', 'want', 'please do',
    'tell me', 'i want', 'show me', 'let me know', 'y', 'ye', 'yess', 'yesh',
]


def strip_html(html: str) -> str:
    return re.sub(re.compile('<.*?>'), '', html or '').strip()


def normalize_input(text: str) -> str:
    t = (text or '').lower().strip()
    for p in FILLER_PHRASES:
        t = re.sub(p, '', t, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', t).strip()


def is_confirmation(text: str) -> bool:
    if not text:
        return False
    t = re.sub(r'[^\w\s]', '', text.lower().strip())
    ws = t.split()
    if not ws:
        return False
    return ws[0] in CONFIRMATION_WORDS or t in CONFIRMATION_WORDS


def is_valid_question(text: str) -> bool:
    """Gibberish / keyboard-mash detection."""
    if not text or len(text) < 2:
        return False
    if len(text) <= 3:
        return True

    tl = text.lower()

    for p in [r'(?:asdf|qwer|zxcv|hjkl|fghj){2,}', r'(.)\1{4,}',
              r'^[a-z]{10,}$', r'[bcdfghjklmnpqrstvwxyz]{7,}']:
        if re.search(p, tl):
            return False

    if len(text) > 8:
        vowels = len(re.findall(r'[aeiou]', tl))
        consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', tl))
        if consonants > 0 and vowels / consonants < 0.2:
            return False

    keyboard_rows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
    consecutive_keyboard = 0
    for i in range(len(tl) - 1):
        char1, char2 = tl[i], tl[i + 1]
        for row in keyboard_rows:
            if char1 in row and char2 in row:
                idx1, idx2 = row.index(char1), row.index(char2)
                if abs(idx1 - idx2) == 1:
                    consecutive_keyboard += 1
                    break
    if consecutive_keyboard > 3:
        return False

    words = re.findall(r'[a-zA-Z]{3,}', tl)
    if len(words) > 0:
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'you', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'what',
            'how', 'when', 'where', 'who', 'why', 'can', 'will', 'would',
            'please', 'thank', 'help', 'need', 'want', 'know', 'about',
            'may', 'paano', 'ano', 'saan', 'kailan', 'bakit', 'magkano',
            'kumusta', 'kamusta', 'salamat', 'po', 'opo', 'unsa', 'ngano',
            'giunsa', 'kinsa', 'asa', 'kanus-a', 'pila', 'mao', 'amo'
        }
        common_count = sum(1 for w in words if w in common_words)
        if len(words) > 3 and common_count / len(words) < 0.2:
            return False

    if re.search(r'([a-z])\1{5,}', tl):
        return False

    if re.findall(r'[bcdfghjklmnpqrstvwxyz]{5,}', tl):
        return False

    return True


def validate_and_sanitize_user_input(text: str):
    if not text or len(text) > 2000:
        return None
    dangerous = [
        r'<script.*?>.*?</script>', r'javascript:', r'data:text/html',
        r'onload=', r'onerror=', r'<iframe', r'<object', r'<embed',
    ]
    for p in dangerous:
        if re.search(p, text, re.IGNORECASE):
            return None
    return text.strip()


def fuzzy_score(a: str, b: str) -> float:
    return max(
        fuzz.token_set_ratio(a, b),
        fuzz.partial_ratio(a, b),
        fuzz.ratio(a, b),
    ) / 100.0


def fuzzy_contains(text: str, keywords: list, threshold: int = 82) -> bool:
    """Fuzzy/typo-tolerant "does any of these keywords appear in text"
    check, used by dates.py's is_date_question/_has_day_word to spot
    day-word and class/office mentions even when misspelled.

    BUG FIX: this used to open with a raw substring test --
        for kw in keywords:
            if kw in t: return True
    -- against EVERY keyword regardless of length. A short single-word
    keyword like "sa" (present in some day/schedule word lists) is a
    substring of plenty of unrelated words in Bisaya/Tagalog -- most
    notably "unsa" ("what"), so `fuzzy_contains(q, [...,'sa',...])`
    returned True for almost any Bisaya question starting with "unsa",
    no matter the actual topic. (The confirmed real-world case of this
    exact bug pattern was a hand-rolled copy of the same substring check
    in dates.py's is_schedule_level_followup, not this function directly
    -- but this shared helper has the identical footgun for any caller
    that ever adds a short keyword to its list.)

    Now: single-word keywords must match an actual WORD boundary (either
    exactly, via `words`, or fuzzily via fuzz.ratio against a real token)
    -- never merely as a substring of a longer, unrelated word.
    Multi-word phrases keep the substring check, since a multi-word
    phrase is long enough that the false-positive risk is negligible and
    phrase-boundary matching isn't worth the complexity.
    """
    t = text.lower()
    words = re.findall(r'[a-z\-]+', t)
    word_set = set(words)
    for kw in keywords:
        kw = kw.lower()
        if ' ' in kw:
            if kw in t:
                return True
            continue
        if kw in word_set:
            return True
        for w in words:
            if len(w) >= 4 and fuzz.ratio(w, kw) >= threshold:
                return True
    return False