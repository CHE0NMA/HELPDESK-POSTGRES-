"""Date/schedule question answering — v3.

Changes from v2:
- Ollama removed; uses `llm_chat` (Groq -> Gemini).
- NEW: genuine department disambiguation. Previously an ambiguous question
  (mentions no clear department, or fuzzy-matches more than one department
  about equally) silently defaulted to "all departments" — which can be
  flat-out wrong if the student meant one specific office. Now uses the
  SAME shared policy as handbook/FAQ (services.disambiguation.DATE_POLICY)
  to decide when to actually ask "which office did you mean?" instead of
  guessing — and, symmetrically, does NOT ask when the question is clearly
  about one department or clearly about everything.

ARCHITECTURE — canonical-first, not compose-from-JSON
-------------------------------------------------------
Builds a plain, 100%-correct sentence FIRST from deterministic DB
lookups, then asks the AI only to paraphrase that already-correct
sentence into something warmer (services.nlg.speak_naturally).
"""
import json
import logging
import re
from datetime import datetime, timedelta
from services.tz import ph_today  # BUG FIX: server clock is UTC, school is Asia/Manila (UTC+8) -- see services/tz.py
from typing import Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz

from ai_providers import circuit_breaker, llm_chat
from services import nlg
from services import schedule as sched
from services.disambiguation import DATE_POLICY, select_contenders, should_disambiguate
from utils.text import fuzzy_contains
from services.grounding import answer_is_grounded

logger = logging.getLogger("pac_helpdesk.dates")

WEEKDAY_NAMES = {
    'monday': 0, 'lunes': 0,
    'tuesday': 1, 'martes': 1,
    'wednesday': 2, 'miyerkules': 2, 'miyerkoles': 2, 'myerkules': 2, 'myerkoles': 2, 'wed': 2,
    'thursday': 3, 'huwebes': 3, 'huebes': 3, 'thu': 3, 'thurs': 3,
    'friday': 4, 'biyernes': 4, 'byernes': 4, 'fri': 4,
    'saturday': 5, 'sabado': 5, 'sabadu': 5, 'sat': 5,
    'sunday': 6, 'linggo': 6, 'dominggo': 6, 'domingo': 6, 'sun': 6,
}
TOMORROW_WORDS = [
    'tomorrow', 'tommorow', 'tomorow', 'tommorrow', '2morrow',
    'bukas', 'ugma', 'ugmang', 'ugma nga adlaw', 'buwas', 'buwas na',
    'sa bukas', 'sa ugma', 'sa umaga bukas',
]
TODAY_WORDS = [
    'today', 'todya', 'tody', '2day',
    'ngayon', 'ngayong', 'ngaon', 'ngaung',
    'karon', 'karong', 'karun', 'karung', 'niining orasa', 'ron',
]
YESTERDAY_WORDS = [
    'yesterday', 'yesturday', 'yesteday',
    'kahapon', 'kahapong', 'gahapon', 'kagahapon', 'kaganina',
]
LAST_WEEK_WORDS = [
    'last week', 'previous week', 'past week', 'the week before',
    'nakaraang linggo', 'linggong nakaraan', 'nakalipas na linggo',
    'niaging semana', 'miaging semana', 'niabot nga semana', 'nilabay nga semana',
    'sa milabay nga semana', 'kaniadtong semana',
]
LAST_MONTH_WORDS = [
    'last month', 'previous month', 'past month', 'the month before',
    'nakaraang buwan', 'buwang nakaraan', 'nakalipas na buwan',
    'niaging bulan', 'miaging buwan', 'nilabay nga bulan',
    'sa milabay nga bulan', 'kaniadtong bulan',
]

DATE_KEYWORDS = [
    'today', 'tomorrow', 'this week', 'next week', 'last week', 'previous week',
    'this month', 'next month', 'last month', 'previous month',
    'open today', 'open tomorrow', 'closed today', 'closed tomorrow',
    'office open', 'office closed', 'school open', 'school closed',
    'class today', 'class tomorrow', 'classes today', 'classes tomorrow',
    'do we have class', 'is there class', 'is the office', 'is the school',
    'is the registrar', 'is there school', 'are there classes',
    'will there be', 'is it open', 'is it closed', 'have class',
    'what day', 'what date', 'schedule for',
    # Tagalog
    'bukas', 'ngayon', 'ngayong', 'may pasok', 'walang pasok',
    'may klase', 'walang klase', 'bukas ang', 'sarado ang',
    'bukas ba', 'ngayon ba', 'may pasok ba', 'araw na ito', 'susunod na linggo',
    'ngayong linggo', 'bukas ba ang', 'sarado ba ang', 'nakaraang linggo',
    'nakaraang buwan', 'kahapon', 'kailan', 'anong araw', 'anong petsa',
    # Bisaya / Cebuano
    'ugma', 'karon', 'karong', 'abli', 'sirado',
    'walay klase', 'aduna bay klase', 'abli ba', 'sirado ba',
    'karong adlawa', 'ugmang adlaw', 'niining semanaha', 'sunod nga semana',
    'karon nga semana', 'karon nga week', 'ning semana', 'sunod nga week',
    'miaging semana', 'niaging semana', 'miaging bulan', 'niaging bulan',
    'kagahapon', 'kanus-a', 'unsa nga adlaw', 'unsa nga petsa',
    'naa bay klase', 'naay klase', 'unsa may klase', 'aduna bay pasok',
    'holiday', 'holidays', 'no class day', 'no classes day',
    'class suspended', 'classes suspended', 'class suspension',
    'school suspension', 'walay pasok', 'pasok ba',
]

OPEN_CLOSED_PATTERNS = [
    r'\b(open|closed|close)\b',
    r'\b(sarado|abli|sirado)\b',
    r'\b(may pasok|walang pasok|may klase|walang klase|walay klase|naa bay klase|naay klase)\b',
    r'\bdo we have\b.{0,15}\b(class|classes|pasok)\b',
    r'\bis there\b.{0,15}\b(class|classes|pasok)\b',
    r'\bare there\b.{0,15}\b(class|classes)\b',
    r'\bwill there be\b.{0,15}\b(class|classes)\b',
    r'\b(is|are)\b.{0,25}\b(open|closed)\b',
    r'\b(office|school|registrar|business office|cashier)\b.{0,20}\b(available|open|closed)\b',
    r'\bschedule\b.{0,30}\b(for|on|this|next|last)\b',
    r'\b(registrar|business office|cashier)\b.{0,20}\b(open|closed|open ba|sarado)\b',
]

# ------------------------------------------------------------------
# Stronger schedule follow-up detection
# ------------------------------------------------------------------
_SCHEDULE_LEVEL_FOLLOWUP_RE = re.compile(
    r'(?i)\b(?:'
    r'all levels?|all departments?|whole school|everyone|all of us|for all|'
    r'so all|also all|including all|all level|'
    r'(?:elementary|high.?school|college|k-?12|shs|jhs|registrar|business)'
    r'(?:\s+(?:too|also|as well|ba|pud|naman))?|'
    r'(?:so|and|what about|how about|kaya|paano|unsa|ano)\s+'
    r'(?:all|everyone|whole|tanan|lahat|tanang level)|'
    r'sa tanan|para sa tanan|sa lahat|para sa lahat|'
    r'college too|elementary also|high school too|for everyone|whole school'
    r')\b'
)

def is_schedule_level_followup(question: str, ctx) -> bool:
    """Very aggressive detector for short follow-ups when schedule context exists."""
    if not ctx or not getattr(ctx, 'last_schedule_query', None):
        return False

    q = (question or '').strip().lower()
    if not q:
        return False

    # Extremely short messages that contain any of these words are almost
    # certainly continuing the previous schedule conversation
    if len(q.split()) <= 6 and any(w in q for w in (
        'all', 'level', 'college', 'elementary', 'high', 'tanan', 'lahat',
        'everyone', 'whole', 'too', 'also', 'sa', 'for', 'school'
    )):
        return True

    return bool(_SCHEDULE_LEVEL_FOLLOWUP_RE.search(q))


def answer_schedule_department_followup(question: str, ctx) -> Optional[str]:
    """Handle short follow-ups that change the department/level of the
    previous schedule answer, or expand it to 'all'.

    Examples that now work:
      - "so all level?"
      - "all levels?"
      - "college too?"
      - "elementary also?"
      - "sa tanan?"
      - "for high school?"
    """
    if not ctx or not getattr(ctx, 'last_schedule_query', None):
        return None

    last = ctx.last_schedule_query
    q = (question or '').strip()
    if not q:
        return None

    # --- 1. Explicit "all levels / whole school" expansion ---------------
    if is_schedule_level_followup(q, ctx):
        # Force all departments
        dept = 'all'
        scope = last.get('scope', 'general')
        query_date = last.get('query_date')
        if not query_date:
            return None

        facts = gather_single_day_facts(query_date, dept)
        canonical = build_canonical_single_day_sentence(facts, scope, dept)
        natural = nlg.speak_naturally(
            canonical,
            q,
            last.get('detected_lang', 'english'),
            extra_note=_RELATIVE_DATE_NOTE,
        )
        if natural and not answer_is_grounded(natural, canonical, min_overlap=0.25):
            natural = canonical

        # Update context so further follow-ups stay correct
        ctx.last_schedule_query = {
            'query_date': query_date,
            'dept': dept,
            'scope': scope,
            'detected_lang': last.get('detected_lang', 'english'),
            'canonical': canonical,
            'question': q,
        }
        return natural

    # --- 2. Normal department switch (existing logic, kept intact) -------
    dept = sched.detect_dept_from_question(q)
    if dept == 'all' or len(q.split()) > 6:
        return None

    if not nlg.classify_followup(q, last.get('question', ''), last.get('canonical', '')):
        return None

    facts = gather_single_day_facts(last['query_date'], dept)
    scope = last.get('scope', 'general')
    canonical = build_canonical_single_day_sentence(facts, scope, dept)
    natural = nlg.speak_naturally(
        canonical,
        q,
        last.get('detected_lang', 'english'),
        extra_note=_RELATIVE_DATE_NOTE,
    )
    if natural and not answer_is_grounded(natural, canonical, min_overlap=0.25):
        natural = canonical

    ctx.last_schedule_query = {
        'query_date': last['query_date'],
        'dept': dept,
        'scope': scope,
        'detected_lang': last.get('detected_lang', 'english'),
        'canonical': canonical,
        'question': q,
    }
    return natural

def _has_day_word(q: str, words: list) -> bool:
    return fuzzy_contains(q, words, threshold=80)


def is_date_question(question: str, ctx=None) -> bool:
    """Return True when the user is clearly asking about a day/schedule,
    OR when they are continuing a previous schedule conversation with a
    new day (e.g. "how about wednesday")."""
    q = question.lower().strip()

    if (_has_day_word(q, TOMORROW_WORDS)
            or _has_day_word(q, TODAY_WORDS)
            or _has_day_word(q, YESTERDAY_WORDS)
            or _has_day_word(q, LAST_WEEK_WORDS)
            or _has_day_word(q, LAST_MONTH_WORDS)):
        return True

    has_weekday = any(
        re.search(rf'\b{name}\b', q, re.I) for name in WEEKDAY_NAMES
    )

    if ctx and getattr(ctx, 'last_schedule_query', None) and has_weekday:
        return True

    if has_weekday and re.search(
        r'\b(i\s+mean[t]?|meant|sorry|not\s+that|correction|actually)\b',
        q, re.I,
    ):
        return True

    if has_weekday and re.search(
        r'\b(how\s+about|what\s+about|and|then|kaya|paano\s+naman|ug)\b',
        q, re.I,
    ):
        return True

    # BUG FIX: the "how about X" follow-up check above only fired when a
    # WEEKDAY name was present, so a bare range follow-up like "how about
    # last week" or "and holidays?" right after a schedule answer fell
    # through entirely and got treated as a brand new (non-date)
    # question -- landing on handbook retrieval instead of continuing
    # the date conversation. Mirrors the weekday check but for
    # short/relative-range follow-ups instead of specific weekdays.
    if ctx and getattr(ctx, 'last_schedule_query', None):
        is_range_word = (
            _has_day_word(q, LAST_WEEK_WORDS) or _has_day_word(q, LAST_MONTH_WORDS)
            or re.search(r'\b(this|next|last|previous)\s+(week|month)\b', q, re.I)
            or 'holiday' in q or 'holidays' in q
        )
        if is_range_word:
            return True
        if len(q.split()) <= 6 and re.search(
            r'\b(how\s+about|what\s+about|and|then|kaya|paano\s+naman|ug|unya)\b',
            q, re.I,
        ):
            return True

    schedule_context = (
        fuzzy_contains(q, sched.CLASS_KEYWORDS)
        or fuzzy_contains(q, sched.OFFICE_KEYWORDS)
        or any(kw in q for kw in (
            'open', 'closed', 'schedule', 'pasok', 'klase', 'holiday',
            'class suspension', 'school suspension', 'office hours',
            'may pasok', 'walang pasok',
        ))
    )
    if has_weekday and schedule_context:
        return True

    for p in OPEN_CLOSED_PATTERNS:
        if re.search(p, q, re.I):
            return True

    if re.search(r'\b(this|next|last|previous)\s+(week|month)\b', q, re.I) and schedule_context:
        return True

    # Class / weather suspension only — NOT disciplinary "grounds for suspension".
    # Topic shape, not handbook facts: stays valid across handbook editions.
    if any(kw in q for kw in (
        'holiday', 'holidays', 'no class day', 'no classes day',
        'class suspended', 'classes suspended', 'class suspension',
        'school suspension', 'work suspension',
    )):
        return True

    if 'suspension' in q or 'suspended' in q:
        # weather / operations cues → schedule
        if re.search(
            r'\b(class|classes|school|work|office|weather|typhoon|storm|'
            r'rain|flood|pasok|klase|holiday)\b',
            q, re.I,
        ):
            return True
        # disciplinary cues → not a date question (handbook owns these)
        if re.search(
            r'\b(grounds?|reason|cause|demerit|offense|offence|expulsion|'
            r'dismissal|disciplinary|violation|fraternity|pregnant|'
            r'behavioral|conduct)\b',
            q, re.I,
        ):
            return False

    return False


def detect_dept_candidates(question: str) -> List[Tuple[str, float]]:
    """Generic fuzzy match of the question against every known department
    label (whatever `sched.DEPT_LABELS` currently contains — no department
    names hardcoded here). Returns [(dept_code, score 0..1), ...] sorted
    descending. Only departments with a non-trivial match are included."""
    q_norm = question.lower()
    scored = []
    for dept_code, label in getattr(sched, 'DEPT_LABELS', {}).items():
        if not label:
            continue
        score = fuzz.partial_ratio(q_norm, label.lower()) / 100.0
        # also credit exact keyword hits from the label's words
        label_words = [w for w in re.findall(r'[a-z]{3,}', label.lower())]
        if label_words and any(w in q_norm for w in label_words):
            score = max(score, 0.55)
        if score >= 0.35:
            scored.append((dept_code, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def check_department_disambiguation(question: str, detected_lang: str = 'english',
                                     ctx=None) -> Optional[Tuple[str, List[Dict]]]:
    """Returns (message, options) if the question is genuinely ambiguous
    between two-or-more specific departments, else None. Only relevant
    when the question doesn't already resolve to a specific department via
    the existing (presumably stronger, handbook-aware) `sched.detect_dept_from_question`.
    """
    if not is_date_question(question, ctx=ctx):
        return None

    resolved = sched.detect_dept_from_question(question)
    if resolved and resolved != 'all':
        return None  # already unambiguous — nothing to disambiguate

    candidates = detect_dept_candidates(question)
    if len(candidates) < 2:
        return None

    scores = [s for _, s in candidates]
    if not should_disambiguate(scores, DATE_POLICY):
        return None

    contenders = select_contenders(candidates, scores, DATE_POLICY)
    if len(contenders) < 2:
        return None

    headers = {
        'english': "Which office/department did you mean?",
        'tagalog': "Aling opisina o departamento ang tinutukoy mo?",
        'bisaya': "Unsa nga opisina o departamento ang imong gipasabot?",
    }
    header = headers.get(detected_lang, headers['english'])

    options = []
    lines = []
    for i, (dept_code, score) in enumerate(contenders, 1):
        label = sched.DEPT_LABELS.get(dept_code, dept_code)
        lines.append(f"{i}. {label}")
        options.append({'label': label, 'dept': dept_code})

    message = f"{header}\n\n" + "\n".join(lines)
    return message, options


def answer_for_selected_department(original_question: str, dept: str,
                                    detected_lang: str = 'english', ctx=None) -> Optional[str]:
    """Answer using exactly the department the student picked from a
    disambiguation prompt, grounded in the original question that
    triggered it."""
    return answer_date_question(original_question, detected_lang, ctx=ctx, forced_dept=dept)


def _next_weekday(today, target_weekday: int):
    delta = (target_weekday - today.weekday()) % 7
    return today + timedelta(days=delta)


def _next_weekday_from_date(start_date, target_weekday: int):
    days_ahead = target_weekday - start_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return start_date + timedelta(days=days_ahead)


def parse_date_with_ai(question: str) -> Optional[Tuple[datetime.date, str, str]]:
    if circuit_breaker.is_open():
        return None
    today = ph_today()
    prompt = f"""{nlg.now_context()}
Today's date is {today.strftime('%B %d, %Y')} ({today.strftime('%A')}), year {today.year}.

The user asked: "{question}"

Extract the date they are asking about and respond in this exact JSON format:
{{"date": "YYYY-MM-DD", "type": "specific|today|tomorrow|this_week|next_week|this_month|next_month", "display": "Month Day, Year"}}

Rules:
- If they ask about "this Thursday" or a specific day, give the exact date
- If they ask about "next Thursday", give the date 7 days after this week's Thursday
- If they ask about "this week", give today's date
- If they ask about "next week", give the date 7 days from today
- If they ask about "this month", give the first day of this month
- If they ask about "next month", give the first day of next month
- If they ask about a date that has passed this year, use next year
- If the date is ambiguous, respond with {{"date": null, "type": "unknown", "display": ""}}

Only respond with the JSON, nothing else."""
    try:
        response = llm_chat(prompt, max_tokens=150)
        if not response:
            return None
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            return None
        data = json.loads(json_match.group())
        if not data.get('date'):
            return None
        parsed_date = datetime.strptime(data['date'], '%Y-%m-%d').date()
        if parsed_date.year < 2000 or parsed_date.year > 2035:
            return None
        return (
            parsed_date,
            data.get('type', 'specific'),
            data.get('display', parsed_date.strftime('%B %d, %Y')),
        )
    except Exception as e:
        logger.error(f"AI date parsing error: {e}")
        return None


def parse_date_with_regex(question: str) -> Tuple[datetime.date, str, str]:
    q = question.lower()
    today = ph_today()
    current_year, next_year = today.year, today.year + 1

    if re.search(
        r'(next\s+week|susunod\s+na\s+linggo|sunod\s+nga\s+semana|next\s+nga\s+week)',
        q, re.IGNORECASE,
    ):
        next_week = today + timedelta(days=7)
        for name, wd in WEEKDAY_NAMES.items():
            if re.search(rf'\b{name}\b', q, re.IGNORECASE):
                target = _next_weekday_from_date(next_week, wd)
                return target, 'specific', target.strftime('%B %d, %Y')
        return next_week, 'next_week', 'next week'

    if _has_day_word(q, LAST_WEEK_WORDS):
        last_week_start = today - timedelta(days=7)
        for name, wd in WEEKDAY_NAMES.items():
            if re.search(rf'\b{name}\b', q, re.IGNORECASE):
                # Nearest occurrence of that weekday within the 7 days
                # ending yesterday, not next week's or today's.
                target = last_week_start + timedelta(days=(wd - last_week_start.weekday()) % 7)
                return target, 'specific', target.strftime('%B %d, %Y')
        return last_week_start, 'last_week', 'last week'

    if _has_day_word(q, LAST_MONTH_WORDS):
        first_this_month = today.replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        return last_month_start, 'last_month', last_month_start.strftime('%B %Y')

    if re.search(r'(this\s+week|ngayong\s+linggo|karong\s+semana)', q, re.IGNORECASE):
        for name, wd in WEEKDAY_NAMES.items():
            if re.search(rf'\b{name}\b', q, re.IGNORECASE):
                target = _next_weekday(today, wd)
                return target, 'specific', target.strftime('%B %d, %Y')
        return today, 'this_week', 'this week'

    for name, wd in WEEKDAY_NAMES.items():
        if re.search(rf'\b{name}\b', q, re.IGNORECASE):
            if re.search(r'(next|sunod|susunod)', q, re.IGNORECASE):
                target = _next_weekday_from_date(today + timedelta(days=7), wd)
            else:
                target = _next_weekday(today, wd)
            return target, 'specific', target.strftime('%B %d, %Y')

    if _has_day_word(q, TOMORROW_WORDS):
        d = today + timedelta(days=1)
        return d, 'tomorrow', d.strftime('%B %d, %Y')
    if _has_day_word(q, YESTERDAY_WORDS):
        d = today - timedelta(days=1)
        return d, 'yesterday', d.strftime('%B %d, %Y')
    if _has_day_word(q, TODAY_WORDS):
        return today, 'today', today.strftime('%B %d, %Y')

    if re.search(
        r'(this\s+week|ngayong\s+linggo|karong\s+semana|karon\s+nga\s+semana|'
        r'niining\s+semanaha|karon\s+nga\s+week|ning\s+semana|'
        r'niini\s+nga\s+semana|this\s+wk)',
        q, re.IGNORECASE,
    ):
        return today, 'this_week', 'this week'
    if re.search(
        r'(next\s+week|susunod\s+na\s+linggo|sunod\s+nga\s+semana|'
        r'sunod\s+nga\s+week|next\s+nga\s+week)',
        q, re.IGNORECASE,
    ):
        next_week = today + timedelta(days=7)
        return next_week, 'next_week', 'next week'
    if re.search(r'(this\s+month|ngayong\s+buwan|karong\s+buwana)', q, re.IGNORECASE):
        return today, 'this_month', today.strftime('%B %Y')
    if re.search(r'(next\s+month|susunod\s+na\s+buwan)', q, re.IGNORECASE):
        nm = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        return nm, 'next_month', nm.strftime('%B %Y')

    dm = re.search(r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b', q)
    if dm:
        for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%m/%d/%y', '%m-%d-%y', '%d/%m/%Y', '%d-%m-%Y']:
            try:
                parsed = datetime.strptime(dm.group(1), fmt).date()
                return parsed, 'specific', parsed.strftime('%B %d, %Y')
            except ValueError:
                continue

    for pattern, day_first in [
        (r'\b(January|February|March|April|May|June|July|August|September|'
         r'October|November|December)\s+(\d{1,2})\b', False),
        (r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|'
         r'August|September|October|November|December)\b', True),
    ]:
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            month_name, day = (
                (m.group(2), int(m.group(1))) if day_first
                else (m.group(1), int(m.group(2)))
            )
            try:
                parsed = datetime.strptime(
                    f"{month_name} {day} {current_year}", '%B %d %Y'
                ).date()
                if parsed < today:
                    parsed = datetime.strptime(
                        f"{month_name} {day} {next_year}", '%B %d %Y'
                    ).date()
                return parsed, 'specific', parsed.strftime('%B %d, %Y')
            except ValueError:
                return today, 'today', today.strftime('%B %d, %Y')

    for name, wd in WEEKDAY_NAMES.items():
        if re.search(rf'\b{name}\b', q, re.IGNORECASE):
            target = _next_weekday(today, wd)
            return target, 'specific', target.strftime('%B %d, %Y')

    return today, 'today', today.strftime('%B %d, %Y')


def parse_date_from_question(question: str) -> Tuple[datetime.date, str, str]:
    """Regex-first, AI as enhancement only."""
    regex_date, regex_type, regex_display = parse_date_with_regex(question)

    q = question.lower()
    explicit_signal = (
        regex_type != 'today'
        or _has_day_word(q, TODAY_WORDS)
    )
    if explicit_signal:
        return regex_date, regex_type, regex_display

    ai_result = parse_date_with_ai(question)
    if ai_result:
        logger.info(
            f"AI parsed date (regex found no explicit signal): "
            f"{ai_result[0]} from '{question}'"
        )
        return ai_result

    return regex_date, regex_type, regex_display


def _has_explicit_date_signal(question: str) -> bool:
    q = question.lower()
    if (_has_day_word(q, TODAY_WORDS)
            or _has_day_word(q, TOMORROW_WORDS)
            or _has_day_word(q, YESTERDAY_WORDS)):
        return True
    for name in WEEKDAY_NAMES:
        if re.search(rf'\b{name}\b', q, re.IGNORECASE):
            return True
    if re.search(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', q):
        return True
    if re.search(
        r'\b(january|february|march|april|may|june|july|august|'
        r'september|october|november|december)\b',
        q, re.IGNORECASE,
    ):
        return True
    if re.search(r'(this|next)\s+(week|month)', q, re.IGNORECASE):
        return True
    return False


def _dept_raw_facts(query_date, dept: str) -> Dict:
    day_of_week = query_date.weekday()
    is_weekend = day_of_week >= 5

    holiday = sched.get_holiday_for_date(query_date, dept)
    if holiday:
        return {
            'source': 'holiday',
            'title': holiday.get('title'),
            'is_closed': bool(holiday.get('is_closed', True)),
            'content_type': holiday.get('content_type'),
            'note': holiday.get('message_english'),
        }

    default = sched.get_default_schedule(dept, 'all', day_of_week)
    if default:
        return {
            'source': 'default_schedule',
            'has_class': default['has_class'],
            'hours': (
                f"{default['start_time']}–{default['end_time']}"
                if default['has_class'] else None
            ),
            'is_weekend': is_weekend,
        }

    return {
        'source': 'assumed',
        'has_class': not is_weekend,
        'is_weekend': is_weekend,
    }


def _dept_status_for_scope(raw: Dict, scope: str) -> Tuple[str, Optional[str]]:
    if raw['source'] == 'holiday':
        if raw['is_closed']:
            return 'closed', raw.get('note')
        if raw.get('content_type') == 'office_hours' and scope == 'class':
            return 'no_class', raw.get('note')
        return 'open', raw.get('note')
    if raw['source'] == 'default_schedule':
        if raw['has_class']:
            return 'open', None
        return ('closed' if raw.get('is_weekend') else 'no_class'), None
    return ('closed' if raw.get('is_weekend') else 'open'), None


def gather_single_day_facts(query_date, dept: str = 'all') -> Dict:
    today = ph_today()
    facts: Dict = {
        'date': query_date.strftime('%A, %B %d, %Y'),
        'today': today.strftime('%A, %B %d, %Y'),
        'is_today': query_date == today,
        'is_weekend': query_date.weekday() >= 5,
        'is_past': query_date < today,
    }

    emergencies = sched.get_emergency_closures(query_date)
    if emergencies:
        e = emergencies[0]
        facts['emergency_closure'] = {
            'title': e['title'],
            'reason': e['reason'],
            'severity': e['severity'],
            'affected_departments': e['affected_departments'],
            'message': e.get('message_english'),
        }
        return facts

    dept_list = [dept] if dept != 'all' else sched.ALL_DEPTS
    facts['departments'] = {
        d: _dept_raw_facts(query_date, d) for d in dept_list
    }
    facts['normal_office_hours'] = sched.get_normal_office_hours()
    return facts


def gather_range_facts(query_date, date_type: str, dept: str = 'all') -> Dict:
    if date_type in ('this_month', 'next_month', 'last_month'):
        first = query_date.replace(day=1)
        last = (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        # BUG FIX: get_events_for_week() previously always used
        # start_date + 6 days internally, so a "this month"/"next month"
        # query only ever saw the FIRST WEEK of the month's events —
        # "do we have holidays this month" on, say, the 20th of the
        # month would silently miss anything after day 7. Pass the real
        # month-end explicitly now that get_events_for_week() accepts
        # end_date.
        events = sched.get_events_for_week(first, 'english', dept, end_date=last)
        events = [e for e in events if first <= e['start_date'] <= last]
        label = first.strftime('%B %Y')
    else:
        events = sched.get_events_for_week(query_date, 'english', dept)
        end = query_date + timedelta(days=6)
        label = (
            f"{query_date.strftime('%B %d')}–{end.strftime('%B %d, %Y')}"
        )

    return {
        'period_label': label,
        'events': [{
            'title': e['title'],
            'date': e['start_date'].strftime('%A, %B %d'),
            'is_closed': e['is_closed'],
            'message': e.get('message'),
        } for e in events],
    }

_STATUS_WORD = {
    ('class', 'open'): "there's class as usual",
    ('class', 'no_class'): "there's no class (the office is still open)",
    ('class', 'closed'): "there's no class and the office is closed too",
    ('office', 'open'): "the office is open",
    ('office', 'no_class'): "the office is open",
    ('office', 'closed'): "the office is closed",
    ('general', 'open'): "open, with class in session",
    ('general', 'no_class'): "the office is open but there's no class",
    ('general', 'closed'): "closed, no class",
}


def _phrase_dept(
    dept_label: str, date_label: str, status: str,
    note: Optional[str], scope: str,
) -> str:
    words = _STATUS_WORD.get((scope, status), f"status: {status}")
    sentence = f"On {date_label}, {dept_label} — {words}."
    if note:
        sentence += f" ({note})"
    return sentence


def build_canonical_single_day_sentence(
    facts: Dict, scope: str, dept: str,
) -> str:
    date_label = facts['date']

    if 'emergency_closure' in facts:
        e = facts['emergency_closure']
        reason = (e.get('reason') or '').replace('_', ' ')
        base = (
            f"On {date_label}, there's an emergency closure "
            f"({reason}): {e['title']}."
        )
        if e.get('message'):
            base += f" {e['message']}"
        else:
            base += " Please follow official announcements for updates."
        return base

    depts = facts.get('departments', {})

    if dept != 'all':
        raw = depts.get(dept)
        if raw is None:
            return (
                f"I don't have schedule data for "
                f"{sched.DEPT_LABELS.get(dept, dept)} on {date_label}."
            )
        status, note = _dept_status_for_scope(raw, scope)
        return _phrase_dept(
            sched.DEPT_LABELS.get(dept, dept), date_label, status, note, scope
        )

    resolved = {
        d: _dept_status_for_scope(depts[d], scope) for d in depts
    }
    unique = set(s for s, _ in resolved.values())

    if len(unique) == 1:
        status = next(iter(unique))
        words = _STATUS_WORD.get((scope, status), f"status: {status}")
        note = next((n for _, n in resolved.values() if n), None)
        sentence = f"On {date_label}, all departments — {words}."
        if note:
            sentence += f" ({note})"
        return sentence

    groups: Dict[str, List[str]] = {}
    for d, (status, _note) in resolved.items():
        groups.setdefault(status, []).append(sched.DEPT_LABELS.get(d, d))
    parts = []
    for status, dept_names in groups.items():
        words = _STATUS_WORD.get((scope, status), f"status: {status}")
        parts.append(f"{', '.join(dept_names)} — {words}")
    return f"On {date_label}: " + "; ".join(parts) + "."


def build_canonical_range_sentence(facts: Dict) -> str:
    label = facts['period_label']
    events = facts.get('events', [])
    if not events:
        hours = sched.get_normal_office_hours()
        return (
            f"Nothing special is scheduled for {label} — just the regular "
            f"school days, office hours {hours['start']}–{hours['end']} "
            f"on {hours['days']}."
        )
    lines = [f"For {label}, here's what's scheduled:"]
    for ev in events:
        lines.append(
            f"- {ev['date']}: {ev['title']}"
            + (" (closed)" if ev['is_closed'] else "")
        )
    return "\n".join(lines)


_RELATIVE_DATE_NOTE = (
    "You may naturally reference how the date relates to the real current "
    "date given above (e.g. say 'today', 'tomorrow', or 'this Thursday' "
    "instead of the full date) since that's just a rephrasing of the same "
    "date, not a new fact."
)


def answer_date_question(
    question: str,
    detected_lang: str = 'english',
    ctx=None,
    forced_dept: str = None,
) -> Optional[str]:
    """
    Main entry point for any date / schedule / holiday / class question.
    Always saves ctx.last_schedule_query so short follow-ups work.
    """
    if not question:
        return None

    # 1. Parse the date
    query_date, date_type, display = parse_date_from_question(question)

    # 2. Detect department / scope
    dept = forced_dept or sched.detect_dept_from_question(question) or 'all'

    # Simple scope detection
    q_lower = question.lower()
    if any(w in q_lower for w in ('class', 'classes', 'pasok', 'klase', 'school day')):
        scope = 'class'
    elif any(w in q_lower for w in ('office', 'registrar', 'business', 'cashier', 'open', 'closed')):
        scope = 'office'
    else:
        scope = 'general'

    # 3. Gather facts + build canonical sentence
    if date_type in ('this_week', 'next_week', 'last_week', 'this_month', 'next_month', 'last_month'):
        facts = gather_range_facts(query_date, date_type, dept)
        canonical = build_canonical_range_sentence(facts)
    else:
        facts = gather_single_day_facts(query_date, dept)
        canonical = build_canonical_single_day_sentence(facts, scope, dept)

    if not canonical:
        return None

    # 4. Natural language rephrasing
    natural = nlg.speak_naturally(
        canonical,
        question,
        detected_lang,
        extra_note=_RELATIVE_DATE_NOTE,
    )

    # 5. Grounding check
    if natural and not answer_is_grounded(natural, canonical, min_overlap=0.25):
        natural = canonical

    final_answer = natural or canonical

    # 6. CRITICAL — save context so short follow-ups work
    if ctx is not None:
        ctx.last_schedule_query = {
            'query_date': query_date,
            'dept': dept,
            'scope': scope,
            'date_type': date_type,
            'detected_lang': detected_lang,
            'canonical': canonical,
            'question': question,
        }
        ctx.update(user_msg=question, bot_response=final_answer)

    return final_answer


