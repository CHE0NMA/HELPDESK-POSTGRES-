"""Pure fact-gathering for schedule/holiday/emergency-closure questions.
No phrasing logic lives here — see services/dates.py for that. Keeping
this file phrasing-free means these facts can be reused (dashboard
widgets, digest emails, etc.) without dragging AI calls along.

--- Postgres migration notes ---
- MySQL `MONTH(x)`/`DAY(x)`/`YEAR(x)` -> Postgres `EXTRACT(MONTH FROM x)`
  etc. (EXTRACT returns numeric, cast/compare accordingly).
- MySQL `DATE_SUB(NOW(), INTERVAL %s DAY)` -> Postgres
  `NOW() - (%s || ' days')::interval`.
- MySQL `ON DUPLICATE KEY UPDATE` -> Postgres `ON CONFLICT ... DO UPDATE`.
  Assumes `question` has a UNIQUE constraint on missing_date_queries
  (that's what the original ON DUPLICATE KEY UPDATE implied) — add one if
  it isn't already there.
- IMPORTANT (behavior change, not just syntax): psycopg2 returns Postgres
  JSON/JSONB columns already deserialized as native Python list/dict, not
  as a JSON string the way pymysql does. `_parse_json_list()` already
  guarded for that (`isinstance(raw, (list, tuple))`), but `_parse_depts()`
  did not — fixed below, since calling `json.loads()` on an already-parsed
  list raises TypeError and used to silently fall back to `['all']`,
  quietly dropping real department data.
"""
import json
import logging
from datetime import timedelta
from typing import Dict, List, Optional

from database import get_db

logger = logging.getLogger("pac_helpdesk.schedule")

DEPT_LABELS = {
    'all': 'all departments', 'college': 'College',
    'highschool': 'High School', 'elementary': 'Elementary',
    'registrar': "Registrar's Office", 'business': 'Business Office',
}

GRADE_LEVELS = {
    'all': 'All Grades', 'k-6': 'Elementary (K-6)',
    '7-10': 'Junior High (7-10)', '11-12': 'Senior High (11-12)', 'college': 'College'
}

GRADE_LEVEL_DETECT = {
    'k-6': ['elementary', 'grade 1', 'grade 2', 'grade 3', 'grade 4', 'grade 5', 'grade 6', 'kinder', 'kindergarten'],
    '7-10': ['junior high', 'grade 7', 'grade 8', 'grade 9', 'grade 10', 'jhs'],
    '11-12': ['senior high', 'grade 11', 'grade 12', 'shs'],
    'college': ['college', 'bsed', 'bsba', 'beed', 'bsit', 'bs', 'ba', 'tertiary', 'undergraduate']
}

DEPT_DETECT = {
    'registrar': ['registrar', 'tor', 'transcript', 'clearance', 'grades'],
    'business': ['business office', 'cashier', 'payment', 'fees', 'tuition', 'cashier office'],
    'college': ['college', 'collegiate', 'tertiary', 'undergraduate', 'bsed', 'bsba', 'beed', 'bsit'],
    'highschool': ['high school', 'highschool', 'senior high', 'shs', 'grade 11', 'grade 12', 'jhs', 'junior high'],
    'elementary': ['elementary', 'grade school', 'primary', 'grade 1', 'grade 2', 'grade 3',
                   'grade 4', 'grade 5', 'grade 6'],
}

ALL_DEPTS = ['college', 'highschool', 'elementary', 'registrar', 'business']
OFFICE_ONLY_DEPTS = {'registrar', 'business'}

CLASS_KEYWORDS = [
    'class', 'classes', 'klase', 'klasi', 'klas', 'clase',
    'may klase', 'walang klase', 'may pasok', 'walang pasok', 'pasok',
    'walay klase', 'aduna bay klase', 'naa bay klase', 'naa bay klasi',
    'naay klase', 'unsa may klase', 'f2f class', 'face to face class',
]
OFFICE_KEYWORDS = [
    'office', 'opisina', 'registrar office', 'cashier', 'business office',
    'admin office', 'bukas ang opisina', 'abli ang opisina', 'open ang office',
]


def detect_dept_from_question(question: str) -> str:
    q = question.lower()
    for dept, keywords in DEPT_DETECT.items():
        for kw in keywords:
            if kw in q:
                return dept
    return 'all'


def detect_grade_level(question: str) -> str:
    q = question.lower()
    for grade, keywords in GRADE_LEVEL_DETECT.items():
        for kw in keywords:
            if kw in q:
                return grade
    return 'all'


def detect_query_scope(question: str) -> str:
    from utils.text import fuzzy_contains
    q = question.lower()
    asks_class = fuzzy_contains(q, CLASS_KEYWORDS)
    asks_office = fuzzy_contains(q, OFFICE_KEYWORDS)
    if asks_class and not asks_office:
        return 'class'
    if asks_office and not asks_class:
        return 'office'
    return 'general'


def get_normal_office_hours() -> dict:
    return {'start': '08:00', 'end': '17:00', 'days': 'Monday–Friday'}


def get_default_schedule(department: str, grade_level: str, day_of_week: int) -> Optional[Dict]:
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT start_time, end_time, has_class
            FROM default_schedules
            WHERE department = %s
              AND (grade_level = %s OR grade_level = 'all')
              AND day_of_week = %s
              AND is_active = 1
            ORDER BY CASE WHEN grade_level = %s THEN 0 ELSE 1 END
            LIMIT 1
        """, (department, grade_level, day_of_week, grade_level))
        row = cur.fetchone()
        if row:
            return {'start_time': row[0], 'end_time': row[1], 'has_class': bool(row[2])}
        return None
    except Exception as e:
        logger.error(f"get_default_schedule error: {e}")
        return None
    finally:
        cur.close()
        conn.close()


def get_emergency_closures(query_date) -> List[Dict]:
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, title, reason, severity, start_date, end_date,
                   affected_departments, affected_grades,
                   message_english, message_tagalog, message_bisaya
            FROM emergency_closures
            WHERE is_active = 1
              AND start_date <= %s
              AND (end_date IS NULL OR end_date >= %s)
            ORDER BY severity DESC, created_at DESC
        """, (query_date, query_date))
        rows = cur.fetchall()
        return [{
            'id': r[0], 'title': r[1], 'reason': r[2], 'severity': r[3],
            'start_date': r[4], 'end_date': r[5],
            'affected_departments': _parse_json_list(r[6]),
            'affected_grades': _parse_json_list(r[7]),
            'message_english': r[8], 'message_tagalog': r[9], 'message_bisaya': r[10],
        } for r in rows]
    except Exception as e:
        logger.error(f"get_emergency_closures error: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def get_holiday_for_date(query_date, department: str = 'all') -> Optional[Dict]:
    """Return the active closed event/holiday that applies on query_date.

    end_date rules:
      - NULL          → single-day event (only start_date)
      - = start_date  → single-day event
      - > start_date  → multi-day range inclusive
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT title, is_closed, message_english, message_tagalog, message_bisaya,
                   content_type, start_date, end_date, departments, grade_levels
            FROM dynamic_schedules
            WHERE is_active = 1
              AND is_closed = 1
              AND (
                -- One-time / ranged events
                (
                  start_date <= %s
                  AND COALESCE(end_date, start_date) >= %s
                )
                -- Yearly repeat (same month+day every year)
                OR (
                  repeat_type = 'yearly'
                  AND EXTRACT(MONTH FROM start_date) = EXTRACT(MONTH FROM %s::date)
                  AND EXTRACT(DAY FROM start_date) = EXTRACT(DAY FROM %s::date)
                  AND (repeat_until_year IS NULL OR repeat_until_year >= EXTRACT(YEAR FROM %s::date))
                )
                -- Monthly repeat (same day-of-month)
                OR (
                  repeat_type = 'monthly'
                  AND EXTRACT(DAY FROM start_date) = EXTRACT(DAY FROM %s::date)
                  AND (repeat_until_year IS NULL OR repeat_until_year >= EXTRACT(YEAR FROM %s::date))
                )
              )
            ORDER BY
              CASE WHEN start_date = %s THEN 0 ELSE 1 END,
              start_date DESC
            LIMIT 1
        """, (
            query_date, query_date,          # one-time range
            query_date, query_date, query_date,  # yearly
            query_date, query_date,          # monthly
            query_date,                      # ORDER BY exact-day priority
        ))
        row = cur.fetchone()
        if not row:
            return None

        depts = _parse_json_list(row[8], default=['all'])
        grades = _parse_json_list(row[9], default=['all'])

        # Department filter
        if department != 'all' and 'all' not in depts and department not in depts:
            return None

        return {
            'title': row[0],
            'is_closed': bool(row[1]),
            'message_english': row[2],
            'message_tagalog': row[3],
            'message_bisaya': row[4],
            'content_type': row[5],
            'start_date': row[6],
            'end_date': row[7],
            'departments': depts,
            'grade_levels': grades,
        }
    except Exception as e:
        logger.error(f"get_holiday_for_date error: {e}")
        return None
    finally:
        cur.close()
        conn.close()


def _parse_json_list(raw, default=None):
    """Safely parse departments / grade_levels.

    Handles:
      - real JSON arrays:  ["all"]
      - double-encoded:    "[\\"all\\"]"
      - plain strings:     all
      - None / empty
    """
    if default is None:
        default = ['all']
    if raw is None or raw == '':
        return list(default)
    if isinstance(raw, (list, tuple)):
        return list(raw)
    try:
        val = json.loads(raw)
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(val, list):
            return val if val else list(default)
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        return list(default)
    except (json.JSONDecodeError, TypeError):
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        return list(default)

def get_known_holiday_for_date(query_date) -> Optional[Dict]:
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT title, start_date, message_english, message_tagalog, message_bisaya
            FROM dynamic_schedules
            WHERE is_active = 1 AND is_closed = 1 AND repeat_type = 'yearly'
              AND (repeat_until_year IS NULL OR repeat_until_year >= EXTRACT(YEAR FROM %s::date))
        """, (query_date,))
        for row in cur.fetchall():
            start_date = row[1]
            if start_date and start_date.month == query_date.month and start_date.day == query_date.day:
                return {'title': row[0], 'message_english': row[2],
                        'message_tagalog': row[3], 'message_bisaya': row[4]}
        return None
    except Exception as e:
        logger.error(f"get_known_holiday_for_date error: {e}")
        return None
    finally:
        cur.close()
        conn.close()


def get_events_for_week(start_date, detected_lang: str = 'english', dept: str = 'all') -> List[Dict]:
    end_date = start_date + timedelta(days=6)
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT title, content_type, start_date, end_date,
                   message_english, message_tagalog, message_bisaya,
                   is_closed, description
            FROM dynamic_schedules
            WHERE is_active=1
              AND ((start_date BETWEEN %s AND %s)
                OR (end_date BETWEEN %s AND %s)
                OR (start_date<=%s AND end_date>=%s))
            ORDER BY start_date ASC
        """, (start_date, end_date, start_date, end_date, start_date, end_date))
        results = []
        for row in cur.fetchall():
            title, ctype, sd, ed, msg_en, msg_tl, msg_ceb, is_closed, desc = row
            depts = _parse_depts(desc)
            if dept != 'all' and 'all' not in depts and dept not in depts:
                continue
            if detected_lang in ['tagalog', 'mixed_te', 'mixed_tb'] and msg_tl:
                msg = msg_tl
            elif detected_lang in ['bisaya', 'mixed_be', 'mixed_tb'] and msg_ceb:
                msg = msg_ceb
            else:
                msg = msg_en or title
            results.append({'title': title, 'content_type': ctype, 'start_date': sd, 'end_date': ed,
                            'message': msg, 'is_closed': bool(is_closed), 'departments': depts})
        return results
    except Exception as e:
        logger.error(f"get_events_for_week error: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def _parse_depts(raw_description) -> list:
    if not raw_description:
        return ['all']
    # Postgres JSON/JSONB columns come back from psycopg2 already
    # deserialized (a real list), not a JSON string — handle both.
    if isinstance(raw_description, (list, tuple)):
        return list(raw_description)
    try:
        parsed = json.loads(raw_description)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return ['all']


def record_missing_date_query(question: str, detected_date):
    """NOTE: missing_date_queries.question has no unique constraint in the
    real schema (confirmed from the MySQL DESCRIBE output — Key column is
    blank), so the original MySQL `ON DUPLICATE KEY UPDATE asked_count =
    asked_count + 1` never actually had anything to conflict on — it was
    always just doing a plain INSERT, and asked_count was never really
    incremented in practice. This matches that real behavior rather than
    quietly introducing dedup that wasn't there before.

    If you'd rather have real dedup, run the ALTER TABLE in
    database/full_schema.sql's trailing comment to add a unique
    constraint on `question`, then swap this back to:

        INSERT INTO missing_date_queries (question, detected_date, asked_count, last_asked)
        VALUES (%s, %s, 1, NOW())
        ON CONFLICT (question) DO UPDATE
            SET asked_count = missing_date_queries.asked_count + 1, last_asked = NOW()
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO missing_date_queries (question, detected_date, asked_count, last_asked)
            VALUES (%s, %s, 1, NOW())
        """, (question[:500], detected_date))
        conn.commit()
    except Exception as e:
        logger.error(f"record_missing_date_query error: {e}")
    finally:
        cur.close()
        conn.close()