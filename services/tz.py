"""services/tz.py — single source of truth for "what day/time is it right
now" from the school's perspective (Asia/Manila, UTC+8) — NOT the
server's own clock, which is UTC on most hosts (Render/Vercel).

Use ph_now()/ph_today() anywhere the answer is user-facing ("today",
"this week", "is there class now") or is being compared against a
timestamp read back from the database, instead of bare datetime.now() /
date.today().

Deliberately returns NAIVE datetimes (tzinfo stripped) so they drop
straight into comparisons against this app's TIMESTAMP (not TIMESTAMPTZ)
columns, which — after database/pool.py's session-timezone fix — are
also naive Asia/Manila wall-clock values. Mixing an aware datetime here
with those naive DB values would raise TypeError, so don't remove the
.replace(tzinfo=None).

Session-timeout bookkeeping (routes/pages.py's _touch_activity /
_session_expired) is fine left as datetime.utcnow() — that only compares
durations, not calendar days, so timezone doesn't matter there.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

PH_TZ = ZoneInfo("Asia/Manila")


def ph_now() -> datetime:
    """Naive datetime representing the current wall-clock time in
    Asia/Manila (UTC+8)."""
    return datetime.now(PH_TZ).replace(tzinfo=None)


def ph_today():
    return ph_now().date()