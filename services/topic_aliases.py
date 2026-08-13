"""
services/topic_aliases.py — data-driven handbook topic routing.

Replaces (and can eventually retire) the hardcoded _FORCED_TOPIC_MATCHES
in handbook.py. Aliases live in handbook_topic_aliases so admins can
fix ranking misses without redeploying code.

Also provides:
  - source_priority: 'handbook_only' | 'faq_ok' | 'schedule_ok'
  - seed_default_aliases(): load the known PAC test-suite mappings once
  - suggest_aliases_from_unanswered(): cluster frequent misses for review
"""
from __future__ import annotations

import logging
import re
import time
from threading import Lock
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("pac_helpdesk.topic_aliases")

# In-process cache so every question doesn't hit the DB.
_CACHE: List[dict] = []
_CACHE_TS: float = 0.0
_CACHE_TTL = 120.0  # seconds
_LOCK = Lock()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS handbook_topic_aliases (
    id              SERIAL PRIMARY KEY,
    topic_key       TEXT NOT NULL,
    match_pattern   TEXT NOT NULL,
    title_needles   TEXT[] NOT NULL DEFAULT '{}',
    content_needles TEXT[] NOT NULL DEFAULT '{}',
    source_priority TEXT NOT NULL DEFAULT 'handbook_only',
    priority        INT NOT NULL DEFAULT 10,
    hit_count       INT NOT NULL DEFAULT 0,
    is_active       SMALLINT NOT NULL DEFAULT 1,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hta_active ON handbook_topic_aliases (is_active, priority DESC);

-- Optional tags / key facts on chunks (safe if columns already exist)
DO $$ BEGIN
    ALTER TABLE handbook_chunks ADD COLUMN IF NOT EXISTS topic_tags TEXT[] DEFAULT '{}';
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE handbook_chunks ADD COLUMN IF NOT EXISTS key_facts JSONB DEFAULT '[]'::jsonb;
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE handbook_chunks ADD COLUMN IF NOT EXISTS embedding JSONB;
EXCEPTION WHEN undefined_table THEN NULL;
END $$;

-- Closed-loop feedback
CREATE TABLE IF NOT EXISTS answer_feedback (
    id              SERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    question_norm   TEXT,
    rating          TEXT,              -- thumbs_up | thumbs_down
    correction      TEXT,
    source_type     TEXT,              -- handbook | faq | schedule
    chunk_ids       INT[],
    faq_id          INT,
    session_id      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_af_created ON answer_feedback (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_af_norm ON answer_feedback (question_norm);

-- Successful grounded answers (stable paraphrase cache)
CREATE TABLE IF NOT EXISTS answer_success_cache (
    id              SERIAL PRIMARY KEY,
    question_norm   TEXT NOT NULL,
    detected_lang   TEXT NOT NULL DEFAULT 'english',
    source_type     TEXT NOT NULL,
    chunk_ids       INT[],
    faq_id          INT,
    answer_text     TEXT NOT NULL,
    hit_count       INT NOT NULL DEFAULT 1,
    last_hit_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT answer_success_cache_q_lang UNIQUE (question_norm, detected_lang)
);
CREATE INDEX IF NOT EXISTS idx_asc_lang ON answer_success_cache (detected_lang);

"""


def ensure_topic_alias_schema(get_db_fn) -> None:
    """Idempotent migration — safe to call on every boot."""
    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute(SCHEMA_SQL)
        conn.commit()
        logger.info("handbook_topic_aliases schema ensured")
    except Exception as e:
        conn.rollback()
        logger.error(f"ensure_topic_alias_schema: {e}")
    finally:
        cur.close()
        conn.close()


# ---------------------------------------------------------------------------
# Seed data (mirrors the old hardcoded forced matches — loaded once)
# ---------------------------------------------------------------------------

_SEED_ALIASES: List[dict] = [
    {
        "topic_key": "vision",
        "match_pattern": r"\b(want to become|aspire|future of|what does .{0,20} want|envision|envisions)\b",
        "title_needles": ["vision"],
        "content_needles": ["envisions", "center of excellence"],
        "source_priority": "handbook_only",
        "priority": 100,
        "notes": "Q: what does PAC want to become / vision",
    },
    {
        "topic_key": "core_values",
        "match_pattern": r"\b(core values|pac values|the pac ones|values of (the )?(school|college|pac))\b",
        "title_needles": ["core values", "core value"],
        "content_needles": ["purpose-driven", "christ-centered"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "third_degree",
        "match_pattern": r"\b(heaviest|worst|most serious).{0,25}(demerit|offense|level|degree)\b|\bthird[ -]?degree\b",
        "title_needles": ["third-degree", "third degree"],
        "content_needles": ["30 demerit"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "second_degree",
        "match_pattern": r"\b(how bad|how serious|what.?s a|what is a).{0,20}second[ -]?degree\b|\bsecond[ -]?degree\b",
        "title_needles": ["second-degree", "second degree"],
        "content_needles": ["15 demerit"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "first_degree",
        "match_pattern": r"\b(first[ -]?degree|light\s+offense)\b",
        "title_needles": ["first-degree", "first degree"],
        "content_needles": ["5 demerit"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "max_fine",
        "match_pattern": r"\b(max(imum)?\s*fine|is there a maximum fine|biggest fine|fine for)\b",
        "title_needles": ["disciplinary measures"],
        "content_needles": ["1,500", "1500", "fines"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "appeal",
        "match_pattern": r"\b(appeal).{0,50}(decision|disciplinary|sanction|how long|window)|how long.{0,30}appeal\b",
        "title_needles": ["prescription and appeal"],
        "content_needles": ["15 days", "appealed within"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "unchaperoned_hike",
        "match_pattern": r"\b(hike|outing).{0,30}(chaperone|without|unchaperoned)|unchaperoned|(without a chaperone)\b",
        "title_needles": ["first-degree", "first degree"],
        "content_needles": ["unchaperoned hike", "unchaperoned"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "visit_dorms",
        "match_pattern": r"\b(visit|visiting).{0,25}(other )?dorms?|other dorms?\b",
        "title_needles": ["first-degree", "first degree"],
        "content_needles": ["visiting in dormitor", "visiting in dormitories"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "cook_dorm",
        "match_pattern": r"\b(cook|cooking).{0,30}(dorm|dormitory|inside)|cook inside\b",
        "title_needles": ["second-degree", "second degree"],
        "content_needles": ["cooking within the dormitory", "cooking within"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "cafeteria_food",
        "match_pattern": r"\b(food|cafeteria).{0,30}(not allowed|prohibited|bawal|banned|restricted)\b",
        "title_needles": ["second-degree", "second degree"],
        "content_needles": ["not sanctioned"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "safe_spaces",
        "match_pattern": r"\b(safe spaces|ra.?11313|safe spaces act|11313)\b",
        "title_needles": ["gender-based sexual harassment", "gender-based", "safe spaces"],
        "content_needles": ["11313", "safe spaces act"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "anti_bullying",
        "match_pattern": r"\b(anti.?bullying|ra.?10627|bullying law|10627)\b",
        "title_needles": ["grievance", "anti-bullying", "bullying"],
        "content_needles": ["10627", "anti-bullying"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "programs",
        "match_pattern": r"\b(programs?|courses?).{0,25}(offer|available|does pac)\b",
        "title_needles": ["academic program offerings", "academic program", "program offerings"],
        "content_needles": ["bachelor of science", "bachelor of"],
        "source_priority": "handbook_only",
        "priority": 80,
    },
    {
        "topic_key": "senior_high_strands",
        "match_pattern": r"\b(senior high|strands?|shs|what strands)\b",
        "title_needles": ["academic program offerings", "academic program", "program offerings"],
        "content_needles": ["stem", "humss", "abm", "grades 11"],
        "source_priority": "handbook_only",
        "priority": 80,
    },
    {
        "topic_key": "mission",
        "match_pattern": r"\b(main mission|what.?s the mission|mission of)\b",
        "title_needles": ["mission"],
        "content_needles": ["human service", "personal integrity"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "philosophy",
        "match_pattern": r"\b(philosophy|school.?s? philosophy)\b",
        "title_needles": ["philosophy"],
        "content_needles": ["source of true education"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "jewelry_dress",
        "match_pattern": r"\b(jewelry|necklace|ring|earrings|piercing|dress code|how .{0,15}dress|proper attire)\b",
        "title_needles": ["dress and appearance", "first-degree", "first degree", "school uniform"],
        "content_needles": ["necklace", "earrings", "jewelry", "uniform", "attire"],
        "source_priority": "handbook_only",
        "priority": 85,
    },
    {
        "topic_key": "campus_location",
        # "campus"/"campuses" is already treated as a real topic elsewhere
        # (HANDBOOK_KEYWORDS, _QUERY_EXPANSIONS, topic_title_cues in
        # handbook.py) but had no forced-topic entry, so questions like
        # "are there any campuses in PAC" / "which locations" fell through
        # to generic fuzzy retrieval, which grabbed unrelated chunks
        # (Brief History, Dress and Appearance, Mission — all of which
        # happen to share generic words like "college"/"page"/"PAC") and
        # let the LLM stitch them into a hallucinated non-sequitur
        # ("Philippines is in the Pacific region... so yes, there's a
        # campus in the Pacific region"). Pin it to the MEETINGS section,
        # which is the chunk that actually states the campus address.
        # NOTE: deliberately does NOT match the bare word "campus" alone --
        # that also matched ordinary "is X allowed on campus?" conduct
        # questions and force-hijacked them to this section too. Only
        # phrasing that actually asks about the campus's identity/location
        # should force this topic.
        "match_pattern": r"\b(campuses|other campus(es)?|how many campuses|"
                          r"campus locations?|where (is|are) the (main )?campus(es)?|"
                          r"which campus(es)?|which locations?|"
                          r"names? of the (other )?campuses?|main campus)\b",
        "title_needles": ["meetings"],
        "content_needles": ["held within the campus", "regular meetings of the members"],
        "source_priority": "handbook_only",
        "priority": 90,
        "notes": "Q: are there any campuses / which locations / campus names",
    },
]


def seed_default_aliases(get_db_fn, force: bool = False) -> int:
    """Insert seed aliases if the table is empty (or force=True)."""
    conn = get_db_fn()
    cur = conn.cursor()
    inserted = 0
    try:
        cur.execute("SELECT COUNT(*) FROM handbook_topic_aliases")
        n = cur.fetchone()[0]
        if n > 0 and not force:
            return 0
        for a in _SEED_ALIASES:
            cur.execute(
                """
                INSERT INTO handbook_topic_aliases
                    (topic_key, match_pattern, title_needles, content_needles,
                     source_priority, priority, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    a["topic_key"],
                    a["match_pattern"],
                    a["title_needles"],
                    a["content_needles"],
                    a["source_priority"],
                    a["priority"],
                    a.get("notes"),
                ),
            )
            inserted += 1
        conn.commit()
        invalidate_alias_cache()
        logger.info(f"Seeded {inserted} topic aliases")
        return inserted
    except Exception as e:
        conn.rollback()
        logger.error(f"seed_default_aliases: {e}")
        return 0
    finally:
        cur.close()
        conn.close()


# ---------------------------------------------------------------------------
# Load / match
# ---------------------------------------------------------------------------

def invalidate_alias_cache():
    global _CACHE, _CACHE_TS
    with _LOCK:
        _CACHE = []
        _CACHE_TS = 0.0


def _load_aliases(get_db_fn) -> List[dict]:
    global _CACHE, _CACHE_TS
    with _LOCK:
        if _CACHE and (time.time() - _CACHE_TS) < _CACHE_TTL:
            return _CACHE

    rows: List[dict] = []
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT id, topic_key, match_pattern, title_needles, content_needles,
                       source_priority, priority
                FROM handbook_topic_aliases
                WHERE is_active = 1
                ORDER BY priority DESC, id ASC
                """
            )
            for r in cur.fetchall():
                rows.append({
                    "id": r[0],
                    "topic_key": r[1],
                    "match_pattern": r[2],
                    "title_needles": list(r[3] or []),
                    "content_needles": list(r[4] or []),
                    "source_priority": r[5] or "handbook_only",
                    "priority": r[6] or 10,
                })
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.warning(f"_load_aliases DB miss, using seed in-memory: {e}")
        # Fallback: in-memory seed so ranking still works before migration
        for i, a in enumerate(_SEED_ALIASES):
            rows.append({"id": -(i + 1), **a})

    with _LOCK:
        _CACHE = rows
        _CACHE_TS = time.time()
    return rows


def match_aliases(question: str, get_db_fn=None) -> List[dict]:
    """Return all aliases whose pattern matches the question, highest priority first."""
    if not question:
        return []
    if get_db_fn is None:
        try:
            from database import get_db
            get_db_fn = get_db
        except Exception:
            get_db_fn = None

    aliases = _load_aliases(get_db_fn) if get_db_fn else [
        {"id": -(i + 1), **a} for i, a in enumerate(_SEED_ALIASES)
    ]
    q = question.lower()
    hits = []
    for a in aliases:
        try:
            if re.search(a["match_pattern"], q, re.I):
                hits.append(a)
        except re.error:
            logger.warning(f"bad alias regex id={a.get('id')}: {a.get('match_pattern')}")
    return hits


def forced_topic_for(question: str, get_db_fn=None) -> Optional[Tuple[List[str], List[str]]]:
    """Drop-in replacement for handbook._forced_topic_for.

    Returns (title_needles, content_needles) from the highest-priority match.
    """
    hits = match_aliases(question, get_db_fn)
    if not hits:
        return None
    best = hits[0]
    return list(best["title_needles"]), list(best["content_needles"])


def is_handbook_only(question: str, get_db_fn=None) -> bool:
    """True when any matching alias demands handbook-only answers."""
    hits = match_aliases(question, get_db_fn)
    return any(h.get("source_priority") == "handbook_only" for h in hits)


def bump_alias_hit(alias_id: int, get_db_fn) -> None:
    if not alias_id or alias_id < 0:
        return
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE handbook_topic_aliases SET hit_count = hit_count + 1, updated_at = NOW() WHERE id = %s",
                (alias_id,),
            )
            conn.commit()
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.debug(f"bump_alias_hit: {e}")


# ---------------------------------------------------------------------------
# Apply boosts to scored handbook chunks (same semantics as old forced boost)
# ---------------------------------------------------------------------------

def apply_alias_boosts(question: str, scored: List[dict], get_db_fn=None) -> List[dict]:
    """Mutate scored chunk dicts with _forced / _forced_content and score multipliers."""
    hits = match_aliases(question, get_db_fn)
    if not hits or not scored:
        return scored

    # Merge needles from all matching aliases (highest priority first)
    title_needles: List[str] = []
    content_needles: List[str] = []
    for h in hits:
        for t in h.get("title_needles") or []:
            if t not in title_needles:
                title_needles.append(t)
        for c in h.get("content_needles") or []:
            if c not in content_needles:
                content_needles.append(c)

    for s in scored:
        title_l = (s.get("section_title") or "").lower()
        blob = f"{title_l} {(s.get('content') or '')}".lower()
        title_hit = any(n.lower() in title_l for n in title_needles)
        content_hit = any(n.lower() in blob for n in content_needles) if content_needles else False
        if title_hit:
            s["score"] = float(s.get("score") or 0) * 20.0
            s["_forced"] = True
        elif content_hit:
            s["score"] = float(s.get("score") or 0) * 3.0
            s["_forced_content"] = True

    if hits and hits[0].get("id"):
        bump_alias_hit(hits[0]["id"], get_db_fn) if get_db_fn else None

    return scored


def apply_alias_boosts_light(question: str, scored: List[dict], get_db_fn=None) -> List[dict]:
    """Second pass after rarity_rerank (smaller multipliers)."""
    hits = match_aliases(question, get_db_fn)
    if not hits or not scored:
        return scored
    title_needles, content_needles = [], []
    for h in hits:
        title_needles.extend(h.get("title_needles") or [])
        content_needles.extend(h.get("content_needles") or [])
    for s in scored:
        title_l = (s.get("section_title") or "").lower()
        blob = f"{title_l} {(s.get('content') or '')}".lower()
        if any(n.lower() in title_l for n in title_needles):
            s["score"] = float(s.get("score") or 0) * 15.0
            s["_forced"] = True
        elif any(n.lower() in blob for n in content_needles):
            s["score"] = float(s.get("score") or 0) * 2.0
            s["_forced_content"] = True
    return scored


# ---------------------------------------------------------------------------
# Suggest new aliases from unanswered traffic
# ---------------------------------------------------------------------------

def suggest_aliases_from_unanswered(get_db_fn, min_count: int = 3, limit: int = 20) -> List[dict]:
    """
    Group frequent unanswered questions and return suggestion dicts for
    the admin UI. Does not auto-insert — admin must approve.
    """
    conn = get_db_fn()
    cur = conn.cursor()
    suggestions = []
    try:
        cur.execute(
            """
            SELECT LOWER(TRIM(question)) AS q, COUNT(*) AS cnt,
                   MAX(question) AS sample
            FROM unanswered_questions
            WHERE question IS NOT NULL AND LENGTH(TRIM(question)) > 5
            GROUP BY LOWER(TRIM(question))
            HAVING COUNT(*) >= %s
            ORDER BY cnt DESC
            LIMIT %s
            """,
            (min_count, limit),
        )
        for q_norm, cnt, sample in cur.fetchall():
            # Skip if an existing alias already matches
            if match_aliases(sample, get_db_fn):
                continue
            suggestions.append({
                "sample_question": sample,
                "frequency": cnt,
                "suggested_topic_key": re.sub(r"[^a-z0-9]+", "_", q_norm)[:40].strip("_"),
                "suggested_pattern": re.escape(sample.lower()[:60]),
                "notes": "Auto-suggested from unanswered_questions",
            })
    except Exception as e:
        logger.error(f"suggest_aliases_from_unanswered: {e}")
    finally:
        cur.close()
        conn.close()
    return suggestions