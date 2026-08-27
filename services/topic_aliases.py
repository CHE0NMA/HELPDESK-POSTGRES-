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


_SEED_ALIASES: List[dict] = [
    # ------------------------------------------------------------------
    # Identity / History / Founders / Logo / Hymn
    # ------------------------------------------------------------------
    {
        "topic_key": "full_name_campuses",
        "match_pattern": (
            r"\b(full name|real (full )?name|name of (this )?school|"
            r"campuses|other campus(es)?|how many campuses|campus locations?|"
            r"where (is|are) the (main )?campus(es)?|which campus(es)?|"
            r"main campus|sindangan|salug|postigo|leon b\.?|"
            r"ano ang (totoong )?pangalan|unsa ang (tinuod )?ngalan|"
            r"asa ang (mga )?campus|pila ka campus|asa ang main campus)\b"
        ),
        "title_needles": ["academic program offerings", "brief history", "meetings"],
        "content_needles": [
            "sindangan", "salug", "leon b. postigo", "philippine advent college, inc",
            "main campus", "lbp campus", "magsaysay"
        ],
        "source_priority": "handbook_only",
        "priority": 110,
        "notes": "Full name + all campuses – keep strict, no bare 'on campus'",
    },
    {
        "topic_key": "original_name",
        "match_pattern": (
            r"\b(called before|original name|formerly known|old name|hillside view|"
            r"before it became|una nga ngalan|kanhi nga ngalan|unsa ang una|"
            r"ano ang dating pangalan)\b"
        ),
        "title_needles": ["brief history"],
        "content_needles": ["hillside view", "originally known", "1975"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "founded_history",
        "match_pattern": (
            r"\b(when did .{0,20} start|who founded|founder|founders|macias|"
            r"when .{0,15} (built|established|founded)|something about 1975|"
            r"tell me more about the founders|macias family|"
            r"kanus-a gitukod|kinsa ang nagtukod|sino ang nagtatag|"
            r"kailan tinayo|kasaysayan|history of pac)\b"
        ),
        "title_needles": ["brief history"],
        "content_needles": ["mariano", "margie", "macias", "1975", "founded"],
        "source_priority": "handbook_only",
        "priority": 105,
    },
    {
        "topic_key": "logo",
        "match_pattern": (
            r"\b(logo|emblem|symbol|what does .{0,15} logo|"
            r"ano ang logo|unsa ang logo|kahulugan ng logo)\b"
        ),
        "title_needles": ["logo and symbols", "logo", "symbols"],
        "content_needles": ["bible", "torch", "1975", "blue", "red", "white"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "hymn",
        "match_pattern": (
            r"\b(hymn|who wrote|composer|lyrics|alma mater|"
            r"kanta sa pac|pac hymn|sino ang sumulat|kinsa ang nagsulat)\b"
        ),
        "title_needles": ["pac hymn", "hymn"],
        "content_needles": ["pilon", "la sage", "composed"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "ellen_white",
        "match_pattern": (
            r"\b(ellen white|education book|greatest want|"
            r"libro ni ellen|white education)\b"
        ),
        "title_needles": ["greatest want", "foreword"],
        "content_needles": ["ellen g. white", "education", "greatest want of the world"],
        "source_priority": "handbook_only",
        "priority": 90,
    },

    # ------------------------------------------------------------------
    # Philosophy / Vision / Mission / Core Values / Goals
    # ------------------------------------------------------------------
    {
        "topic_key": "vision",
        "match_pattern": (
            r"\b(want to become|aspire|future of|what does .{0,20} want|"
            r"envision|envisions|vision of|ano ang vision|unsa ang vision|"
            r"gusto mahimo|kung unsa ang gusto)\b"
        ),
        "title_needles": ["vision"],
        "content_needles": ["envisions", "center of excellence"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "core_values",
        "match_pattern": (
            r"\b(core values|pac values|the pac ones|values of (the )?(school|college|pac)|"
            r"p-a-c values|purpose-driven|ano ang core values|unsa ang core values|"
            r"mga pagpapahalaga|mga values)\b"
        ),
        "title_needles": ["core values", "core value"],
        "content_needles": ["purpose-driven", "christ-centered", "academic excellence oriented"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "christ_centered",
        "match_pattern": (
            r"\b(christ-?centered|christ centered living|christ.?centered|"
            r"centered on christ|kinabuhi nga kristohanon|"
            r"pamumuhay na kristiyano|christ centered)\b"
        ),
        "title_needles": ["core values", "declaration of principles"],
        "content_needles": ["christ-centered", "christ centered", "c – christ-centered"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "mission",
        "match_pattern": (
            r"\b(main mission|what.?s the mission|mission of|"
            r"ano ang mission|unsa ang mission|misyon)\b"
        ),
        "title_needles": ["mission"],
        "content_needles": ["human service", "personal integrity"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "philosophy",
        "match_pattern": (
            r"\b(philosophy|school.?s? philosophy|philosophy about|"
            r"pilosopiya|ano ang philosophy|unsa ang philosophy)\b"
        ),
        "title_needles": ["philosophy"],
        "content_needles": ["source of true education", "god is the source"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "goals",
        "match_pattern": (
            r"\b(goals students should|goals by graduation|what (are|should).{0,20}goals|"
            r"mga layunin|mga goals|unsa ang goals)\b"
        ),
        "title_needles": ["goals"],
        "content_needles": ["kindness", "patience", "spiritual life", "scriptures"],
        "source_priority": "handbook_only",
        "priority": 85,
    },

    # ------------------------------------------------------------------
    # Demerit / Discipline / Probation / Appeal / Fine
    # ------------------------------------------------------------------
    {
        "topic_key": "merit_points_start",
        "match_pattern": (
            r"\b(how many points|start with|merit points|points do we start|30 points|"
            r"pila ka points|ilan ang points|mga puntos)\b"
        ),
        "title_needles": ["student's behavioral code", "behavioral code"],
        "content_needles": ["30 points", "merit of 30", "beginning of the semester"],
        "source_priority": "handbook_only",
        "priority": 105,
    },
    {
        "topic_key": "first_degree",
        "match_pattern": (
            r"\b(first[ -]?degree|light\s+offense|examples of (light|first)|"
            r"unang degree|una nga degree|banay nga sala)\b"
        ),
        "title_needles": ["first-degree", "first degree"],
        "content_needles": ["5 demerit", "public displays of affection", "unchaperoned"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "second_degree",
        "match_pattern": (
            r"\b(how bad|how serious|what.?s a|what is a).{0,20}second[ -]?degree\b|"
            r"\bsecond[ -]?degree|"
            r"ikaduha nga degree|pangalawang degree\b"
        ),
        "title_needles": ["second-degree", "second degree"],
        "content_needles": ["15 demerit"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "third_degree",
        "match_pattern": (
            r"\b(heaviest|worst|most serious).{0,25}(demerit|offense|level|degree)\b|"
            r"\bthird[ -]?degree|"
            r"ikatulo nga degree|pangatlong degree|pinaka bug-at\b"
        ),
        "title_needles": ["third-degree", "third degree"],
        "content_needles": ["30 demerit"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "max_fine",
        "match_pattern": (
            r"\b(max(imum)?\s*fine|is there a maximum fine|biggest fine|fine for|"
            r"community service instead of fine|"
            r"pinaka taas nga fine|unsa ang maximum fine|maximum nga multa)\b"
        ),
        "title_needles": ["disciplinary measures", "student's behavioral code"],
        "content_needles": ["1,500", "1500", "fines", "community service"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "appeal",
        "match_pattern": (
            r"\b(appeal).{0,50}(decision|disciplinary|sanction|how long|window)|"
            r"how long.{0,30}appeal|"
            r"mag-apela|pag-apela|unsaon pag-apela|ilan araw ang appeal\b"
        ),
        "title_needles": ["prescription and appeal"],
        "content_needles": ["15 days", "appealed within"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "probation_20",
        "match_pattern": (
            r"\b(20 or more demerits|20\+ demerits|get 20|probation|"
            r"demerit.{0,25}(end|near).{0,15}(term|semester)|"
            r"(end|near).{0,15}(term|semester).{0,25}demerit|"
            r"malapit sa katapusan.{0,20}demerit|"
            r"hapit na matapos.{0,20}demerit|"
            r"duol sa katapusan|near the end of the term)\b"
        ),
        "title_needles": ["disciplinary measures"],
        "content_needles": ["20 demerits", "placed on probation", "one semester"],
        "source_priority": "handbook_only",
        "priority": 100,
    },

    # ------------------------------------------------------------------
    # Dress / Conduct / Dorm / PDA / Curfew / Music / Worship
    # ------------------------------------------------------------------
    {
        "topic_key": "jewelry_dress",
        "match_pattern": (
            r"\b(jewelry|necklace|ring|earrings|piercing|dress code|"
            r"how .{0,15}dress|proper attire|inappropriate attire|"
            r"slippers|short pants|crop tops|tsinelas|damit|"
            r"ano ang damit|pananamit|bisti|pwede ba mag jewelry|"
            r"pwede ba magdala ug singsing|allowed to wear|"
            r"slippers allowed|are slippers|"
            # Bisaya / Tagalog natural phrasings
            r"unsay (dapat )?suoton|unsaon pagbisti|unsa ang (dapat )?isul-ob|"
            r"unsa (man )?ang dress code|dress code diri|suoton diri|"
            r"ano ang (dapat )?isuot|paano magdamit|anong damit|"
            r"mga lalaki|sa lalaki|for (the )?men|for (the )?guys|"
            r"mga babae|sa babae|for (the )?ladies|for (the )?girls)\b"
        ),
        "title_needles": ["dress and appearance", "first-degree", "first degree", "school uniform"],
        "content_needles": [
            "necklace", "earrings", "jewelry", "wedding ring", "slippers",
            "short pants", "crop tops", "uniform", "attire",
            "ladies", "men", "dress modestly", "simple but neat"
        ],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "unchaperoned_hike",
        "match_pattern": (
            r"\b(hike|outing).{0,30}(chaperone|without|unchaperoned)|unchaperoned|"
            r"(without a chaperone)|walay chaperone|walang chaperone\b"
        ),
        "title_needles": ["first-degree", "first degree"],
        "content_needles": ["unchaperoned hike", "unchaperoned"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "visit_dorms",
        "match_pattern": (
            r"\b(visit|visiting).{0,25}(other )?dorms?|other dorms?|"
            r"bisita sa laing dorm|pumunta sa ibang dorm\b"
        ),
        "title_needles": ["first-degree", "first degree"],
        "content_needles": ["visiting in dormitor", "visiting in dormitories"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "cook_dorm",
        "match_pattern": (
            r"\b(cook|cooking).{0,30}(dorm|dormitory|inside)|cook inside|"
            r"magluto sa dorm|pagluto sulod sa dorm\b"
        ),
        "title_needles": ["second-degree", "second degree"],
        "content_needles": ["cooking within the dormitory", "cooking within"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "cafeteria_food",
        "match_pattern": (
            r"\b(food|cafeteria).{0,30}(not allowed|prohibited|bawal|banned|restricted)|"
            r"bawal nga pagkaon|dili allowed nga pagkaon\b"
        ),
        "title_needles": ["second-degree", "second degree"],
        "content_needles": ["not sanctioned"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "pda",
        "match_pattern": (
            r"\b(public display of affection|pda|going together|sweethearts|"
            r"kissing|necking|magholding hands|maghalok|"
            r"public display|affection)\b"
        ),
        "title_needles": ["first-degree", "first degree", "third-degree", "third degree"],
        "content_needles": ["public displays of affection", "going together", "kissing", "necking"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "curfew",
        "match_pattern": (
            r"\b(curfew|9:30|9\.30|dorm.*(hours|time)|curfew hours|"
            r"oras sa pag-uli|oras sa dorm|kanus-a ang curfew)\b"
        ),
        "title_needles": ["first-degree", "first degree", "dormitory"],
        "content_needles": ["9:30", "curfew", "4:30"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "music_songs",
        "match_pattern": (
            r"\b(music|songs|questionable music|indecent songs|"
            r"rules? about music|musika|kanta|awit|"
            r"bawal nga kanta|dili allowed nga music)\b"
        ),
        "title_needles": ["first-degree", "first degree"],
        "content_needles": ["questionable music", "indecent songs"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "worship_absences",
        "match_pattern": (
            r"\b(miss worship|worship (too many|absences)|worship absences|"
            r"too many times.*worship|kulang sa worship|"
            r"pila ka times miss worship|absent sa worship)\b"
        ),
        "title_needles": ["dormitory student's pledge", "dormitory", "first-degree"],
        "content_needles": ["worship absences", "20 absences", "30 worship"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
{
    "topic_key": "staff_people",
    "match_pattern": (
        r"(?i)\b("
        r"who is|sino ang|kinsa ang|who's|who are|"
        r"president|dean|principal|registrar|cashier|"
        r"guidance|counselor|librarian|nurse|"
        r"teacher|instructor|professor|faculty|"
        r"staff|employee|personnel|office head|"
        r"director|coordinator|chairperson|"
        r"contact (person|of)|number of|email of|"
        r"office of the|sa opisina ni|"
        r"sino si|kinsa si"
        r")\b"
    ),
    "title_needles": [
        "administrative staff", "faculty", "personnel",
        "office of the president", "dean", "registrar",
        "guidance office", "student affairs"
    ],
    "content_needles": [
        "president", "dean", "registrar", "cashier",
        "guidance counselor", "principal", "faculty"
    ],
    "source_priority": "handbook_only",
    "priority": 115,
},
    # ------------------------------------------------------------------
    # Legal / Safe Spaces / Anti-Bullying / Hazing / Social Media
    # ------------------------------------------------------------------
    {
        "topic_key": "safe_spaces",
        "match_pattern": (
            r"\b(safe spaces|ra.?11313|safe spaces act|11313|"
            r"gender-based|gbsh|committee for safe|"
            r"batas sa safe spaces)\b"
        ),
        "title_needles": ["gender-based sexual harassment", "gender-based", "safe spaces"],
        "content_needles": ["11313", "safe spaces act", "gender-based harassment committee"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        # BUG FIX: only matched "anti-bullying"/"bullying law"/the RA
        # number -- a bare "bully"/"bullies"/"bullied" (e.g. "what would
        # happen if I bully someone") never matched at all. Added bare
        # word forms + common Filipino phrasing. NOTE: if this alias set
        # was already seeded into handbook_topic_aliases, this in-code
        # seed change alone won't update the live DB rows -- re-run
        # seed_default_aliases(force=True) or edit the row from the
        # admin dashboard for it to take effect.
        "topic_key": "anti_bullying",
        "match_pattern": (
            r"\b(anti.?bullying|ra.?10627|bullying law|10627|"
            r"batas anti-bullying|anti bullying|"
            r"bully|bullies|bullied|bullying|"
            r"nambu-?bully|nanbu-?bully|pambu-?bully|bina-?bully|"
            r"nanintimidar|pagdaugdaug)\b"
        ),
        "title_needles": ["grievance", "anti-bullying", "bullying", "third-degree"],
        "content_needles": ["10627", "anti-bullying", "bullying"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "hazing",
        "match_pattern": (
            r"\b(hazing|fraternity|sorority|"
            r"frat|soro|pag-hazing)\b"
        ),
        "title_needles": ["third-degree", "third degree"],
        "content_needles": ["fraternity", "sorority", "hazing", "never be allowed"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "social_media",
        "match_pattern": (
            r"\b(social media|cyber|online (harassment|libel)|"
            r"responsible use of social|facebook|tiktok rules)\b"
        ),
        "title_needles": ["third-degree", "third degree"],
        "content_needles": ["social media", "cyber libel", "responsible use"],
        "source_priority": "handbook_only",
        "priority": 90,
    },

    # ------------------------------------------------------------------
    # Academic Programs / Strands / TESDA / Campuses for programs
    # ------------------------------------------------------------------
    {
        "topic_key": "programs",
        "match_pattern": (
            r"\b(programs?|courses?|kurso).{0,25}(offer|available|does pac|meron)|"
            r"nursing|midwifery|business administration|education programs?|"
            r"tesda|it or computer|computer science|information technology|"
            r"bsba|bsit|bpa|beed|bsed|bped|"
            r"which campus(es)? .{0,20}offer|"
            r"naa bay nursing|naa bay midwifery|naa bay education|"
            r"meron bang nursing|meron bang midwifery|"
            r"ano ang mga program|unsa ang mga kurso|"
            r"is there nursing|is there midwifery|"
            r"do they have education|is business administration available)\b"
        ),
        "title_needles": ["academic program offerings", "academic program", "program offerings"],
        "content_needles": [
            "bachelor of science in nursing", "midwifery", "bachelor of early childhood",
            "bachelor of elementary education", "bachelor of science in secondary education",
            "bachelor of science in business administration", "information technology",
            "computer science", "stem", "humss", "tesda", "housekeeping", "caregiving",
            "main campus", "salug"
        ],
        "source_priority": "handbook_only",
        "priority": 110,
    },
    {
        "topic_key": "senior_high_strands",
        "match_pattern": (
            r"\b(senior high|strands?|shs|what strands|"
            r"mga strand|unsa ang strands|ano ang strands)\b"
        ),
        "title_needles": ["academic program offerings", "academic program", "program offerings"],
        "content_needles": ["stem", "humss", "abm", "gas", "grades 11"],
        "source_priority": "handbook_only",
        "priority": 90,
    },

    # ------------------------------------------------------------------
    # Academic policies / Fees / Scholarships / Incomplete / Residency
    # ------------------------------------------------------------------
    {
        "topic_key": "incomplete_grades",
        "match_pattern": (
            r"\b(incomplete grades?|deadline for incomplete|"
            r"how long to complete an incomplete|complete an incomplete|"
            r"inc grade|hanggang kailan ang inc|pila ka tuig ang incomplete)\b"
        ),
        "title_needles": ["completion of incomplete", "schedule of examinations", "grading"],
        "content_needles": ["within one (1) year", "one year", "incomplete grade"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "residency_limit",
        "match_pattern": (
            r"\b(residency (limit|requirement)|maximum .{0,15} years|"
            r"finish(ing)? a degree|complete the program|"
            r"how long .{0,20}(finish|complete|degree)|"
            r"residency limit|maximum of 5 years|"
            r"hanggang ilan taon|pila ka tuig makahuman)\b"
        ),
        "title_needles": ["admission policies", "residency", "guidelines for graduation and academic honor", "academic honors"],
        "content_needles": ["maximum of 5 years", "5 years to complete", "residency"],
        "source_priority": "handbook_only",
        "priority": 105,
    },
    {
        "topic_key": "drug_test",
        "match_pattern": (
            r"\b(drug test|first.?year.{0,20}drug|"
            r"drug test sa first year|required drug test)\b"
        ),
        "title_needles": ["admission policies", "first-year"],
        "content_needles": ["drug test", "first-year students will be required"],
        "source_priority": "handbook_only",
        "priority": 95,
    },
    {
        "topic_key": "sibling_scholarship",
        "match_pattern": (
            r"\b(scholarship.{0,20}sibling|siblings?|family scholar|"
            r"discount.{0,15}sibling|scholarship para sa magkapatid|"
            r"discount sa siblings)\b"
        ),
        "title_needles": ["family scholar", "scholarship"],
        "content_needles": ["four siblings", "three siblings", "two siblings", "youngest sibling"],
        "source_priority": "handbook_only",
        "priority": 100,
    },
    {
        "topic_key": "dorm_fees",
        "match_pattern": (
            r"\b(dormitory (fees?|charges?)|dorm fees?|bed space|"
            r"bayad sa dorm|pila ang dorm)\b"
        ),
        "title_needles": ["dormitory charges", "dormitory"],
        "content_needles": ["717.99", "938.91", "331.38", "electricity"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "week_spiritual",
        "match_pattern": (
            r"\b(week of spiritual emphasis|spiritual emphasis|"
            r"semana sa espirituhanong paghatag og gibug-aton)\b"
        ),
        "title_needles": ["multifaith", "first-degree"],
        "content_needles": ["week of spiritual emphasis", "spiritual emphasis"],
        "source_priority": "handbook_only",
        "priority": 90,
    },
    {
        "topic_key": "morphe",
        "match_pattern": (
            r"\b(morphe|2008|manual of regulations|"
            r"morphe of 2008)\b"
        ),
        "title_needles": ["school discipline under morphe", "morphe"],
        "content_needles": ["morphe of 2008", "manual of regulations"],
        "source_priority": "handbook_only",
        "priority": 85,
    },
    {
        "topic_key": "2025_edition",
        "match_pattern": (
            r"\b(2025 edition|school year 2025|2025-2026|2025.?2026|"
            r"sy 2025|bag-ong edition|latest handbook)\b"
        ),
        "title_needles": ["gender-based", "foreword", "academic program"],
        "content_needles": ["2025", "2025-2026", "sy 2025"],
        "source_priority": "handbook_only",
        "priority": 80,
    },

    # ------------------------------------------------------------------
    # People / Staff / Faculty / Offices (for future data you will insert)
    # ------------------------------------------------------------------
    {
        "topic_key": "staff_people",
        "match_pattern": (
            r"\b(who is the|who's the|name of the|"
            r"president|dean|registrar|cashier|guidance counselor|"
            r"faculty|teacher|instructor|staff|personnel|"
            r"office of the|director of|"
            r"kinsa ang|sino ang|sino ba ang|"
            r"unsa ang ngalan sa|ano ang pangalan ng)\b"
        ),
        "title_needles": [
            "faculty", "staff", "personnel", "administration",
            "office of the president", "dean", "registrar", "guidance"
        ],
        "content_needles": [
            "president", "dean", "registrar", "cashier", "guidance",
            "faculty", "staff", "director"
        ],
        "source_priority": "handbook_only",
        "priority": 85,
        "notes": "Starter alias for when you add staff/faculty sections or FAQ rows with photos",
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
    # Callers (e.g. handbook._forced_topic_for) already normalize before
    # calling in, but normalize here too -- defense in depth for any other
    # caller, and cheap/idempotent if already done. Fixes vernacular
    # spelling (bulan/bulana, suoton, haxing, ...) and unwraps hypothetical
    # ("what would happen if I ___") framing into direct-question keywords.
    try:
        from services.language import expand_query as _expand_query
    except Exception:
        try:
            from language import expand_query as _expand_query
        except Exception:
            _expand_query = None
    q = (_expand_query(question) if _expand_query else question).lower()
    hits = []
    for a in aliases:
        try:
            if re.search(a["match_pattern"], q, re.I):
                hits.append(a)
        except re.error:
            logger.warning(f"bad alias regex id={a.get('id')}: {a.get('match_pattern')}")
    return hits


_hit_dedup_lock = Lock()
_recent_alias_hits: Dict[tuple, float] = {}
_HIT_DEDUP_WINDOW = 3.0  # seconds — handbook.py calls forced_topic_for()
# several times (once early, once in main retrieval, and once per
# candidate chunk inside chunk_is_relevant_hard's loop) for what is
# logically ONE student question. Bumping hit_count on every call would
# inflate it 5-20x per real question. This collapses those repeat calls
# within one answer-pipeline run into a single increment, without
# caching (and thus staling) the needles themselves — those are always
# recomputed fresh.


def _should_bump(alias_id: int, question: str) -> bool:
    key = (alias_id, question)
    now = time.time()
    with _hit_dedup_lock:
        last = _recent_alias_hits.get(key)
        if last is not None and (now - last) < _HIT_DEDUP_WINDOW:
            return False
        _recent_alias_hits[key] = now
        if len(_recent_alias_hits) > 2000:  # opportunistic cleanup
            cutoff = now - _HIT_DEDUP_WINDOW
            for k, ts in list(_recent_alias_hits.items()):
                if ts < cutoff:
                    del _recent_alias_hits[k]
    return True


def forced_topic_for(question: str, get_db_fn=None) -> Optional[Tuple[List[str], List[str]]]:
    """Drop-in replacement for handbook._forced_topic_for.

    Returns (title_needles, content_needles) from the highest-priority match.

    BUG FIX: this now also records that the alias actually fired
    (hit_count), which used to be dead code — apply_alias_boosts()/
    bump_alias_hit() existed but nothing in handbook.py ever called them,
    since handbook.py already does its own inline scoring boost and only
    ever used this read-only lookup. That left hit_count permanently
    stuck at 0 in the admin dashboard, with no way to see which aliases
    were actually earning their keep. get_db_fn now also resolves to the
    real database.get_db by default (handbook.py's caller never passed
    one), matching the pattern match_aliases() already uses — otherwise
    bump_alias_hit() would silently no-op forever.
    """
    if get_db_fn is None:
        try:
            from database import get_db
            get_db_fn = get_db
        except Exception:
            get_db_fn = None

    hits = match_aliases(question, get_db_fn)
    if not hits:
        return None
    best = hits[0]
    if get_db_fn is not None and _should_bump(best["id"], question):
        bump_alias_hit(best["id"], get_db_fn)
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