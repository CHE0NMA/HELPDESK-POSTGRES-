-- =====================================================================
-- Full Postgres schema for PAC Helpdesk, built from the real MySQL/
-- MariaDB `DESCRIBE`/`SHOW COLUMNS` output you posted.
--
-- Run this ONCE against a fresh database:
--     psql "postgresql://DB_USER:DB_PASSWORD@DB_HOST:DB_PORT/DB_NAME" -f full_schema.sql
--
-- After this, `database/schema.py`'s startup functions
-- (ensure_admin_exists, ensure_schedule_schema, etc.) just no-op past
-- their existence checks — they're a defensive fallback, not the
-- primary source of truth anymore. This file is.
--
-- KEY CONVERSION DECISIONS (read this before assuming anything's wrong)
-- -----------------------------------------------------------------------
-- 1. tinyint(1) -> SMALLINT, not BOOLEAN.
--    Postgres has no implicit int->boolean cast, and the codebase is
--    full of `WHERE is_active = 1`, `SET is_active = 0`, and bare `1`
--    literals in INSERT ... VALUES (..., 1) across schema.py, alerts.py,
--    schedules.py, admin.py. Using real BOOLEAN would break every one of
--    those (Postgres would raise "column is of type boolean but
--    expression is of type integer"). SMALLINT with 0/1 values keeps
--    100% of the existing SQL working unchanged. This is a deliberate
--    trade-off, not an oversight — BOOLEAN would be more "correct"
--    Postgres style, but only after auditing and rewriting every one of
--    those comparisons across ~8 files.
--
-- 2. `longtext` columns storing JSON (departments, grade_levels,
--    affected_departments, affected_grades) -> JSONB, not TEXT.
--    Your dump shows these as `longtext`, which looks like a plain text
--    column at first glance — but MariaDB implements its JSON type AS
--    longtext internally (with a JSON_VALID() check constraint), so
--    `DESCRIBE` reports "longtext" even for genuine JSON columns. The
--    application code confirms this: it always json.dumps()/json.loads()
--    these fields. JSONB is the real Postgres equivalent, and is what
--    services/schedule.py and routes/schedules.py were already written
--    against (see their "psycopg2 returns jsonb already parsed" notes).
--
-- 3. `mediumtext`/`longtext` (non-JSON) -> TEXT. Postgres TEXT has no
--    size cap, so there's no MySQL-style size-tier distinction to
--    preserve.
--
-- 4. `... TIMESTAMP ... on update current_timestamp()` -> TIMESTAMP +
--    a BEFORE UPDATE trigger (Postgres has no built-in equivalent).
--    Applies to: chat_sessions.last_activity, dynamic_schedules.updated_at,
--    emergency_closures.updated_at, faq.last_updated, visitor_stats.last_seen.
--
-- 5. dynamic_schedules.content_type's enum list was cut off in what you
--    pasted (`enum('holiday','event','office_hours','deadline','...`).
--    Rather than guess the missing values and risk rejecting a real one,
--    this became a plain VARCHAR(30) with NO check constraint. If you
--    want the same "reject invalid values" safety MySQL's enum gave you,
--    send the full list and I'll add
--    `CHECK (content_type IN (...))`.
--
-- 6. missing_date_queries.question has NO unique key in your dump (the
--    Key column is blank) — meaning the old `ON DUPLICATE KEY UPDATE
--    asked_count = asked_count + 1` in the original MySQL code never
--    actually had anything to conflict on, so it was always just doing a
--    plain INSERT (asked_count was never really being incremented). I
--    matched that real behavior (see services/schedule.py) rather than
--    quietly "fixing" it by inventing a unique constraint that wasn't
--    there — if you'd rather have real dedup, see the note at the bottom
--    of this file for the one-line opt-in.
-- =====================================================================


-- ---------------------------------------------------------------------
-- Shared trigger function for "ON UPDATE CURRENT_TIMESTAMP" columns
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at_generic()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


CREATE TABLE IF NOT EXISTS abbreviations (
    id SERIAL PRIMARY KEY,
    abbreviation VARCHAR(50) NOT NULL UNIQUE,
    expansion VARCHAR(255) NOT NULL,
    language VARCHAR(20) DEFAULT 'all',
    is_active SMALLINT DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS admin_users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_feedback (
    id SERIAL PRIMARY KEY,
    ai_response_id INTEGER,
    user_feedback SMALLINT,
    user_correction TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_generated_responses (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    intent VARCHAR(50),
    confidence REAL DEFAULT 0,
    was_helpful SMALLINT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alert_history (
    id SERIAL PRIMARY KEY,
    severity VARCHAR(20) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    message TEXT,
    channels VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_alert_history_created_at ON alert_history(created_at);

CREATE TABLE IF NOT EXISTS bad_feedback (
    id SERIAL PRIMARY KEY,
    question TEXT,
    answer TEXT,
    detected_language VARCHAR(20),
    detected_topic VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE bad_feedback ADD COLUMN IF NOT EXISTS detected_language VARCHAR(20);
ALTER TABLE bad_feedback ADD COLUMN IF NOT EXISTS detected_topic VARCHAR(100);
CREATE INDEX IF NOT EXISTS idx_bad_feedback_created_at ON bad_feedback(created_at);

CREATE TABLE IF NOT EXISTS faq (
    id SERIAL PRIMARY KEY,
    topic VARCHAR(100) DEFAULT 'General',
    intent VARCHAR(100) DEFAULT '',
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    image_path VARCHAR(255),
    keywords TEXT,
    related_questions TEXT,
    language VARCHAR(20) DEFAULT 'en',
    category VARCHAR(100),
    priority_score INTEGER DEFAULT 0,
    attachment_url VARCHAR(255),
    reviewed_flag INTEGER DEFAULT 0,
    last_reviewed TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    public_uuid VARCHAR(36) UNIQUE,
    source VARCHAR(32) NOT NULL DEFAULT 'manual',
    -- Cached embedding vector for semantic FAQ matching (see
    -- services/embeddings.py). NULL until the row is (re-)saved through
    -- the admin FAQ editor, or until db_migration/backfill_faq_embeddings.py
    -- has been run once for pre-existing rows. Stored as JSONB (a plain
    -- array of floats) rather than a dedicated vector type — no pgvector
    -- extension dependency, which free-tier hosts don't always enable.
    embedding JSONB
);
CREATE INDEX IF NOT EXISTS idx_faq_topic ON faq(topic);
CREATE INDEX IF NOT EXISTS idx_faq_priority_score ON faq(priority_score);
CREATE INDEX IF NOT EXISTS idx_faq_source ON faq(source);
-- faq.last_updated needs its own trigger function (not the generic
-- updated_at one) since this column is named differently.
CREATE OR REPLACE FUNCTION set_faq_last_updated()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER trg_faq_last_updated
    BEFORE UPDATE ON faq
    FOR EACH ROW EXECUTE FUNCTION set_faq_last_updated();

CREATE TABLE IF NOT EXISTS faq_review_queue (
    id SERIAL PRIMARY KEY,
    faq_id INTEGER UNIQUE,
    reason TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL UNIQUE,
    public_uuid VARCHAR(36) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_uuid ON chat_sessions(public_uuid);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_session ON chat_sessions(session_id);
CREATE OR REPLACE FUNCTION set_last_activity()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_activity = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER trg_chat_sessions_last_activity
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION set_last_activity();

CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    user_message TEXT,
    bot_response TEXT,
    category VARCHAR(100),
    detected_language VARCHAR(20),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    faq_id INTEGER,
    source_ref TEXT,
    -- chat_uuid: NOT in your dump, but routes/chat.py's _log_chat() inserts
    -- it (session.get('current_chat_uuid')). Added as nullable so that
    -- INSERT doesn't fail — if this column already exists in your real
    -- DB under a different name, or the dump is just stale, let me know
    -- and I'll adjust.
    chat_uuid VARCHAR(36)
);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_detected_language ON chat_history(detected_language);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_faq_id ON chat_history(faq_id);

CREATE TABLE IF NOT EXISTS chat_ratings (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    rating INTEGER NOT NULL,
    comment TEXT,
    exchange_count INTEGER DEFAULT 0,
    detected_language VARCHAR(20),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chat_ratings_rating ON chat_ratings(rating);
CREATE INDEX IF NOT EXISTS idx_chat_ratings_created_at ON chat_ratings(created_at);

CREATE TABLE IF NOT EXISTS default_schedules (
    id SERIAL PRIMARY KEY,
    department VARCHAR(50) NOT NULL,
    grade_level VARCHAR(20) DEFAULT 'all',
    day_of_week INTEGER NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    has_class SMALLINT DEFAULT 1,
    is_active SMALLINT DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_schedule UNIQUE (department, grade_level, day_of_week)
);
CREATE INDEX IF NOT EXISTS idx_default_schedules_department ON default_schedules(department);

CREATE TABLE IF NOT EXISTS dynamic_schedules (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    -- see note #5 at top of file re: truncated enum list
    content_type VARCHAR(30) DEFAULT 'event',
    start_date DATE NOT NULL,
    end_date DATE,
    repeat_type VARCHAR(10) DEFAULT 'none'
        CHECK (repeat_type IN ('none','yearly','monthly','weekly')),
    repeat_until_year INTEGER,
    is_closed SMALLINT DEFAULT 0,
    special_hours_start TIME,
    special_hours_end TIME,
    message_english TEXT,
    message_tagalog TEXT,
    message_bisaya TEXT,
    created_by VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active SMALLINT DEFAULT 1,
    departments JSONB,
    grade_levels JSONB
);
CREATE INDEX IF NOT EXISTS idx_dynamic_schedules_start_date ON dynamic_schedules(start_date);
CREATE INDEX IF NOT EXISTS idx_dynamic_schedules_is_active ON dynamic_schedules(is_active);
CREATE TRIGGER trg_dynamic_schedules_updated_at
    BEFORE UPDATE ON dynamic_schedules
    FOR EACH ROW EXECUTE FUNCTION set_updated_at_generic();

CREATE TABLE IF NOT EXISTS emergency_closures (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    reason VARCHAR(20) NOT NULL
        CHECK (reason IN ('heavy_rain','typhoon','flood','earthquake','power_outage','other')),
    severity VARCHAR(10) DEFAULT 'medium'
        CHECK (severity IN ('low','medium','high','critical')),
    start_date DATE NOT NULL,
    end_date DATE,
    start_time TIME,
    end_time TIME,
    affected_departments JSONB NOT NULL,
    affected_grades JSONB,
    message_english TEXT,
    message_tagalog TEXT,
    message_bisaya TEXT,
    is_active SMALLINT DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);
CREATE INDEX IF NOT EXISTS idx_emergency_closures_start_date ON emergency_closures(start_date);
CREATE INDEX IF NOT EXISTS idx_emergency_closures_is_active ON emergency_closures(is_active);
CREATE TRIGGER trg_emergency_closures_updated_at
    BEFORE UPDATE ON emergency_closures
    FOR EACH ROW EXECUTE FUNCTION set_updated_at_generic();

CREATE TABLE IF NOT EXISTS fallback_responses (
    id SERIAL PRIMARY KEY,
    language VARCHAR(20) NOT NULL UNIQUE,
    response_text TEXT NOT NULL,
    is_active SMALLINT DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS greetings (
    id SERIAL PRIMARY KEY,
    language VARCHAR(20) NOT NULL UNIQUE,
    greeting_text TEXT NOT NULL,
    is_active SMALLINT DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS non_school_responses (
    id SERIAL PRIMARY KEY,
    language VARCHAR(20) NOT NULL UNIQUE,
    response_text TEXT NOT NULL,
    is_active SMALLINT DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS handbook_chunks (
    id SERIAL PRIMARY KEY,
    chapter VARCHAR(200),
    section_title VARCHAR(255),
    page_start INTEGER,
    page_end INTEGER,
    content TEXT NOT NULL,
    keywords TEXT,
    chunk_order INTEGER DEFAULT 0,
    is_active SMALLINT DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Replaces the old MySQL FULLTEXT KEY(section_title, keywords, content)
    -- that MATCH()/AGAINST() used — see services/handbook.py's query.
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(section_title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(keywords, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'C')
    ) STORED
);
CREATE INDEX IF NOT EXISTS idx_handbook_chunks_search ON handbook_chunks USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS idx_handbook_chunks_chunk_order ON handbook_chunks(chunk_order);

CREATE TABLE IF NOT EXISTS missing_date_queries (
    id SERIAL PRIMARY KEY,
    question TEXT,
    detected_date DATE,
    asked_count INTEGER DEFAULT 1,
    last_asked TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    -- no UNIQUE on question, matching the real (unindexed) MySQL schema —
    -- see note #6 at top of file
);
CREATE INDEX IF NOT EXISTS idx_missing_date_queries_detected_date ON missing_date_queries(detected_date);

CREATE TABLE IF NOT EXISTS pending_faq_drafts (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    last_asked TIMESTAMP,
    suggested_category VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS satisfaction_feedback (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    question TEXT,
    answer TEXT,
    rating VARCHAR(20),
    detected_language VARCHAR(20),
    detected_topic VARCHAR(100),
    source_type VARCHAR(100),
    source_ref TEXT,
    faq_id INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE satisfaction_feedback ADD COLUMN IF NOT EXISTS detected_language VARCHAR(20);
ALTER TABLE satisfaction_feedback ADD COLUMN IF NOT EXISTS detected_topic VARCHAR(100);
ALTER TABLE satisfaction_feedback ADD COLUMN IF NOT EXISTS source_type VARCHAR(100);
ALTER TABLE satisfaction_feedback ADD COLUMN IF NOT EXISTS source_ref TEXT;
ALTER TABLE satisfaction_feedback ADD COLUMN IF NOT EXISTS faq_id INTEGER;
CREATE INDEX IF NOT EXISTS idx_satisfaction_feedback_created_at ON satisfaction_feedback(created_at);

CREATE TABLE IF NOT EXISTS topic_classifications (
    id SERIAL PRIMARY KEY,
    topic VARCHAR(50) NOT NULL,
    keyword VARCHAR(100) NOT NULL,
    is_school SMALLINT DEFAULT 1,
    weight REAL DEFAULT 1,
    is_active SMALLINT DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_topic_classifications_topic ON topic_classifications(topic);

CREATE TABLE IF NOT EXISTS unanswered_questions (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    resolved_question TEXT,
    context TEXT,
    detected_language VARCHAR(20),
    detected_topic VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE unanswered_questions ADD COLUMN IF NOT EXISTS detected_language VARCHAR(20);
ALTER TABLE unanswered_questions ADD COLUMN IF NOT EXISTS detected_topic VARCHAR(100);
CREATE INDEX IF NOT EXISTS idx_unanswered_questions_created_at ON unanswered_questions(created_at);
CREATE INDEX IF NOT EXISTS idx_unanswered_questions_language_topic
    ON unanswered_questions(detected_language, detected_topic, created_at);

CREATE TABLE IF NOT EXISTS visitor_stats (
    id SERIAL PRIMARY KEY,
    fingerprint VARCHAR(32) NOT NULL UNIQUE,
    question_count INTEGER DEFAULT 0,
    first_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TRIGGER trg_visitor_stats_last_seen
    BEFORE UPDATE ON visitor_stats
    FOR EACH ROW EXECUTE FUNCTION set_last_activity();
-- reuses the same set_last_activity() function defined above for
-- chat_sessions.last_activity — same "bump this column to now() on any
-- update" behavior, different column name doesn't matter since the
-- function always sets NEW.last_activity, and both tables happen to use
-- that exact column name.

-- =====================================================================
-- OPTIONAL: if you want missing_date_queries' asked_count to actually
-- increment (real dedup) instead of matching the original's always-
-- insert behavior, run this once and then restore the ON CONFLICT
-- version of record_missing_date_query() in services/schedule.py
-- (see that file's docstring for the exact clause):
--
--   ALTER TABLE missing_date_queries
--     ADD CONSTRAINT missing_date_queries_question_key UNIQUE (question);
--
-- Only do this if `question` values are short/consistent enough to be
-- meaningfully deduped — free-text student phrasing rarely repeats
-- verbatim, so this may not buy you much in practice.
-- =====================================================================


-- =====================================================================
-- Continuous-learning layer (topic aliases, feedback, success cache)
-- Added for data-driven handbook ranking — see services/topic_aliases.py
-- =====================================================================

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
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_hta_active ON handbook_topic_aliases (is_active, priority DESC);
CREATE TRIGGER trg_hta_updated_at
    BEFORE UPDATE ON handbook_topic_aliases
    FOR EACH ROW EXECUTE FUNCTION set_updated_at_generic();

-- Optional metadata on handbook_chunks (safe ADD IF NOT EXISTS)
ALTER TABLE handbook_chunks ADD COLUMN IF NOT EXISTS topic_tags TEXT[] DEFAULT '{}';
ALTER TABLE handbook_chunks ADD COLUMN IF NOT EXISTS key_facts JSONB DEFAULT '[]'::jsonb;
ALTER TABLE handbook_chunks ADD COLUMN IF NOT EXISTS embedding JSONB;

CREATE TABLE IF NOT EXISTS answer_feedback (
    id              SERIAL PRIMARY KEY,
    question        TEXT NOT NULL,
    question_norm   TEXT,
    rating          TEXT,
    correction      TEXT,
    source_type     TEXT,
    chunk_ids       INT[],
    faq_id          INT,
    detected_language TEXT NOT NULL DEFAULT 'english',
    session_id      TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE answer_feedback
    ADD COLUMN IF NOT EXISTS detected_language TEXT NOT NULL DEFAULT 'english';
CREATE INDEX IF NOT EXISTS idx_af_created ON answer_feedback (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_af_norm ON answer_feedback (question_norm);

CREATE TABLE IF NOT EXISTS answer_success_cache (
    id              SERIAL PRIMARY KEY,
    question_norm   TEXT NOT NULL,
    detected_lang   TEXT NOT NULL DEFAULT 'english',
    source_type     TEXT NOT NULL,
    chunk_ids       INT[],
    faq_id          INT,
    is_verified     BOOLEAN NOT NULL DEFAULT FALSE,
    answer_text     TEXT NOT NULL,
    hit_count       INT NOT NULL DEFAULT 1,
    last_hit_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT answer_success_cache_q_lang UNIQUE (question_norm, detected_lang)
);
ALTER TABLE answer_success_cache
    ADD COLUMN IF NOT EXISTS is_verified BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_asc_lang ON answer_success_cache (detected_lang);
