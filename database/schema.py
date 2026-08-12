"""One-time schema setup + lightweight migrations, run at startup.

--- Update ---
database/full_schema.sql now exists and is the authoritative schema —
built directly from the real MySQL DESCRIBE output rather than inferred
from usage. Run it once against a fresh DB:
    psql "postgresql://..." -f database/full_schema.sql
Everything below is now a defensive fallback for partial/older
databases, not the primary source of truth. Aligned to match
full_schema.sql's types (SMALLINT for the old tinyint(1) flag columns,
not BOOLEAN — see full_schema.sql's header comment for why).

--- Postgres migration notes ---
- MySQL AUTO_INCREMENT PRIMARY KEY -> Postgres SERIAL PRIMARY KEY.
- MySQL ENUM(...) column type -> Postgres has no inline enum type in the
  same spot; using VARCHAR + CHECK constraint instead (functionally
  equivalent, and avoids the extra step of CREATE TYPE ... AS ENUM).
- MySQL `INDEX name(cols)` / `UNIQUE KEY name(cols)` declared INSIDE
  CREATE TABLE is not valid Postgres syntax — indexes are now separate
  CREATE INDEX statements after the table, and unique constraints use
  `CONSTRAINT name UNIQUE (...)`.
- MySQL `SHOW COLUMNS FROM t LIKE 'x'` / `SHOW TABLES LIKE 'x'` -> queried
  from information_schema / to_regclass() instead.
- MySQL `... TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE
  CURRENT_TIMESTAMP` has no direct Postgres equivalent — Postgres commits
  to `DEFAULT CURRENT_TIMESTAMP` for INSERT only; the "ON UPDATE" half is
  reimplemented as a BEFORE UPDATE trigger calling set_updated_at().
- MySQL `INSERT IGNORE` -> Postgres `INSERT ... ON CONFLICT DO NOTHING`.
- MySQL tinyint(1) flag columns (is_active, has_class, etc.) -> SMALLINT,
  not BOOLEAN. See full_schema.sql's header for why.
"""
import logging
import os

from werkzeug.security import generate_password_hash

from database import get_db

logger = logging.getLogger("pac_helpdesk.database.schema")


def ensure_core_tables():
    """Create every table the app needs if it doesn't exist yet.

    Free-tier fallback when full_schema.sql was never run against the
    Render Postgres instance. Mirrors database/full_schema.sql.
    Safe to call every boot — CREATE TABLE IF NOT EXISTS is a no-op
    once tables exist.
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE OR REPLACE FUNCTION set_updated_at_generic()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        cur.execute("""
            CREATE OR REPLACE FUNCTION set_last_activity()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.last_activity = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        cur.execute("""
            CREATE OR REPLACE FUNCTION set_faq_last_updated()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.last_updated = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        cur.execute("""
            CREATE OR REPLACE FUNCTION set_visitor_last_seen()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.last_seen = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
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
                embedding JSONB
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faq_topic ON faq(topic)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faq_priority_score ON faq(priority_score)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faq_source ON faq(source)")
        cur.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_faq_last_updated'
                ) THEN
                    CREATE TRIGGER trg_faq_last_updated
                        BEFORE UPDATE ON faq
                        FOR EACH ROW EXECUTE FUNCTION set_faq_last_updated();
                END IF;
            END $$;
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS unanswered_questions (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                resolved_question TEXT,
                context TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_unanswered_questions_created_at "
            "ON unanswered_questions(created_at)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL UNIQUE,
                public_uuid VARCHAR(36) NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ip_address VARCHAR(45),
                user_agent TEXT
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_uuid ON chat_sessions(public_uuid)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_session ON chat_sessions(session_id)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100),
                user_message TEXT,
                bot_response TEXT,
                category VARCHAR(100),
                detected_language VARCHAR(20),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                faq_id INTEGER,
                chat_uuid VARCHAR(36)
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_ratings (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100),
                rating INTEGER NOT NULL,
                comment TEXT,
                exchange_count INTEGER DEFAULT 0,
                detected_language VARCHAR(20),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_ratings_created_at ON chat_ratings(created_at)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS satisfaction_feedback (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100),
                question TEXT,
                answer TEXT,
                rating VARCHAR(20),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_satisfaction_feedback_created_at "
            "ON satisfaction_feedback(created_at)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS visitor_stats (
                id SERIAL PRIMARY KEY,
                fingerprint VARCHAR(32) NOT NULL UNIQUE,
                question_count INTEGER DEFAULT 0,
                first_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_visitor_stats_last_seen'
                ) THEN
                    CREATE TRIGGER trg_visitor_stats_last_seen
                        BEFORE UPDATE ON visitor_stats
                        FOR EACH ROW EXECUTE FUNCTION set_visitor_last_seen();
                END IF;
            END $$;
        """)

        cur.execute("""
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
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_handbook_chunks_chunk_order "
            "ON handbook_chunks(chunk_order)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id SERIAL PRIMARY KEY,
                severity VARCHAR(20) NOT NULL,
                subject VARCHAR(255) NOT NULL,
                message TEXT,
                channels VARCHAR(100),
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_alert_history_created_at ON alert_history(created_at)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS bad_feedback (
                id SERIAL PRIMARY KEY,
                question TEXT,
                answer TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS faq_review_queue (
                id SERIAL PRIMARY KEY,
                faq_id INTEGER UNIQUE,
                reason TEXT,
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS pending_faq_drafts (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_asked TIMESTAMP,
                suggested_category VARCHAR(100),
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS missing_date_queries (
                id SERIAL PRIMARY KEY,
                question TEXT,
                detected_date DATE,
                asked_count INTEGER DEFAULT 1,
                last_asked TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS abbreviations (
                id SERIAL PRIMARY KEY,
                abbreviation VARCHAR(50) NOT NULL UNIQUE,
                expansion VARCHAR(255) NOT NULL,
                language VARCHAR(20) DEFAULT 'all',
                is_active SMALLINT DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS greetings (
                id SERIAL PRIMARY KEY,
                language VARCHAR(20) NOT NULL UNIQUE,
                greeting_text TEXT NOT NULL,
                is_active SMALLINT DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS fallback_responses (
                id SERIAL PRIMARY KEY,
                language VARCHAR(20) NOT NULL UNIQUE,
                response_text TEXT NOT NULL,
                is_active SMALLINT DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS non_school_responses (
                id SERIAL PRIMARY KEY,
                language VARCHAR(20) NOT NULL UNIQUE,
                response_text TEXT NOT NULL,
                is_active SMALLINT DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_feedback (
                id SERIAL PRIMARY KEY,
                ai_response_id INTEGER,
                user_feedback SMALLINT,
                user_correction TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_generated_responses (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                intent VARCHAR(50),
                confidence REAL DEFAULT 0,
                was_helpful SMALLINT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS topic_classifications (
                id SERIAL PRIMARY KEY,
                topic VARCHAR(50) NOT NULL,
                keyword VARCHAR(100) NOT NULL,
                is_school SMALLINT DEFAULT 1,
                weight REAL DEFAULT 1,
                is_active SMALLINT DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_schedules (
                id SERIAL PRIMARY KEY,
                title VARCHAR(200) NOT NULL,
                description TEXT,
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
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dynamic_schedules_start_date "
            "ON dynamic_schedules(start_date)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_dynamic_schedules_is_active "
            "ON dynamic_schedules(is_active)"
        )
        cur.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_dynamic_schedules_updated_at'
                ) THEN
                    CREATE TRIGGER trg_dynamic_schedules_updated_at
                        BEFORE UPDATE ON dynamic_schedules
                        FOR EACH ROW EXECUTE FUNCTION set_updated_at_generic();
                END IF;
            END $$;
        """)

        cur.execute("""
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
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_emergency_closures_start_date "
            "ON emergency_closures(start_date)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_emergency_closures_is_active "
            "ON emergency_closures(is_active)"
        )

        cur.execute("""
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
            )
        """)

        conn.commit()
        logger.info("Core tables ensured (full set)")
    except Exception as e:
        err = str(e).lower()
        if "tuple concurrently updated" in err or "deadlock detected" in err:
            logger.warning(f"ensure_core_tables: concurrent DDL, skipping: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        else:
            logger.error(f"ensure_core_tables error: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
    return cur.fetchone()[0] is not None


def _column_exists(cur, table_name: str, column_name: str) -> bool:
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = %s AND column_name = %s
    """, (table_name, column_name))
    return cur.fetchone() is not None


def ensure_updated_at_trigger_fn():
    """Creates (once) the trigger function that stands in for MySQL's
    `ON UPDATE CURRENT_TIMESTAMP`.
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE OR REPLACE FUNCTION set_updated_at()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        conn.commit()
    except Exception as e:
        err = str(e).lower()
        if "tuple concurrently updated" in err or "deadlock detected" in err:
            logger.warning(f"ensure_updated_at_trigger_fn: concurrent DDL, skipping: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        else:
            logger.error(f"ensure_updated_at_trigger_fn error: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def ensure_admin_exists():
    conn = get_db()
    cur = conn.cursor()
    try:
        admin_password = os.environ.get('ADMIN_PASSWORD', 'admin123')
        cur.execute("SELECT id FROM admin_users WHERE username=%s", ("admin212004",))
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO admin_users (username, password) VALUES (%s, %s)",
                ("admin212004", generate_password_hash(admin_password)),
            )
            conn.commit()
            logger.info("Admin user created")
    finally:
        cur.close()
        conn.close()


def ensure_schedule_schema():
    """Idempotent schedule-table migrations + seed data.

    Tables themselves are created in ensure_core_tables(). This only adds
    missing columns/triggers and seeds default_schedules when empty.
    """
    ensure_updated_at_trigger_fn()
    conn = get_db()
    cur = conn.cursor()
    try:
        if _table_exists(cur, 'dynamic_schedules'):
            if not _column_exists(cur, 'dynamic_schedules', 'grade_levels'):
                cur.execute(
                    "ALTER TABLE dynamic_schedules ADD COLUMN grade_levels JSONB NULL"
                )
                conn.commit()
                logger.info("dynamic_schedules.grade_levels column added")
            if not _column_exists(cur, 'dynamic_schedules', 'departments'):
                cur.execute(
                    "ALTER TABLE dynamic_schedules ADD COLUMN departments JSONB NULL"
                )
                conn.commit()
                logger.info("dynamic_schedules.departments column added")

        if _table_exists(cur, 'emergency_closures'):
            cur.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger
                        WHERE tgname = 'trg_emergency_closures_updated_at'
                    ) THEN
                        CREATE TRIGGER trg_emergency_closures_updated_at
                            BEFORE UPDATE ON emergency_closures
                            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
                    END IF;
                END $$;
            """)
            conn.commit()

        if _table_exists(cur, 'default_schedules'):
            cur.execute("SELECT COUNT(*) FROM default_schedules")
            if cur.fetchone()[0] == 0:
                _seed_default_schedules(cur, conn)
    except Exception as e:
        err = str(e).lower()
        if "tuple concurrently updated" in err or "deadlock detected" in err:
            logger.warning(f"ensure_schedule_schema: concurrent DDL, skipping: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        else:
            logger.error(f"ensure_schedule_schema error: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def _seed_default_schedules(cur, conn):
    # day_of_week here MUST match Python's date.weekday(): Monday=0,
    # Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6.
    # (The previous version of this seed data used a 0=Sunday convention,
    # which silently shifted every lookup by one day -- e.g. Monday was
    # reading the row meant for Sunday, which is why "no class Monday"
    # could show up with zero holiday data involved.)
    # has_class is SMALLINT (0/1), not BOOLEAN — Postgres rejects True/False here.
    rows = [
        ('college', 'all', 0, '08:00', '17:00', 1),   # Monday
        ('college', 'all', 1, '08:00', '17:00', 1),   # Tuesday
        ('college', 'all', 2, '08:00', '17:00', 1),   # Wednesday
        ('college', 'all', 3, '08:00', '17:00', 1),   # Thursday
        ('college', 'all', 4, '08:00', '17:00', 1),   # Friday
        ('college', 'all', 5, '00:00', '00:00', 0),  # Saturday
        ('college', 'all', 6, '00:00', '00:00', 0),  # Sunday
        ('highschool', '7-10', 0, '07:30', '16:30', 1),
        ('highschool', '7-10', 1, '07:30', '16:30', 1),
        ('highschool', '7-10', 2, '07:30', '16:30', 1),
        ('highschool', '7-10', 3, '07:30', '16:30', 1),
        ('highschool', '7-10', 4, '07:30', '16:30', 1),
        ('highschool', '7-10', 5, '00:00', '00:00', 0),
        ('highschool', '7-10', 6, '00:00', '00:00', 0),
        ('highschool', '11-12', 0, '08:00', '17:00', 1),
        ('highschool', '11-12', 1, '08:00', '17:00', 1),
        ('highschool', '11-12', 2, '08:00', '17:00', 1),
        ('highschool', '11-12', 3, '08:00', '17:00', 1),
        ('highschool', '11-12', 4, '08:00', '17:00', 1),
        ('highschool', '11-12', 5, '00:00', '00:00', 0),
        ('highschool', '11-12', 6, '00:00', '00:00', 0),
        ('elementary', 'k-6', 0, '07:30', '16:00', 1),
        ('elementary', 'k-6', 1, '07:30', '16:00', 1),
        ('elementary', 'k-6', 2, '07:30', '16:00', 1),
        ('elementary', 'k-6', 3, '07:30', '16:00', 1),
        ('elementary', 'k-6', 4, '07:30', '16:00', 1),
        ('elementary', 'k-6', 5, '00:00', '00:00', 0),
        ('elementary', 'k-6', 6, '00:00', '00:00', 0),
        ('registrar', 'all', 0, '08:00', '17:00', 1),
        ('registrar', 'all', 1, '08:00', '17:00', 1),
        ('registrar', 'all', 2, '08:00', '17:00', 1),
        ('registrar', 'all', 3, '08:00', '17:00', 1),
        ('registrar', 'all', 4, '08:00', '17:00', 1),
        ('registrar', 'all', 5, '00:00', '00:00', 0),
        ('registrar', 'all', 6, '00:00', '00:00', 0),
        ('business', 'all', 0, '08:00', '17:00', 1),
        ('business', 'all', 1, '08:00', '17:00', 1),
        ('business', 'all', 2, '08:00', '17:00', 1),
        ('business', 'all', 3, '08:00', '17:00', 1),
        ('business', 'all', 4, '08:00', '17:00', 1),
        ('business', 'all', 5, '00:00', '00:00', 0),
        ('business', 'all', 6, '00:00', '00:00', 0),
    ]
    for dept, grade, dow, start_t, end_t, has_class in rows:
        cur.execute("""
            INSERT INTO default_schedules
            (department, grade_level, day_of_week, start_time, end_time, has_class)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT ON CONSTRAINT unique_schedule DO NOTHING
        """, (dept, grade, dow, start_t, end_t, has_class))
    conn.commit()
    logger.info("Default schedules seeded (Monday=0..Sunday=6 convention)")


def migrate_create_chat_sessions():
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL UNIQUE,
                public_uuid VARCHAR(36) NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address VARCHAR(45), user_agent TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_uuid ON chat_sessions(public_uuid)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_session ON chat_sessions(session_id)")
        ensure_updated_at_trigger_fn()
        cur.execute("""
            DROP TRIGGER IF EXISTS trg_chat_sessions_last_activity ON chat_sessions;
        """)
        cur.execute("""
            CREATE OR REPLACE FUNCTION set_last_activity()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.last_activity = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        cur.execute("""
            CREATE TRIGGER trg_chat_sessions_last_activity
            BEFORE UPDATE ON chat_sessions
            FOR EACH ROW EXECUTE FUNCTION set_last_activity();
        """)
        conn.commit()
    except Exception as e:
        err = str(e).lower()
        if "tuple concurrently updated" in err or "deadlock detected" in err:
            logger.warning(f"migrate_create_chat_sessions: concurrent DDL, skipping: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        else:
            logger.error(f"migrate_create_chat_sessions error: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def migrate_add_uuid_to_faq():
    import uuid
    conn = get_db()
    cur = conn.cursor()
    try:
        if not _column_exists(cur, 'faq', 'public_uuid'):
            cur.execute("ALTER TABLE faq ADD COLUMN public_uuid VARCHAR(36) UNIQUE")
            conn.commit()
            cur.execute("SELECT id FROM faq WHERE public_uuid IS NULL")
            rows = cur.fetchall()
            for (faq_id,) in rows:
                cur.execute("UPDATE faq SET public_uuid = %s WHERE id = %s", (str(uuid.uuid4()), faq_id))
            conn.commit()
            logger.info(f"Populated UUIDs for {len(rows)} existing FAQs")
    except Exception as e:
        err = str(e).lower()
        if "tuple concurrently updated" in err or "deadlock detected" in err:
            logger.warning(f"migrate_add_uuid_to_faq: concurrent DDL, skipping: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        else:
            logger.error(f"migrate_add_uuid_to_faq error: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def ensure_faq_embedding_column():
    """Adds faq.embedding (JSONB) if it's not already there."""
    conn = get_db()
    cur = conn.cursor()
    try:
        if not _column_exists(cur, 'faq', 'embedding'):
            cur.execute("ALTER TABLE faq ADD COLUMN embedding JSONB")
            conn.commit()
            logger.info("faq.embedding column added")
    except Exception as e:
        err = str(e).lower()
        if "tuple concurrently updated" in err or "deadlock detected" in err:
            logger.warning(f"ensure_faq_embedding_column: concurrent DDL, skipping: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        else:
            logger.error(f"ensure_faq_embedding_column error: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def ensure_handbook_fulltext_search():
    """Postgres full-text search column + GIN index."""
    conn = get_db()
    cur = conn.cursor()
    try:
        if not _table_exists(cur, 'handbook_chunks'):
            logger.warning(
                "handbook_chunks table doesn't exist yet — skipping "
                "full-text search column setup."
            )
            return
        cur.execute("""
            ALTER TABLE handbook_chunks
            ADD COLUMN IF NOT EXISTS search_vector tsvector
            GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(section_title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(keywords, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(content, '')), 'C')
            ) STORED
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_handbook_chunks_search "
            "ON handbook_chunks USING GIN (search_vector)"
        )
        conn.commit()
        logger.info("handbook_chunks full-text search column/index ensured")
    except Exception as e:
        err = str(e).lower()
        if "tuple concurrently updated" in err or "deadlock detected" in err:
            logger.warning(f"ensure_handbook_fulltext_search: concurrent DDL, skipping: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        else:
            logger.error(f"ensure_handbook_fulltext_search error: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def ensure_learning_layer():
    """Topic aliases + answer memory tables. Delegates to services.topic_aliases
    so seed data and runtime schema stay in one place. Safe every boot.
    """
    try:
        from database.learning_schema import ensure_learning_schema
        ensure_learning_schema(get_db)
    except ImportError:
        try:
            from services.topic_aliases import ensure_topic_alias_schema, seed_default_aliases
            ensure_topic_alias_schema(get_db)
            seed_default_aliases(get_db, force=False)
        except Exception as e:
            logger.error(f"ensure_learning_layer: {e}")
