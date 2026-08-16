"""Admin alerts (email) + scheduled maintenance jobs. Email HTML template
kept minimal here — bring back the fully-styled version from the old
_build_email_html() if you want it; the logic that decides WHEN to alert
is what matters for the scheduler jobs below.

--- Postgres migration notes ---
- MySQL `DATE_SUB(NOW(), INTERVAL %s DAY)` -> Postgres
  `NOW() - (%s || ' days')::interval` (parameterized) or
  `NOW() - INTERVAL '90 days'` (literal, when the number is fixed in code).
- MySQL `ON DUPLICATE KEY UPDATE created_at = NOW()` -> Postgres
  `ON CONFLICT (faq_id) DO UPDATE SET created_at = NOW()`. This assumes
  faq_id has a UNIQUE constraint on faq_review_queue — if it doesn't yet,
  add one (`ALTER TABLE faq_review_queue ADD CONSTRAINT
  faq_review_queue_faq_id_key UNIQUE (faq_id);`) or swap the conflict
  target for whatever the real unique key/constraint name is.
"""
import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from database import get_db
from services.tz import ph_now  # BUG FIX: was comparing against server-UTC datetime.now() -- see services/tz.py

logger = logging.getLogger("pac_helpdesk.alerts")

import os
ADMIN_ALERT_EMAIL = os.environ.get('ADMIN_ALERT_EMAIL', '')
ADMIN_ALERT_SMTP_HOST = os.environ.get('ADMIN_ALERT_SMTP_HOST', 'smtp.gmail.com')
ADMIN_ALERT_SMTP_PORT = int(os.environ.get('ADMIN_ALERT_SMTP_PORT', 587))
ADMIN_ALERT_SMTP_USER = os.environ.get('ADMIN_ALERT_SMTP_USER', '')
ADMIN_ALERT_SMTP_PASS = os.environ.get('ADMIN_ALERT_SMTP_PASS', '')


def send_admin_alert(subject: str, message: str, severity: str = 'info') -> bool:
    if not subject or not message:
        return False
    logger.info(f"ADMIN ALERT [{severity.upper()}]: {subject} — {message}")

    success = False
    if ADMIN_ALERT_EMAIL and ADMIN_ALERT_SMTP_USER and ADMIN_ALERT_SMTP_PASS:
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = ADMIN_ALERT_SMTP_USER
            msg['To'] = ADMIN_ALERT_EMAIL
            msg['Subject'] = f"[PAC Helpdesk] {subject}"
            msg.attach(MIMEText(f"{subject}\n\n{message}", 'plain'))
            with smtplib.SMTP(ADMIN_ALERT_SMTP_HOST, ADMIN_ALERT_SMTP_PORT) as server:
                server.starttls()
                server.login(ADMIN_ALERT_SMTP_USER, ADMIN_ALERT_SMTP_PASS)
                server.send_message(msg)
            success = True
        except Exception as e:
            logger.error(f"Failed to send admin alert email: {e}")

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO alert_history (severity, subject, message, channels, created_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (severity, subject[:255], message[:2000], 'Email' if success else 'none'))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save alert to history: {e}")

    return success


def send_frequent_unanswered_alert(question: str, count: int) -> None:
    send_admin_alert(
        f"Frequent unanswered question ({count}x)",
        f'Question: "{question}"\n\nAsked {count} times without a matching FAQ answer.\n'
        f'Check /unanswered on the dashboard to review and add it.',
        severity='warning' if count >= 10 else 'info',
    )


def send_stale_faq_alert(faq_id: int, question: str, days_old: int) -> None:
    send_admin_alert(
        f"Stale FAQ needs review ({days_old} days old)",
        f'FAQ ID: {faq_id}\nQuestion: "{question}"\n\nNot updated in {days_old} days — review on /dashboard.',
        severity='warning',
    )


def batch_process_unanswered():
    """Alert on frequently-repeated unanswered questions and create FAQ drafts."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT question, COUNT(*) as cnt, MAX(created_at) as last_asked
            FROM unanswered_questions
            GROUP BY LOWER(TRIM(question)) HAVING COUNT(*) >= 3 ORDER BY cnt DESC
        """)
        for question, count, last_asked in cur.fetchall():
            if count >= 5:
                send_frequent_unanswered_alert(question, count)
            cur.execute("SELECT id FROM pending_faq_drafts WHERE question = %s", (question,))
            if cur.fetchone():
                continue
            cur.execute("""
                SELECT resolved_question FROM unanswered_questions
                WHERE LOWER(TRIM(question)) = LOWER(TRIM(%s)) AND resolved_question IS NOT NULL
                ORDER BY created_at DESC LIMIT 1
            """, (question,))
            row = cur.fetchone()
            cur.execute("""
                INSERT INTO pending_faq_drafts (question, resolved_question, frequency, last_asked, suggested_category)
                VALUES (%s, %s, %s, %s, 'General')
            """, (question, row[0] if row else None, count, last_asked))
            conn.commit()
    except Exception as e:
        logger.error(f"Batch process error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def check_stale_faqs():
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, question, last_updated FROM faq
            WHERE last_updated < NOW() - INTERVAL '90 days' AND reviewed_flag = 0
        """)
        for faq_id, question, last_updated in cur.fetchall():
            days_old = (ph_now() - last_updated).days if last_updated else 90
            if days_old > 180:
                send_stale_faq_alert(faq_id, question, days_old)
            cur.execute("""
                INSERT INTO faq_review_queue (faq_id, reason, created_at) VALUES (%s, %s, NOW())
                ON CONFLICT (faq_id) DO UPDATE SET created_at = NOW()
            """, (faq_id, f'Not updated since {last_updated.date()} ({days_old} days)'))
            conn.commit()
    except Exception as e:
        logger.error(f"Stale FAQ check error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def promote_high_frequency_faqs():
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE faq f
            SET priority_score = LEAST(f.priority_score + 5, 100)
            FROM (
                SELECT faq_id, COUNT(*) as match_count FROM chat_history
                WHERE faq_id IS NOT NULL AND created_at > NOW() - INTERVAL '30 days'
                GROUP BY faq_id HAVING COUNT(*) > 10
            ) AS matched
            WHERE matched.faq_id = f.id AND f.priority_score < 50
        """)
        conn.commit()
        if cur.rowcount > 0:
            logger.info(f"Auto-promoted {cur.rowcount} FAQs")
    except Exception as e:
        logger.error(f"Promote FAQs error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def gather_digest_stats(days: int = 7) -> dict:
    conn = get_db()
    cur = conn.cursor()
    stats = {}
    try:
        cur.execute(
            "SELECT COUNT(*) FROM chat_history WHERE created_at >= NOW() - (%s || ' days')::interval",
            (days,))
        stats['total_chats'] = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM unanswered_questions WHERE created_at >= NOW() - (%s || ' days')::interval",
            (days,))
        stats['unanswered_total'] = cur.fetchone()[0]
        cur.execute(
            "SELECT ROUND(AVG(rating)::numeric, 2), COUNT(*) FROM chat_ratings "
            "WHERE created_at >= NOW() - (%s || ' days')::interval",
            (days,))
        row = cur.fetchone()
        stats['avg_rating'] = float(row[0] or 0)
        stats['rating_count'] = row[1] or 0
        cur.execute("SELECT COUNT(*) FROM faq WHERE last_updated < NOW() - INTERVAL '90 days' AND reviewed_flag = 0")
        stats['stale_faqs'] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM faq")
        stats['total_faqs'] = cur.fetchone()[0]
    except Exception as e:
        logger.error(f"gather_digest_stats error: {e}")
    finally:
        cur.close()
        conn.close()
    return stats


def send_frequent_missing_date_alert(question: str, count: int) -> None:
    send_admin_alert(
        f"Frequently-asked date has no schedule data ({count}x)",
        f'Question: "{question}"\n\nAsked {count} times, but no matching holiday/'
        f'schedule entry exists for the date(s) involved.\n'
        f'Check /holidays on the dashboard to add the missing entry.',
        severity='warning' if count >= 10 else 'info',
    )


def check_frequent_missing_dates(min_count: int = 3):
    """Parallel to batch_process_unanswered() above, but for dates instead
    of FAQs — missing_date_queries (services/schedule.py's
    record_missing_date_query) was accumulating with nothing acting on
    it. Same "asked repeatedly with no data to answer it" pattern as
    unanswered FAQ questions, so it gets the same treatment: alert admins
    once it's been asked often enough to be worth fixing.

    NOTE: as documented in services/schedule.py and database/full_schema.sql,
    missing_date_queries.question has no unique constraint in the real
    schema, so record_missing_date_query() always inserts a new row
    rather than incrementing a count on a matching one — this job groups
    by question at query time instead of relying on a pre-aggregated
    asked_count column, so it still works correctly despite that.
    """
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT question, COUNT(*) as cnt, MAX(last_asked) as last_seen
            FROM missing_date_queries
            WHERE question IS NOT NULL AND question != ''
            GROUP BY LOWER(TRIM(question))
            HAVING COUNT(*) >= %s
            ORDER BY cnt DESC
        """, (min_count,))
        for question, count, _last_seen in cur.fetchall():
            send_frequent_missing_date_alert(question, count)
    except Exception as e:
        logger.error(f"check_frequent_missing_dates error: {e}")
    finally:
        cur.close()
        conn.close()


def send_digest(period_label: str, days: int = 7) -> bool:
    """Emails (or, if SMTP isn't configured, just logs to alert_history)
    a short usage summary — chats, unanswered count, avg rating, stale
    FAQs. Triggered manually from the alert log page, or wire it into a
    scheduled job in app.py if you want it automatic."""
    stats = gather_digest_stats(days)
    subject = f"{period_label} Report"
    message = (
        f"{stats.get('total_chats', 0)} chats, "
        f"{stats.get('unanswered_total', 0)} unanswered, "
        f"avg rating {stats.get('avg_rating', 0)}/5, "
        f"{stats.get('stale_faqs', 0)} stale FAQs need review."
    )
    return send_admin_alert(subject, message, severity='info')