import logging
import time
from datetime import datetime

from flask import Blueprint, jsonify, request

from ai_providers import ai_cache, circuit_breaker, faq_cache, model_metrics
from database import get_db, pool_stats
from extensions import csrf
from services.alerts import send_digest
from utils.security import admin_required

logger = logging.getLogger("pac_helpdesk.routes.api")
bp = Blueprint('api', __name__)

@bp.route("/ping")
def ping():
    return "pong", 200

@bp.route('/health')
def health_check():
    health = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {}
    }
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        health['components']['database'] = 'healthy'
    except Exception as e:
        health['components']['database'] = f'unhealthy: {e}'
        health['status'] = 'degraded'
    health['components']['ai'] = {
        'ready': not circuit_breaker.is_open(),
        'circuit': circuit_breaker.get_state()
    }
    return jsonify(health)


@bp.route('/api/model_status')
def api_model_status():
    return jsonify({
        'ready': not circuit_breaker.is_open(),
        'circuit_breaker': circuit_breaker.get_state(),
        'metrics': model_metrics.get_stats(),
        'cache_size': len(ai_cache.cache),
    })


@bp.route('/api/admin/retrieval_status')
def api_retrieval_status():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403

    from services.embeddings import (
        EMBEDDING_MODEL,
        get_embedding_coverage,
        get_embedding_status,
    )
    from services.ai_retrieval import get_query_expansion_metrics
    try:
        embedding_status = get_embedding_status()
        corpora = get_embedding_coverage(get_db)
    except Exception:
        logger.exception("Failed to load retrieval readiness status")
        return jsonify({'error': 'Retrieval status unavailable'}), 503

    return jsonify({
        'embedding_model': EMBEDDING_MODEL,
        'embedding_provider': embedding_status,
        'corpora': corpora,
        'provider_calls': model_metrics.get_provider_stats(),
        'query_expansion': get_query_expansion_metrics(),
        'metrics_scope': 'process-local; resets when the worker restarts',
    })


@bp.route('/api/ai/status')
def ai_status():
    from ai_providers import llm_chat
    try:
        test_response = llm_chat("Hello", max_tokens=5)
        status = "healthy" if test_response else "degraded"
    except Exception as e:
        status = f"unhealthy: {e}"
    return jsonify({
        'status': status,
        'circuit_breaker': circuit_breaker.get_state(),
        'metrics': model_metrics.get_stats()
    })


@bp.route('/metrics')
def metrics():
    stats = model_metrics.get_stats()
    return jsonify({
        'model_ready': 1 if not circuit_breaker.is_open() else 0,
        'ai_success_rate': stats['success_rate'],
        'cache_hit_rate': stats['cache_hit_rate'],
    })


@bp.route('/api/db_pool_stats')
def db_pool_stats():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify(pool_stats())


@bp.route('/api/model/reset_circuit', methods=['POST'])
@csrf.exempt
def reset_circuit():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    circuit_breaker.failures = 0
    circuit_breaker.half_open = False
    return jsonify({'success': True})


@bp.route('/api/model/clear_cache', methods=['POST'])
@csrf.exempt
def clear_ai_cache():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    ai_cache.invalidate()
    faq_cache.invalidate()
    return jsonify({'success': True})


@bp.route('/api/check_admin')
def check_admin():
    return jsonify({'is_admin': admin_required()})



@bp.route('/api/analytics/chart_data')
def analytics_chart_data():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM chat_history
            WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """)
        chat_dates = cur.fetchall()

        total_questions = 0
        total_visitors = 0
        cur.execute("SELECT SUM(question_count), COUNT(*) FROM visitor_stats")
        vrow = cur.fetchone()
        total_questions = int(vrow[0] or 0)
        total_visitors = int(vrow[1] or 0)
        if total_questions == 0 and total_visitors == 0:
            cur.execute("SELECT COUNT(*), COUNT(DISTINCT session_id) FROM chat_history")
            fallback_row = cur.fetchone()
            total_questions = int(fallback_row[0] or 0)
            total_visitors = int(fallback_row[1] or 0)

        cur.execute("SELECT COUNT(*) FROM unanswered_questions")
        unanswered = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM bad_feedback")
        bad_feedback = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM faq")
        total_faqs = cur.fetchone()[0]


        cur.execute("""
            SELECT COALESCE(NULLIF(TRIM(category), ''), NULLIF(TRIM(topic), ''), 'General') AS cat,
                   COUNT(*) as count
            FROM faq
            GROUP BY cat
            ORDER BY count DESC
            LIMIT 25
        """)
        categories = cur.fetchall()

        cur.execute("SELECT rating, COUNT(*) FROM satisfaction_feedback GROUP BY rating")
        satisfaction = {r[0]: r[1] for r in cur.fetchall()}

        from services.popular_questions import get_popular_knowledge_questions
        popular = get_popular_knowledge_questions(
            get_db,
            limit=20,
            days=None,
            connection=conn,
        )

        total_q = total_questions
        return jsonify({
            'activity': [{'date': str(r[0]), 'count': r[1]} for r in chat_dates],
            'popular': [{'question': r['question'], 'count': r['count']} for r in popular],
            'categories': [{'name': r[0], 'count': r[1]} for r in categories],
            'stats': {
                'total_queries': total_q,
                'total_visitors': total_visitors,
                'unanswered': unanswered,
                'faqs': total_faqs,
                'bad_feedback': bad_feedback,
                'answered': max(0, total_q - unanswered),
                'thumbs_up': satisfaction.get('thumbs_up', 0),
                'thumbs_down': satisfaction.get('thumbs_down', 0),
            },
        })
    finally:
        cur.close()
        conn.close()


@bp.route('/api/response_times')
def api_response_times():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403

    from ai_providers import model_metrics
    timing = model_metrics.get_answer_timing_stats()
    total_tracked = timing['cache_hit_count'] + timing['computed_count']

    return jsonify({
        'average_response_time_cache_ms': timing['avg_cache_ms'],
        'average_response_time_db_ms': timing['avg_db_ms'],
        'cache_hit_count': timing['cache_hit_count'],
        'computed_count': timing['computed_count'],
        'total_queries_tracked': total_tracked,
        'note': (
            'Classified by end-to-end response time (>300ms = a real AI '
            'provider call happened) since the fast path can be served by '
            'faq_cache, answer_success_cache, or admin corrections -- not '
            'just the in-memory AI cache.'
        ),
    })



@bp.route('/api/chat_ratings_data')
def chat_ratings_data():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(*) as total,
                   ROUND(AVG(rating), 2) as avg_rating,
                   SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) as five_star,
                   SUM(CASE WHEN rating = 4 THEN 1 ELSE 0 END) as four_star,
                   SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) as three_star,
                   SUM(CASE WHEN rating = 2 THEN 1 ELSE 0 END) as two_star,
                   SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as one_star
            FROM chat_ratings
        """)
        stats = cur.fetchone()

        cur.execute("""
            SELECT DATE(created_at) as day,
                   ROUND(AVG(rating), 2) as avg_r,
                   COUNT(*) as count
            FROM chat_ratings
            WHERE created_at >= CURRENT_DATE - INTERVAL '14 days'
            GROUP BY DATE(created_at)
            ORDER BY day ASC
        """)
        trend = cur.fetchall()

        cur.execute("""
            SELECT id, rating, comment, detected_language, created_at
            FROM chat_ratings
            WHERE comment IS NOT NULL AND comment != ''
            ORDER BY created_at DESC
            LIMIT 50
        """)
        comments = cur.fetchall()

        return jsonify({
            'stats': {
                'total': stats[0] or 0,
                'avg_rating': float(stats[1] or 0),
                'breakdown': {
                    '5': stats[2] or 0,
                    '4': stats[3] or 0,
                    '3': stats[4] or 0,
                    '2': stats[5] or 0,
                    '1': stats[6] or 0,
                },
            },
            'trend': [
                {'day': str(r[0]), 'avg': float(r[1]), 'count': r[2]}
                for r in trend
            ],
            'comments': [
                {
                    'id': r[0],
                    'rating': r[1],
                    'comment': r[2],
                    'language': r[3],
                    'created_at': r[4].strftime('%b %d, %Y %H:%M') if r[4] else ''
                }
                for r in comments
            ],
        })
    finally:
        cur.close()
        conn.close()


_badge_counts_cache = {'data': None, 'ts': 0.0}
_BADGE_COUNTS_TTL = 8


def _compute_admin_badge_counts() -> dict:
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM unanswered_questions")
        unanswered = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM satisfaction_feedback
            WHERE rating = 'thumbs_down'
              AND created_at > NOW() - INTERVAL '7 days'
        """)
        sat_reports = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM bad_feedback WHERE created_at > NOW() - INTERVAL '7 days'"
        )
        bad_reports = cur.fetchone()[0]
        reports = sat_reports + bad_reports

        cur.execute("SELECT COUNT(*) FROM faq_review_queue WHERE status = 'pending'")
        review = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM pending_faq_drafts WHERE status = 'pending'")
        drafts = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM alert_history WHERE created_at > NOW() - INTERVAL '24 hours'"
        )
        alerts_new = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM chat_ratings WHERE rating <= 2 AND created_at > NOW() - INTERVAL '7 days'"
        )
        low_ratings = cur.fetchone()[0]

        return {
            "unanswered": unanswered,
            "reports": reports,
            "review_queue": review,
            "pending_drafts": drafts,
            "alerts_new": alerts_new,
            "low_ratings": low_ratings,
            "total": unanswered + reports + review + drafts,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def _get_admin_badge_counts() -> dict:
    now = time.time()
    cached = _badge_counts_cache['data']
    if cached and now - _badge_counts_cache['ts'] < _BADGE_COUNTS_TTL:
        return cached
    try:
        data = _compute_admin_badge_counts()
        _badge_counts_cache.update(data=data, ts=now)
        return data
    except Exception as e:
        logger.error(f"admin badge counts error: {e}")
        if cached:
            # Serve the last good snapshot rather than zeroing every
            # badge out over one transient query failure.
            return cached
        return {
            'unanswered': 0, 'reports': 0, 'review_queue': 0, 'pending_drafts': 0,
            'alerts_new': 0, 'low_ratings': 0, 'total': 0, 'degraded': True,
        }


@bp.route('/api/admin/alert_counts')
def api_alert_counts():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    data = dict(_get_admin_badge_counts())
    data['health_issues'] = 0
    return jsonify(data)


@bp.route('/api/admin/alerts', methods=['GET'])
def api_get_alerts():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, severity, subject, message, channels, created_at
            FROM alert_history
            ORDER BY created_at DESC
            LIMIT 200
        """)
        rows = cur.fetchall()
        return jsonify({
            'alerts': [{
                'time': r[5].strftime('%Y-%m-%d %H:%M:%S') if r[5] else '',
                'severity': r[1],
                'subject': r[2],
                'message': r[3],
                'channels': (r[4] or '').split(',') if r[4] else [],
            } for r in rows]
        })
    finally:
        cur.close()
        conn.close()


@bp.route('/api/admin/alerts', methods=['DELETE'])
@csrf.exempt
def api_clear_alerts():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM alert_history")
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'success': True})


@bp.route('/api/admin/send_digest', methods=['POST'])
@csrf.exempt
def api_send_digest():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    period = request.args.get('period', 'weekly')
    ok = send_digest("Daily", days=1) if period == 'daily' else send_digest("Weekly", days=7)
    return jsonify({'sent': ok})


@bp.route('/api/public_schedules')
def api_public_schedules():
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, title, content_type, start_date, end_date, is_closed,
                   special_hours_start, special_hours_end,
                   message_english, message_tagalog, message_bisaya
            FROM dynamic_schedules
            WHERE is_active = 1
              AND (start_date >= CURRENT_DATE OR end_date >= CURRENT_DATE)
              AND start_date <= CURRENT_DATE + INTERVAL '30 days'
            ORDER BY start_date ASC
        """)
        return jsonify({
            'schedules': [{
                'id': r[0],
                'title': r[1],
                'content_type': r[2],
                'start_date': r[3].strftime('%Y-%m-%d') if r[3] else None,
                'end_date': r[4].strftime('%Y-%m-%d') if r[4] else None,
                'is_closed': bool(r[5]),
                'special_hours_start': str(r[6]) if r[6] else None,
                'special_hours_end': str(r[7]) if r[7] else None,
                'message_english': r[8],
                'message_tagalog': r[9],
                'message_bisaya': r[10],
            } for r in cur.fetchall()]
        })
    except Exception as e:
        logger.error(f"API public schedules error: {e}")
        return jsonify({'error': str(e), 'schedules': []}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/report', methods=['POST'])
@csrf.exempt
def api_report():
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '')[:500]
    answer = (data.get('answer') or '')[:2000]
    reason = (data.get('reason') or 'other')[:50]
    details = (data.get('details') or '')[:300]
    session_id = (data.get('session_id') or '')[:100]
    if not question:
        return jsonify({'ok': False, 'error': 'question required'}), 400
    from services.language import classify_topic, detect_language
    detected_lang = detect_language(question)
    detected_topic = classify_topic(question)
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO bad_feedback "
            "(question, answer, detected_language, detected_topic) "
            "VALUES (%s, %s, %s, %s)",
            (
                question, f"[{reason}] {details}\n---\n{answer}"[:2000],
                detected_lang, detected_topic,
            ),
        )
        try:
            from services.answer_memory import record_feedback
            record_feedback(
                get_db, question, 'thumbs_down',
                correction=f"{reason}: {details}".strip(': '),
                session_id=session_id or None,
                source_type='report',
            )
        except Exception:
            pass
        conn.commit()
        return jsonify({'ok': True})
    except Exception as e:
        conn.rollback()
        logger.error(f"api_report: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()

@bp.route('/api/featured')
def api_featured():
    """Most-asked questions for the landing page Featured grid."""
    try:
        from services.popular_questions import get_popular_knowledge_questions
        items = get_popular_knowledge_questions(get_db, limit=8, days=60)
        if not items:
            from services.popular_questions import has_question_context
            conn = get_db()
            cur = conn.cursor()
            try:
                cur.execute("""
                    SELECT question, topic
                    FROM faq
                    WHERE question IS NOT NULL AND question != ''
                    ORDER BY priority_score DESC NULLS LAST, id DESC
                    LIMIT 40
                """)
                for q, topic in cur.fetchall():
                    if has_question_context(q):
                        items.append({
                            'question': q,
                            'category': topic or 'FAQ',
                            'count': 0,
                        })
                    if len(items) >= 8:
                        break
            finally:
                cur.close()
                conn.close()
        return jsonify(items)
    except Exception as e:
        logger.error(f"api_featured error: {e}")
        return jsonify([])


@bp.route('/api/recent')
def api_recent():
    """Recently added / updated FAQs for landing New & Updated strip."""
    conn = get_db()
    cur = conn.cursor()
    try:
        try:
            cur.execute("""
                SELECT question, COALESCE(topic, category, 'General') AS cat,
                       COALESCE(last_updated, created_at, NOW()) AS touched
                FROM faq
                WHERE question IS NOT NULL AND question != ''
                ORDER BY COALESCE(last_updated, created_at, NOW()) DESC
                LIMIT 12
            """)
        except Exception:
            conn.rollback()
            cur.execute("""
                SELECT question, COALESCE(topic, 'General'), NOW()
                FROM faq
                WHERE question IS NOT NULL AND question != ''
                ORDER BY id DESC
                LIMIT 12
            """)
        rows = cur.fetchall()
        today = datetime.now().date()
        items = []
        for question, cat, touched in rows:
            try:
                day = touched.date() if hasattr(touched, 'date') else today
                days_ago = max(0, (today - day).days)
            except Exception:
                days_ago = 0
            items.append({
                'question': question,
                'category': cat or 'General',
                'days_ago': days_ago,
                'is_new': days_ago <= 7,
            })
        return jsonify(items)
    except Exception as e:
        logger.error(f"api_recent error: {e}")
        return jsonify([])
    finally:
        cur.close()
        conn.close()


@bp.route('/api/handbook_stats')
def api_handbook_stats():
    """Counts for landing handbook spotlight."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM handbook_chunks WHERE is_active = 1")
        chunk_count = cur.fetchone()[0] or 0
        try:
            cur.execute("""
                SELECT COUNT(DISTINCT chapter)
                FROM handbook_chunks
                WHERE is_active = 1 AND chapter IS NOT NULL AND chapter != ''
            """)
            chapter_count = cur.fetchone()[0] or 0
        except Exception:
            chapter_count = 0
        return jsonify({
            'chunk_count': chunk_count,
            'chapter_count': chapter_count,
        })
    except Exception as e:
        logger.error(f"api_handbook_stats error: {e}")
        return jsonify({'chunk_count': 0, 'chapter_count': 0})
    finally:
        cur.close()
        conn.close()


@bp.route('/api/my_chats')
def api_my_chats():
    """Resolve chat UUIDs (from browser localStorage) into sidebar titles."""
    raw = (request.args.get('uuids') or '').strip()
    if not raw:
        return jsonify({'chats': []})
    uuids = [u.strip() for u in raw.split(',') if u.strip()][:40]
    if not uuids:
        return jsonify({'chats': []})

    conn = get_db()
    cur = conn.cursor()
    try:
        placeholders = ','.join(['%s'] * len(uuids))
        cur.execute(f"""
            SELECT public_uuid, session_id
            FROM chat_sessions
            WHERE public_uuid IN ({placeholders})
        """, uuids)
        session_map = {r[0]: r[1] for r in cur.fetchall()}
        if not session_map:
            return jsonify({'chats': []})

        out = []
        for uuid in uuids:
            sid = session_map.get(uuid)
            if not sid:
                continue
            cur.execute("""
                SELECT user_message FROM chat_history
                WHERE session_id = %s
                ORDER BY created_at ASC LIMIT 1
            """, (sid,))
            first = cur.fetchone()
            cur.execute("""
                SELECT MAX(created_at) FROM chat_history WHERE session_id = %s
            """, (sid,))
            last_row = cur.fetchone()
            title = (first[0] if first else 'New conversation') or 'New conversation'
            last_at = last_row[0] if last_row else None
            out.append({
                'uuid': uuid,
                'title': title[:80],
                'last_at': last_at.isoformat() if last_at else None,
            })
        return jsonify({'chats': out})
    except Exception as e:
        logger.error(f"api_my_chats error: {e}")
        return jsonify({'chats': []})
    finally:
        cur.close()
        conn.close()

@bp.route('/api/chat_history/<string:chat_uuid>')
def api_chat_history(chat_uuid):
    """JSON equivalent of routes/pages.py's /chat/<uuid> page, for the
    Flutter app (which can't render the server-side Jinja chat.html
    template). Web and app both end up reading the exact same
    chat_history rows, so a conversation started on one is fully
    readable on the other — that's the "connect the app and web" part:
    both clients hit this same backend/session-cookie/DB, there is no
    separate app-only data path.

    Looks up chat_sessions by public_uuid (same as pages.py) rather than
    trusting the caller's own session cookie for the lookup, so a chat
    opened from the app's local "recent chats" list (saved client-side,
    same as chat.html's localStorage list) resolves correctly even if
    the app's cookie jar was cleared/reinstalled -- matching how
    /api/my_chats already works.
    """
    from services.formatting import format_bot_message

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT session_id FROM chat_sessions WHERE public_uuid = %s",
            (chat_uuid,),
        )
        row = cur.fetchone()
        if not row:
            return jsonify({'ok': False, 'error': 'not_found', 'messages': []}), 404
        session_id = row[0]

        cur.execute(
            """
            SELECT user_message, bot_response, category, created_at
            FROM chat_history
            WHERE session_id = %s
            ORDER BY created_at ASC
            LIMIT 200
            """,
            (session_id,),
        )
        messages = []
        for user_message, bot_response, category, created_at in cur.fetchall():
            try:
                bot_html = format_bot_message(bot_response, show_images=True)
            except Exception:
                bot_html = bot_response or ''
            messages.append({
                'user': user_message,
                'bot': bot_html,
                'category': category,
                'created_at': created_at.isoformat() if created_at else None,
            })
        return jsonify({'ok': True, 'chat_uuid': chat_uuid, 'messages': messages})
    except Exception as e:
        logger.error(f"api_chat_history error: {e}")
        return jsonify({'ok': False, 'error': str(e), 'messages': []}), 500
    finally:
        cur.close()
        conn.close()


@bp.route("/api/admin/realtime")
def api_admin_realtime():
    """Lightweight payload for admin live badges (poll every 8-15s).
    Shares the cached _get_admin_badge_counts() helper above with
    /api/admin/alert_counts so two badge-polling pages open at once don't
    double the DB hits."""
    if not admin_required():
        return jsonify({"error": "Unauthorized"}), 403
    data = dict(_get_admin_badge_counts())
    data["ts"] = datetime.now().isoformat()
    return jsonify(data)

@bp.route('/api/visitor_stats')
def api_visitor_stats():
    """Public landing-page stats. Falls back to chat_history when visitor_stats is empty."""
    conn = get_db()
    cur = conn.cursor()
    try:
        total_visitors = 0
        total_questions = 0
        today_visitors = 0

        try:
            cur.execute("""
                SELECT COUNT(*) AS total_visitors,
                       COALESCE(SUM(question_count), 0) AS total_questions,
                       COUNT(CASE WHEN DATE(last_seen) = CURRENT_DATE THEN 1 END) AS today_visitors
                FROM visitor_stats
            """)
            row = cur.fetchone()
            total_visitors = int(row[0] or 0)
            total_questions = int(row[1] or 0)
            today_visitors = int(row[2] or 0)
        except Exception:
            conn.rollback()

        # Fallback: real chat activity
        if total_visitors == 0 and total_questions == 0:
            try:
                cur.execute("""
                    SELECT COUNT(DISTINCT session_id),
                           COUNT(*),
                           COUNT(DISTINCT CASE WHEN DATE(created_at) = CURRENT_DATE
                                               THEN session_id END)
                    FROM chat_history
                """)
                row = cur.fetchone()
                total_visitors = int(row[0] or 0)
                total_questions = int(row[1] or 0)
                today_visitors = int(row[2] or 0)
            except Exception:
                conn.rollback()

        avg_rating = 0.0
        rating_count = 0
        try:
            cur.execute("""
                SELECT ROUND(AVG(rating)::numeric, 1), COUNT(*)
                FROM chat_ratings
                WHERE rating BETWEEN 1 AND 5
            """)
            rrow = cur.fetchone()
            avg_rating = float(rrow[0] or 0)
            rating_count = int(rrow[1] or 0)
        except Exception:
            conn.rollback()

        return jsonify({
            'total_visitors': total_visitors,
            'total_questions': total_questions,
            'today_visitors': today_visitors,
            'students_helped': total_visitors or total_questions,
            'active_today': today_visitors,
            'avg_rating': avg_rating,
            'rating_count': rating_count,
        })
    finally:
        cur.close()
        conn.close()


@bp.route('/api/public_reviews')
def api_public_reviews():
    """Recent 4–5★ comments for the public landing page."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT rating, comment, created_at
            FROM chat_ratings
            WHERE rating >= 4
              AND comment IS NOT NULL
              AND TRIM(comment) != ''
              AND LENGTH(TRIM(comment)) >= 8
            ORDER BY created_at DESC
            LIMIT 12
        """)
        rows = cur.fetchall()
        reviews = []
        for rating, comment, created_at in rows:
            text = (comment or '').strip()
            if len(text) > 180:
                text = text[:177] + '…'
            reviews.append({
                'rating': int(rating),
                'comment': text,
                'when': created_at.strftime('%b %Y') if created_at else '',
            })
        return jsonify({'reviews': reviews})
    except Exception as e:
        logger.error(f"api_public_reviews: {e}")
        return jsonify({'reviews': []})
    finally:
        cur.close()
        conn.close()


_SUGGEST_LIMIT = 40         
_SUGGEST_WINDOW = 60.0       
_suggest_hits = {}          


def _suggest_rate_ok(ip: str) -> bool:
    import time
    now = time.time()
    bucket = _suggest_hits.get(ip) or []
    bucket = [t for t in bucket if now - t < _SUGGEST_WINDOW]
    if len(bucket) >= _SUGGEST_LIMIT:
        _suggest_hits[ip] = bucket
        return False
    bucket.append(now)
    _suggest_hits[ip] = bucket
    if len(_suggest_hits) > 5000:
        cutoff = now - _SUGGEST_WINDOW
        dead = [k for k, v in _suggest_hits.items() if not v or v[-1] < cutoff]
        for k in dead[:500]:
            _suggest_hits.pop(k, None)
    return True


@bp.route('/api/suggest')
def api_suggest():
    """Public typeahead: FAQ questions + handbook section titles.

    Query params:
      q      – search string (min 2 chars)
      limit  – max items per source (default 6, max 10)

    Returns only short labels (no full answer/content bodies) so the
    payload stays small and we don't leak long handbook text to every
    keystroke. Shape:
      {
        "faqs": [{"question": "...", "category": "..."}],
        "handbook": [{"section_title": "...", "chapter": "...", "page_start": 12}]
      }
    """
    ip = (request.headers.get('X-Forwarded-For') or request.remote_addr or 'unknown').split(',')[0].strip()
    if not _suggest_rate_ok(ip):
        return jsonify({'error': 'rate_limited', 'faqs': [], 'handbook': []}), 429

    q = (request.args.get('q') or '').strip()
    if len(q) < 2:
        return jsonify({'faqs': [], 'handbook': []})

    try:
        limit = min(10, max(1, int(request.args.get('limit', 6))))
    except (TypeError, ValueError):
        limit = 6

    q = q[:80]
    like = f'%{q}%'
    prefix = f'{q}%'

    conn = get_db()
    cur = conn.cursor()
    faqs, handbook = [], []
    try:
        try:
            cur.execute("""
                SELECT question, COALESCE(category, topic, 'General') AS cat
                FROM faq
                WHERE question IS NOT NULL AND question != ''
                  AND (question ILIKE %s
                       OR COALESCE(keywords, '') ILIKE %s
                       OR COALESCE(category, '') ILIKE %s
                       OR COALESCE(topic, '') ILIKE %s)
                ORDER BY
                  CASE WHEN question ILIKE %s THEN 0
                       WHEN question ILIKE %s THEN 1
                       ELSE 2 END,
                  priority_score DESC NULLS LAST,
                  id DESC
                LIMIT %s
            """, (like, like, like, like, prefix, like, limit))
            for r in cur.fetchall():
                text = (r[0] or '').strip()
                if not text:
                    continue
                faqs.append({
                    'question': text[:120],
                    'category': (r[1] or 'General')[:40],
                })
        except Exception as e:
            logger.warning(f"api_suggest faq: {e}")
            conn.rollback()

        try:
            cur.execute("""
                SELECT section_title, chapter, page_start
                FROM handbook_chunks
                WHERE is_active = 1
                  AND (section_title ILIKE %s
                       OR chapter ILIKE %s
                       OR COALESCE(keywords, '') ILIKE %s
                       OR content ILIKE %s)
                ORDER BY
                  CASE WHEN section_title ILIKE %s THEN 0
                       WHEN section_title ILIKE %s THEN 1
                       ELSE 2 END,
                  chunk_order ASC NULLS LAST,
                  id ASC
                LIMIT %s
            """, (like, like, like, like, prefix, like, limit))
            for r in cur.fetchall():
                title = (r[0] or '').strip()
                if not title:
                    continue
                handbook.append({
                    'section_title': title[:120],
                    'chapter': (r[1] or '')[:60],
                    'page_start': r[2],
                })
        except Exception as e:
            logger.warning(f"api_suggest handbook: {e}")
            conn.rollback()

        return jsonify({'faqs': faqs, 'handbook': handbook})
    finally:
        cur.close()
        conn.close()

@bp.route("/api/health")
def health():
    """Lightweight liveness check used by cron keep-alive and monitoring."""
    try:
        from database import get_db
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return {"status": "ok", "db": "alive"}, 200
        finally:
            conn.close()
    except Exception as e:
        return {"status": "error", "db": str(e)}, 503