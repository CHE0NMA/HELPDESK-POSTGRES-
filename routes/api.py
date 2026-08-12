import logging
from datetime import datetime, timedelta

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

        cur.execute("""
            SELECT user_message, COUNT(*) as freq
            FROM chat_history
            GROUP BY user_message
            ORDER BY freq DESC
            LIMIT 5
        """)
        popular = cur.fetchall()

        cur.execute("SELECT SUM(question_count), COUNT(*) FROM visitor_stats")
        vrow = cur.fetchone()

        cur.execute("SELECT COUNT(*) FROM unanswered_questions")
        unanswered = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM bad_feedback")
        bad_feedback = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM faq")
        total_faqs = cur.fetchone()[0]

        cur.execute("""
            SELECT category, COUNT(*) as count
            FROM faq
            WHERE category IS NOT NULL AND category != ''
            GROUP BY category
            ORDER BY count DESC
            LIMIT 5
        """)
        categories = cur.fetchall()

        cur.execute("SELECT rating, COUNT(*) FROM satisfaction_feedback GROUP BY rating")
        satisfaction = {r[0]: r[1] for r in cur.fetchall()}

        total_q = int(vrow[0] or 0)
        return jsonify({
            'activity': [{'date': str(r[0]), 'count': r[1]} for r in chat_dates],
            'popular': [{'question': r[0], 'count': r[1]} for r in popular],
            'categories': [{'name': r[0], 'count': r[1]} for r in categories],
            'stats': {
                'total_queries': total_q,
                'total_visitors': int(vrow[1] or 0),
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

    return jsonify({
        'average_response_time_cache_ms': 0,
        'average_response_time_db_ms': 0,
        'cache_hit_count': 0,
        'total_queries_tracked': 0,
        'note': 'Per-request timing tracking not yet wired up in the modular version.',
    })


@bp.route('/api/visitor_stats')
def api_visitor_stats():
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(*) AS total_visitors,
                   SUM(question_count) AS total_questions,
                   COUNT(CASE WHEN DATE(last_seen) = CURRENT_DATE THEN 1 END) AS today_visitors
            FROM visitor_stats
        """)
        row = cur.fetchone()
        return jsonify({
            'total_visitors': row[0] or 0,
            'total_questions': row[1] or 0,
            'today_visitors': row[2] or 0
        })
    finally:
        cur.close()
        conn.close()


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


@bp.route('/api/admin/alert_counts')
def api_alert_counts():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM unanswered_questions")
        unanswered_count = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM satisfaction_feedback
            WHERE rating = 'thumbs_down'
              AND created_at > NOW() - INTERVAL '7 days'
        """)
        reports_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM faq_review_queue WHERE status = 'pending'")
        review_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM pending_faq_drafts WHERE status = 'pending'")
        draft_count = cur.fetchone()[0]

        return jsonify({
            'unanswered': unanswered_count,
            'reports': reports_count,
            'review_queue': review_count,
            'pending_drafts': draft_count,
            'health_issues': 0,
            'total': unanswered_count + reports_count + review_count + draft_count,
        })
    finally:
        cur.close()
        conn.close()


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
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO bad_feedback (question, answer) VALUES (%s, %s)",
            (question, f"[{reason}] {details}\n---\n{answer}"[:2000]),
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
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT user_message, COUNT(*) AS freq
            FROM chat_history
            WHERE created_at >= NOW() - INTERVAL '60 days'
              AND LENGTH(user_message) > 8
              AND LOWER(user_message) NOT IN (
                'hi','hello','hey','yes','no','ok','thanks','thank you',
                'kumusta','kamusta','musta'
              )
            GROUP BY user_message
            ORDER BY freq DESC
            LIMIT 8
        """)
        rows = cur.fetchall()
        items = []
        for msg, freq in rows:
            items.append({'question': msg, 'category': '', 'count': freq})
        if not items:
            cur.execute("""
                SELECT question, topic
                FROM faq
                WHERE question IS NOT NULL AND question != ''
                ORDER BY priority_score DESC NULLS LAST, id DESC
                LIMIT 8
            """)
            for q, topic in cur.fetchall():
                items.append({'question': q, 'category': topic or 'General', 'count': 0})
        return jsonify(items)
    except Exception as e:
        logger.error(f"api_featured error: {e}")
        return jsonify([])
    finally:
        cur.close()
        conn.close()


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

@bp.route("/api/admin/realtime")
def api_admin_realtime():
    """Lightweight payload for admin live badges (poll every 8–15s)."""
    if not admin_required():
        return jsonify({"error": "Unauthorized"}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM unanswered_questions")
        unanswered = cur.fetchone()[0]

        cur.execute(
            """
            SELECT COUNT(*) FROM satisfaction_feedback
            WHERE rating = 'thumbs_down'
              AND created_at > NOW() - INTERVAL '7 days'
            """
        )
        reports = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM faq_review_queue WHERE status = 'pending'"
        )
        review = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM pending_faq_drafts WHERE status = 'pending'"
        )
        drafts = cur.fetchone()[0]

        return jsonify({
            "unanswered": unanswered,
            "reports": reports,
            "review_queue": review,
            "pending_drafts": drafts,
            "total": unanswered + reports + review + drafts,
            "ts": datetime.now().isoformat(),
        })
    finally:
        cur.close()
        conn.close()