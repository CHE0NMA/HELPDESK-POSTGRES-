import json
import logging

from flask import Blueprint, jsonify, request, session

from ai_providers import ai_cache, faq_cache
from database import get_db
from extensions import csrf
from utils.security import admin_required

logger = logging.getLogger("pac_helpdesk.routes.schedules")
bp = Blueprint('schedules', __name__)


def invalidate_schedule_cache():
    for kw in ['today', 'tomorrow', 'week', 'open', 'closed', 'schedule',
               'bukas', 'ngayon', 'karon', 'ugma', 'registrar', 'business']:
        faq_cache.invalidate_pattern(kw)
    ai_cache.invalidate()


@bp.route('/holidays')
def holiday_management():
    from flask import redirect, render_template
    if not session.get('admin'):
        return redirect('/admin')
    return render_template('admin/holiday_management.html')


@bp.route('/api/dynamic_schedules', methods=['GET'])
def api_get_schedules():
    if not session.get('admin'):
        return jsonify({'error': 'Unauthorized', 'schedules': []}), 401
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, title, description, content_type, start_date, end_date, repeat_type,
                   repeat_until_year, is_closed, special_hours_start, special_hours_end,
                   message_english, message_tagalog, message_bisaya, created_at
            FROM dynamic_schedules
            WHERE is_active = 1
              AND (end_date >= CURRENT_DATE OR end_date IS NULL)
              AND start_date <= CURRENT_DATE + INTERVAL '90 days'
            ORDER BY start_date ASC
            LIMIT 100
        """)
        return jsonify({'schedules': [{
            'id': r[0], 'title': r[1], 'description': r[2], 'content_type': r[3],
            'start_date': r[4].strftime('%Y-%m-%d') if r[4] else None,
            'end_date': r[5].strftime('%Y-%m-%d') if r[5] else None,
            'repeat_type': r[6], 'repeat_until_year': r[7], 'is_closed': bool(r[8]),
            'special_hours_start': str(r[9]) if r[9] else None,
            'special_hours_end': str(r[10]) if r[10] else None,
            'message_english': r[11], 'message_tagalog': r[12], 'message_bisaya': r[13],
        } for r in cur.fetchall()]})
    except Exception as e:
        logger.error(f"API get schedules error: {e}")
        return jsonify({'error': str(e), 'schedules': []}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/dynamic_schedules', methods=['POST'])
@csrf.exempt
def api_create_schedule():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json()
    if not data or not data.get('title') or not data.get('start_date'):
        return jsonify({'error': 'title and start_date are required'}), 400

    conn = get_db()
    cur = conn.cursor()
    try:
        departments = json.dumps(data.get('departments') or ['all'])
        grade_levels = json.dumps(data.get('grade_levels') or ['all'])
        description = json.dumps({
            'departments': json.loads(departments),
            'grade_levels': json.loads(grade_levels)
        })
        cur.execute("""
            INSERT INTO dynamic_schedules
            (title, description, content_type, start_date, end_date, repeat_type, repeat_until_year,
             is_closed, special_hours_start, special_hours_end, message_english, message_tagalog,
             message_bisaya, departments, grade_levels, created_by, is_active)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1)
        """, (
            data.get('title'), description, data.get('content_type', 'event'),
            data.get('start_date'), data.get('end_date'),
            data.get('repeat_type', 'none'), data.get('repeat_until_year'),
            1 if data.get('is_closed') else 0,
            data.get('special_hours_start'), data.get('special_hours_end'),
            data.get('message_english'), data.get('message_tagalog'), data.get('message_bisaya'),
            departments, grade_levels, session.get('admin_user', 'admin')
        ))
        conn.commit()
        invalidate_schedule_cache()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        logger.error(f"API create schedule error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/dynamic_schedules/<int:schedule_id>', methods=['PUT'])
@csrf.exempt
def api_update_schedule(schedule_id):
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE dynamic_schedules
            SET title=%s, description=%s, content_type=%s, start_date=%s, end_date=%s,
                repeat_type=%s, repeat_until_year=%s, is_closed=%s,
                special_hours_start=%s, special_hours_end=%s,
                message_english=%s, message_tagalog=%s, message_bisaya=%s,
                grade_levels=%s
            WHERE id=%s
        """, (
            data.get('title'), data.get('description'), data.get('content_type'),
            data.get('start_date'), data.get('end_date'),
            data.get('repeat_type', 'none'), data.get('repeat_until_year'),
            1 if data.get('is_closed') else 0,
            data.get('special_hours_start'), data.get('special_hours_end'),
            data.get('message_english'), data.get('message_tagalog'), data.get('message_bisaya'),
            json.dumps(data.get('grade_levels', ['all'])) if data.get('grade_levels') else None,
            schedule_id
        ))
        conn.commit()
        invalidate_schedule_cache()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        logger.error(f"API update schedule error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/dynamic_schedules/<int:schedule_id>', methods=['DELETE'])
@csrf.exempt
def api_delete_schedule(schedule_id):
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE dynamic_schedules SET is_active=0 WHERE id=%s", (schedule_id,))
        conn.commit()
        invalidate_schedule_cache()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/emergency_closures', methods=['GET'])
def api_get_emergency_closures():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, title, reason, severity, start_date, end_date, affected_departments,
                   affected_grades, message_english, message_tagalog, message_bisaya, created_at
            FROM emergency_closures
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 50
        """)
        rows = cur.fetchall()
        return jsonify({'closures': [{
            'id': r[0], 'title': r[1], 'reason': r[2], 'severity': r[3],
            'start_date': r[4].strftime('%Y-%m-%d') if r[4] else None,
            'end_date': r[5].strftime('%Y-%m-%d') if r[5] else None,
            'affected_departments': json.loads(r[6]) if r[6] else ['all'],
            'affected_grades': json.loads(r[7]) if r[7] else ['all'],
            'message_english': r[8], 'message_tagalog': r[9], 'message_bisaya': r[10],
            'created_at': r[11].strftime('%Y-%m-%d %H:%M') if r[11] else None,
        } for r in rows]})
    finally:
        cur.close()
        conn.close()


@bp.route('/api/emergency_closures', methods=['POST'])
@csrf.exempt
def api_create_emergency_closure():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO emergency_closures
            (title, reason, severity, start_date, end_date, affected_departments, affected_grades,
             message_english, message_tagalog, message_bisaya, created_by)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data.get('title'), data.get('reason'), data.get('severity', 'medium'),
            data.get('start_date'), data.get('end_date'),
            json.dumps(data.get('affected_departments', ['all'])),
            json.dumps(data.get('affected_grades', ['all'])),
            data.get('message_english'), data.get('message_tagalog'), data.get('message_bisaya'),
            session.get('admin_user', 'admin')
        ))
        conn.commit()
        invalidate_schedule_cache()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        logger.error(f"API create emergency closure error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/emergency_closures/<int:closure_id>', methods=['DELETE'])
@csrf.exempt
def api_delete_emergency_closure(closure_id):
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE emergency_closures SET is_active = 0 WHERE id = %s", (closure_id,))
        conn.commit()
        invalidate_schedule_cache()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/missing_date_queries')
def api_missing_date_queries():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, question, detected_date, asked_count, last_asked
        FROM missing_date_queries
        ORDER BY asked_count DESC, last_asked DESC
        LIMIT 50
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify({'queries': [{
        'id': r[0],
        'question': r[1],
        'detected_date': r[2].strftime('%Y-%m-%d') if r[2] else None,
        'asked_count': r[3],
        'last_asked': r[4].strftime('%Y-%m-%d %H:%M') if r[4] else None
    } for r in rows]})

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