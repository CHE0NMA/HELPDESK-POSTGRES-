import logging
import os
import time
import uuid
import json
from datetime import datetime, timezone
from services.handbook_ingest import parse_pasted_faqs, mass_insert_faqs
from flask import Blueprint, jsonify, redirect, render_template, request, session, make_response
from werkzeug.security import check_password_hash
from threading import Thread
from ai_providers import faq_cache
from database import get_db
from extensions import csrf
from services.language import invalidate_response_tables
from utils.security import (admin_required, attempts_left, clear_attempts, client_ip,
                             is_locked_out, record_failed_attempt)
from utils.text import fuzzy_score, normalize_input

from database import get_db

logger = logging.getLogger("pac_helpdesk.routes.admin")
bp = Blueprint('admin', __name__)


def check_duplicate_faq(question, threshold=85):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, question FROM faq")
    existing = cur.fetchall()
    cur.close()
    conn.close()
    norm_new = normalize_input(question)
    from fuzzywuzzy import fuzz
    return [{'id': fid, 'question': eq, 'similarity': fuzz.token_set_ratio(norm_new, normalize_input(eq))}
            for fid, eq in existing if fuzz.token_set_ratio(norm_new, normalize_input(eq)) >= threshold]

@bp.route('/admin', methods=['GET', 'POST'])
def admin():
    ip = client_ip()
    if request.method == 'POST':
        locked, remaining = is_locked_out(ip)
        if locked:
            return render_template('admin_login.html',
                                    error=f'Too many attempts. Try again in {remaining // 60}m {remaining % 60}s.',
                                    lockout=True)
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '')
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT password FROM admin_users WHERE username=%s", (u,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and check_password_hash(row[0], p):
            clear_attempts(ip)
            session['admin'] = True
            session.permanent = True
            session['admin_ip'] = ip
            session['login_time'] = datetime.now(timezone.utc).isoformat()
            logger.info(f"Admin login successful from {ip}")
            return redirect('/dashboard')
        record_failed_attempt(ip)
        locked, remaining = is_locked_out(ip)
        if locked:
            return render_template('admin_login.html', error=f'Account locked for {remaining // 60} min.', lockout=True)
        logger.warning(f"Failed admin login from {ip}")
        return render_template('admin_login.html',
                                error=f'Invalid credentials. {attempts_left(ip)} attempt(s) left.')
    return render_template('admin_login.html')


@bp.route('/admin/logout')
def admin_logout():
    from services.conversation import conversation_contexts
    sid = session.get('persistent_session_id')
    if sid and sid in conversation_contexts:
        del conversation_contexts[sid]
    session.clear()
    return redirect('/admin')


_dash_cache = {'data': None, 'ts': 0}


def get_dashboard_stats(ttl=30):
    if _dash_cache['data'] and time.time() - _dash_cache['ts'] < ttl:
        return _dash_cache['data']
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM faq ORDER BY priority_score DESC, id DESC")
        faqs = cur.fetchall()
        cur.execute("SELECT DISTINCT topic FROM faq WHERE topic IS NOT NULL AND topic != ''")
        topics = [t[0] for t in cur.fetchall()]
        cur.execute("SELECT COUNT(*) FROM faq")
        total_faqs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM unanswered_questions")
        unanswered_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chat_history")
        total_chats = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM satisfaction_feedback WHERE rating='thumbs_down'")
        reports_count = cur.fetchone()[0]
        result = {'faqs': faqs, 'topics': topics, 'total_faqs': total_faqs,
                  'unanswered_count': unanswered_count, 'total_chats': total_chats,
                  'reports_count': reports_count}
    finally:
        cur.close()
        conn.close()
    _dash_cache.update(data=result, ts=time.time())
    return result


def invalidate_dashboard_cache():
    _dash_cache['data'] = None


@bp.route('/dashboard')
def dashboard():
    if not admin_required():
        return redirect('/admin')
    stats = get_dashboard_stats()
    return render_template('dashboard.html', faqs=stats['faqs'], topics=stats['topics'],
                            total_faqs=stats['total_faqs'], unanswered_count=stats['unanswered_count'],
                            reports_count=stats['reports_count'], total_chats=stats['total_chats'])


@bp.route('/add_faq', methods=['POST'])
@csrf.exempt
def add_faq():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    d = request.form
    question = d.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    dups = check_duplicate_faq(question)
    if dups:
        return jsonify({'error': f'Duplicate: "{dups[0]["question"]}" ({dups[0]["similarity"]}% similar)'}), 409

    f = request.files.get('image')
    stored_name = None
    if f and f.filename:
        stored_name = f"{uuid.uuid4().hex}_{f.filename}"
        f.save(os.path.join(os.environ.get('UPLOAD_FOLDER', 'uploads'), stored_name))

    public_uuid = str(uuid.uuid4())
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO faq (topic, intent, question, answer, image_path, keywords, related_questions,
                              language, category, priority_score, attachment_url, public_uuid)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (d.get('topic', 'General'), d.get('intent', ''), question, d.get('answer', ''),
              f"uploads/{stored_name}" if stored_name else None, d.get('keywords', ''),
              d.get('related_questions', ''), d.get('language', 'en'), d.get('category', 'General'),
              int(d.get('priority_score', 0)), d.get('attachment_url'), public_uuid))
        conn.commit()
        faq_cache.invalidate()
        invalidate_response_tables()
        invalidate_dashboard_cache()
        return jsonify({'success': True, 'message': 'FAQ added successfully', 'uuid': public_uuid})
    except Exception as e:
        conn.rollback()
        logger.error(f"Add FAQ error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/update_faq', methods=['POST'])
@csrf.exempt
def update_faq():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    d = request.form
    faq_uuid = d.get('faq_uuid')
    if not faq_uuid:
        return jsonify({'error': 'FAQ UUID missing'}), 400

    f = request.files.get('image')
    stored_name = None
    if f and f.filename:
        stored_name = f"{uuid.uuid4().hex}_{f.filename}"
        f.save(os.path.join(os.environ.get('UPLOAD_FOLDER', 'uploads'), stored_name))

    conn = get_db()
    cur = conn.cursor()
    try:
        if stored_name:
            cur.execute("""
                UPDATE faq SET topic=%s, intent=%s, question=%s, answer=%s, image_path=%s,
                             keywords=%s, related_questions=%s, language=%s, category=%s,
                             priority_score=%s, attachment_url=%s
                WHERE public_uuid=%s
            """, (d.get('topic', 'General'), d.get('intent', ''), d.get('question', ''), d.get('answer', ''),
                  f"uploads/{stored_name}", d.get('keywords', ''), d.get('related_questions', ''),
                  d.get('language', 'en'), d.get('category', 'General'), int(d.get('priority_score', 0)),
                  d.get('attachment_url'), faq_uuid))
        else:
            cur.execute("""
                UPDATE faq SET topic=%s, intent=%s, question=%s, answer=%s,
                             keywords=%s, related_questions=%s, language=%s, category=%s,
                             priority_score=%s, attachment_url=%s
                WHERE public_uuid=%s
            """, (d.get('topic', 'General'), d.get('intent', ''), d.get('question', ''), d.get('answer', ''),
                  d.get('keywords', ''), d.get('related_questions', ''), d.get('language', 'en'),
                  d.get('category', 'General'), int(d.get('priority_score', 0)), d.get('attachment_url'), faq_uuid))
        conn.commit()
        faq_cache.invalidate()
        invalidate_response_tables()
        invalidate_dashboard_cache()
        return jsonify({'success': True, 'message': 'FAQ updated successfully'})
    except Exception as e:
        conn.rollback()
        logger.error(f"Update FAQ error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/delete_faq/<string:public_uuid>', methods=['GET', 'POST'])
@csrf.exempt
def delete_faq(public_uuid):
    if not admin_required():
        return (jsonify({'error': 'Unauthorized'}), 403) if request.method == 'POST' else redirect('/admin')
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM faq WHERE public_uuid = %s", (public_uuid,))
        conn.commit()
        if cur.rowcount == 0:
            return (jsonify({'error': 'FAQ not found'}), 404) if request.method == 'POST' else redirect('/dashboard')
        faq_cache.invalidate()
        invalidate_response_tables()
        invalidate_dashboard_cache()
        return jsonify({'success': True}) if request.method == 'POST' else redirect('/dashboard')
    finally:
        cur.close()
        conn.close()


@bp.route('/faq/<string:public_uuid>')
def view_faq_by_uuid(public_uuid):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, topic, intent, question, answer, keywords, related_questions,
                   priority_score, image_path, public_uuid
            FROM faq WHERE public_uuid = %s
        """, (public_uuid,))
        row = cur.fetchone()
        if not row:
            return render_template('404.html'), 404
        faq = {'id': row[0], 'topic': row[1], 'intent': row[2], 'question': row[3], 'answer': row[4],
               'keywords': row[5], 'related_questions': row[6], 'priority_score': row[7],
               'image_path': row[8], 'public_uuid': row[9]}
        try:
            cur.execute("INSERT INTO faq_views (faq_id, viewed_at, ip_address) VALUES (%s, NOW(), %s)",
                        (faq['id'], request.remote_addr))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log FAQ view: {e}")
        return render_template('faq_view.html', faq=faq)
    finally:
        cur.close()
        conn.close()


@bp.route('/unanswered')
def unanswered():
    if not admin_required():
        return redirect('/admin')
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, question, resolved_question, context, created_at
        FROM unanswered_questions ORDER BY created_at DESC
    """)
    today = datetime.today()
    questions = [{'id': r[0], 'question': r[1],
                  'resolved_question': r[2] if (r[2] and r[2].strip() != (r[1] or '').strip()) else None,
                  'context': r[3], 'days_ago': (today.date() - r[4].date()).days if r[4] else 0,
                  'is_today': (today.date() - r[4].date()).days == 0 if r[4] else False}
                 for r in cur.fetchall()]
    cur.execute("SELECT COUNT(*) FROM unanswered_questions")
    unanswered_count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return render_template('unanswered.html', questions=questions, unanswered_count=unanswered_count)


@bp.route('/unanswered_count')
def unanswered_count():
    if not admin_required():
        return jsonify({'count': 0}), 401
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM unanswered_questions")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return jsonify({'count': count})


@bp.route('/delete_unanswered/<int:qid>')
def delete_unanswered(qid):
    if not admin_required():
        return redirect('/admin')
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM unanswered_questions WHERE id=%s", (qid,))
    conn.commit()
    cur.close()
    conn.close()
    return redirect('/unanswered')


@bp.route('/answer_question/<int:qid>', methods=['GET', 'POST'])
@csrf.exempt
def answer_question(qid):
    if not admin_required():
        return redirect('/admin')
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT question, resolved_question FROM unanswered_questions WHERE id=%s", (qid,))
    q = cur.fetchone()
    if not q:
        cur.close()
        conn.close()
        return redirect('/unanswered')

    raw_question, resolved_question = q
    default_question = (resolved_question or raw_question or '').strip()

    if request.method == 'POST':
        d = request.form
        final_question = (d.get('question') or '').strip() or default_question
        dups = check_duplicate_faq(final_question)
        if dups:
            cur.close()
            conn.close()
            return render_template('unanswered.html', error=f'Similar FAQ: {dups[0]["question"]}',
                                    question=default_question)
        cur.execute("""
            INSERT INTO faq (topic, intent, question, answer, keywords, related_questions,
                              language, category, priority_score, attachment_url)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (d.get('topic', 'General'), d.get('intent', ''), final_question, d.get('answer', ''),
              d.get('keywords', ''), d.get('related_questions', ''), d.get('language', 'en'),
              d.get('category', 'General'), int(d.get('priority_score', 0)), d.get('attachment_url')))
        cur.execute("DELETE FROM unanswered_questions WHERE id=%s", (qid,))
        conn.commit()
        faq_cache.invalidate()
        invalidate_response_tables()
        invalidate_dashboard_cache()
        cur.close()
        conn.close()
        return redirect('/dashboard')

    cur.close()
    conn.close()
    return render_template('unanswered.html', question=default_question)


@bp.route('/reports')
def reports():
    if not admin_required():
        return redirect('/admin')
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, question, answer, created_at FROM satisfaction_feedback "
                "WHERE rating='thumbs_down' ORDER BY created_at DESC")
    today = datetime.today()
    feedbacks = [{'id': f[0], 'question': f[1], 'answer': f[2],
                  'days_ago': (today.date() - f[3].date()).days if f[3] else 0} for f in cur.fetchall()]
    new_reports_count = sum(1 for f in feedbacks if f['days_ago'] == 0)
    cur.execute("SELECT COUNT(*) FROM unanswered_questions")
    unanswered_count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return render_template('reports.html', feedbacks=feedbacks, new_reports_count=new_reports_count,
                            unanswered_count=unanswered_count)


@bp.route('/reports_count')
def reports_count():
    if not admin_required():
        return jsonify({'count': 0}), 401
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM satisfaction_feedback WHERE rating='thumbs_down'")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return jsonify({'count': count})


@bp.route('/delete_feedback/<int:fid>', methods=['POST'])
@csrf.exempt
def delete_feedback(fid):
    if not admin_required():
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM satisfaction_feedback WHERE id=%s", (fid,))
        conn.commit()
        if cur.rowcount == 0:
            return jsonify({'status': 'error', 'message': 'Feedback not found'}), 404
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Delete feedback error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/history')
def history():
    if not admin_required():
        return redirect('/admin')
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM chat_history ORDER BY created_at DESC LIMIT 500")
    chats = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('history.html', chats=chats)


@bp.route('/analytics')
def analytics_page():
    if not admin_required():
        return redirect('/admin')
    return render_template('analytics.html')


@bp.route('/ratings')
def ratings_page():
    if not admin_required():
        return redirect('/admin')
    return render_template('ratings.html')


@bp.route('/handbook')
def handbook_management():
    if not session.get('admin'):
        return redirect('/admin')
    return render_template('handbook_management.html')


@bp.route('/alert_log')
def alert_log():
    if not admin_required():
        return redirect('/admin')
    return render_template('alert_log.html')


@bp.route('/api/ai_feedback', methods=['POST'])
@csrf.exempt
def ai_feedback():
    data = request.get_json()
    response_id = data.get('response_id')
    helpful = data.get('helpful')
    correction = data.get('correction', '')
    if not response_id:
        return jsonify({'error': 'No response_id'}), 400
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE ai_generated_responses SET was_helpful=%s WHERE id=%s", (helpful, response_id))
    if correction:
        cur.execute(
            "INSERT INTO ai_feedback (ai_response_id, user_feedback, user_correction) VALUES (%s,%s,%s)",
            (response_id, helpful, correction))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'success': True})


@bp.route('/mark_reviewed/<int:faq_id>', methods=['POST'])
@csrf.exempt
def mark_reviewed(faq_id):
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE faq SET reviewed_flag=1, last_reviewed=NOW() WHERE id=%s", (faq_id,))
    cur.execute("UPDATE faq_review_queue SET status='completed' WHERE faq_id=%s", (faq_id,))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'success': True})


@bp.route('/reject_draft/<int:draft_id>', methods=['POST'])
@csrf.exempt
def reject_draft(draft_id):
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE pending_faq_drafts SET status='rejected' WHERE id=%s", (draft_id,))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'success': True})


@bp.route('/approve_draft/<int:draft_id>', methods=['POST'])
@csrf.exempt
def approve_draft(draft_id):
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT question, suggested_category FROM pending_faq_drafts WHERE id=%s", (draft_id,))
    draft = cur.fetchone()
    if draft:
        question, category = draft
        cur.execute("INSERT INTO faq (question, answer, category, topic, priority_score) "
                    "VALUES (%s,'To be filled',%s,'General',0)", (question, category))
        cur.execute("UPDATE pending_faq_drafts SET status='approved' WHERE id=%s", (draft_id,))
        conn.commit()
        faq_cache.invalidate()
        invalidate_response_tables()
        invalidate_dashboard_cache()
    cur.close()
    conn.close()
    return jsonify({'success': True})


@bp.route('/api/pending_drafts')
def api_pending_drafts():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, question, frequency, suggested_category FROM pending_faq_drafts WHERE status='pending'")
    drafts = [{'id': r[0], 'question': r[1], 'frequency': r[2], 'suggested_category': r[3]} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return jsonify({'drafts': drafts})


@bp.route('/api/review_queue')
def api_review_queue():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, faq_id, reason FROM faq_review_queue WHERE status='pending'")
    reviews = [{'id': r[0], 'faq_id': r[1], 'reason': r[2]} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return jsonify({'reviews': reviews})


@bp.route('/api/categories')
def api_categories():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT category, question FROM faq WHERE category IS NOT NULL AND category != '' "
                "ORDER BY category, priority_score DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    grouped = {}
    for cat, question in rows:
        grouped.setdefault(cat, []).append({'question': question})
    return jsonify(grouped)


@bp.route('/api/export/faq')
def export_faq():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    import json
    from flask import current_app, send_file
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, topic, intent, question, answer, keywords, priority_score FROM faq")
    faqs = cur.fetchall()
    cur.close()
    conn.close()
    export_data = {
        'export_date': datetime.now().isoformat(), 'version': '4.0',
        'faqs': [{'id': f[0], 'topic': f[1], 'intent': f[2], 'question': f[3], 'answer': f[4],
                  'keywords': f[5], 'priority_score': f[6]} for f in faqs],
    }
    fname = f"faq_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    fpath = os.path.join(current_app.config['BACKUP_FOLDER'], fname)
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    return send_file(fpath, as_attachment=True, download_name=fname)


@bp.route('/api/export/faq/csv')
def export_faq_csv():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    import csv
    from io import StringIO
    from flask import Response
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, topic, intent, question, answer, keywords, priority_score FROM faq")
    faqs = cur.fetchall()
    cur.close()
    conn.close()
    out = StringIO()
    w = csv.writer(out)
    w.writerow(['ID', 'Topic', 'Intent', 'Question', 'Answer', 'Keywords', 'Priority Score'])
    for f in faqs:
        w.writerow(f)
    out.seek(0)
    return Response(out.getvalue(), mimetype='text/csv',
                     headers={"Content-Disposition": "attachment;filename=faq_export.csv"})


@bp.route('/api/faq/bulk_import', methods=['POST'])
@csrf.exempt
def bulk_import_faq():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    import json
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    if not f.filename.endswith('.json'):
        return jsonify({'error': 'JSON only'}), 400
    data = json.load(f)
    conn = get_db()
    cur = conn.cursor()
    imported = 0
    for item in data.get('faqs', []):
        cur.execute(
            "INSERT INTO faq (topic,intent,question,answer,keywords,priority_score) VALUES (%s,%s,%s,%s,%s,%s)",
            (item.get('topic', 'General'), item.get('intent', ''), item.get('question'), item.get('answer'),
             item.get('keywords', ''), item.get('priority_score', 0)))
        imported += 1
    conn.commit()
    cur.close()
    conn.close()
    faq_cache.invalidate()
    invalidate_response_tables()
    invalidate_dashboard_cache()
    return jsonify({'success': True, 'imported': imported})

@bp.route('/api/check_admin')
def api_check_admin():
    return jsonify({'is_admin': bool(session.get('admin'))})


@bp.route('/api/admin/handbook_chunks')
def api_handbook_chunks():
    if not session.get('admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, chapter, section_title, page_start, page_end,
                   content, keywords, is_active, chunk_order
            FROM handbook_chunks
            ORDER BY is_active DESC, chunk_order ASC, id ASC
        """)
        rows = cur.fetchall()
        chunks = []
        for r in rows:
            content = r[5] or ''
            preview = (content[:180] + '…') if len(content) > 180 else content
            chunks.append({
                'id': r[0],
                'chapter': r[1],
                'section_title': r[2],
                'page_start': r[3],
                'page_end': r[4],
                'content': content,
                'preview': preview,
                'keywords': r[6] or '',
                'is_active': bool(r[7]),
                'chunk_order': r[8],
            })
        return jsonify({'chunks': chunks})
    except Exception as e:
        logger.error(f"handbook_chunks list error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/admin/handbook_chunks/<int:chunk_id>', methods=['PUT'])
def api_update_handbook_chunk(chunk_id):
    if not session.get('admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json(silent=True) or {}
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE handbook_chunks
            SET section_title = %s,
                chapter = %s,
                content = %s,
                keywords = %s
            WHERE id = %s
        """, (
            (data.get('section_title') or '')[:255],
            (data.get('chapter') or '')[:200],
            data.get('content') or '',
            (data.get('keywords') or '')[:500],
            chunk_id,
        ))
        conn.commit()
        if cur.rowcount == 0:
            return jsonify({'error': 'Section not found'}), 404
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        logger.error(f"handbook_chunks update error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/admin/handbook_chunks/<int:chunk_id>', methods=['DELETE'])
def api_delete_handbook_chunk(chunk_id):
    if not session.get('admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM handbook_chunks WHERE id = %s", (chunk_id,))
        conn.commit()
        if cur.rowcount == 0:
            return jsonify({'error': 'Section not found'}), 404
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        logger.error(f"handbook_chunks delete error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/admin/upload_handbook', methods=['POST'])
def api_upload_handbook():
    if not session.get('admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    f = request.files.get('handbook')
    if not f or not f.filename:
        return jsonify({'error': 'No PDF uploaded'}), 400
    if not f.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'PDF only'}), 400

    upload_dir = os.environ.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = f"handbook_{uuid.uuid4().hex}.pdf"
    path = os.path.join(upload_dir, safe_name)
    f.save(path)

    def _ingest():
        try:
            from services.handbook_ingest import ingest_handbook
            n = ingest_handbook(path, get_db)
            logger.info(f"Handbook ingested: {n} chunks from {safe_name}")
        except Exception as e:
            logger.error(f"Handbook ingest failed: {e}")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    Thread(target=_ingest, daemon=True).start()
    return jsonify({
        'success': True,
        'message': 'Handbook uploaded and is processing. Refresh in ~15–60s.',
    })

@bp.route('/api/admin/export_handbook', methods=['GET'])
def export_handbook():
    """Export all active (and optionally inactive) handbook chunks as JSON."""
    include_inactive = request.args.get('include_inactive', '0') == '1'

    conn = get_db()
    cur = conn.cursor()
    try:
        if include_inactive:
            cur.execute("""
                SELECT id, chapter, section_title, page_start, page_end,
                       content, keywords, chunk_order, is_active
                FROM handbook_chunks
                ORDER BY chunk_order, id
            """)
        else:
            cur.execute("""
                SELECT id, chapter, section_title, page_start, page_end,
                       content, keywords, chunk_order, is_active
                FROM handbook_chunks
                WHERE is_active = 1
                ORDER BY chunk_order, id
            """)

        rows = cur.fetchall()
        chunks = []
        for r in rows:
            chunks.append({
                'id': r[0],
                'chapter': r[1],
                'section_title': r[2],
                'page_start': r[3],
                'page_end': r[4],
                'content': r[5],
                'keywords': r[6],
                'chunk_order': r[7],
                'is_active': bool(r[8]),
            })

        from datetime import datetime
        payload = {
            'exported_at': datetime.utcnow().isoformat() + 'Z',
            'total_chunks': len(chunks),
            'active_chunks': sum(1 for c in chunks if c['is_active']),
            'chunks': chunks,
        }

        response = make_response(json.dumps(payload, ensure_ascii=False, indent=2))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        filename = f"handbook_chunks_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    finally:
        cur.close()
        conn.close()

@bp.route('/api/admin/import_handbook_chunks', methods=['POST'])
def api_import_handbook_chunks():
    """Import a pre-chunked handbook JSON (same format as the export)."""
    if not session.get('admin'):
        return jsonify({'error': 'Unauthorized'}), 401

    f = request.files.get('chunks_file')
    if not f or not f.filename:
        return jsonify({'error': 'No JSON file uploaded'}), 400
    if not f.filename.lower().endswith('.json'):
        return jsonify({'error': 'JSON only'}), 400

    try:
        data = json.load(f)
    except Exception as e:
        return jsonify({'error': f'Invalid JSON: {e}'}), 400

    chunks = data.get('chunks')
    if not isinstance(chunks, list) or not chunks:
        return jsonify({'error': 'JSON must contain a non-empty "chunks" array'}), 400

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute('UPDATE handbook_chunks SET is_active = 0')

        inserted = 0
        for i, ch in enumerate(chunks):
            cur.execute(
                """
                INSERT INTO handbook_chunks
                (chapter, section_title, page_start, page_end, content, keywords, chunk_order, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
                """,
                (
                    (ch.get('chapter') or '')[:200],
                    (ch.get('section_title') or '')[:255],
                    ch.get('page_start'),
                    ch.get('page_end'),
                    ch.get('content') or '',
                    (ch.get('keywords') or '')[:500],
                    ch.get('chunk_order', i),
                ),
            )
            inserted += 1

        conn.commit()
        logger.info(f"Imported {inserted} pre-chunked handbook sections")
        return jsonify({
            'success': True,
            'message': f'Successfully imported {inserted} sections.',
            'imported': inserted,
        })
    except Exception as e:
        conn.rollback()
        logger.error(f"Import handbook chunks failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
        conn.close()



    @app.route('/api/admin/mass_insert_handbook_faqs', methods=['POST'])
    @admin_required
    def api_mass_insert_handbook_faqs():
        data = request.get_json(silent=True) or {}
        text = data.get('text') or ''
        retire = bool(data.get('retire_previous', True))
        faqs = parse_pasted_faqs(text)
        if not faqs:
            return jsonify(success=False, error='No valid FAQ pairs found'), 400
        n = mass_insert_faqs(get_db, faqs, retire_previous=retire)
        return jsonify(success=True, inserted=n, message=f'Inserted {n} handbook FAQs')