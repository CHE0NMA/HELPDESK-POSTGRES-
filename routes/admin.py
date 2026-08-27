import logging
import os
import time
import uuid
import json
from datetime import datetime, timezone
from services.tz import ph_now  # BUG FIX: was using server-UTC datetime.now()/today() for "days ago" -- see services/tz.py
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

        # BUG FIX: nothing was ever calling this for FAQs added/edited
        # through the admin panel -- only a one-time backfill script
        # populated `embedding`, so every FAQ added since then had
        # embedding=NULL forever and silently fell back to fuzzy-only
        # matching (services/answer_engine.py's SEMANTIC_BLEND never
        # applied to it). Best-effort: never blocks the save if
        # GEMINI_API_KEY is unset or the call fails.
        try:
            from services.embeddings import store_embedding_for_faq
            store_embedding_for_faq(get_db, question, d.get('keywords', ''), public_uuid=public_uuid)
        except Exception as e:
            logger.debug(f"add_faq embedding backfill skipped: {e}")

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

        try:
            from services.embeddings import store_embedding_for_faq
            store_embedding_for_faq(get_db, d.get('question', ''), d.get('keywords', ''), public_uuid=faq_uuid)
        except Exception as e:
            logger.debug(f"update_faq embedding backfill skipped: {e}")

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
    today = ph_now()
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
    """
    BUG FIX: this used to read ONLY satisfaction_feedback (thumbs-down
    chat ratings). The detailed "report an issue" flow -- routes/chat.py's
    /feedback and routes/api.py's /api/report, which capture a reason +
    free-text details -- both insert into a completely separate table,
    bad_feedback, which nothing here ever displayed. Admins could see a
    raw count of it on /api/analytics/chart_data but never the actual
    reported question/answer/reason. Now merges both sources so nothing
    submitted through either flow is invisible. IDs are prefixed
    ('sf_'/'bf_') so delete_feedback() below knows which table a given
    row came from.
    """
    if not admin_required():
        return redirect('/admin')
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, question, answer, created_at FROM satisfaction_feedback "
                "WHERE rating='thumbs_down' ORDER BY created_at DESC")
    sat_rows = cur.fetchall()
    cur.execute("SELECT id, question, answer, created_at FROM bad_feedback ORDER BY created_at DESC")
    bad_rows = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM unanswered_questions")
    unanswered_count = cur.fetchone()[0]
    cur.close()
    conn.close()

    today = ph_now()

    def _row(prefix, r):
        return {'id': f'{prefix}{r[0]}', 'question': r[1], 'answer': r[2],
                'days_ago': (today.date() - r[3].date()).days if r[3] else 0}

    feedbacks = [_row('sf_', r) for r in sat_rows] + [_row('bf_', r) for r in bad_rows]
    feedbacks.sort(key=lambda f: f['days_ago'])
    new_reports_count = sum(1 for f in feedbacks if f['days_ago'] == 0)

    return render_template('reports.html', feedbacks=feedbacks, new_reports_count=new_reports_count,
                            unanswered_count=unanswered_count)


@bp.route('/reports_count')
def reports_count():
    if not admin_required():
        return jsonify({'count': 0}), 401
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM satisfaction_feedback WHERE rating='thumbs_down'")
    sat_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM bad_feedback")
    bad_count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return jsonify({'count': sat_count + bad_count})


@bp.route('/delete_feedback/<string:fid>', methods=['POST'])
@csrf.exempt
def delete_feedback(fid):
    """fid is now prefixed ('sf_123' / 'bf_45') by the merged /reports
    view above, so this knows which of the two tables to delete from."""
    if not admin_required():
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    if fid.startswith('sf_'):
        table, raw_id = 'satisfaction_feedback', fid[3:]
    elif fid.startswith('bf_'):
        table, raw_id = 'bad_feedback', fid[3:]
    else:
        return jsonify({'status': 'error', 'message': 'Unknown feedback id'}), 400
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(f"DELETE FROM {table} WHERE id=%s", (raw_id,))
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


@bp.route('/api/admin/add_correction', methods=['POST'])
@csrf.exempt
def add_correction():
    """
    The "fix it directly" feature: takes a question + the correct answer
    (typically typed in right next to a reported entry on /reports) and
    saves it as a permanent override in answer_corrections. From then on,
    that exact question is answered from this correction FIRST -- before
    FAQ matching, handbook retrieval, date logic, or the success cache
    (see services/faq.py's find_answer()) -- reworded naturally per
    question/language through the same extraction pipeline used for FAQ
    answers, but grounded strictly in the corrected text.

    If `source_report_id` is included (the prefixed id from the merged
    /reports view, e.g. "sf_123" or "bf_45"), that report is deleted once
    the correction is saved -- fixing a reported answer resolves it.
    """
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    d = request.get_json(silent=True) or request.form
    question = (d.get('question') or '').strip()
    answer = (d.get('answer') or '').strip()
    if not question or not answer:
        return jsonify({'error': 'question and answer are required'}), 400

    from services.corrections import save_correction
    faq_id = d.get('faq_id')
    ok = save_correction(
        get_db, question, answer,
        created_by=session.get('admin_user', 'admin'),
        faq_id=int(faq_id) if faq_id else None,
        source_report_id=d.get('source_report_id'),
    )
    if not ok:
        return jsonify({'error': 'Could not save correction'}), 500

    source_report_id = (d.get('source_report_id') or '').strip()
    if source_report_id.startswith('sf_'):
        _delete_report_row('satisfaction_feedback', source_report_id[3:])
    elif source_report_id.startswith('bf_'):
        _delete_report_row('bad_feedback', source_report_id[3:])

    return jsonify({'success': True})


def _delete_report_row(table, raw_id):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(f"DELETE FROM {table} WHERE id=%s", (raw_id,))
        conn.commit()
    except Exception as e:
        logger.warning(f"add_correction: could not auto-resolve source report: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


@bp.route('/corrections')
def corrections_management():
    """Admin page listing every standing correction, so it's easy to see
    (and clean up) what's currently overriding normal matching."""
    if not admin_required():
        return redirect('/admin')
    from services.corrections import list_corrections
    return render_template('corrections_management.html', corrections=list_corrections(get_db))


@bp.route('/api/admin/corrections/<int:correction_id>', methods=['DELETE'])
@csrf.exempt
def delete_correction_route(correction_id):
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    from services.corrections import delete_correction
    ok = delete_correction(get_db, correction_id)
    if not ok:
        return jsonify({'error': 'Correction not found'}), 404
    return jsonify({'success': True})


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
    cur.execute("""
        SELECT topic, question
        FROM faq
        WHERE topic IS NOT NULL AND topic != ''
        ORDER BY topic, priority_score DESC NULLS LAST
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    grouped = {}
    for topic, question in rows:
        grouped.setdefault(topic, []).append({'question': question})
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

@bp.route('/api/admin/mass_insert_handbook_faqs', methods=['POST'])
@csrf.exempt
def api_mass_insert_handbook_faqs():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json(silent=True) or {}
    text = data.get('text') or ''
    retire = bool(data.get('retire_previous', True))
    faqs = parse_pasted_faqs(text)
    if not faqs:
        return jsonify(success=False, error='No valid FAQ pairs found'), 400
    n = mass_insert_faqs(get_db, faqs, retire_previous=retire)
    return jsonify(success=True, inserted=n, message=f'Inserted {n} handbook FAQs')



# ---- Topic aliases (learning layer) ----
@bp.route("/api/admin/topic_aliases", methods=["GET"])
def api_list_topic_aliases():
    if not admin_required():
        return jsonify({"error": "Unauthorized"}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, topic_key, match_pattern, title_needles, content_needles,
                   source_priority, priority, hit_count, is_active, notes
            FROM handbook_topic_aliases
            ORDER BY priority DESC, id ASC
            """
        )
        rows = cur.fetchall()
        return jsonify({
            "aliases": [{
                "id": r[0], "topic_key": r[1], "match_pattern": r[2],
                "title_needles": list(r[3] or []), "content_needles": list(r[4] or []),
                "source_priority": r[5], "priority": r[6], "hit_count": r[7],
                "is_active": bool(r[8]), "notes": r[9],
            } for r in rows]
        })
    finally:
        cur.close()
        conn.close()


@bp.route("/api/admin/topic_aliases", methods=["POST"])
@csrf.exempt
def api_create_topic_alias():
    if not admin_required():
        return jsonify({"error": "Unauthorized"}), 403
    data = request.get_json(silent=True) or {}
    if not data.get("topic_key") or not data.get("match_pattern"):
        return jsonify({"error": "topic_key and match_pattern required"}), 400
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO handbook_topic_aliases
                (topic_key, match_pattern, title_needles, content_needles,
                 source_priority, priority, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                data["topic_key"][:80],
                data["match_pattern"][:500],
                data.get("title_needles") or [],
                data.get("content_needles") or [],
                data.get("source_priority") or "handbook_only",
                int(data.get("priority") or 50),
                data.get("notes"),
            ),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        try:
            from services.topic_aliases import invalidate_alias_cache
            invalidate_alias_cache()
        except Exception:
            pass
        return jsonify({"success": True, "id": new_id})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route("/api/admin/topic_aliases/<int:alias_id>", methods=["PUT"])
@csrf.exempt
def api_update_topic_alias(alias_id):
    if not admin_required():
        return jsonify({"error": "Unauthorized"}), 403
    data = request.get_json(silent=True) or {}
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE handbook_topic_aliases SET
                topic_key = COALESCE(%s, topic_key),
                match_pattern = COALESCE(%s, match_pattern),
                title_needles = COALESCE(%s, title_needles),
                content_needles = COALESCE(%s, content_needles),
                source_priority = COALESCE(%s, source_priority),
                priority = COALESCE(%s, priority),
                is_active = COALESCE(%s, is_active),
                notes = COALESCE(%s, notes),
                updated_at = NOW()
            WHERE id = %s
            """,
            (
                data.get("topic_key"),
                data.get("match_pattern"),
                data.get("title_needles"),
                data.get("content_needles"),
                data.get("source_priority"),
                data.get("priority"),
                data.get("is_active"),
                data.get("notes"),
                alias_id,
            ),
        )
        conn.commit()
        try:
            from services.topic_aliases import invalidate_alias_cache
            invalidate_alias_cache()
        except Exception:
            pass
        return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route("/api/admin/topic_aliases/<int:alias_id>", methods=["DELETE"])
@csrf.exempt
def api_delete_topic_alias(alias_id):
    if not admin_required():
        return jsonify({"error": "Unauthorized"}), 403
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE handbook_topic_aliases SET is_active = 0 WHERE id = %s", (alias_id,))
        conn.commit()
        try:
            from services.topic_aliases import invalidate_alias_cache
            invalidate_alias_cache()
        except Exception:
            pass
        return jsonify({"success": True})
    finally:
        cur.close()
        conn.close()


@bp.route("/api/admin/alias_suggestions")
def api_alias_suggestions():
    if not admin_required():
        return jsonify({"error": "Unauthorized"}), 403
    from services.topic_aliases import suggest_aliases_from_unanswered
    tips = suggest_aliases_from_unanswered(get_db, min_count=3, limit=30)
    return jsonify({"suggestions": tips})


@bp.route("/api/admin/seed_aliases", methods=["POST"])
@csrf.exempt
def api_seed_aliases():
    if not admin_required():
        return jsonify({"error": "Unauthorized"}), 403
    from services.topic_aliases import seed_default_aliases
    n = seed_default_aliases(get_db, force=bool((request.get_json(silent=True) or {}).get("force")))
    return jsonify({"seeded": n})


@bp.route('/api/admin/search_sources')
def api_admin_search_sources():
    """Search FAQ + handbook for the Reports/Corrections fix modals.

    Query param `q` is required. Returns up to ~10 FAQ hits and ~10 handbook
    chunks so an admin can pick grounded text instead of typing from scratch.
    """
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    q = (request.args.get('q') or '').strip()
    if len(q) < 2:
        return jsonify({'faqs': [], 'handbook': []})

    conn = get_db()
    cur = conn.cursor()
    faqs, handbook = [], []
    try:
        like = f'%{q}%'
        try:
            # FAQ table may not have is_active — fall back without it if needed
            cur.execute("""
                SELECT id, question, answer, category
                FROM faq
                WHERE (question ILIKE %s OR answer ILIKE %s
                       OR COALESCE(category,'') ILIKE %s
                       OR COALESCE(topic,'') ILIKE %s)
                ORDER BY
                  CASE WHEN question ILIKE %s THEN 0 ELSE 1 END,
                  id DESC
                LIMIT 10
            """, (like, like, like, like, f'{q}%'))
            for r in cur.fetchall():
                ans = r[2] or ''
                faqs.append({
                    'id': r[0],
                    'question': r[1] or '',
                    'answer': ans,
                    'preview': (ans[:220] + '…') if len(ans) > 220 else ans,
                    'category': r[3] or '',
                })
        except Exception as e:
            logger.warning(f"search_sources faq: {e}")
            conn.rollback()

        try:
            cur.execute("""
                SELECT id, section_title, chapter, page_start, content
                FROM handbook_chunks
                WHERE is_active = 1
                  AND (section_title ILIKE %s OR content ILIKE %s OR chapter ILIKE %s
                       OR COALESCE(keywords,'') ILIKE %s)
                ORDER BY
                  CASE WHEN section_title ILIKE %s THEN 0 ELSE 1 END,
                  chunk_order ASC NULLS LAST, id ASC
                LIMIT 10
            """, (like, like, like, like, f'{q}%'))
            for r in cur.fetchall():
                content = r[4] or ''
                handbook.append({
                    'id': r[0],
                    'section_title': r[1] or '',
                    'chapter': r[2] or '',
                    'page_start': r[3],
                    'content': content,
                    'preview': (content[:220] + '…') if len(content) > 220 else content,
                })
        except Exception as e:
            logger.warning(f"search_sources handbook: {e}")
            conn.rollback()

        return jsonify({'faqs': faqs, 'handbook': handbook})
    finally:
        cur.close()
        conn.close()

@bp.route('/api/admin/delete_faqs', methods=['POST'])
@csrf.exempt
def delete_faqs_bulk():
    if not admin_required():
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json(silent=True) or {}
    uuids = data.get('uuids') or data.get('ids') or []
    if not isinstance(uuids, list):
        return jsonify({'error': 'uuids must be a list'}), 400
    uuids = [str(u).strip() for u in uuids if u and str(u).strip() not in ('None', 'null')]
    if not uuids:
        return jsonify({'error': 'No valid UUIDs provided'}), 400
    if len(uuids) > 200:
        return jsonify({'error': 'Max 200 FAQs per request'}), 400

    conn = get_db()
    cur = conn.cursor()
    deleted = []
    try:
        for public_uuid in uuids:
            cur.execute("DELETE FROM faq WHERE public_uuid = %s", (public_uuid,))
            if cur.rowcount > 0:
                deleted.append(public_uuid)
        conn.commit()
        if deleted:
            faq_cache.invalidate()
            invalidate_response_tables()
            invalidate_dashboard_cache()
        return jsonify({'success': True, 'deleted': deleted, 'count': len(deleted)})
    except Exception as e:
        conn.rollback()
        logger.error(f"Bulk delete FAQ error: {e}")
        return jsonify({'error': str(e)}), 500
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