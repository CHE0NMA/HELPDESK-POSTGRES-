"""Page routes that render HTML (landing page, chat UI, chat-by-uuid).

Preserves ?q= across /chat → /new_chat → /chat/<uuid> so landing-page
question cards auto-ask in chat. Passes `prefill` into chat.html.
"""
import logging
from datetime import datetime, timedelta
from urllib.parse import urlencode

from flask import Blueprint, redirect, render_template, request, send_from_directory, session

from database import get_db
from services.formatting import format_bot_message

logger = logging.getLogger("pac_helpdesk.routes.pages")
bp = Blueprint('pages', __name__)

SESSION_TIMEOUT_HOURS = 6


def _touch_activity():
    session['last_activity'] = datetime.utcnow().isoformat()


def _session_expired() -> bool:
    last = session.get('last_activity')
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
    except (ValueError, TypeError):
        return False
    return datetime.utcnow() - last_dt > timedelta(hours=SESSION_TIMEOUT_HOURS)


def _reset_session_for_new_chat():
    from services.conversation import conversation_contexts
    sid = session.get('persistent_session_id')
    if sid and sid in conversation_contexts:
        del conversation_contexts[sid]
    for k in ['last_topic', 'last_intent', 'last_category', 'conversation_turn',
              'persistent_session_id', 'chat_session', 'current_chat_uuid', 'last_activity']:
        session.pop(k, None)


def _prefill_query_string() -> str:
    """Keep q / prefill across redirects."""
    q = (request.args.get('q') or request.args.get('prefill') or '').strip()
    if not q:
        return ''
    return '?' + urlencode({'q': q[:500]})


@bp.route('/')
def landing():
    return render_template('landing.html')


@bp.route('/uploads/<filename>')
def uploaded_file(filename):
    if '..' in filename or filename.startswith('/'):
        return {'error': 'Invalid filename'}, 400
    from flask import current_app
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename, as_attachment=False)


@bp.route('/chat')
def chat_ui():
    qs = _prefill_query_string()
    if _session_expired():
        _reset_session_for_new_chat()
        return redirect('/new_chat' + qs)

    session_id = session.get('persistent_session_id')
    chat_uuid = session.get('current_chat_uuid')

    if session_id and not chat_uuid:
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute("SELECT public_uuid FROM chat_sessions WHERE session_id = %s", (session_id,))
            row = cur.fetchone()
            if row:
                session['current_chat_uuid'] = row[0]
                return redirect(f'/chat/{row[0]}{qs}')
        except Exception as e:
            logger.error(f"chat_ui error: {e}")
        finally:
            cur.close()
            conn.close()

    if chat_uuid:
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute("SELECT session_id FROM chat_sessions WHERE public_uuid = %s", (chat_uuid,))
            if cur.fetchone():
                return redirect(f'/chat/{chat_uuid}{qs}')
        except Exception as e:
            logger.error(f"chat_ui error: {e}")
        finally:
            cur.close()
            conn.close()

    return redirect('/new_chat' + qs)


@bp.route('/new_chat')
def new_chat():
    import uuid
    from services.conversation import conversation_contexts

    qs = _prefill_query_string()

    sid = session.get('persistent_session_id')
    if sid and sid in conversation_contexts:
        del conversation_contexts[sid]
    for k in ['last_topic', 'last_intent', 'last_category', 'conversation_turn']:
        session.pop(k, None)

    chat_uuid = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_sessions (session_id, public_uuid, ip_address, user_agent)
            VALUES (%s, %s, %s, %s)
        """, (session_id, chat_uuid, request.remote_addr, request.headers.get('User-Agent', '')))
        conn.commit()
    except Exception as e:
        logger.error(f"new_chat error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    session['persistent_session_id'] = session_id
    session['chat_session'] = session_id
    session['current_chat_uuid'] = chat_uuid
    session.permanent = True
    _touch_activity()

    return redirect(f'/chat/{chat_uuid}{qs}')


@bp.route('/chat/<string:chat_uuid>')
def chat_with_uuid(chat_uuid):
    qs_prefill = (request.args.get('q') or request.args.get('prefill') or '').strip()[:500]

    if _session_expired():
        _reset_session_for_new_chat()
        extra = ('?' + urlencode({'q': qs_prefill})) if qs_prefill else ''
        return redirect('/new_chat' + extra)

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT session_id FROM chat_sessions WHERE public_uuid = %s", (chat_uuid,))
        row = cur.fetchone()
        if not row:
            return render_template('404.html'), 404

        session_id = row[0]
        cur.execute(
            "SELECT user_message, bot_response FROM chat_history "
            "WHERE session_id=%s ORDER BY created_at ASC LIMIT 100",
            (session_id,)
        )
        raw_chats = cur.fetchall()

        chats = []
        for user_message, bot_response in raw_chats:
            try:
                bot_html = format_bot_message(bot_response, show_images=True)
            except Exception as fmt_err:
                logger.error(f"format_bot_message failed for a history row: {fmt_err}", exc_info=True)
                bot_html = bot_response or ''
            chats.append({'user': user_message, 'bot': bot_html})

        session['current_chat_uuid'] = chat_uuid
        session['persistent_session_id'] = session_id
        _touch_activity()

        return render_template(
            'chat.html',
            chats=chats,
            chat_uuid=chat_uuid,
            prefill=qs_prefill,
        )
    except Exception as e:
        logger.error(f"chat_with_uuid error: {e}", exc_info=True)
        return redirect('/chat')
    finally:
        cur.close()
        conn.close()


@bp.route('/privacy')
def privacy_policy():
    return render_template('privacy.html')


@bp.route('/terms')
def terms_of_use():
    return render_template('terms.html')