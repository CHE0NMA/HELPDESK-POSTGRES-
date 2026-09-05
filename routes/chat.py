"""Student-facing chat endpoints.

--- Postgres migration notes ---
- MySQL `DATE_SUB(NOW(), INTERVAL 30 DAY)` -> Postgres
  `NOW() - INTERVAL '30 days'`.
- MySQL `MATCH(user_message) AGAINST (%s IN NATURAL LANGUAGE MODE)` has no
  direct Postgres equivalent. Rather than require a schema migration for
  chat_history (adding a stored tsvector column, like handbook_chunks
  got), related_suggestions() below builds the tsvector on the fly with
  `to_tsvector('english', user_message) @@ plainto_tsquery(...)` — slower
  without an index on a big table, but correct, and needs zero schema
  changes. If this table gets large, consider adding a generated
  `search_vector` column + GIN index the same way schema.py did for
  handbook_chunks, then swap this back to querying that column directly.
- MySQL `LIKE` is case-insensitive by default for non-binary text columns;
  Postgres `LIKE` is case-sensitive. Swapped the fallback query to `ILIKE`
  to keep the same "not case-sensitive" matching behavior.
"""
import logging
import re
import uuid
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request, session

from database import get_db
from extensions import csrf
from services import language as lang_service
from services.conversation import get_conversation_context, conversation_contexts
from services.faq import find_answer, log_unanswered_standalone
from services.formatting import format_bot_message
from utils.security import check_session_rate_limit
from utils.text import strip_html

logger = logging.getLogger("pac_helpdesk.routes.chat")
bp = Blueprint('chat', __name__)

SESSION_MAX_MESSAGES = 30

SESSION_TIMEOUT_HOURS = 6

GREETING_RX = re.compile(
    r'\b(hello|hi|hey|good morning|good afternoon|good evening|how are you'
    r'|kamusta|kumusta|magandang umaga|magandang hapon|magandang gabi|musta'
    r'|maayong buntag|maayong hapon|maayong gabii)\b', re.IGNORECASE)

MATH_PATTERNS = [
    r'\d+\s*[\+\*\/x]\s*\d+',
    # Subtraction, specifically excluded from matching a 4-digit year range
    # ("2025-2026", "SY 2025-2026") -- (?<!\d) / (?!\d) stop a partial match
    # from landing inside a longer digit run, and the explicit \d{4} block
    # on either side of the dash rules out exactly the "YYYY-YYYY" shape.
    r'(?<!\d)(?!\d{4}\s*-\s*\d{4}\b)\d+\s*-\s*\d+(?!\d)',
    r'\d+\s*times\s*\d+', r'what is \d+',
    r'calculate', r'math', r'plus', r'minus', r'times', r'divided by',
]


def _log_chat(session_id, user_msg, bot_response, category, detected_lang, faq_id=None):
    conn = get_db()
    cur = conn.cursor()
    try:
        chat_uuid = session.get('current_chat_uuid')
        cur.execute("""
            INSERT INTO chat_history
            (session_id, user_message, bot_response, category, detected_language, faq_id, chat_uuid, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """, (session_id, user_msg, bot_response[:2000], category, detected_lang, faq_id, chat_uuid))
        conn.commit()
    except Exception as e:
        logger.error(f"_log_chat error: {e}")
    finally:
        cur.close()
        conn.close()


def _should_show_image(question: str, answer: str, conversation_turn: int = 1) -> bool:
    if conversation_turn > 1 or len(answer) < 100:
        return False
    q = question.lower()
    visual_keywords = ['show', 'see', 'picture', 'image', 'photo', 'drawing', 'diagram',
                        'chart', 'graph', 'visual', 'look', 'view', 'example of', 'sample of',
                        'want to see', 'wanna see']
    return any(kw in q for kw in visual_keywords)


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


def _start_new_chat_session():
    """Same bookkeeping as GET /new_chat, callable mid-request so a
    message that arrives after 6+ idle hours lands in a fresh
    conversation instead of quietly bolting onto the old one.
    Returns the new chat_uuid.
    """
    old_sid = session.get('persistent_session_id')
    if old_sid and old_sid in conversation_contexts:
        del conversation_contexts[old_sid]
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
        logger.error(f"_start_new_chat_session error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    session['persistent_session_id'] = session_id
    session['chat_session'] = session_id
    session['current_chat_uuid'] = chat_uuid
    session.permanent = True
    return chat_uuid


@bp.route('/chat', methods=['POST'])
@csrf.exempt
def chat():
    try:
        started_new_chat = False
        if 'persistent_session_id' not in session or _session_expired():
            _start_new_chat_session()
            started_new_chat = True

        session_id = session['persistent_session_id']
        session['chat_session'] = session_id
        _touch_activity()

        allowed, _ = check_session_rate_limit(session_id, SESSION_MAX_MESSAGES)
        if not allowed:
            return jsonify({'reply': "You're sending messages too fast. Please wait a moment."})

        user_msg = (request.get_json(silent=True) or {}).get('message', '').strip()
        if not user_msg:
            return jsonify({'reply': 'Please type a message.'})

        # Pass the conversation's established language as a prior — matters
        # most for short replies ("opo", "3") that carry no language signal
        # on their own. See services/language.py's detect_language().
        ctx_for_lang = get_conversation_context(session_id)
        detected_lang = lang_service.detect_language(user_msg, prior_lang=ctx_for_lang.last_detected_lang)
        ctx_for_lang.last_detected_lang = detected_lang

        if any(re.search(p, user_msg.lower()) for p in MATH_PATTERNS):
            reply = "I can only answer PAC-related questions."
            _log_chat(session_id, user_msg, reply, 'Math', detected_lang)
            return jsonify({'reply': f'<div class="text">{reply}</div>', 'category': 'Math',
                             'chat_uuid': session.get('current_chat_uuid'), 'new_chat': started_new_chat})

        if GREETING_RX.search(user_msg) or user_msg.lower().strip() in ['hi', 'hello', 'hey', 'kamusta', 'kumusta', 'musta']:
            reply = lang_service.get_greeting(detected_lang)
            _log_chat(session_id, user_msg, reply, 'Greeting', detected_lang)
            get_conversation_context(session_id).update(user_msg=user_msg, bot_response=reply)
            return jsonify({'reply': f'<div class="text">{reply}</div>', 'category': 'Greeting',
                             'chat_uuid': session.get('current_chat_uuid'), 'new_chat': started_new_chat})


        result = find_answer(user_msg, return_topic=True, session_id=session_id)
        if isinstance(result, tuple):
            answer, topic = result[0], (result[1] if len(result) > 1 else 'Unknown')
            faq_id = result[2] if len(result) > 2 else None
            category = result[3] if len(result) > 3 else None
        else:
            answer, topic, faq_id, category = result, 'Unknown', None, None

        if topic and topic not in ('Unknown', 'Clarification', 'Disambiguation', 'Follow-up'):
            session['last_topic'] = topic
            if category:
                session['last_category'] = category

        _log_chat(session_id, user_msg, strip_html(answer), category or '', detected_lang, faq_id=faq_id)

        conversation_turn = session.get('conversation_turn', 0) + 1
        session['conversation_turn'] = conversation_turn
        # Format ALL replies (handbook, FAQ, schedule, etc.) the same way
        show_images = (
                topic not in ('Disambiguation',)
                and _should_show_image(user_msg, answer, conversation_turn)
        )
        formatted = format_bot_message(answer or '', show_images=show_images)

        return jsonify({'reply': f'<div class="text">{formatted}</div>', 'category': category,
                        'chat_uuid': session.get('current_chat_uuid'), 'new_chat': started_new_chat})

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({'reply': 'Sorry, an error occurred. Please try again.'})


@bp.route('/api/chat/stream', methods=['POST'])
@csrf.exempt
def chat_stream():
    from flask import Response, stream_with_context
    import json
    import time as _time

    data = request.get_json(silent=True) or {}
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'error': 'No message'}), 400

    def generate():
        yield f"data: {json.dumps({'type': 'start', 'message': 'Processing…'})}\n\n"
        result = find_answer(user_msg, return_topic=True)
        answer = result[0] if isinstance(result, tuple) else result
        for i in range(0, len(answer), 50):
            yield f"data: {json.dumps({'type': 'chunk', 'content': answer[i:i + 50]})}\n\n"
            _time.sleep(0.01)
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                     headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@bp.route('/api/satisfaction', methods=['POST'])
@csrf.exempt
def satisfaction_feedback():
    data = request.get_json(silent=True) or {}
    rating = data.get('rating')
    question = data.get('question')
    answer = strip_html(data.get('answer') or '')
    sid = data.get('session_id') or session.get('chat_session') or 'anonymous'
    if not rating or not question:
        return jsonify({'status': 'error', 'message': 'Missing data'}), 400
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO satisfaction_feedback (session_id, question, answer, rating, created_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (sid, question[:500], (answer or '')[:1000], rating))
        conn.commit()
        # Continuous-learning: lock in good answers / drop bad cache entries
        try:
            from services.answer_memory import promote_from_thumbs_up, record_feedback
            if rating == 'thumbs_up':
                promote_from_thumbs_up(
                    get_db, question, answer or '',
                    detected_lang='english', source_type='chat',
                )
            else:
                record_feedback(
                    get_db, question, 'thumbs_down',
                    session_id=sid, source_type='chat',
                )
        except Exception as mem_err:
            logger.debug(f"answer_memory hook: {mem_err}")
        if rating == 'thumbs_down':
            log_unanswered_standalone(question, session_id=sid)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Satisfaction feedback error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/api/chat_rating', methods=['POST'])
@csrf.exempt
def submit_chat_rating():
    data = request.get_json(silent=True) or {}
    rating = data.get('rating')
    comment = (data.get('comment') or '').strip()
    sid = data.get('session_id') or session.get('persistent_session_id') or 'anonymous'
    if not rating or not isinstance(rating, int) or not (1 <= rating <= 5):
        return jsonify({'status': 'error', 'message': 'Invalid rating'}), 400
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_ratings (session_id, rating, comment, exchange_count, detected_language, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
        """, (sid, rating, comment[:500] if comment else None,
              data.get('exchange_count', 0), data.get('detected_language', 'english')))
        conn.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Chat rating error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        cur.close()
        conn.close()


@bp.route('/feedback', methods=['POST'])
@csrf.exempt
def feedback():
    data = request.get_json(silent=True) or {}
    q, a = data.get('question'), strip_html(data.get('answer') or '')
    if q and a:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO bad_feedback (question, answer) VALUES (%s, %s)", (q, a))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 400


@bp.route('/suggested_questions')
def suggested_questions():
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT user_message, COUNT(*) as frequency
            FROM chat_history
            WHERE created_at >= NOW() - INTERVAL '30 days'
            GROUP BY user_message
            ORDER BY frequency DESC
            LIMIT 10
        """)
        rows = cur.fetchall()
        suggestions = []
        for row in rows:
            msg = row[0].strip()
            if len(msg) > 5 and msg.lower() not in ['hi', 'hello', 'hey', 'yes', 'no', 'ok', 'thanks', 'thank you']:
                suggestions.append({'user_message': msg, 'frequency': row[1]})
        return jsonify(suggestions)
    except Exception as e:
        logger.error(f"Suggested questions error: {e}")
        return jsonify([])
    finally:
        cur.close()
        conn.close()


@bp.route('/related_suggestions', methods=['POST'])
@csrf.exempt
def related_suggestions():
    data = request.get_json(silent=True) or {}
    user_msg = data.get('user_message', '').strip()
    category = data.get('category', '')
    if not user_msg:
        return jsonify([])

    conn = get_db()
    cur = conn.cursor()
    try:
        try:
            cur.execute("""
                SELECT user_message FROM (
                    SELECT user_message, MAX(created_at) AS last_asked
                    FROM chat_history
                    WHERE to_tsvector('english', user_message) @@ plainto_tsquery('english', %s)
                      AND user_message != %s AND LENGTH(user_message) > 5
                    GROUP BY user_message
                ) t
                ORDER BY last_asked DESC LIMIT 5
            """, (user_msg, user_msg))
            rows = cur.fetchall()
        except Exception:
            # Postgres aborts the whole transaction on a failed statement —
            # unlike MySQL, the connection can't run another query until
            # this is rolled back.
            conn.rollback()
            cur.execute("""
                SELECT user_message FROM (
                    SELECT user_message, MAX(created_at) AS last_asked
                    FROM chat_history
                    WHERE user_message ILIKE %s AND user_message != %s AND LENGTH(user_message) > 5
                    GROUP BY user_message
                ) t
                ORDER BY last_asked DESC LIMIT 5
            """, (f'%{user_msg[:20]}%', user_msg))
            rows = cur.fetchall()

        if rows:
            return jsonify([{'user_message': r[0]} for r in rows])

        if category and category != 'Unknown':
            cur.execute("""
                SELECT user_message FROM chat_history
                WHERE category = %s AND user_message != %s AND LENGTH(user_message) > 5
                ORDER BY created_at DESC LIMIT 5
            """, (category, user_msg))
            rows = cur.fetchall()
            if rows:
                seen = set()
                deduped = []
                for r in rows:
                    if r[0] not in seen:
                        seen.add(r[0])
                        deduped.append(r[0])
                return jsonify([{'user_message': m} for m in deduped])

        cur.execute("""
            SELECT user_message FROM (
                SELECT user_message, MAX(created_at) AS last_asked
                FROM chat_history
                WHERE user_message != %s AND LENGTH(user_message) > 5
                GROUP BY user_message
            ) t
            ORDER BY last_asked DESC LIMIT 5
        """, (user_msg,))
        rows = cur.fetchall()
        return jsonify([{'user_message': r[0]} for r in rows])
    except Exception as e:
        logger.error(f"Related suggestions error: {e}")
        return jsonify([])
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