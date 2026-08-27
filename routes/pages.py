"""Page routes that render HTML (landing page, chat UI, chat-by-uuid).

Preserves ?q= across /chat → /new_chat → /chat/<uuid> so landing-page
question cards auto-ask in chat. Passes `prefill` into chat.html.
"""
import logging
import uuid
from datetime import datetime, timedelta
from urllib.parse import urlencode

from flask import Blueprint, redirect, render_template, request, send_from_directory, session

from database import get_db
from services.formatting import format_bot_message

logger = logging.getLogger("pac_helpdesk.routes.pages")
bp = Blueprint("pages", __name__)

SESSION_TIMEOUT_HOURS = 6


def _touch_activity():
    try:
        session["last_activity"] = datetime.utcnow().isoformat()
    except Exception:
        pass


def _session_expired() -> bool:
    last = session.get("last_activity")
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
    except (ValueError, TypeError):
        return False
    try:
        return datetime.utcnow() - last_dt > timedelta(hours=SESSION_TIMEOUT_HOURS)
    except Exception:
        return False


def _reset_session_for_new_chat():
    try:
        from services.conversation import delete_ctx
        sid = session.get("persistent_session_id")
        if sid:
            delete_ctx(sid)
    except Exception as e:
        logger.debug(f"_reset_session_for_new_chat delete_ctx: {e}")
    for k in [
        "last_topic", "last_intent", "last_category", "conversation_turn",
        "persistent_session_id", "chat_session", "current_chat_uuid", "last_activity",
    ]:
        session.pop(k, None)


def _prefill_query_string() -> str:
    q = (request.args.get("q") or request.args.get("prefill") or "").strip()
    if not q:
        return ""
    return "?" + urlencode({"q": q[:500]})


@bp.route("/")
def landing():
    try:
        return render_template("landing.html")
    except Exception as e:
        logger.error(f"landing error: {e}", exc_info=True)
        return (
            "<!doctype html><html><body>"
            "<h1>PAC Helpdesk</h1>"
            "<p>Temporary load issue. <a href='/'>Refresh</a> or "
            "<a href='/new_chat'>start chat</a>.</p>"
            "</body></html>"
        ), 200


@bp.route("/uploads/<filename>")
def uploaded_file(filename):
    if ".." in filename or filename.startswith("/"):
        return {"error": "Invalid filename"}, 400
    from flask import current_app
    return send_from_directory(
        current_app.config["UPLOAD_FOLDER"], filename, as_attachment=False
    )


@bp.route("/chat")
def chat_ui():
    try:
        qs = _prefill_query_string()
        if _session_expired():
            _reset_session_for_new_chat()
            return redirect("/new_chat" + qs)

        session_id = session.get("persistent_session_id")
        chat_uuid = session.get("current_chat_uuid")

        if session_id and not chat_uuid:
            try:
                conn = get_db()
                cur = conn.cursor()
                try:
                    cur.execute(
                        "SELECT public_uuid FROM chat_sessions WHERE session_id = %s",
                        (session_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        session["current_chat_uuid"] = row[0]
                        return redirect(f"/chat/{row[0]}{qs}")
                finally:
                    cur.close()
                    conn.close()
            except Exception as e:
                logger.error(f"chat_ui lookup by session_id: {e}")

        if chat_uuid:
            try:
                conn = get_db()
                cur = conn.cursor()
                try:
                    cur.execute(
                        "SELECT session_id FROM chat_sessions WHERE public_uuid = %s",
                        (chat_uuid,),
                    )
                    if cur.fetchone():
                        return redirect(f"/chat/{chat_uuid}{qs}")
                finally:
                    cur.close()
                    conn.close()
            except Exception as e:
                logger.error(f"chat_ui lookup by chat_uuid: {e}")

        return redirect("/new_chat" + qs)
    except Exception as e:
        logger.error(f"chat_ui fatal: {e}", exc_info=True)
        return redirect("/new_chat")


@bp.route("/new_chat")
def new_chat():
    qs = _prefill_query_string()
    try:
        try:
            from services.conversation import delete_ctx
            sid = session.get("persistent_session_id")
            if sid:
                delete_ctx(sid)
        except Exception as e:
            logger.debug(f"new_chat delete_ctx: {e}")

        for k in ["last_topic", "last_intent", "last_category", "conversation_turn"]:
            session.pop(k, None)

        chat_uuid = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        try:
            conn = get_db()
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (session_id, public_uuid, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        chat_uuid,
                        request.remote_addr,
                        (request.headers.get("User-Agent") or "")[:500],
                    ),
                )
                conn.commit()
            except Exception as e:
                logger.error(f"new_chat INSERT chat_sessions: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                try:
                    cur.close()
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"new_chat get_db: {e}")

        try:
            session["persistent_session_id"] = session_id
            session["chat_session"] = session_id
            session["current_chat_uuid"] = chat_uuid
            session.permanent = True
            _touch_activity()
        except Exception as e:
            logger.error(f"new_chat session assign: {e}")

        return redirect(f"/chat/{chat_uuid}{qs}")
    except Exception as e:
        logger.error(f"new_chat fatal: {e}", exc_info=True)
        try:
            return render_template(
                "chat.html",
                chats=[],
                chat_uuid=str(uuid.uuid4()),
                prefill=(request.args.get("q") or "")[:500],
                session_id=str(uuid.uuid4()),
            )
        except Exception:
            return redirect("/")


@bp.route("/chat/<string:chat_uuid>")
def chat_with_uuid(chat_uuid):
    qs_prefill = (request.args.get("q") or request.args.get("prefill") or "").strip()[:500]

    try:
        if _session_expired():
            _reset_session_for_new_chat()
            extra = ("?" + urlencode({"q": qs_prefill})) if qs_prefill else ""
            return redirect("/new_chat" + extra)

        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT session_id FROM chat_sessions WHERE public_uuid = %s",
                (chat_uuid,),
            )
            row = cur.fetchone()
            if not row:
                return redirect(
                    "/new_chat" + (("?" + urlencode({"q": qs_prefill})) if qs_prefill else "")
                )

            session_id = row[0]
            chats = []
            try:
                cur.execute(
                    "SELECT user_message, bot_response FROM chat_history "
                    "WHERE session_id=%s ORDER BY created_at ASC LIMIT 100",
                    (session_id,),
                )
                raw_chats = cur.fetchall()
                for user_message, bot_response in raw_chats:
                    try:
                        bot_html = format_bot_message(bot_response, show_images=True)
                    except Exception as fmt_err:
                        logger.error(f"format_bot_message failed: {fmt_err}", exc_info=True)
                        bot_html = bot_response or ""
                    chats.append({"user": user_message, "bot": bot_html})
            except Exception as hist_err:
                logger.error(f"chat history load: {hist_err}")
                try:
                    conn.rollback()
                except Exception:
                    pass

            try:
                session["current_chat_uuid"] = chat_uuid
                session["persistent_session_id"] = session_id
                _touch_activity()
            except Exception:
                pass

            return render_template(
                "chat.html",
                chats=chats,
                chat_uuid=chat_uuid,
                prefill=qs_prefill,
                session_id=session_id,
            )
        finally:
            try:
                cur.close()
                conn.close()
            except Exception:
                pass
    except Exception as e:
        logger.error(f"chat_with_uuid error: {e}", exc_info=True)
        try:
            return render_template(
                "chat.html",
                chats=[],
                chat_uuid=chat_uuid or str(uuid.uuid4()),
                prefill=qs_prefill,
                session_id=session.get("persistent_session_id") or str(uuid.uuid4()),
            )
        except Exception:
            return redirect("/new_chat")


@bp.route("/privacy")
def privacy_policy():
    try:
        return render_template("privacy.html")
    except Exception:
        return redirect("/")


@bp.route("/terms")
def terms_of_use():
    try:
        return render_template("terms.html")
    except Exception:
        return redirect("/")

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
