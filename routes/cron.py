"""Cron endpoints for Vercel (and any external cron caller).
All jobs that used to live in APScheduler run here, behind CRON_SECRET.
"""
import logging
import os

from flask import Blueprint, jsonify, request

from extensions import csrf

logger = logging.getLogger("pac_helpdesk.routes.cron")
bp = Blueprint("cron", __name__)

CRON_SECRET = os.environ.get("CRON_SECRET", "")


def _authorized() -> bool:
    if not CRON_SECRET:
        # Fail closed in production; allow empty secret only in local dev
        if os.environ.get("FLASK_ENV") == "production" or os.environ.get("VERCEL"):
            return False
        return True
    header = request.headers.get("Authorization", "")
    token = header.replace("Bearer ", "").strip()
    # Vercel Cron sends: Authorization: Bearer <CRON_SECRET>
    # Also accept ?secret= for simple external cron services
    q = request.args.get("secret", "")
    return token == CRON_SECRET or q == CRON_SECRET


@bp.route("/api/cron/maintenance", methods=["GET", "POST"])
@csrf.exempt
def cron_maintenance():
    """Single combined job — best for free-tier (1 cron per day)."""
    if not _authorized():
        return jsonify({"error": "Unauthorized"}), 401

    results = {}
    try:
        from services.alerts import (
            batch_process_unanswered,
            check_stale_faqs,
            promote_high_frequency_faqs,
            check_frequent_missing_dates,
        )
        from utils.security import cleanup_old_sessions

        try:
            batch_process_unanswered()
            results["batch_unanswered"] = "ok"
        except Exception as e:
            results["batch_unanswered"] = str(e)
            logger.error(f"batch_unanswered: {e}")

        try:
            check_stale_faqs()
            results["stale_faqs"] = "ok"
        except Exception as e:
            results["stale_faqs"] = str(e)
            logger.error(f"stale_faqs: {e}")

        try:
            promote_high_frequency_faqs()
            results["promote_faqs"] = "ok"
        except Exception as e:
            results["promote_faqs"] = str(e)

        try:
            check_frequent_missing_dates()
            results["missing_dates"] = "ok"
        except Exception as e:
            results["missing_dates"] = str(e)

        try:
            cleanup_old_sessions()
            results["cleanup_sessions"] = "ok"
        except Exception as e:
            results["cleanup_sessions"] = str(e)

        # BUG FIX: on Vercel, app.py's APScheduler block (the only place
        # run_self_heal_pass was previously wired up) never runs at all --
        # so the "answer it correctly enough times and it fixes itself"
        # loop was silently dead in production unless an admin manually
        # clicked "Run now" on /self_heal. This cron job is the Vercel-safe
        # equivalent of the interval job database/learning_schema.py sets
        # up for non-serverless deployments.
        try:
            from database import get_db
            from services.self_heal import run_self_heal_pass
            n = run_self_heal_pass(get_db, min_count=2, limit=50)
            results["self_heal"] = f"ok ({n} draft(s))"
        except Exception as e:
            results["self_heal"] = str(e)
            logger.error(f"self_heal: {e}")

        return jsonify({"ok": True, "results": results})
    except Exception as e:
        logger.error(f"cron_maintenance: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.route("/api/cron/batch_unanswered", methods=["GET", "POST"])
@csrf.exempt
def cron_batch_unanswered():
    if not _authorized():
        return jsonify({"error": "Unauthorized"}), 401
    from services.alerts import batch_process_unanswered
    batch_process_unanswered()
    return jsonify({"ok": True})


@bp.route("/api/cron/stale_faqs", methods=["GET", "POST"])
@csrf.exempt
def cron_stale_faqs():
    if not _authorized():
        return jsonify({"error": "Unauthorized"}), 401
    from services.alerts import check_stale_faqs
    check_stale_faqs()
    return jsonify({"ok": True})


@bp.route("/api/cron/self_heal", methods=["GET", "POST"])
@csrf.exempt
def cron_self_heal():
    if not _authorized():
        return jsonify({"error": "Unauthorized"}), 401
    from database import get_db
    from services.self_heal import run_self_heal_pass
    n = run_self_heal_pass(get_db, min_count=2, limit=50)
    return jsonify({"ok": True, "drafts_created_or_updated": n})

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