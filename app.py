"""App factory — Vercel + Render Postgres compatible.
On Vercel: no APScheduler, no background threads, no file logging.
"""
import logging
import os
import time
from datetime import timedelta

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

from config import UPLOAD_FOLDER, validate_config
from extensions import csrf
from database import init_pool
from database.schema import (
    ensure_core_tables, ensure_admin_exists, ensure_schedule_schema,
    migrate_create_chat_sessions, migrate_add_uuid_to_faq,
    ensure_handbook_fulltext_search, ensure_faq_embedding_column,
    ensure_learning_layer,
)
from services.language import warm_response_tables

from routes.chat import bp as chat_bp
from routes.admin import bp as admin_bp
from routes.schedules import bp as schedules_bp
from routes.api import bp as api_bp
from routes.pages import bp as pages_bp
from routes.cron import bp as cron_bp  # NEW


def _is_vercel() -> bool:
    return bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"))


def create_app():
    if not os.environ.get("FLASK_SECRET_KEY"):
        raise EnvironmentError(
            "FLASK_SECRET_KEY must be set. Generate one with:\n"
            "python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

    app = Flask(__name__)
    app.start_time = time.time()
    app.secret_key = os.environ.get("FLASK_SECRET_KEY")

    # Logging: console only on Vercel (no persistent disk)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)
    app.logger.handlers.clear()
    app.logger.addHandler(console)
    app.logger.setLevel(logging.INFO)

    if not _is_vercel() and not app.debug:
        try:
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                "pac_helpdesk.log", maxBytes=10485760, backupCount=10
            )
            handler.setFormatter(fmt)
            handler.setLevel(logging.INFO)
            app.logger.addHandler(handler)
        except Exception:
            pass

    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
        PERMANENT_SESSION_LIFETIME=timedelta(
            hours=int(os.environ.get("SESSION_TIMEOUT_HOURS", 168))
        ),
        WTF_CSRF_TIME_LIMIT=3600,
        UPLOAD_FOLDER=UPLOAD_FOLDER,
        BACKUP_FOLDER=os.path.join(os.path.dirname(__file__), "backups"),
        PREFERRED_URL_SCHEME=(
            "https" if os.environ.get("FLASK_ENV") == "production" else "http"
        ),
        SEND_FILE_MAX_AGE_DEFAULT=86400,
    )

    if not _is_vercel():
        os.makedirs(app.config["BACKUP_FOLDER"], exist_ok=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    csrf.init_app(app)

    Limiter(
        get_remote_address,
        app=app,
        default_limits=[
            os.environ.get("RATE_LIMIT_DEFAULT", "200 per day;50 per hour")
        ],
        storage_uri="memory://",
    )

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(schedules_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(pages_bp)
    app.register_blueprint(cron_bp)

    # DB + schema (idempotent — safe on every cold start)
    init_pool()
    ensure_core_tables()
    ensure_admin_exists()
    ensure_schedule_schema()
    migrate_create_chat_sessions()
    migrate_add_uuid_to_faq()
    ensure_handbook_fulltext_search()
    ensure_faq_embedding_column()
    ensure_learning_layer()

    # Skip heavy warm-up threads on Vercel
    if not _is_vercel():
        from threading import Thread
        Thread(target=warm_response_tables, daemon=True).start()

        # Local / traditional host only: APScheduler
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from services.alerts import (
                batch_process_unanswered,
                check_stale_faqs,
                promote_high_frequency_faqs,
                check_frequent_missing_dates,
            )
            from utils.security import cleanup_old_sessions

            scheduler = BackgroundScheduler()
            scheduler.add_job(func=check_stale_faqs, trigger="interval", days=1)
            scheduler.add_job(
                func=promote_high_frequency_faqs, trigger="interval", days=7
            )
            scheduler.add_job(
                func=batch_process_unanswered, trigger="interval", minutes=30
            )
            scheduler.add_job(
                func=check_frequent_missing_dates, trigger="interval", minutes=30
            )
            scheduler.add_job(
                func=cleanup_old_sessions, trigger="interval", days=7
            )
            scheduler.start()
            app.scheduler = scheduler
        except Exception as e:
            app.logger.warning(f"Scheduler not started: {e}")

    return app


# Vercel + local entrypoint: must expose top-level `app`
app = create_app()


if __name__ == "__main__":
    validate_config()
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    print("=" * 55)
    print("PAC Helpdesk — modular edition")
    print(f"   URL : http://0.0.0.0:{port}")
    print(f"   Mode: {'Development' if debug_mode else 'Production'}")
    print(f"   Vercel: {_is_vercel()}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)