"""App factory. Compare this to the old 2500+ line app.py — everything
that used to live here (AI provider calls, DB pool, date logic, handbook
logic, FAQ logic, routes) now lives in ai_providers/, database/, services/,
and routes/. This file's only job is to wire them together.

--- Postgres migration note ---
Added a call to database.schema.ensure_handbook_fulltext_search() in the
startup sequence below — it creates the tsvector column + GIN index that
services/handbook.py's search query now depends on. Safe to run every
boot (no-ops once already applied). Everything else here was already
portable (init_pool()/ensure_admin_exists()/etc. are the same function
names, just backed by psycopg2 now).
"""
import logging
import os
import signal
import threading
import time
from datetime import timedelta
from logging.handlers import RotatingFileHandler
from threading import Thread

from flask import Flask, request, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from apscheduler.schedulers.background import BackgroundScheduler

from config import UPLOAD_FOLDER, validate_config
from extensions import csrf
from database import init_pool
from database.schema import (
    ensure_admin_exists, ensure_schedule_schema, migrate_create_chat_sessions,
    migrate_add_uuid_to_faq, ensure_handbook_fulltext_search, ensure_faq_embedding_column,
)
from services.alerts import (
    batch_process_unanswered, check_stale_faqs, promote_high_frequency_faqs,
    check_frequent_missing_dates,
)
from services.language import warm_response_tables
from utils.security import cleanup_old_sessions

from routes.chat import bp as chat_bp
from routes.admin import bp as admin_bp
from routes.schedules import bp as schedules_bp
from routes.api import bp as api_bp
from routes.pages import bp as pages_bp


def create_app():
    if not os.environ.get('FLASK_SECRET_KEY'):
        raise EnvironmentError(
            "FLASK_SECRET_KEY must be set. Generate one with:\n"
            "python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )

    app = Flask(__name__)
    app.start_time = time.time()
    app.secret_key = os.environ.get('FLASK_SECRET_KEY')

    if not app.debug:
        handler = RotatingFileHandler('pac_helpdesk.log', maxBytes=10485760, backupCount=10)
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(fmt)
        handler.setLevel(logging.INFO)
        app.logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        console.setLevel(logging.INFO)
        app.logger.addHandler(console)

    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        SESSION_COOKIE_SECURE=os.environ.get('FLASK_ENV') == 'production',
        PERMANENT_SESSION_LIFETIME=timedelta(hours=int(os.environ.get('SESSION_TIMEOUT_HOURS', 168))),
        WTF_CSRF_TIME_LIMIT=3600,
        UPLOAD_FOLDER=UPLOAD_FOLDER,
        BACKUP_FOLDER=os.path.join(os.path.dirname(__file__), 'backups'),
        PREFERRED_URL_SCHEME='https' if os.environ.get('FLASK_ENV') == 'production' else 'http',
        SEND_FILE_MAX_AGE_DEFAULT=86400,
    )
    os.makedirs(app.config['BACKUP_FOLDER'], exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    csrf.init_app(app)  # routes that must stay exempt (JSON APIs posted
                        # from JS without a CSRF token) call @csrf.exempt
                        # directly above their own @bp.route — see
                        # routes/chat.py, routes/admin.py, routes/schedules.py

    Limiter(get_remote_address, app=app,
            default_limits=[os.environ.get('RATE_LIMIT_DEFAULT', '200 per day;50 per hour')],
            storage_uri="memory://")

    @app.after_request
    def security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        return response

    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(schedules_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(pages_bp)

    init_pool()
    ensure_admin_exists()
    ensure_schedule_schema()
    migrate_create_chat_sessions()
    migrate_add_uuid_to_faq()
    ensure_handbook_fulltext_search()
    ensure_faq_embedding_column()
    Thread(target=warm_response_tables, daemon=True).start()

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=check_stale_faqs, trigger="interval", days=1)
    scheduler.add_job(func=promote_high_frequency_faqs, trigger="interval", days=7)
    scheduler.add_job(func=batch_process_unanswered, trigger="interval", minutes=30)
    scheduler.add_job(func=check_frequent_missing_dates, trigger="interval", minutes=30)
    scheduler.add_job(func=cleanup_old_sessions, trigger="interval", days=7)
    scheduler.start()
    app.scheduler = scheduler

    if threading.current_thread() is threading.main_thread():
        def graceful_shutdown(signum, frame):
            app.logger.info("Shutdown signal received…")
            scheduler.shutdown()
            os._exit(0)
        signal.signal(signal.SIGTERM, graceful_shutdown)
        signal.signal(signal.SIGINT, graceful_shutdown)

    return app


if __name__ == '__main__':
    validate_config()
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    print("=" * 55)
    print("PAC Helpdesk v4.0 — modular edition")
    print(f"   URL : http://{'0.0.0.0' if debug_mode else 'localhost'}:{port}")
    print(f"   Mode: {'Development' if debug_mode else 'Production'}")
    print("=" * 55)
    app.run(host="0.0.0.0" if debug_mode else "127.0.0.1", port=port, debug=debug_mode, threaded=True)