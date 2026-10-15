"""App factory — Vercel + Render Postgres compatible.
On Vercel: no APScheduler, no background threads, no file logging.
"""
import logging
import os
from datetime import timedelta

from flask import Flask, jsonify, render_template, request
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
from routes.cron import bp as cron_bp


def _is_vercel() -> bool:
    return bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"))


def create_app():
    if not os.environ.get("FLASK_SECRET_KEY"):
        raise EnvironmentError(
            "FLASK_SECRET_KEY must be set. Generate one with:\n"
            "python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY")

    class FixVercelPath:
        """Vercel's rewrite ("/(.*)" -> "/app.py?__vercel_path__=/$1") always
        invokes this function with a literal PATH_INFO of "/app.py" — the
        various X-Vercel-*/X-Original-Url/REQUEST_URI headers this used to
        rely on are NOT reliably present on the current runtime, so that
        approach silently fell back to "/" for every request. Query strings
        ARE reliably passed through, so vercel.json now smuggles the real
        path in as ?__vercel_path__=..., and we recover it from there.
        """
        def __init__(self, app):
            self.app = app

        def __call__(self, environ, start_response):
            from urllib.parse import parse_qsl, urlencode, unquote

            path = environ.get("PATH_INFO", "") or ""
            qs = environ.get("QUERY_STRING", "") or ""

            if path in ("/app.py", "app.py") or path.endswith("/app.py"):
                pairs = parse_qsl(qs, keep_blank_values=True)
                real_path = None
                remaining = []
                for key, value in pairs:
                    if key == "__vercel_path__" and real_path is None:
                        real_path = value
                    else:
                        remaining.append((key, value))

                environ["PATH_INFO"] = unquote(real_path) if real_path else "/"
                environ["QUERY_STRING"] = urlencode(remaining)

            return self.app(environ, start_response)

    app.wsgi_app = FixVercelPath(app.wsgi_app)

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

    # CORS for the Flutter app. Native Android/iOS builds aren't subject to
    # browser CORS at all (only Flutter *Web* is), but this is what makes a
    # `flutter run -d chrome` build -- or any other cross-origin client --
    # work against this same backend without a proxy. Off by default (no
    # env var set = no behavior change from before); set
    # CORS_ALLOWED_ORIGINS to a comma-separated list of exact origins the
    # app is served from, e.g.:
    #   CORS_ALLOWED_ORIGINS=http://localhost:5173,https://app.pachelpdesk.example
    # Deliberately NOT a wildcard: /chat and friends rely on the Flask
    # session cookie for conversation continuity, and
    # Access-Control-Allow-Credentials only works with an explicit origin,
    # never "*".
    _cors_origins = [
        o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()
    ]

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        if _cors_origins:
            from flask import request as _req
            origin = _req.headers.get("Origin")
            if origin and origin in _cors_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Vary"] = "Origin"
                if _req.method == "OPTIONS":
                    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
                    response.headers["Access-Control-Max-Age"] = "3600"
        return response

    if _cors_origins:
        @app.route("/<path:_any>", methods=["OPTIONS"])
        def _cors_preflight(_any):
            return "", 204

    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(schedules_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(pages_bp)
    app.register_blueprint(cron_bp)

    def wants_json_error():
        return (
            request.path.startswith("/api/")
            or request.is_json
            or request.accept_mimetypes.best == "application/json"
        )

    @app.errorhandler(404)
    def page_not_found(_error):
        if wants_json_error():
            return jsonify({"error": "not_found", "message": "The requested resource was not found."}), 404
        return render_template(
            "errors/error.html",
            status_code=404,
            title="This page is missing.",
            message="The address may be outdated, or the page may have moved.",
        ), 404

    @app.errorhandler(500)
    def internal_server_error(_error):
        if wants_json_error():
            return jsonify({"error": "internal_server_error", "message": "The request could not be completed."}), 500
        return render_template(
            "errors/error.html",
            status_code=500,
            title="We hit a snag.",
            message="The help desk could not complete that request. Please try again in a moment.",
        ), 500

    @app.route("/__debug_path")
    def __debug_path():
        from flask import request
        return {
            "path": request.path,
            "url": request.url,
            "environ_PATH_INFO": request.environ.get("PATH_INFO"),
            "environ_QUERY_STRING": request.environ.get("QUERY_STRING"),
            "rules": [str(r) for r in app.url_map.iter_rules()][:40],
        }

    try:
        init_pool()
    except Exception as e:
        app.logger.error(f"init_pool failed (will retry on request): {e}")

    for _name, _fn in (
        ("ensure_core_tables", ensure_core_tables),
        ("ensure_admin_exists", ensure_admin_exists),
        ("ensure_schedule_schema", ensure_schedule_schema),
        ("migrate_create_chat_sessions", migrate_create_chat_sessions),
        ("migrate_add_uuid_to_faq", migrate_add_uuid_to_faq),
        ("ensure_handbook_fulltext_search", ensure_handbook_fulltext_search),
        ("ensure_faq_embedding_column", ensure_faq_embedding_column),
        ("ensure_learning_layer", ensure_learning_layer),
    ):
        try:
            _fn()
        except Exception as e:
            app.logger.error(f"{_name} failed (non-fatal on startup): {e}")

    if not _is_vercel():
        from threading import Thread
        Thread(target=warm_response_tables, daemon=True).start()

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
            app.extensions["pac_helpdesk_scheduler"] = scheduler

            try:
                from database.learning_schema import wire_into_app
                from database import get_db
                wire_into_app(app, get_db, scheduler=scheduler)
            except Exception as e:
                app.logger.warning(f"learning_schema wire_into_app failed: {e}")
        except Exception as e:
            app.logger.warning(f"Scheduler not started: {e}")

    return app


app = create_app()


if __name__ == "__main__":
    validate_config()
    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    print("=" * 55)
    print("PAC Helpdesk — modular edition")
    print(f"   URL : http://0.0.0.0:{port}")
    print(f"   Mode: {'Development' if debug_mode else 'Production'}")
    print(f"   Vercel: {_is_vercel()}")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)