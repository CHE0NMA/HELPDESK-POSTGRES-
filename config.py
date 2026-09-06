"""
Central configuration — reads everything from environment variables.

Supports both:
  - Local development  (FLASK_ENV=development or no FLASK_ENV)
  - Online / Render    (FLASK_ENV=production or VERCEL / RENDER env present)

Database notes:
- Prefers individual DB_* vars (DB_HOST, DB_USER, etc.)
- Also accepts a single DATABASE_URL (Render style) if the individual vars are missing
- SSL is required online, disabled/prefer locally
"""
import os
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------
def _is_online() -> bool:
    """True when running on Render, Vercel, or explicitly in production."""
    return bool(
        os.environ.get("VERCEL")
        or os.environ.get("VERCEL_ENV")
        or os.environ.get("RENDER")
        or os.environ.get("FLASK_ENV") == "production"
    )


IS_ONLINE = _is_online()
IS_LOCAL = not IS_ONLINE


def _build_database_config() -> dict:
    """
    Build DATABASE_CONFIG for local or online use.

    Priority controlled by DB_TARGET:
      - "neon" / "online"  → use DATABASE_URL (Neon)
      - "local" (default)  → use individual DB_* vars
    When running on Vercel/Render (IS_ONLINE), always prefer DATABASE_URL
    if it exists.
    """
    target = (os.environ.get("DB_TARGET") or "local").lower().strip()
    database_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")

    # Force Neon when explicitly requested or when deployed online
    use_neon = target in ("neon", "online") or (IS_ONLINE and database_url)

    if use_neon and database_url:
        parsed = urlparse(database_url)
        return {
            "host": parsed.hostname or "localhost",
            "user": parsed.username or "postgres",
            "password": parsed.password or "",
            "database": (parsed.path or "/").lstrip("/") or "neondb",
            "port": int(parsed.port or 5432),
            "sslmode": "require",
        }

    # Local (individual vars)
    host = os.environ.get("DB_HOST") or "localhost"
    user = os.environ.get("DB_USER") or "postgres"
    password = os.environ.get("DB_PASSWORD") or ""
    database = os.environ.get("DB_NAME") or "school_chatbot"
    port = int(os.environ.get("DB_PORT") or 5432)
    sslmode = os.environ.get("DB_SSLMODE") or ("require" if IS_ONLINE else "disable")

    return {
        "host": host,
        "user": user,
        "password": password,
        "database": database,
        "port": port,
        "sslmode": sslmode,
    }

DATABASE_CONFIG = _build_database_config()


# ---------------------------------------------------------------------
# File uploads
# ---------------------------------------------------------------------
UPLOAD_FOLDER = os.environ.get(
    "UPLOAD_FOLDER",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads"),
)


# ---------------------------------------------------------------------
# AI provider selection
# ---------------------------------------------------------------------
# One of: 'groq' | 'gemini'
AI_PROVIDER = os.environ.get("AI_PROVIDER", "groq").lower()

# Groq (supports multiple keys)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_2", "")
GROQ_API_KEY_3 = os.environ.get("GROQ_API_KEY_3", "")
GROQ_API_KEY_4 = os.environ.get("GROQ_API_KEY_4", "")
GROQ_API_KEY_5 = os.environ.get("GROQ_API_KEY_5", "")
GROQ_API_KEY_6 = os.environ.get("GROQ_API_KEY_6", "")
GROQ_API_KEY_7 = os.environ.get("GROQ_API_KEY_7", "")
GROQ_API_KEY_8 = os.environ.get("GROQ_API_KEY_8", "")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Gemini (supports multiple keys)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_KEY_2 = os.environ.get("GEMINI_API_KEY_2", "")
GEMINI_API_KEY_3 = os.environ.get("GEMINI_API_KEY_3", "")
GEMINI_API_KEY_4 = os.environ.get("GEMINI_API_KEY_4", "")
GEMINI_API_KEY_5 = os.environ.get("GEMINI_API_KEY_5", "")
GEMINI_API_KEY_6 = os.environ.get("GEMINI_API_KEY_6", "")
GEMINI_API_KEY_7 = os.environ.get("GEMINI_API_KEY_7", "")
GEMINI_API_KEY_8 = os.environ.get("GEMINI_API_KEY_8", "")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


# ---------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------
REQUIRED_ENV_VARS = {
    "groq": ["GROQ_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
}


def validate_config():
    """Called once at startup. Raises early with a clear message."""
    missing = []
    for var in REQUIRED_ENV_VARS.get(AI_PROVIDER, []):
        if not globals().get(var):
            missing.append(var)

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables for AI_PROVIDER='{AI_PROVIDER}': "
            f"{', '.join(missing)}\n"
            f"Set them in your .env file or environment before starting the app."
        )

    if not os.environ.get("FLASK_SECRET_KEY"):
        raise EnvironmentError(
            "FLASK_SECRET_KEY must be set in environment variables.\n"
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

    if not DATABASE_CONFIG["password"] and IS_ONLINE:
        print(
            "WARNING: DB_PASSWORD is empty while running online. "
            "Double check this is intentional."
        )