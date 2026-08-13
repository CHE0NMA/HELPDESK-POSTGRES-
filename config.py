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


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------
def _build_database_config() -> dict:
    """
    Build DATABASE_CONFIG for local or online use.

    Priority:
      1. Individual DB_* environment variables
      2. DATABASE_URL (Render / Heroku style) if individual host is missing
      3. Sensible local defaults
    """
    host = os.environ.get("DB_HOST")
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    database = os.environ.get("DB_NAME")
    port = os.environ.get("DB_PORT")
    sslmode = os.environ.get("DB_SSLMODE")

    # Fall back to DATABASE_URL if the host looks like a full URL or is missing
    database_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if database_url and (not host or host.startswith("postgresql://") or host.startswith("postgres://")):
        parsed = urlparse(database_url)
        host = parsed.hostname or host
        user = parsed.username or user
        password = parsed.password or password
        database = (parsed.path or "/").lstrip("/") or database
        port = str(parsed.port) if parsed.port else port
        # Render always needs SSL
        if sslmode is None:
            sslmode = "require"

    # Final defaults
    host = host or "localhost"
    user = user or "postgres"
    password = password or ""
    database = database or "school_chatbot"
    port = int(port or 5432)

    # SSL policy
    if sslmode is None:
        # Online (Render) → require SSL
        # Local → disable so plain local Postgres works
        sslmode = "require" if IS_ONLINE else "disable"

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
GROQ_API_KEY_9 = os.environ.get("GROQ_API_KEY_9", "")
GROQ_API_KEY_10 = os.environ.get("GROQ_API_KEY_10", "")
GROQ_API_KEY_11 = os.environ.get("GROQ_API_KEY_11", "")
GROQ_API_KEY_12 = os.environ.get("GROQ_API_KEY_12", "")
GROQ_API_KEY_13 = os.environ.get("GROQ_API_KEY_13", "")
GROQ_API_KEY_14 = os.environ.get("GROQ_API_KEY_14", "")
GROQ_API_KEY_15 = os.environ.get("GROQ_API_KEY_15", "")
GROQ_API_KEY_16 = os.environ.get("GROQ_API_KEY_16", "")
GROQ_API_KEY_17 = os.environ.get("GROQ_API_KEY_17", "")
GROQ_API_KEY_18 = os.environ.get("GROQ_API_KEY_18", "")
GROQ_API_KEY_19 = os.environ.get("GROQ_API_KEY_19", "")
GROQ_API_KEY_20 = os.environ.get("GROQ_API_KEY_20", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
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