"""
Central configuration — reads everything from environment variables.

--- Postgres migration notes ---
- MYSQL_CONFIG -> DATABASE_CONFIG. Env var names changed from
  MYSQL_HOST/MYSQL_USER/MYSQL_PASSWORD/MYSQL_DB/MYSQL_PORT to
  DB_HOST/DB_USER/DB_PASSWORD/DB_NAME/DB_PORT (matches the .env you're
  using — this replaces the placeholder config.py from before, which
  guessed at this file's contents since it wasn't uploaded yet).
- Default port changed from 3306 to 5432.
- `db` key renamed to `database` (that's the psycopg2/DBUtils kwarg name;
  pymysql used `db`).
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------
DATABASE_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': os.environ.get('DB_NAME', 'school_chatbot'),
    'port': int(os.environ.get('DB_PORT', 5432)),
}

# ---------------------------------------------------------------------
# File uploads
# ---------------------------------------------------------------------
UPLOAD_FOLDER = os.environ.get(
    'UPLOAD_FOLDER',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
)

# ---------------------------------------------------------------------
# AI provider selection
# ---------------------------------------------------------------------
# One of: 'groq' | 'gemini'
AI_PROVIDER = os.environ.get('AI_PROVIDER', 'groq').lower()

# Groq (supports multiple keys)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_API_KEY_2 = os.environ.get('GROQ_API_KEY_2', '')
GROQ_API_KEY_3 = os.environ.get('GROQ_API_KEY_3', '')
GROQ_API_KEY_4 = os.environ.get('GROQ_API_KEY_4', '')
GROQ_API_KEY_5 = os.environ.get('GROQ_API_KEY_5', '')
GROQ_API_KEY_6 = os.environ.get('GROQ_API_KEY_6', '')
GROQ_API_KEY_7 = os.environ.get('GROQ_API_KEY_7', '')
GROQ_API_KEY_8 = os.environ.get('GROQ_API_KEY_8', '')
GROQ_API_KEY_9 = os.environ.get('GROQ_API_KEY_9', '')
GROQ_API_KEY_10 = os.environ.get('GROQ_API_KEY_10', '')
GROQ_API_KEY_11 = os.environ.get('GROQ_API_KEY_11', '')
GROQ_API_KEY_12 = os.environ.get('GROQ_API_KEY_12', '')
GROQ_API_KEY_13 = os.environ.get('GROQ_API_KEY_13', '')
GROQ_API_KEY_14 = os.environ.get('GROQ_API_KEY_14', '')
GROQ_API_KEY_15 = os.environ.get('GROQ_API_KEY_15', '')
GROQ_API_KEY_16 = os.environ.get('GROQ_API_KEY_16', '')
GROQ_API_KEY_17 = os.environ.get('GROQ_API_KEY_17', '')
GROQ_API_KEY_18 = os.environ.get('GROQ_API_KEY_18', '')
GROQ_API_KEY_19 = os.environ.get('GROQ_API_KEY_19', '')
GROQ_API_KEY_20 = os.environ.get('GROQ_API_KEY_20', '')
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')

# Gemini
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3.5-flash-lite')


# ---------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------
REQUIRED_ENV_VARS = {
    'groq': ['GROQ_API_KEY'],
    'gemini': ['GEMINI_API_KEY'],
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

    if not os.environ.get('FLASK_SECRET_KEY'):
        raise EnvironmentError(
            "FLASK_SECRET_KEY must be set in environment variables.\n"
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

    if not DATABASE_CONFIG['password'] and os.environ.get('FLASK_ENV') == 'production':
        print("WARNING: DB_PASSWORD is empty while FLASK_ENV=production. "
              "Double check this is intentional.")