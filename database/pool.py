from contextlib import contextmanager
import os

import psycopg2
import psycopg2.extensions
from dbutils.pooled_db import PooledDB

from config import DATABASE_CONFIG

_pool = None


def init_pool():
    global _pool
    if _pool is not None:
        return _pool

    kwargs = {
        "creator": psycopg2,
        "maxconnections": 10,
        "mincached": 0,
        "maxcached": 5,
        "maxshared": 0,
        "blocking": True,
        "host": DATABASE_CONFIG["host"],
        "user": DATABASE_CONFIG["user"],
        "password": DATABASE_CONFIG["password"],
        "database": DATABASE_CONFIG["database"],
        "port": DATABASE_CONFIG["port"],
        # BUG FIX: every NOW()/CURRENT_DATE/CURRENT_TIMESTAMP call in the
        # app (90+ call sites across schedule.py, alerts.py, admin.py,
        # chat.py...) was implicitly evaluated in the DB server's own
        # timezone -- UTC on most managed Postgres hosts. For a
        # Philippines-based school (UTC+8), "today" on the DB side lagged
        # real PH local time by up to 8 hours: still "yesterday" from
        # 12:00am-7:59am PH time, exactly when students ask "is there
        # class today" before heading to school. Setting the session
        # timezone here fixes it for every query, everywhere, without
        # touching each call site individually. See services/tz.py for
        # the matching fix on the Python side.
        "options": "-c timezone=Asia/Manila",
    }

    sslmode = DATABASE_CONFIG.get("sslmode") or os.environ.get("DB_SSLMODE", "require")
    if sslmode:
        kwargs["sslmode"] = sslmode

    _pool = PooledDB(**kwargs)
    return _pool


def get_db():
    """Return a connection from the pool. Caller must close it."""
    if _pool is None:
        init_pool()
    return _pool.connection()


@contextmanager
def get_db_cursor(commit=False):
    """Optional convenience context manager (used by some services)."""
    conn = get_db()
    try:
        cur = conn.cursor()
        try:
            yield cur
            if commit:
                conn.commit()
        finally:
            cur.close()
    finally:
        conn.close()


def pool_stats():
    """Simple diagnostics (optional, used by health checks / admin)."""
    if _pool is None:
        return {"status": "not_initialized"}
    return {
        "status": "ok",
        "maxconnections": getattr(_pool, "_maxconnections", None),
    }