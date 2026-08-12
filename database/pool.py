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