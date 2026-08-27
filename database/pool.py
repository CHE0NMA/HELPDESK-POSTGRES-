"""
database/pool.py — safe for Vercel + Render Postgres

Goals:
- Survive idle connection kills from Render
- Survive Vercel cold starts / function freezes
- Never turn a working DB into a permanent 500
- Keep the original behaviour that was working
"""

from contextlib import contextmanager
import logging
import os
import time

import psycopg2
from dbutils.pooled_db import PooledDB
from psycopg2 import OperationalError, InterfaceError

from config import DATABASE_CONFIG

logger = logging.getLogger("pac_helpdesk.database.pool")

_pool = None
_IS_VERCEL = bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"))


def init_pool():
    global _pool
    if _pool is not None:
        return _pool

    # Small pool on Vercel, larger locally
    max_conn = 3 if _IS_VERCEL else 10
    max_cached = 1 if _IS_VERCEL else 4

    kwargs = {
        "creator": psycopg2,
        "maxconnections": max_conn,
        "mincached": 0,
        "maxcached": max_cached,
        "maxshared": 0,
        "blocking": True,
        "host": DATABASE_CONFIG["host"],
        "user": DATABASE_CONFIG["user"],
        "password": DATABASE_CONFIG["password"],
        "database": DATABASE_CONFIG["database"],
        "port": DATABASE_CONFIG["port"],
        "options": "-c timezone=Asia/Manila",
        # Help keep connections alive longer
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
        "connect_timeout": 10,
    }

    sslmode = DATABASE_CONFIG.get("sslmode") or os.environ.get("DB_SSLMODE", "require")
    if sslmode:
        kwargs["sslmode"] = sslmode

    _pool = PooledDB(**kwargs)
    logger.info("DB pool initialised (vercel=%s, maxconnections=%s)", _IS_VERCEL, max_conn)
    return _pool


def _is_alive(conn) -> bool:
    """Return True if the connection can still run a simple query."""
    try:
        if conn is None or conn.closed:
            return False
        # Use a short timeout so a hung connection doesn't block forever
        old_autocommit = conn.autocommit
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        finally:
            try:
                conn.autocommit = old_autocommit
            except Exception:
                pass
    except Exception:
        return False


def get_db(retries: int = 2):
    """
    Get a usable connection.
    - Tries the pool first
    - If the connection is dead, discards it and retries
    - If the pool itself is broken, falls back to a fresh direct connection
    - Always returns a real connection or raises the *original* error
    """
    if _pool is None:
        init_pool()

    last_err = None

    for attempt in range(retries + 1):
        conn = None
        try:
            conn = _pool.connection()

            if _is_alive(conn):
                return conn

            # Dead connection → close it so the pool does not reuse it
            logger.warning("Discarding dead pooled connection (attempt %s)", attempt + 1)
            try:
                conn.close()
            except Exception:
                pass

        except (OperationalError, InterfaceError) as e:
            last_err = e
            logger.warning("Pool connection error (attempt %s): %s", attempt + 1, e)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            last_err = e
            logger.error("Unexpected error from pool: %s", e)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

        if attempt < retries:
            time.sleep(0.15 * (attempt + 1))

    # ---------------------------------------------------------------
    # Fallback: open a completely fresh connection outside the pool.
    # This recovers from a corrupted pool after long idle periods.
    # ---------------------------------------------------------------
    logger.warning("Pool exhausted or broken – opening direct connection")
    try:
        direct_kwargs = {
            "host": DATABASE_CONFIG["host"],
            "user": DATABASE_CONFIG["user"],
            "password": DATABASE_CONFIG["password"],
            "database": DATABASE_CONFIG["database"],
            "port": DATABASE_CONFIG["port"],
            "options": "-c timezone=Asia/Manila",
            "connect_timeout": 10,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }
        sslmode = DATABASE_CONFIG.get("sslmode") or os.environ.get("DB_SSLMODE", "require")
        if sslmode:
            direct_kwargs["sslmode"] = sslmode

        conn = psycopg2.connect(**direct_kwargs)
        if _is_alive(conn):
            return conn
        conn.close()
    except Exception as e:
        last_err = e
        logger.error("Direct connection also failed: %s", e)

    # Re-raise the real underlying error so logs are useful
    if last_err is not None:
        raise last_err
    raise OperationalError("Could not obtain a database connection")


@contextmanager
def get_db_cursor(commit: bool = False):
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
        try:
            conn.close()
        except Exception:
            pass


def pool_stats():
    if _pool is None:
        return {"status": "not_initialized"}
    return {
        "status": "ok",
        "maxconnections": getattr(_pool, "_maxconnections", None),
        "vercel": _IS_VERCEL,
    }