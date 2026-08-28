"""
database/pool.py — reliable version for local + Vercel + Render
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

    max_conn = 3 if _IS_VERCEL else 8
    max_cached = 1 if _IS_VERCEL else 3

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
    logger.info("DB pool ready (vercel=%s max=%s)", _IS_VERCEL, max_conn)
    return _pool


def get_db(retries: int = 2):
    """
    Get a connection from the pool.
    Only retries when we actually get an OperationalError / InterfaceError.
    Does NOT run a health-check that marks good connections as dead.
    """
    if _pool is None:
        init_pool()

    last_err = None

    for attempt in range(retries + 1):
        try:
            conn = _pool.connection()
            return conn          # ← just return it. No SELECT 1 check.
        except (OperationalError, InterfaceError) as e:
            last_err = e
            logger.warning("DB connection failed (attempt %s): %s", attempt + 1, e)
            if attempt < retries:
                time.sleep(0.2 * (attempt + 1))
        except Exception as e:
            logger.error("Unexpected pool error: %s", e)
            raise

    # Last resort: open a brand-new connection outside the pool
    logger.warning("Pool failed – trying direct connection")
    try:
        kwargs = {
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
            kwargs["sslmode"] = sslmode

        return psycopg2.connect(**kwargs)
    except Exception as e:
        if last_err:
            raise last_err from e
        raise


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