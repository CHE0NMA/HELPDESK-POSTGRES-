"""
database/pool.py — Vercel + Render Postgres safe connection pool.

Designed for:
- Serverless cold starts / idle freezes on Vercel
- Render Postgres killing idle connections
- Multiple concurrent students (chat + admin)
"""

from contextlib import contextmanager
import logging
import os
import time

import psycopg2
import psycopg2.extensions
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

    # Vercel: keep the pool tiny (each concurrent function instance
    # gets its own pool). Non-Vercel (local / long-running) can be larger.
    max_conn = 2 if _IS_VERCEL else 8
    max_cached = 1 if _IS_VERCEL else 3

    kwargs = {
        "creator": psycopg2,
        "maxconnections": max_conn,
        "mincached": 0,          # never keep idle connections
        "maxcached": max_cached,
        "maxshared": 0,
        "blocking": True,        # wait instead of exploding under load
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
        "connect_timeout": 8,
    }

    sslmode = DATABASE_CONFIG.get("sslmode") or os.environ.get("DB_SSLMODE", "require")
    if sslmode:
        kwargs["sslmode"] = sslmode

    _pool = PooledDB(**kwargs)
    logger.info(
        "DB pool ready (vercel=%s maxconnections=%s)",
        _IS_VERCEL,
        max_conn,
    )
    return _pool


def _is_connection_alive(conn) -> bool:
    try:
        if conn.closed:
            return False
        old = conn.autocommit
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        finally:
            conn.autocommit = old
        return True
    except (OperationalError, InterfaceError, psycopg2.Error):
        return False
    except Exception:
        return False


def get_db(retries: int = 2):
    """
    Return a *live* connection.
    Automatically discards dead connections and retries.
    Caller MUST call conn.close() when finished.
    """
    if _pool is None:
        init_pool()

    last_err = None
    for attempt in range(retries + 1):
        conn = None
        try:
            conn = _pool.connection()
            if _is_connection_alive(conn):
                return conn

            logger.warning(
                "Discarding dead DB connection (attempt %s/%s)",
                attempt + 1, retries + 1,
            )
            try:
                conn.close()
            except Exception:
                pass
            if attempt < retries:
                time.sleep(0.12 * (attempt + 1))
        except (OperationalError, InterfaceError) as e:
            last_err = e
            logger.warning(
                "Pool get failed (attempt %s/%s): %s",
                attempt + 1, retries + 1, e,
            )
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            if attempt < retries:
                time.sleep(0.18 * (attempt + 1))
        except Exception as e:
            last_err = e
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            raise

    raise OperationalError(
        f"Could not obtain a live database connection after {retries + 1} attempts"
    ) from last_err


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