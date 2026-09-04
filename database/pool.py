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


def _connection_is_alive(conn) -> bool:
    """Real liveness check for a pooled connection.

    BUG FIX (cold-start 500s): DBUtils' PooledDB defaults to ping=1
    ("validate on checkout"), but its ping mechanism calls
    connection.ping() -- a method plain psycopg2 connections don't have.
    DBUtils catches the resulting AttributeError, silently sets its
    internal ping flag to 0, and never checks again for the lifetime of
    that pooled connection. In other words: the pool's built-in
    validation is a complete no-op for psycopg2, even though it looks
    configured correctly (this is a well-known DBUtils+psycopg2 gotcha).

    So a connection that's been idle in the pool -- across a Vercel cold
    start, or just a gap of a few minutes between requests -- can come
    back from _pool.connection() looking perfectly fine, while actually
    being dead at the TCP level (closed server-side by Render's Postgres,
    or dropped by a NAT/proxy idle timeout, or torn down when Vercel
    recycled the previous invocation's network state). The FIRST real
    query on it is what discovers this -- which is exactly the "first
    load 500s, refresh works" symptom: request 1 draws the stale
    connection and fails, request 2 gets a connection the pool has since
    replaced (or the retry loop below succeeds) and works fine.

    This does the same thing SQLAlchemy's pool_pre_ping does: an actual
    cheap query, not a driver-level ping that may not exist. Costs one
    extra tiny round-trip per checkout (sub-millisecond to a few ms for a
    same-region DB) -- worth it against a full request failing outright.
    """
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT 1")
        finally:
            cur.close()
        return True
    except Exception:
        return False


def get_db(retries: int = 2):
    """
    Get a connection from the pool.
    Retries when we get an OperationalError / InterfaceError, OR when
    the connection fails the liveness check below (see
    _connection_is_alive's docstring for why that check exists and why
    DBUtils' own ping=1 default doesn't already cover this for psycopg2).
    """
    if _pool is None:
        init_pool()

    last_err = None

    for attempt in range(retries + 1):
        try:
            conn = _pool.connection()
            if not _connection_is_alive(conn):
                logger.warning(
                    "DB connection failed liveness check (attempt %s) -- "
                    "discarding and retrying with a new one", attempt + 1,
                )
                try:
                    conn.close()
                except Exception:
                    pass
                last_err = OperationalError("pooled connection failed SELECT 1 liveness check")
                if attempt < retries:
                    time.sleep(0.2 * (attempt + 1))
                continue
            return conn
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