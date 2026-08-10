"""Connection pool. Kept tiny and dependency-free (no Flask import) so any
module can `from database import get_db` without pulling in app.py.

Usage pattern used everywhere else in the codebase (kept as-is on purpose,
since switching every call site to a context manager is a bigger, separate
refactor):

    conn = get_db()
    cur = conn.cursor()
    try:
        ...
    finally:
        cur.close()
        conn.close()

For NEW code, prefer the `db_cursor()` context manager below — it closes
the cursor/connection automatically even on exception, which was one of
the "connection leak on error paths" issues raised in review.

--- Postgres migration notes ---
- pymysql -> psycopg2. `db=` kwarg becomes `database=`.
- No `cursorclass=pymysql.cursors.Cursor` equivalent needed: psycopg2's
  default cursor already returns plain tuples (same row[0]-style access
  used throughout the codebase), so nothing else had to change in the
  many `cur.fetchall()` call sites.
- `conn.rollback()` behavior is the same; psycopg2 also requires an
  explicit rollback after a failed statement before the connection can be
  reused for a new query (a transaction that hit an error is left in an
  aborted state) — the try/except/rollback in db_cursor() already does
  this, but be aware if you add new bare `cur.execute()` calls elsewhere
  outside this helper, wrap them the same way.
"""
from contextlib import contextmanager

import psycopg2
import psycopg2.extensions
from dbutils.pooled_db import PooledDB

from config import DATABASE_CONFIG

_pool = None


def init_pool():
    global _pool
    if _pool is not None:
        return _pool
    _pool = PooledDB(
        creator=psycopg2,
        maxconnections=20,
        mincached=2,
        maxcached=10,
        maxshared=0,
        blocking=True,
        host=DATABASE_CONFIG['host'],
        user=DATABASE_CONFIG['user'],
        password=DATABASE_CONFIG['password'],
        database=DATABASE_CONFIG['database'],
        port=DATABASE_CONFIG['port'],
        # psycopg2 connections default to autocommit=False already, but
        # being explicit here matches the old pymysql config intent.
    )
    return _pool


def get_db():
    if _pool is None:
        init_pool()
    conn = _pool.connection()
    conn.autocommit = False
    return conn


@contextmanager
def db_cursor(commit: bool = False):
    """with db_cursor() as cur: ...  — connection+cursor always closed,
    optionally auto-commits on clean exit."""
    conn = get_db()
    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def pool_stats() -> dict:
    if _pool is None:
        return {}
    return {
        'max_connections': _pool._maxconnections,
        'min_cached': _pool._mincached,
        'max_cached': _pool._maxcached,
    }