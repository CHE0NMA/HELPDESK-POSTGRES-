#!/usr/bin/env python3
"""
Restore a JSON backup produced by export_db.py into a Postgres database.

Prerequisite: the target DB must already have all the tables created
(same structure as the source) — this script only loads DATA, it doesn't
create tables. Easiest way to get there: start the app once against the
new empty DB so app.py's startup migrations create what they know how to,
then either manually create the rest from your real schema, or (better)
use `pg_dump --schema-only` against the OLD database once while it's
still alive, and load that into the new one before running this script.

Usage:
    python import_db.py backup_20260101_120000.json

Safe to re-run: uses ON CONFLICT (primary key) DO NOTHING per row, so
importing the same file twice won't duplicate rows. It does NOT delete
existing rows in the target DB first — if you want a clean slate,
truncate the tables yourself before running this.

Runs in autocommit mode: each row is its own transaction, so one bad row
(e.g. a table whose structure doesn't quite match the export) is skipped
and reported instead of rolling back everything imported before it.
"""
import json
import sys

import psycopg2
import psycopg2.extras

from config import DATABASE_CONFIG


def get_pk_column(cur, table):
    cur.execute("""
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = %s::regclass AND i.indisprimary
    """, (table,))
    row = cur.fetchone()
    return row[0] if row else None


def get_serial_columns(cur, table):
    """Columns backed by a sequence (SERIAL/BIGSERIAL/IDENTITY) — after
    inserting rows with explicit IDs, the sequence needs to be fast-
    forwarded so the NEXT app-generated insert doesn't collide."""
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s
          AND column_default LIKE 'nextval(%%'
    """, (table,))
    return [r[0] for r in cur.fetchall()]


def wrap_json_values(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        out[k] = psycopg2.extras.Json(v) if isinstance(v, (dict, list)) else v
    return out


def import_all(input_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        dump = json.load(f)

    conn = psycopg2.connect(**DATABASE_CONFIG)
    conn.autocommit = True  # each statement commits on its own; one bad
                             # row can't poison/rollback the whole import
    cur = conn.cursor()

    tables = list(dump['tables'].keys())

    print("Disabling triggers (so foreign-key order doesn't matter)…")
    for table in tables:
        try:
            cur.execute(f'ALTER TABLE "{table}" DISABLE TRIGGER ALL')
        except Exception as e:
            print(f"  (couldn't disable triggers on {table} — needs table-owner "
                  f"privilege; import order will matter for this table: {e})")

    total_inserted = 0
    for table, rows in dump['tables'].items():
        if not rows:
            continue
        pk = get_pk_column(cur, table)
        columns = list(rows[0].keys())
        col_list = ', '.join(f'"{c}"' for c in columns)
        placeholders = ', '.join(['%s'] * len(columns))
        conflict_clause = f'ON CONFLICT ("{pk}") DO NOTHING' if pk else ''
        sql = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders}) {conflict_clause}'

        inserted = 0
        for row in rows:
            try:
                cur.execute(sql, [wrap_json_values(row)[c] for c in columns])
                inserted += cur.rowcount
            except Exception as e:
                print(f"  ! skipped a row in {table}: {e}")
        total_inserted += inserted
        print(f"  {table}: inserted {inserted}/{len(rows)} rows")

        for seq_col in get_serial_columns(cur, table):
            try:
                cur.execute(
                    'SELECT setval(pg_get_serial_sequence(%s, %s), '
                    f'COALESCE((SELECT MAX("{seq_col}") FROM "{table}"), 1))',
                    (table, seq_col)
                )
            except Exception as e:
                print(f"  (couldn't reset sequence for {table}.{seq_col}: {e})")

    print("Re-enabling triggers…")
    for table in tables:
        try:
            cur.execute(f'ALTER TABLE "{table}" ENABLE TRIGGER ALL')
        except Exception as e:
            print(f"  (couldn't re-enable triggers on {table}: {e})")

    cur.close()
    conn.close()
    print(f"\nImported {total_inserted} rows across {len(tables)} tables from {input_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python import_db.py <backup_file.json>")
        sys.exit(1)
    import_all(sys.argv[1])