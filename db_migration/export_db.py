#!/usr/bin/env python3
"""
Export every table's data from the Postgres database into one JSON file.
Run this before your free-tier database expires / gets wiped.

Usage:
    python export_db.py                    # writes backup_<timestamp>.json
    python export_db.py my_backup.json      # writes to a specific path

Reads connection info from config.DATABASE_CONFIG — the same place the
app itself gets it from — so run this from inside the project with your
real .env / environment variables loaded, same as running the app.

NOTE: this exports DATA only, not table structure. The destination
database needs the tables already created (either let app.py's startup
migrations create the ones it knows how to, or — better — use
`pg_dump --schema-only` once on the source DB to capture the exact
CREATE TABLE statements for everything, then import_db.py to load the
data into that structure). If you have shell access to run `pg_dump`
directly against your database, prefer that over this script — it
captures structure AND data in one step with no risk of drift from
whatever this script guesses.
"""
import json
import sys
from datetime import date, datetime, time
from decimal import Decimal

import psycopg2
import psycopg2.extras

from config import DATABASE_CONFIG


def _json_default(obj):
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode('utf-8', errors='replace')
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def get_all_tables(cur):
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    return [r[0] for r in cur.fetchall()]


def export_all(output_path: str):
    conn = psycopg2.connect(**DATABASE_CONFIG)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    tables = get_all_tables(cur)

    if not tables:
        print("No tables found in 'public' schema — is DATABASE_CONFIG pointing at the right DB?")
        cur.close()
        conn.close()
        return

    dump = {'exported_at': datetime.utcnow().isoformat() + 'Z', 'tables': {}}
    total_rows = 0

    print(f"Found {len(tables)} tables. Exporting…")
    for table in tables:
        cur.execute(f'SELECT * FROM "{table}"')
        rows = [dict(r) for r in cur.fetchall()]
        dump['tables'][table] = rows
        total_rows += len(rows)
        print(f"  {table}: {len(rows)} rows")

    cur.close()
    conn.close()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dump, f, default=_json_default, ensure_ascii=False, indent=2)

    print(f"\nExported {len(tables)} tables, {total_rows} total rows -> {output_path}")


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    export_all(out)