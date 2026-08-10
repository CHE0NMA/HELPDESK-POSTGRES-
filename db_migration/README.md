# Migrating between free-tier databases

Two ways to do this, in order of preference.

## Option A (recommended): `pg_dump` / `pg_restore`

These ship with Postgres and work against basically every managed host
(Render, Railway, Supabase, Neon, ElephantSQL, etc.), even free tiers —
you run them from your own machine against the host's connection string,
no shell access on the host needed. They capture **structure and data
together** — tables, indexes, sequences, admin users, ratings, chat
history, everything — with zero risk of the copy drifting from the real
schema (unlike any script, including the one below, that has to guess at
structure).

```bash
# 1. While the OLD database is still alive:
pg_dump "postgresql://DB_USER:DB_PASSWORD@DB_HOST:DB_PORT/DB_NAME" -F c -f backup.dump

# 2. Once you have the NEW database:
pg_restore -d "postgresql://DB_USER:DB_PASSWORD@NEW_HOST:DB_PORT/DB_NAME" --clean --if-exists backup.dump
```

Fill in the connection details from your `.env` (old) and your new host's
dashboard (new). `--clean --if-exists` means it's safe to run against a
database that already has some/all of these tables — it drops and
recreates them from the dump rather than erroring.

If your host doesn't expose a direct Postgres connection string for
`pg_dump` (rare, but some free tiers restrict this), fall back to Option B.

## Option B: `export_db.py` / `import_db.py` (included here)

Goes through your app's own DB connection instead of `pg_dump`, so it
works even on hosts that block direct external Postgres access. Exports
**data only** (not table structure) as one JSON file — every table,
including `admin_users`, `chat_ratings`, `faq`, `chat_history`, all of it.

```bash
# 1. While the OLD database is still alive (uses config.DATABASE_CONFIG,
#    same as the app — point your .env at the OLD db when you run this):
python db_migration/export_db.py

# 2. Point .env at the NEW database, start the app once so its startup
#    migrations create the tables it knows how to (ensure_admin_exists(),
#    ensure_schedule_schema(), migrate_create_chat_sessions(),
#    migrate_add_uuid_to_faq()) — then stop it again.
#
#    IMPORTANT: this only creates a few of your tables. The rest (faq,
#    chat_history, satisfaction_feedback, handbook_chunks, etc.) need to
#    exist already — see the note in export_db.py's docstring. If you
#    don't have their CREATE TABLE statements handy, run
#    `pg_dump --schema-only` against the OLD db once (while it's still up)
#    to capture them, and load that into the new db before step 3.

# 3. Load the data in:
python db_migration/import_db.py backup_20260101_120000.json
```

Safe to re-run — uses `ON CONFLICT ... DO NOTHING`, so importing the same
file twice won't duplicate rows. It temporarily disables triggers
(including the internal ones that enforce foreign keys) so table import
order doesn't matter, then re-enables them and fixes up auto-increment
sequences at the end so new rows created by the app afterward don't
collide with imported IDs.

**This needs your tables to already exist on the target** — it's a data
copier, not a schema creator. That's the one thing Option A gets you for
free that this doesn't.