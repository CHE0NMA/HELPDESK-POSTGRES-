#!/usr/bin/env python3
"""
backfill_embeddings.py — one-time (or periodic) job that computes and
stores a semantic embedding for every FAQ and handbook_chunks row that
doesn't have one yet.

WHY THIS EXISTS
-----------------
Going forward, new FAQs added through mass_insert_faqs() (bulk AI-generated
FAQ import) and new handbook sections added through import_chunks_json()
get an embedding automatically at insert time — see services/embeddings.py
and services/handbook_ingest.py. This script exists for the rows that
predate that (anything already sitting in the DB), and as a safety net you
can re-run any time — it only touches rows where embedding IS NULL, so
running it twice in a row does nothing the second time.

Once a row has an embedding, it gets matched by MEANING, not just
keywords — this is what lets the bot answer a hypothetical or paraphrased
question about any topic (not just ones with a hand-written alias) without
anyone hand-tuning a pattern for it. See services/answer_engine.py's
SEMANTIC_BLEND and services/handbook.py's get_top_handbook_chunks() for
where these embeddings actually get used.

REQUIREMENTS
------------
- GEMINI_API_KEY must be set (same key services/embeddings.py already
  looks for). If it isn't, this script logs why and exits without
  touching the database — nothing breaks, matching just stays
  fuzzy/keyword-only until you set it and re-run.
- Safe to run against a live database — each row is a small, independent
  UPDATE; nothing is locked for the duration of the run.

USAGE
-----
    python backfill_embeddings.py                 # both tables, all rows
    python backfill_embeddings.py --faqs-only
    python backfill_embeddings.py --chunks-only
    python backfill_embeddings.py --limit 200      # e.g. to test on a slice first

COST NOTE
---------
Each row = 1 Gemini embedContent call. For ~500 new FAQs this is ~500
calls, one time, not per student message — Gemini's free tier is
generous enough for this in practice, but if you're backfilling
thousands of rows at once, --limit lets you do it in batches.
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_embeddings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faqs-only", action="store_true", help="Only backfill the faq table")
    parser.add_argument("--chunks-only", action="store_true", help="Only backfill handbook_chunks")
    parser.add_argument("--limit", type=int, default=None,
                         help="Max rows per table this run (omit for all)")
    args = parser.parse_args()

    try:
        from database import get_db
    except ImportError:
        logger.error("Could not import database.get_db — run this from the project root "
                      "(same place you'd run the Flask app from).")
        sys.exit(1)

    from services.embeddings import (
        is_configured, backfill_faq_embeddings, backfill_chunk_embeddings,
        backfill_correction_embeddings,
    )

    if not is_configured():
        logger.error(
            "GEMINI_API_KEY is not set — nothing to do. Set it (Google AI Studio has a "
            "free tier) and re-run. Matching will keep working fuzzy/keyword-only until then."
        )
        sys.exit(1)

    total = 0
    if not args.chunks_only:
        n = backfill_faq_embeddings(get_db, limit=args.limit)
        logger.info(f"FAQ rows embedded this run: {n}")
        total += n
    if not args.faqs_only:
        n = backfill_chunk_embeddings(get_db, limit=args.limit)
        logger.info(f"Handbook chunk rows embedded this run: {n}")
        total += n
    if not args.faqs_only and not args.chunks_only:
        n = backfill_correction_embeddings(get_db, limit=args.limit)
        logger.info(f"Admin correction rows embedded this run: {n}")
        total += n

    logger.info(f"Done. {total} row(s) updated.")


if __name__ == "__main__":
    main()