"""
Diagnostic script: checks whether the system can properly fetch data
when a user asks a question.

Checks:
  1. DB connection + FAQ table
  2. Handbook chunks table
  3. Embedding column presence
  4. Hybrid-search index availability (pgvector)
  5. AI provider keys configured (Groq / Gemini)
  6. Embedding generation (get_embedding call)
  7. Full FAQ retrieval pipeline (_get_all_faqs + _get_faq_candidates)

Run with:
    cd c:\\Users\\HP\\PycharmProjects\\HELPDESK_POSGRES
    python check_data_fetch.py
"""

import sys
import os
import traceback

# ── load .env so DATABASE_CONFIG / keys are populated ─────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[ENV]  .env loaded ✓")
except ImportError:
    print("[ENV]  python-dotenv not installed; relying on shell environment")

# ── 1. Database connection ─────────────────────────────────────────────────────
print("\n── 1. Database connection ──────────────────────────────────────────────")
try:
    from database import init_pool, get_db
    init_pool()
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT version()")
    ver = cur.fetchone()
    print(f"  ✓ Connected  –  {ver[0][:60]}")
    cur.close()
    conn.close()
    DB_OK = True
except Exception as exc:
    print(f"  ✗ FAILED: {exc}")
    DB_OK = False

# ── 2. FAQ table ───────────────────────────────────────────────────────────────
print("\n── 2. FAQ table ────────────────────────────────────────────────────────")
FAQ_COUNT = 0
if DB_OK:
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM faq")
        FAQ_COUNT = cur.fetchone()[0]
        cur.execute("SELECT id, topic FROM faq LIMIT 3")
        samples = cur.fetchall()
        print(f"  ✓ faq rows: {FAQ_COUNT}")
        for s in samples:
            print(f"      id={s[0]}  topic={s[1]}")
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        DB_OK = False
else:
    print("  SKIPPED (no DB)")

# ── 3. handbook_chunks table ───────────────────────────────────────────────────
print("\n── 3. handbook_chunks table ────────────────────────────────────────────")
if DB_OK:
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM handbook_chunks WHERE is_active = 1")
        hb_count = cur.fetchone()[0]
        print(f"  ✓ active handbook_chunks: {hb_count}")
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")

# ── 4. pgvector / hybrid-search indexes ───────────────────────────────────────
print("\n── 4. pgvector / hybrid-search indexes ─────────────────────────────────")
if DB_OK:
    try:
        from services.hybrid_search import get_hybrid_search_status
        status = get_hybrid_search_status(get_db)
        enabled = status.get("enabled")
        available = status.get("available")
        ext_ver = status.get("extension_version", "n/a")
        indexes = status.get("indexes", {})
        flag = "✓" if available else ("⚠ disabled/partial" if enabled else "⚠ PGVECTOR_HYBRID_SEARCH not set")
        print(f"  {flag}  enabled={enabled}  available={available}  pgvector={ext_ver}")
        for k, v in indexes.items():
            mark = "✓" if v else "✗"
            print(f"      {mark}  {k}")
        if not enabled:
            print("  → Set PGVECTOR_HYBRID_SEARCH=true in .env to enable vector/FTS search")
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")

# ── 5. AI provider keys ────────────────────────────────────────────────────────
print("\n── 5. AI provider keys ─────────────────────────────────────────────────")
try:
    from config import AI_PROVIDER, GROQ_API_KEY, GEMINI_API_KEY
    from ai_providers.providers import GROQ_API_KEYS, GEMINI_API_KEYS
    print(f"  AI_PROVIDER = {AI_PROVIDER}")
    print(f"  Groq  keys configured : {len(GROQ_API_KEYS)}  {'✓' if GROQ_API_KEYS else '✗ MISSING'}")
    print(f"  Gemini keys configured: {len(GEMINI_API_KEYS)}  {'✓' if GEMINI_API_KEYS else '⚠ none (optional)'}")
    KEYS_OK = bool(GROQ_API_KEYS) if AI_PROVIDER == "groq" else bool(GEMINI_API_KEYS)
except Exception as exc:
    print(f"  ✗ FAILED: {exc}")
    KEYS_OK = False

# ── 6. Embedding generation ────────────────────────────────────────────────────
print("\n── 6. Embedding generation ─────────────────────────────────────────────")
EMBED_OK = False
try:
    from services.embeddings import get_embedding
    test_text = "What is the dress code policy?"
    vec = get_embedding(test_text)
    if vec and len(vec) > 0:
        print(f"  ✓ embedding generated  dim={len(vec)}  first3={[round(v,4) for v in vec[:3]]}")
        EMBED_OK = True
    else:
        print("  ✗ get_embedding returned empty/None")
except Exception as exc:
    print(f"  ✗ FAILED: {exc}")
    traceback.print_exc()

# ── 7. Full FAQ retrieval pipeline ────────────────────────────────────────────
print("\n── 7. FAQ retrieval pipeline ───────────────────────────────────────────")
TEST_QUESTION = "What is the dress code policy for students?"
if DB_OK and FAQ_COUNT > 0:
    try:
        from services.faq import _get_all_faqs, _get_faq_candidates
        from services.language import detect_language

        all_faqs = _get_all_faqs()
        print(f"  ✓ _get_all_faqs() returned {len(all_faqs)} rows")

        detected_lang = detect_language(TEST_QUESTION)
        print(f"  ✓ detect_language = '{detected_lang}'")

        q_embedding = None
        if EMBED_OK:
            from services.embeddings import get_embedding
            q_embedding = get_embedding(TEST_QUESTION)

        candidates = _get_faq_candidates(
            TEST_QUESTION, detected_lang, all_faqs,
            query_embedding=q_embedding,
        )
        print(f"  ✓ _get_faq_candidates() found {len(candidates)} candidate(s)")
        for c in candidates[:3]:
            row = c.get('faq') or ()
            topic = row[3] if len(row) > 3 else '?'
            q_text = row[1][:60] if len(row) > 1 and row[1] else '?'
            print(f"      score={c.get('combined', 0):.3f}  topic={topic!r}  q={q_text!r}")

        if not candidates:
            print("  ⚠ No candidates returned — check FAQ data or scoring thresholds")
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        traceback.print_exc()
else:
    print("  SKIPPED (DB not ready or FAQ table empty)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
checks = {
    "DB connection":          DB_OK,
    "FAQ rows present":       FAQ_COUNT > 0,
    "AI provider keys":       KEYS_OK,
    "Embedding generation":   EMBED_OK,
}
all_ok = True
for name, ok in checks.items():
    mark = "✓" if ok else "✗"
    if not ok:
        all_ok = False
    print(f"  {mark}  {name}")

print()
if all_ok:
    print("✅ System looks healthy — data fetching pipeline is functional.")
else:
    print("⚠  One or more checks failed. See details above.")
