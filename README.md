# PAC Helpdesk — modular edition

Two things changed from the old single-file `app.py`:

## 1. Code organization

The old `app.py` (2500+ lines) is now split like this:

```
pac_helpdesk/
├── app.py                    # app factory only — wires everything together
├── config.py                 # ⚠️ PLACEHOLDER — drop your real config.py in here
│
├── ai_providers/              # everything about talking to Groq/Gemini/Cloudflare/Ollama
│   ├── circuit_breaker.py
│   ├── metrics.py
│   ├── cache.py               # AICache + FAQCache (TTL-based)
│   └── providers.py           # the actual API calls + NEW fallback chain
│
├── database/
│   ├── pool.py                 # get_db(), db_cursor() context manager
│   └── schema.py                # startup migrations
│
├── services/                  # business logic, no Flask/route code
│   ├── conversation.py         # per-session chat memory
│   ├── dates.py                 # ⭐ rewritten — see below
│   ├── schedule.py              # pure DB fact-lookups for dates.py
│   ├── handbook.py              # student handbook Q&A (AI-grounded)
│   ├── answer_engine.py         # FAQ fuzzy matching + AI extraction
│   ├── faq.py                   # find_answer() — ties it all together
│   ├── language.py              # language detection, greetings, abbreviations
│   ├── alerts.py                # admin email alerts + scheduled jobs
│   └── handbook_ingest.py       # PDF → handbook_chunks ingestion
│
├── routes/                    # Flask blueprints — thin, call into services/
│   ├── chat.py                  # /chat, /api/satisfaction, /api/chat_rating...
│   ├── admin.py                  # /admin, /dashboard, FAQ CRUD
│   ├── schedules.py               # /api/dynamic_schedules, /api/emergency_closures
│   └── api.py                     # /health, /metrics, /api/model_status...
│
└── utils/
    ├── text.py                    # normalize_input, is_valid_question, fuzzy helpers
    └── security.py                  # login lockout, rate limiting, fingerprinting
```

**What's fully ported:** the entire chat pipeline (dates → follow-up →
handbook → FAQ), all of `ai_providers`, `database`, `utils`, and the
highest-traffic admin/schedule endpoints.

**What's a starting pattern, not a full port:** a handful of secondary
admin routes (bulk import/export, reports/history pages, greetings &
abbreviations editors, handbook chunk editor, alert log, digest sending).
These are plain CRUD that don't interact with the AI/date changes — copy
them from the old `app.py` into `routes/admin.py` / `routes/api.py`
following the exact pattern already there (`if not admin_required(): ...`
→ `get_db()`/cursor block). `templates/` and `static/` are unchanged and
just need to be copied over as-is.

**Also addressed from the review, while restructuring:**
- Model fallback chain (`ai_providers/providers.py: call_with_fallback`) —
  if the configured provider fails, it now tries the others instead of
  just returning `""`.
- TTL-based cache cleanup (`ai_providers/cache.py`) instead of relying on
  maxsize eviction alone.
- `database/pool.py` adds a `db_cursor()` context manager for new code
  (existing call sites' manual `try/finally` were left as-is — that's a
  separate, larger refactor).
- The `/test_email_design` public debug route from the old app was
  **dropped**, not ported (it was flagged as a real exposure).

## 2. Natural-sounding date answers ⭐

This was the actual ask, so here's the reasoning in more detail.

**The old code** (`_day_phrase`, `build_class_office_response`,
`_build_formatted_response`, `build_date_response`,
`build_detailed_date_range_response` — ~400 lines total) picked from a
fixed list of pre-written sentences using a hash of the date as a
"random" seed. That's why it read like a bot: it *was* one, reading off
cue cards.

**The new version** (`services/dates.py`) works the same way your
handbook and FAQ answers already do:

1. **Gather facts — deterministically, no AI involved.**
   `gather_single_day_facts()` / `gather_range_facts()` still do exactly
   what the old code did: check emergency closures, named holidays, and
   per-department/grade schedules straight from the DB. This part is
   *not* allowed to be guessed — accuracy matters more than tone.

2. **Phrase it — AI, grounded strictly in those facts.**
   `phrase_naturally()` hands the facts as JSON to the model with
   instructions to reply the way a person would text back — 1-3 short
   sentences, answering what was actually asked (not reciting every
   department's status when the student only asked about one), no
   invented details.

3. **Verify — a second AI pass checks the phrasing didn't drift.**
   Same pattern as `answer_handbook_question()`'s verification step: ask
   the model "does this reply state only what's in the facts?" and retry
   once if it says no.

4. **Fallback — if the AI is unreachable** (circuit breaker open),
   `_fallback_sentence()` gives one honest, plain sentence built from the
   facts. Not a variety pack of pre-written lines — just the truth,
   plainly stated, so the bot never goes silent.

Example — old vs. new, same underlying facts:

> **Old:** "🚫 High School — no class, everything's closed" (picked from a
> template array)
>
> **New:** "No classes at the High School tomorrow — it's the founding
> anniversary, so the office is closed too. Enjoy the day off!"

The same `facts → phrase → verify` approach is what `handbook.py` and
`faq.py`/`answer_engine.py` already did — dates were the odd one out with
hand-written templates, so this brings it in line with the rest of the
bot's voice instead of introducing a fourth style.

## Before running

1. Drop your real `config.py` into the project root (see the placeholder
   file for the exact names it needs to export).
2. Copy `templates/` and `static/` from the old project unchanged.
3. Port the remaining secondary admin routes noted above if you use them.
4. `pip install -r requirements.txt` (same dependencies as before —
   nothing new was added except what the fallback chain already needed).
