# PAC Helpdesk

Flask-based student helpdesk for Philippine Advent College (PAC). It serves a public landing page and chat, answers handbook and FAQ questions, handles schedule queries, and provides authenticated admin tools for content, reports, corrections, ratings, and self-heal drafts.

## Runtime

- `app.py` creates the Flask app, configures security and rate limiting, registers blueprints, initializes the Postgres pool, and runs startup schema checks.
- `routes/` contains Flask blueprints for chat, admin pages, schedules, APIs, public pages, and cron jobs.
- `services/` contains the FAQ, handbook, date/schedule, language, retrieval, grounding, and conversation logic.
- `database/` contains the Postgres connection pool and schema setup. `database/full_schema.sql` is the reference schema; startup also runs idempotent setup/migrations.
- `ai_providers/` contains Groq/Gemini adapters, key rotation, fallback routing, circuit breaking, caching, and model metrics.
- `templates/public/`, `templates/admin/`, and `templates/errors/` group browser templates by purpose. Static assets are under `static/`.

## Answering Flow

The chat pipeline classifies the question, handles schedule/date requests and follow-ups, and searches handbook sections and FAQs. For school-related questions, a cached model-driven interpretation identifies the intended meaning, requested answer focus, constraints, and alternative search queries; it is not a source of facts and does not rely on topic-specific question rules. Those search queries join the original wording in FAQ and handbook retrieval. FAQ and handbook matching use lexical scores; semantic matching uses Gemini embeddings when the required Gemini key and stored content embeddings are available. Handbook retrieval asks an AI section router to choose from the current database section titles before running targeted section lookup, then merges lexical and semantic candidates. Topic-specific rules are emergency fallbacks only when the router is unavailable; selected sections receive a ranking boost, with a separate boost for strong semantic matches. Generated answers and their self-checks receive the interpretation so they can distinguish the requested fact from related content. Newly retrieved FAQ answers, including exact-question hits, and handbook answers must pass language, grounding, and direct-intent checks; if validation is unavailable, ambiguous, or unsuccessful, the system withholds the answer instead of returning unvalidated source text. An unrecognized source-routing verdict also abstains. If interpretation is unavailable, retrieval continues, but candidate answers still must pass validation.

Dates and closures are looked up from Postgres rather than inferred by the language model. The public API returns JSON; browser 404 and 500 errors use the shared PAC error page, while API errors remain JSON.

## Learning Loop

Learning updates the helpdesk's reviewed knowledge and retrieval data; it does not fine-tune a model or let student submissions become trusted answers automatically. Thumbs-up answers can enter the successful-answer cache. Thumbs-down reports are stored for review, and students may optionally suggest corrected wording. That suggestion is queued beside any existing self-heal draft and remains unverified. Admins can compare suggestions, edit the answer, and explicitly approve it into the corrections store. Repeated unanswered questions are also retried against current FAQ/handbook content to create drafts for review. Approved corrections are matched against high-confidence paraphrases; dismissals and low-confidence matches do not alter trusted knowledge.

## AI Providers

`AI_PROVIDER` selects the primary chat provider (`groq` by default, or `gemini`). The other provider is used as fallback. Defaults are:

- Groq primary model: `openai/gpt-oss-120b` (`GROQ_MODEL`)
- Groq fast model: `openai/gpt-oss-20b` (`FAST_GROQ_MODEL`)
- Gemini fallback model: `gemini-3.5-flash-lite` (`GEMINI_MODEL`)
- Gemini embedding model: `models/gemini-embedding-001`

GPT-OSS requests use low reasoning effort and suppress reasoning output so short classification and verification calls still return visible answers. Gemini chat and embedding calls rotate through configured keys after quota or authentication failures. Keep provider keys private; never commit `.env`.

## Setup

Use Python 3.10 or newer. Install dependencies in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local `.env` with the required settings. `FLASK_SECRET_KEY` and a key for the selected `AI_PROVIDER` are required. Gemini embeddings require `GEMINI_API_KEY`, even when Groq is the primary chat provider.

```dotenv
FLASK_SECRET_KEY=replace-with-a-long-random-value
AI_PROVIDER=groq
GROQ_API_KEY=your-groq-key
GEMINI_API_KEY=your-gemini-key

DB_HOST=localhost
DB_PORT=5432
DB_NAME=school_chatbot
DB_USER=postgres
DB_PASSWORD=your-local-postgres-password
DB_SSLMODE=disable
```

Additional keys can be configured as `GROQ_API_KEY_2` through `GROQ_API_KEY_8` and `GEMINI_API_KEY_2` through `GEMINI_API_KEY_8`. Model IDs can be overridden with `GROQ_MODEL`, `FAST_GROQ_MODEL`, and `GEMINI_MODEL`.

FAQ image uploads in hosted/read-only deployments use Cloudinary. Configure either the Cloudinary-provided `CLOUDINARY_URL` in the format `cloudinary://API_KEY:API_SECRET@CLOUD_NAME`, or set `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, and `CLOUDINARY_API_SECRET` separately. A value of just `cloudinary` is not a valid URL. Local development continues to save images under `UPLOAD_FOLDER`.

For hosted Postgres, set `DATABASE_URL` (or `POSTGRES_URL`) and `DB_TARGET=neon`, or run in a supported online environment where the app automatically prefers the URL. Hosted connections use SSL.

Start the app after Postgres is reachable:

```powershell
python app.py
```

The default local URL is `http://127.0.0.1:5001`.

## Tests

`test_chatbot.py` is a live integration/regression runner, not an offline unit suite. Start the app and database first, then run:

```powershell
python test_chatbot.py
```

It sends the mixed question set to `/chat`, checks required/forbidden answer details for known regressions, and writes `test_results_mixed.json` and `test_results_mixed.md`. Calls use real provider keys and can take several minutes. Reports separate explicitly verified answers from answers that passed only basic checks, fallback replies, incorrect answers, and request errors. Add expected facts for every question where the source supports a stable check; unconfigured questions are marked unverified, not factual successes. The suite cannot guarantee accuracy for unseen questions.

### Retrieval Readiness

Authenticated admins can request `GET /api/admin/retrieval_status` to see FAQ and active-handbook embedding coverage, whether each corpus has complete semantic coverage, whether the optional pgvector/FTS/trigram indexes are enabled and available, Gemini embedding call/cache/failure counters, retrieval-expansion and question-interpretation call/cache/failure/latency counters, and separate Groq/Gemini chat-provider success/failure counts. `semantic_ready` now means every row in that corpus has an embedding; partial coverage remains visible in `coverage_percent`. Coverage is read from Postgres; provider, embedding, and retrieval/interpretation counters are process-local and reset when a worker restarts. The endpoint does not return API keys or raw provider responses. Unanswered questions also record the detected language and broad topic to help admins prioritize recurring gaps; approving one as an FAQ keeps the original student phrasing as a related question for later retrieval.

### Retrieval Evaluation and Multilingual FAQ Data

Measure FAQ retrieval separately from answer quality with a reviewed case file:

```json
[
  {
    "question": "Unsaon pag-enroll sa PAC?",
    "expected_faq_id": 1,
    "language": "bisaya"
  }
]
```

Run the cases against the configured database and embedding service with:

```powershell
python .\evaluate_faq_retrieval.py .\retrieval_cases.json
```

The report includes Recall@1, Recall@3, and MRR overall and by language. The offline unit suite also checks the canonical English questions in the checked-in FAQ backup; that backup contains 15 unique question texts repeated across 45 rows, so it is only a smoke baseline, not a 100+ multilingual gold set. Do not treat generated translations as verified test data: use real student phrasings, map each to the correct FAQ ID, and have a bilingual reviewer approve them. Answer correctness still needs its own fact-level checks.

The existing FAQ `related_questions` field is enough for the current corpus; a separate variants table is not needed yet. Add reviewed English, Bisaya, and Tagalog phrasings in the admin FAQ editor, separated by `|`. They are fuzzy-matched and included in the FAQ embedding. A dedicated table becomes useful if variant-level approval, source, language, or usage analytics are needed.

For larger FAQ/handbook corpora, optional indexed hybrid retrieval is available through `database/migrations/20261122_hybrid_pgvector.sql`. Back up the database and verify that the Postgres provider supports pgvector 0.7+ before running it manually; it uses 3072-dimension half-precision vectors, matching the current Gemini embedding model default. The migration adds FAQ full-text/trigram indexes and HNSW indexes for FAQ and handbook vectors. After applying it, set `PGVECTOR_HYBRID_SEARCH=1`; until then the app stays on the JSONB/Python path and makes no extra database calls. Check embedding coverage and run `python .\services\backfill_embeddings.py` for missing vectors. Use `--refresh` to recompute existing vectors after changing embedded text or handbook key facts; this makes additional Gemini calls and should be run as a deliberate maintenance job.

Question interpretation is general rather than tied to a catalog of question types: it should distinguish requests such as a creator's name from a request for a song's contents, or a submission location from enrollment requirements. This model-generated focus guides retrieval and answer validation; the FAQ or handbook remains the factual authority. If the source has context but omits the specific fact requested, the system declines to guess instead of substituting a nearby fact. An unanswered phrasing can then be reviewed by an admin, who can add the missing authoritative content without adding new question-form rules.

## CI/CD

`.github/workflows/ci-cd.yml` runs on pull requests targeting `main`, pushes to `main`, and manual dispatches. Its checks install `requirements.txt`, compile the Python modules, and run the offline template tests. The live chatbot regression runner is intentionally not part of CI because it needs a running Postgres-backed app and makes real model calls.

After checks pass, pushes to `main` are eligible for a Vercel production deploy. Add these GitHub repository or `production` environment secrets to enable the workflow deploy job:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

Without these secrets, the workflow reports that deployment was skipped. The app's database URL and AI provider keys belong in Vercel environment variables, not GitHub Actions.

If the Vercel project is already connected to GitHub for automatic deployments, disable its automatic Git deployments before enabling the workflow secrets; otherwise the same commit can deploy twice. Vercel project metadata under `.vercel/` is ignored and should not be committed.

## Production security configuration

- Set `ADMIN_PASSWORD` to a strong secret before the first database bootstrap. It is only used to create the initial `admin212004` account; rotate an existing account's password through the admin profile.
- Set `CRON_SECRET`; cron requests must send `Authorization: Bearer <secret>`. URL query-string secrets are not accepted.
- Set `RATELIMIT_STORAGE_URI` to a shared Redis-compatible storage URI (for example `rediss://<host>:<port>/0`) in production. The in-memory fallback is per-instance and the app logs a warning when it is used in production.
- `TRUSTED_PROXY_HOPS` controls trusted forwarding hops outside Vercel. Leave it at `0` unless the app is behind a known, trusted proxy.
