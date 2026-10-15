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

The chat pipeline classifies the question, handles schedule/date requests and follow-ups, and searches handbook sections and FAQs. Search-time language normalization expands supported vernacular spellings and hypothetical wording. FAQ and handbook matching use lexical scores; semantic matching uses Gemini embeddings when the required Gemini key and stored content embeddings are available. An AI section router helps select handbook topics. Generated answers are checked against their sources and question relevance; when the AI check is unavailable, the generated answer is rejected in favor of a source-based fallback where available.

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

Authenticated admins can request `GET /api/admin/retrieval_status` to see FAQ and active-handbook embedding coverage, whether each corpus currently has semantic matches available, Gemini embedding call/cache/failure counters, and separate Groq/Gemini chat-provider success/failure counts. Coverage is read from Postgres; provider and embedding call counters are process-local and reset when a worker restarts. The endpoint does not return API keys or raw provider responses.

## CI/CD

`.github/workflows/ci-cd.yml` runs on pull requests targeting `main`, pushes to `main`, and manual dispatches. Its checks install `requirements.txt`, compile the Python modules, and run the offline template tests. The live chatbot regression runner is intentionally not part of CI because it needs a running Postgres-backed app and makes real model calls.

After checks pass, pushes to `main` are eligible for a Vercel production deploy. Add these GitHub repository or `production` environment secrets to enable the workflow deploy job:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

Without these secrets, the workflow reports that deployment was skipped. The app's database URL and AI provider keys belong in Vercel environment variables, not GitHub Actions.

If the Vercel project is already connected to GitHub for automatic deployments, disable its automatic Git deployments before enabling the workflow secrets; otherwise the same commit can deploy twice. Vercel project metadata under `.vercel/` is ignored and should not be committed.
