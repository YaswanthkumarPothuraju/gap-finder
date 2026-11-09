# GapFinder AI

A local FastAPI study-diagnostics app. Upload a syllabus to create a topic map, add notes or textbook chapters, take quizzes grounded in cited passages, review topic-level mastery, and generate a time-bounded revision plan. Persistence uses SQLAlchemy models and engine transactions over the existing local SQLite database.

## Run on Windows

Use Python 3.13 for this app. Python 3.14 is not currently supported by the
pure-Python Pydantic compatibility dependency used here.

From this folder in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:DISABLE_SQLALCHEMY_CEXT = "1"
python -m pip install --no-binary sqlalchemy -r requirements.txt
python -m uvicorn main:app --reload
```

SQLAlchemy is installed without its optional compiled extensions because Windows Application Control may block native modules in some environments.

Open <http://127.0.0.1:8000>. The local demo login is `demo` / `demo`.

The app stores uploads' extracted text and quiz data in `gapfinder.db` in this folder. Original files are not retained. Scanned PDFs require OCR and are rejected with an explanation. Uploads are limited to 10 MB.

## AI provider

Without a key, the app uses local lexical retrieval and a basic deterministic short-answer grader. Quiz generation skips syllabus-title entries and requires sufficiently relevant study passages; retrieval searches uploaded passage text and section headings, including when a passage was not mapped to the topic. Fallback answers are selected from sentences most relevant to each topic. If a provider denies access or hits rate limits during generation, the app automatically retries with backoff and falls back to grounded local questions if limits persist. Choose **OpenAI** or **Groq** in the dashboard, enter a key issued by that provider, and click **Save key**. The default provider is OpenAI; Groq is used only when explicitly selected. Groq uses its OpenAI-compatible chat endpoint and `llama-3.3-70b-versatile` by default (or other Groq models like `llama-3.1-8b-instant`). Groq embeddings are not used, so retrieval remains local and lexical.

```powershell
$env:GAPFINDER_API_KEY = "your-key"
$env:GAPFINDER_PROVIDER = "groq"
# Optional: override the header name and prefix used when sending the key.
# OpenAI-compatible providers commonly expect: Authorization: Bearer <key>
$env:GAPFINDER_API_KEY_HEADER = "Authorization"
$env:GAPFINDER_API_KEY_PREFIX = "Bearer"
$env:GAPFINDER_CHAT_MODEL = "llama-3.3-70b-versatile"
# Optional for a compatible provider:
$env:GAPFINDER_API_BASE_URL = "https://api.groq.com/openai/v1"
python -m uvicorn main:app --reload
```

If your provider expects a different header (for example `X-API-Key` or `api-key`), set `GAPFINDER_API_KEY_HEADER` to that name and leave `GAPFINDER_API_KEY_PREFIX` blank when the value should be sent raw. If the key is already formatted as `Bearer abc123`, set `GAPFINDER_API_KEY` to the full value and keep the prefix blank to avoid double-prefixing.

The app sends retrieved passages to that provider when AI features are enabled. Uploaded passage text is treated as untrusted evidence, and returned citation IDs are checked against the retrieved set. Provider output still requires evaluation; the verifier is not a guarantee of factual correctness.

Dashboard-entered API keys are held in server process memory, not stored in the browser cookie or on disk; re-enter the key after restarting the app. Use **Test provider and model access** on the dashboard to check whether the configured model is listed for the saved key. Quiz errors identify whether question generation or answer verification failed and include the configured provider and model.

For anything beyond local development, set `GAPFINDER_DEMO_USERNAME`, `GAPFINDER_DEMO_PASSWORD`, and a stable, private `GAPFINDER_SESSION_SECRET`. Set `GAPFINDER_COOKIE_SECURE=true` behind HTTPS. This is a single-user prototype, not a production multi-tenant deployment.

## Current scope

- PDF, TXT, and Markdown upload, text extraction, chunking, and citations.
- Hierarchical syllabus heading/topic extraction, editable topic names, and topic merging.
- Topic map deletion that keeps uploaded files and quiz history while removing the map's revision plan.
- Topic-mapped diagnostic quizzes (up to 30 topics per attempt, with additional questions weighted by past-paper frequency) and three-question targeted probes.
- Deterministic mastery, confidence labels, gap reports, and bounded study plans.
- Material-scoped explain helper and optional OpenAI-compatible generation, grading, and embeddings.
- Local SQLite persistence and demo login.

OCR, multi-user accounts, teacher dashboards, external LMS integrations, and validated model quality metrics are not included.
