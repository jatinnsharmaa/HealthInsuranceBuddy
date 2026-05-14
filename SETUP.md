# Setup Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [gh](https://cli.github.com/) (GitHub CLI, for first-time push)

## API keys required

Get these before starting:

| Key | Where | Free tier |
|-----|-------|-----------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Pay-per-use |
| `LLAMA_CLOUD_API_KEY` | [cloud.llamaindex.ai](https://cloud.llamaindex.ai) | 1,000 pages/day |
| `COHERE_API_KEY` | [dashboard.cohere.com](https://dashboard.cohere.com) | Production key (not trial) |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) | 1 free serverless index |

---

## Backend setup

```bash
cd backend

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env and fill in your API keys
```

`.env` values to fill in:
```
ANTHROPIC_API_KEY=sk-ant-...
LLAMA_CLOUD_API_KEY=llx-...
COHERE_API_KEY=...
PINECONE_API_KEY=pcsk_...
```

The remaining values in `.env` can stay as defaults.

---

## Ingest the policy PDF

Place your Care Insurance Supreme PDF at:
```
backend/data/pdfs/care-insurance-sample.pdf
```

Then run ingestion (one-time, ~60 seconds):

```bash
cd backend
uv run python -c "
from app.config import get_settings
from app.ingestion.parser import parse_policy_pdf
from app.ingestion.url_extractor import extract_deferred_urls
from app.ingestion.chunker import chunk_pages
from app.ingestion.indexer import build_and_store_indexes, get_pinecone_index

s = get_settings()
policy_id = 'care-insurance-sample'

print('Parsing...')
pages, n = parse_policy_pdf('data/pdfs/care-insurance-sample.pdf', s.llama_cloud_api_key)
print(f'{n} pages parsed')

print('Chunking...')
deferred_urls = extract_deferred_urls(pages)
leaf_nodes, all_nodes = chunk_pages(pages, policy_id, anthropic_api_key=s.anthropic_api_key)

print('Indexing...')
idx = get_pinecone_index(s.pinecone_api_key, s.pinecone_index_name)
stats = build_and_store_indexes(leaf_nodes, all_nodes, policy_id,
    s.cohere_api_key, s.pinecone_api_key, s.pinecone_index_name, s.data_dir)
print(f'Done. {stats[\"leaf_count\"]} chunks indexed.')
"
```

---

## Run the backend

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/api/ingest/sample`

---

## Frontend setup

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Run the frontend:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Run tests (before eval)

```bash
cd backend
uv run pytest tests/ -v --tb=short
```

All 85 tests must pass before running eval. Fails indicate misconfigured API keys, missing index, or stale files.

---

## Run eval

```bash
cd backend

# Fresh run — all 4 retrieval modes in parallel
uv run python eval/run_eval.py \
  --policy_id care-insurance-sample \
  --output eval/results/ \
  --fresh

# Resume interrupted run (skips already-completed questions)
uv run python eval/run_eval.py \
  --policy_id care-insurance-sample \
  --output eval/results/
```

Results saved to `eval/results/eval_summary_<timestamp>.json`.

---

## Common issues

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY missing` in Ragas | Ragas defaulted to OpenAI — the eval script sets Haiku explicitly, check you're running from `backend/` |
| `BM25 index not found` | Re-run ingestion |
| Cohere `TooManyRequestsError` | You're on a trial key — get a production key from [dashboard.cohere.com](https://dashboard.cohere.com) |
| `400 Bad Request` from Anthropic in eval | Agent memory accumulated — eval creates fresh agent per question, should not happen in current version |
| PDF viewer shows blank | Check `NEXT_PUBLIC_API_URL` in `frontend/.env.local` matches your backend port |
