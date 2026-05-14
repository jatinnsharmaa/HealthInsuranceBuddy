# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Agentic RAG system for interpreting Indian health insurance policies. Users ask coverage questions; the system retrieves policy clauses, optionally fetches live insurer/IRDAI web data, and returns a grounded answer with clause+page citations.

## Commands

### Backend (Python, uv)

```bash
cd backend

# Run dev server (auto-reload)
uv run uvicorn app.main:app --reload --port 8000

# Run eval across 4 retrieval modes (checkpointed — safe to re-run)
uv run python eval/run_eval.py --policy_id care-insurance-sample --output eval/results/

# Eval flags
#   --modes dense hybrid dense_rerank hybrid_rerank   (subset of modes)
#   --dataset eval/golden_dataset.csv                 (id, question, ground_truth_answer, supporting_clause, expected_verdict, category, live_data_dependent)
#   --fresh                                            (delete stale checkpoints before run)

# Add a dependency
uv add <package>
```

### Frontend (Next.js)

```bash
cd frontend
npm run dev        # dev server on :3000
npm run build      # production build
```

## Architecture

Two-agent system + deterministic retrieval layer:

```
User question
    │
    ▼
Orchestrator Agent (Claude Haiku)            app/agents/orchestrator.py
    │
    ├─ Tool 1: retrieve_from_policy()        ← deterministic Python, no LLM
    │          Cohere embed → Pinecone → BM25 → RRF → Cohere rerank
    │          Always called exactly once with the verbatim user question
    │
    └─ Tool 2: search_web()                  ← delegates to Web Navigator Agent
               Only called for 5 trigger conditions (hospital network, grievance
               officer, ombudsman, Smart Select, URL deferral in retrieved chunk)
               Hard cap: max 2 calls. Second call must include RETRY feedback.
                   │
                   ▼
               Web Navigator Agent (Claude Haiku)    app/agents/web_navigator.py
               Tool: web_navigate(start_url, goal)   app/agents/web_tools.py
               Crawls up to 3 hops. Returns raw content + failure trace.
               Never evaluates. Never answers. Raw output only.
```

The Orchestrator independently evaluates web results (not self-graded) and generates the final answer with Verdict + Sources + Disclaimer.

## Key Files

| File | Purpose |
|------|---------|
| `app/agents/orchestrator.py` | Orchestrator/Evaluator Agent — main entry point for chat |
| `app/agents/web_navigator.py` | Web Navigator Agent — navigation specialist |
| `app/agents/web_tools.py` | `web_navigate_impl()` — multi-hop crawler logic |
| `app/prompts/orchestrator_prompt.py` | CO-STAR structured prompt, XML tags, state machine, few-shot example |
| `app/prompts/web_navigator_prompt.py` | RISEN structured prompt for web navigation |
| `app/retrieval/retriever.py` | 4 retrieval modes: dense / hybrid / dense_rerank / hybrid_rerank |
| `app/ingestion/chunker.py` | MarkdownElementNodeParser → HierarchicalNodeParser, tags every chunk |
| `app/feedback/store.py` | SQLite feedback DB — `queries` + `feedback` tables, cost tracking |
| `app/timing.py` | Per-step timing logger — `timing.t("label")` emits `[+Xs | total Ys]` to stdout |
| `app/api/chat.py` | SSE streaming endpoint — auto-logs every query to feedback DB |
| `app/api/feedback.py` | `POST /api/feedback` — thumbs up/down + optional text |
| `eval/run_eval.py` | Ragas eval runner across 4 retrieval modes, checkpointed |

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/chat` | SSE stream — `agent_step`, `chunk`, `done`, `error` events |
| `POST` | `/api/feedback` | User thumbs up/down feedback, linked to query by `query_id` |
| `POST` | `/api/ingest` | Upload PDF, runs ingestion in background, returns `{job_id, policy_id}` |
| `GET` | `/api/ingest/status/{job_id}` | Poll ingestion status: queued → parsing → chunking → indexing → ready |
| `GET` | `/api/ingest/sample` | Metadata for the pre-indexed sample policy |
| `GET` | `/pdfs/*` | Static mount of `data/pdfs/` for frontend PDF viewer |

## Agent Prompts: Critical Constraints

Both prompts use XML-structured CO-STAR/RISEN frameworks. Do not flatten them to prose — the structure is intentional.

**Orchestrator:** `retrieve_from_policy` called exactly once (verbatim query). `search_web` called only for the 5 explicit trigger conditions in `<web_triggers>`. Second `search_web` call must use RETRY format: `"RETRY: previous attempt returned {reason}. Try a different approach for: {goal}"`.

**Web Navigator:** `web_navigate` called exactly once. Returns raw output only — no evaluation, no summarising.

## Retrieval Layer

Four switchable modes in `retriever.py`. Default is `hybrid_rerank` (BM25 + Cohere dense, top-20 candidates fused via RRF, then Cohere Rerank to top-5). Controlled via `RETRIEVAL_MODE` in `.env`.

Every chunk is tagged at ingestion with `policy_id`, `section`, `sub_clause`, `page_number`, `chunk_type`. Chunks with URL deferrals ("visit careinsurance.com/...") are flagged `URL_DEFERRAL: {url}` — the Orchestrator uses these to decide whether to call `search_web`.

## Feedback & Query Logging

Every chat request is automatically logged to `data/feedback.db` (SQLite). Two tables:
- `queries` — full question, answer, retrieval latency, web fetch status, token counts, estimated cost (`input_tokens * $1 + output_tokens * $5` per million)
- `feedback` — thumbs up/down signal linked to `query_id`, optional text

DB is initialized at FastAPI lifespan startup via `init_db()` in `app/feedback/store.py`.

## Ingestion

Run once per policy PDF via `POST /api/ingest` or manually. Output:
- Pinecone namespace `{policy_id}`
- BM25 index: `data/bm25_indexes/{policy_id}.json` (`corpus` + `nodes` arrays)
- Parent node store: `data/bm25_indexes/{policy_id}_all_nodes.json`

Sample policy pre-indexed at `care-insurance-sample` namespace. PDF must be at `data/pdfs/care-insurance-sample.pdf`.

## Environment

Copy `backend/.env.example` → `backend/.env`.

| Key | Default | Notes |
|-----|---------|-------|
| `ANTHROPIC_API_KEY` | — | Required |
| `LLAMA_CLOUD_API_KEY` | — | Required for PDF parsing |
| `COHERE_API_KEY` | — | Required |
| `PINECONE_API_KEY` | — | Required |
| `RETRIEVAL_MODE` | `hybrid_rerank` | dense / hybrid / dense_rerank / hybrid_rerank |
| `RETRIEVAL_TOP_K` | `5` | Final results returned to LLM |
| `RETRIEVAL_CANDIDATE_K` | `20` | Candidate pool before reranking |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated |

Pinecone index: `health-insurance-policies`, serverless, us-east-1, 1024 dims, cosine.

## Frontend Note

The frontend AGENTS.md warns: **this is Next.js 16.x with breaking changes**. Read `node_modules/next/dist/docs/` before modifying frontend routing or APIs. PDF viewer uses `react-pdf` with `pdfjs-dist` — worker is loaded from unpkg CDN in `PDFViewer.tsx`.

## Latency Profile

On a policy question with no web trigger (avg across 70 eval questions):
- Full question latency: ~18s avg (4s min, 60s max)
- `retrieve_from_policy` (Cohere embed + Pinecone + BM25 + rerank): ~3s
- Web Navigator: not triggered for ~61/70 eval questions

Token streaming: `stream_events()` emits workflow lifecycle events but not LLM token deltas in LlamaIndex 0.14.x — `stream_chat()` awaits the full response and streams it in one chunk.
