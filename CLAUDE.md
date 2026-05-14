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

# Run ingestion pipeline (one-time, after adding a PDF to data/pdfs/)
uv run python -c "
from app.config import get_settings
from app.ingestion.parser import parse_policy_pdf
from app.ingestion.url_extractor import extract_deferred_urls
from app.ingestion.chunker import chunk_pages
from app.ingestion.indexer import build_and_store_indexes, get_pinecone_index
# ... see ingestion section below
"

# Run eval across 4 retrieval modes
uv run python eval/run_eval.py --policy_id care-insurance-sample --output eval/results/

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
| `app/timing.py` | Per-step timing logger — `timing.t("label")` emits `[+Xs | total Ys]` to stdout |
| `app/api/chat.py` | SSE streaming endpoint — wraps `orchestrator.stream_chat()` |
| `eval/run_eval.py` | Ragas eval runner across 4 retrieval modes |

## Agent Prompts: Critical Constraints

Both prompts use XML-structured CO-STAR/RISEN frameworks. Do not flatten them to prose — the structure is intentional.

**Orchestrator:** `retrieve_from_policy` called exactly once (verbatim query). `search_web` called only for the 5 explicit trigger conditions in `<web_triggers>`. Second `search_web` call must use RETRY format: `"RETRY: previous attempt returned {reason}. Try a different approach for: {goal}"`.

**Web Navigator:** `web_navigate` called exactly once. Returns raw output only — no evaluation, no summarising.

## Retrieval Layer

Four switchable modes in `retriever.py`. Default is `hybrid_rerank` (BM25 + Cohere dense, top-20 candidates fused via RRF, then Cohere Rerank to top-5). Controlled via `RETRIEVAL_MODE` in `.env`.

Every chunk is tagged at ingestion with `policy_id`, `section`, `sub_clause`, `page_number`, `chunk_type`. Chunks with URL deferrals ("visit careinsurance.com/...") are flagged `URL_DEFERRAL: {url}` — the Orchestrator uses these to decide whether to call `search_web`.

## Ingestion

Run once per policy PDF. Output: Pinecone namespace `{policy_id}` + JSON BM25 index at `data/bm25_indexes/{policy_id}.json` + parent node store at `data/bm25_indexes/{policy_id}_all_nodes.json`.

Sample policy pre-indexed at `care-insurance-sample` namespace. PDF must be at `data/pdfs/care-insurance-sample.pdf`.

## Environment

Copy `backend/.env.example` → `backend/.env`. Required keys: `ANTHROPIC_API_KEY`, `LLAMA_CLOUD_API_KEY`, `COHERE_API_KEY`, `PINECONE_API_KEY`. Pinecone index name: `health-insurance-policies`, serverless, 1024 dims, cosine.

## Frontend Note

The frontend AGENTS.md warns: **this is Next.js 16.x with breaking changes**. Read `node_modules/next/dist/docs/` before modifying frontend routing or APIs. PDF viewer uses `react-pdf` with `pdfjs-dist` — worker is loaded from unpkg CDN in `PDFViewer.tsx`.

## Latency Profile (as of last timing run)

On a policy question with no web trigger:
- Orchestrator LLM (thinks + retrieve call + generates answer): ~38s total
- retrieve_from_policy (Cohere embed + Pinecone + BM25 + rerank): ~3s
- Web Navigator: not triggered for 61/70 eval questions

Token streaming via `stream_events()` emits workflow lifecycle events but not LLM token deltas in LlamaIndex 0.14.x — the fallback in `stream_chat()` awaits the full response and streams it in one chunk.
