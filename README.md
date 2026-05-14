# Health Insurance Policy Interpreter

An agentic RAG system that answers Indian health insurance policy questions in plain English — with exact clause citations, live insurer website lookups, and a two-pane UI where the PDF scrolls to the cited page.

---

## What it does

Indian health insurance policies run 40–80 pages of legal and medical jargon. This tool lets policyholders ask natural language questions and get grounded, cited answers — before buying, after buying, or at claim time.

**Example queries it handles:**
- "Is maternity covered under this policy?" → Verdict: Partial + clause references
- "Is Apollo Hospital Bannerghatta Road in Care's cashless network?" → live web lookup
- "Hernia surgery — PED waiting or 24-month named ailment?" → multi-hop reasoning
- "What happens if I opted Smart Select and hit a non-network hospital?" → co-pay calculation

---

## Architecture

Two-agent system with deterministic retrieval:

```
User Question
    │
    ▼
Orchestrator Agent (Claude Haiku)
    ├── Tool 1: retrieve_from_policy(query)   ← deterministic, no LLM
    │           Cohere embed → Pinecone → BM25 → RRF → Cohere Rerank
    │
    └── Tool 2: search_web(intent, url, goal) ← delegates to Web Navigator
                Only for 5 trigger conditions (hospital network, grievance,
                ombudsman, Smart Select, URL deferral in policy text)
                Multi-hop crawler: explicit URL → crawl links → root domain
                Returns raw content + full failure trace

Web Navigator Agent (Claude Haiku) — called only when needed
    └── Tool: web_navigate(start_url, goal)
```

**Retrieval modes (4, switchable):**

| Mode | What it does |
|------|-------------|
| `dense` | Cohere embed + Pinecone query |
| `hybrid` | BM25 + dense, fused via RRF |
| `dense_rerank` | Dense top-20 → Cohere Rerank → top-5 |
| `hybrid_rerank` | Hybrid top-20 → Cohere Rerank → top-5 |

**Eval results (70 questions, Care Insurance Supreme):**

| Mode | Faithfulness | Context Precision | Context Recall | Answer Relevancy | Answer Correctness |
|------|-------------|------------------|----------------|-----------------|-------------------|
| dense | 0.622 | 0.768 | 0.610 | 0.472 | 0.531 |
| hybrid_rerank | 0.607 | 0.735 | 0.528 | 0.498 | 0.531 |
| dense_rerank | 0.631 | 0.676 | 0.544 | 0.445 | 0.528 |
| hybrid | 0.585 | 0.647 | 0.552 | 0.471 | 0.507 |

---

## Stack

| Component | Technology |
|-----------|-----------|
| PDF parsing | LlamaParse (agentic mode, markdown output) |
| Embedding | Cohere embed-english-v3.0 (1024 dims) |
| Vector store | Pinecone serverless (cosine similarity) |
| Sparse retrieval | BM25 (rank-bm25, JSON-serialised) |
| Reranking | Cohere Rerank v3.0 |
| Agents | LlamaIndex AgentWorkflow |
| LLM | Claude Haiku 4.5 (answers + eval judge) |
| Web crawling | httpx + custom multi-hop navigator |
| Backend | FastAPI + uvicorn |
| Frontend | Next.js 16 + react-pdf + Tailwind |
| Feedback store | SQLite via aiosqlite |
| Eval framework | Ragas (faithfulness, context precision/recall, answer relevancy/correctness) |

---

## Key features

- **Grounded answers** — every claim cites `[Clause X.X, Page N]` or a fetched live URL
- **Live data routing** — policy text defers ~7 URLs to insurer/IRDAI websites; agent fetches them
- **Two-pane UI** — citation chips scroll the PDF viewer to the exact page
- **Feedback loop** — thumbs up/down + free text logged per query for continuous improvement
- **Pre-flight test suite** — 85 tests gate every eval run (API connectivity, retrieval quality, agent behavior, checkpoint integrity)
- **Prompt caching** — orchestrator system prompt cached via Anthropic API (90% off repeated input tokens)

---

## Project structure

```
HealthInsuranceBuddy/
├── backend/
│   ├── app/
│   │   ├── agents/          # Orchestrator + Web Navigator agents
│   │   ├── ingestion/       # LlamaParse → chunker → Pinecone indexer
│   │   ├── retrieval/       # dense, sparse, fusion, rerank, retriever
│   │   ├── prompts/         # XML-structured CO-STAR / RISEN prompts
│   │   ├── feedback/        # SQLite feedback store
│   │   └── api/             # FastAPI routes: /ingest, /chat (SSE), /feedback
│   ├── eval/
│   │   ├── run_eval.py      # 4-mode parallel eval with checkpointing
│   │   └── golden_dataset.csv
│   └── tests/               # Pre-flight test suite (85 tests)
└── frontend/
    ├── app/                 # Next.js pages: landing, processing, chat
    └── components/          # PDFViewer, ChatPane, MessageBubble, FeedbackButtons
```

---

## Compliance note

This tool processes health-related queries. Under India's DPDP Act 2023:
- No PII collected at sign-up — sessions are anonymous
- Uploaded PDFs are not persisted beyond the session
- Conversation logs retained for evaluation only, deleted after 30 days
- Tool is informational only — not IRDAI-regulated advisory
