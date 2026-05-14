# Project Initiation Document: Health Insurance Policy Interpreter

---

## 1. Problem Statement

Health insurance policies in India run 40-80 pages of legal and medical jargon. Buyers sign without reading them, and when a claim gets denied, they have no way to check whether the rejection is actually valid. The asymmetry between insurer and policyholder is total.

A plain-English interpreter that answers questions like "is knee surgery covered if arthritis is listed as a pre-existing condition" or "what's the waiting period for maternity benefits" is small, useful, and absent from the market.

---

## 2. Objectives and Scope

### 2.1 Target Users

Policyholders, across three stages:

- **Pre-purchase:** Check eligibility for themselves and family; identify rigid clauses.
- **Post-purchase:** Verify coverage for a condition or treatment.
- **At claim time:** Determine whether a denial is valid.

---

## 3. Use Cases

**3.1 Instant Coverage Verdict**
User describes a situation. The agent decomposes it, pulls definitions + benefits + exclusions + optional covers, and returns Yes / No / Partial with exact page and clause references, plus next steps.

**3.2 Scenario Simulator**
"Run my upcoming hernia surgery through the policy." The agent applies deductibles, co-pays, room rules, waiting periods, and recharge logic end to end.

**3.3 Red Flag Scanner**
User uploads a treatment estimate or diagnosis. The agent highlights likely denial reasons or reductions before the claim is filed.

**3.4 Policy Health Check**
At renewal or post-purchase: "Review my policy for gaps given my age and conditions." The agent surfaces coverage weaknesses proactively.

**3.5 Document and Web Bridge**
"Does the latest IRDAI rule change my AYUSH cover?" or "Show current network hospitals for my city." The agent crosses the static policy document with live web data to answer.

---

## 4. Technical Architecture

### 4.1 Approach: Agentic RAG

The core design is not simple RAG (chunk → embed → store → retrieve) but agentic RAG with decision loops and sub-agents:

- Query rewriting / decomposition
- Routing to the right source (vector DB, web, API calls)
- Response generation and conversational persistent memory
- Validation (e.g., Corrective RAG checks grounding and completeness)
- Loop until good or admit limits

### 4.2 Proposed Stack

| Component | Tool |
|---|---|
| PDF parsing | LlamaParse (Agentic mode) |
| Indexing + retrieval | LlamaIndex with hybrid search |
| Sparse retrieval | BM25 |
| Dense retrieval | OpenAI / Cohere embeddings |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cohere Rerank |
| Orchestration | LlamaIndex Workflows |
| Reasoning + generation | Claude Sonnet |
| Vector storage | Pinecone (sparse-dense index) |
| Observability + tracing | Langfuse |
| Offline evaluation | Ragas |

**Why hybrid + reranker.** Insurance documents are half prose, half tables. They contain clause numbers ("Clause 3.2," "Annexure II"), exact waiting periods ("24 months"), and named exclusions. Dense embeddings handle the semantic side but silently miss exact-term matches. BM25 catches those. RRF fusion combines the two candidate lists. A cross-encoder reranker (Cohere) then re-scores the top 20 for precision before sending the top 5 to the LLM.

This is the minimum viable retrieval stack for a document-grounded product in 2026. Pure vector retrieval is a known failure mode on text-and-table corpora.

### 4.3 Tools

The agent decides which tools to call based on the question. A factual question about a waiting period only needs `policy_query_engine`. A hospital coverage question needs `policy_query_engine` + `web_fetch`. A dispute question needs all of them.

| Tool | Purpose |
|---|---|
| `policy_query_engine` | Retrieves from the parsed, indexed policy document (hybrid + reranked) |
| `domain_knowledge_engine` | Retrieves from IRDAI circulars, Ombudsman awards |
| `web_fetch` | Fetches live URLs referenced in the policy |
| `hospital_network_checker` | Structured query against insurer's live network list |
| `irdai_regulation_lookup` | Checks current IRDAI guidelines (these also change) |

### 4.4 Implementation Notes (Recommended)

The following are recommended technical choices for the build, validated against current LlamaIndex documentation and verified for compatibility with the rest of the stack.

**Parsing.** LlamaParse in Agentic mode, outputting markdown. Agentic mode handles table-heavy documents better than the default mode and preserves structural cues that downstream chunking relies on.

**Chunking.** Two-step approach. First, `MarkdownElementNodeParser` to preserve tables as intact nodes (essential for Schedule of Benefits, sub-limit tables, exclusion lists). Then `HierarchicalNodeParser` with default chunk sizes [2048, 512, 128] on the remaining text nodes. The hierarchy enables AutoMergingRetriever to expand back to parent context when warranted.

**Indexing.** Pinecone with cosine similarity metric, 1024 dimensions (matches Cohere embed-english-v3). Use ServerlessSpec for the index. Cosine is the standard pairing with Cohere embeddings per Pinecone's own integration documentation.

**Metadata tagging.** Every chunk tagged at ingestion with:

- `policy_id`: unique identifier for the policy
- `section`: top-level section name (e.g., "Waiting Periods")
- `sub_clause`: sub-clause reference (e.g., "3.2.1")
- `page_number`: source page for citation
- `chunk_type`: one of `clause`, `table`, `definition`

**Retrieval.** The retrieval layer is built modularly so each component (BM25, dense embeddings, Cohere rerank) can be toggled independently. Final configuration to be selected by comparing four setups against the eval set. AutoMergingRetriever sits on top of whichever retrieval mode wins, expanding to parent nodes when a majority of children for a parent are retrieved. Metadata filters kept flat (single-value, no nested AND/OR conditions) to avoid known compatibility issues between LlamaIndex's MetadataFilters and the Pinecone chat engine.

| Configuration | What's in it | What we're testing |
|---|---|---|
| A. Dense-only baseline | Cohere embed + Pinecone, top 5 retrieved directly | Is anything beyond pure dense even needed? |
| B. Hybrid only | BM25 + dense, top 5 fused via RRF | Does sparse retrieval help over pure dense? |
| C. Dense + rerank | Dense top 20 → Cohere Rerank → top 5 | Does rerank help over dense alone? |
| D. Hybrid + rerank | BM25 + dense top 20 → Cohere Rerank → top 5 | Combined effect of hybrid plus reranking |

```python
def retrieve(query, mode="hybrid_rerank"):
    if mode == "dense":
        return dense_retrieve(query, k=5)
    elif mode == "hybrid":
        return hybrid_retrieve(query, k=5)
    elif mode == "dense_rerank":
        candidates = dense_retrieve(query, k=20)
        return cohere_rerank(candidates, k=5)
    elif mode == "hybrid_rerank":
        candidates = hybrid_retrieve(query, k=20)
        return cohere_rerank(candidates, k=5)
```

Intent: building this comparison upfront forces the final retrieval choice to be evidence-based, not assumed. If a simpler configuration matches the full stack within eval tolerance, we ship the simpler version. The complexity we keep has to earn its place.

**Evaluation framework.** Ragas is used for scoring the agent's outputs against the golden dataset, not for generating the dataset. Generation is done via a separate LLM prompt with our category and persona specification (see Section 7). After each retrieval configuration runs through the 70 golden questions, Ragas computes the following metrics on the agent's responses:

| Metric | What it measures |
|---|---|
| `faithfulness` | Whether the agent's answer is grounded in the retrieved context (no hallucination) |
| `answer_relevancy` | Whether the answer addresses the question asked |
| `context_precision` | Whether the retrieved chunks are relevant to the question |
| `context_recall` | Whether all relevant information from the ground truth is present in the retrieved context |
| `answer_correctness` | Whether the final answer matches the ground truth (factually and semantically) |

The four retrieval configurations (A through D in the previous section) are scored on these five metrics. The winning configuration is the one that maximizes faithfulness and answer_correctness without significant degradation on the other three. Langfuse captures the per-query traces (tool calls, retrieved contexts, latency, token usage) for the failure analysis.

---

## 5. Live Data Routing: Problem and Design

### 5.1 Why Pure RAG Fails

Indian health insurance PDFs are systematically designed to defer dynamic data to websites. This isn't accidental. It's how insurers maintain flexibility without reprinting policy documents. In all three cases below, a pure RAG system returns an outdated or incomplete answer. Worse, it returns it confidently, which is the dangerous failure mode for a health insurance product specifically.

Agentic routing is not an edge case feature. It is a core correctness requirement.

### 5.2 Live-Data Trigger Map (Sample)

| Trigger question | PDF answer? | Fetch needed | URL |
|---|---|---|---|
| "Is hospital X blacklisted?" | Partially, stale | Yes | careinsurance.com |
| "Does Smart Select apply at X?" | No, explicitly incomplete | Yes | careinsurance.com/smart-select-network-locator.html |
| "Can I get cashless at X?" | No | Yes | careinsurance.com network list |
| "Who do I escalate my grievance to?" | Possibly stale | Yes | careinsurance.com/customer-grievance-redressal.html |
| "What is the Ombudsman contact for my state?" | Possibly stale | Yes | irda.gov.in |

### 5.3 Detailed Routing Cases (Sample)

**Case 1: Excluded hospitals list (Annexure II)**

The document lists ~60 blacklisted hospitals. Then at the bottom: *"For an updated list of Hospitals, please visit the Company's website."* The PDF has a static list dated June 2025. The live list may differ.

Agent behavior:
> User: "Is Prakash Hospital Noida covered?"
> → Check Annexure II → found, listed as excluded
> → BUT policy itself says this list may be outdated
> → Tool call: `web_fetch(careinsurance.com/excluded-hospitals)`
> → Cross-reference live list
> → Answer: "As of the policy document (June 2025), Prakash Hospital Sector 33 Noida is excluded. Verified against live site [date]: still excluded / no longer listed / site unreachable, verify directly."

**Case 2: Smart Select network hospitals (Annexure III)**

Annexure III lists ~30 hospitals where the 20% co-payment under Smart Select is waived. The document explicitly says: *"The below is a Non-exhaustive list... Please check the latest & complete list on https://www.careinsurance.com/smart-select-network-locator.html"*

The document is openly incomplete. If a user asks "Does my Smart Select benefit apply at Apollo Chennai?" the PDF cannot answer. The agent must fetch the live locator.

**Case 3: Network providers for cashless (Clause 6.1.2 vii)**

*"The Company may modify the list of Network Providers... For an updated list of Network Providers and the extent of Cashless Facilities available at each Network Provider, the Insured Person may refer to the list of Network Providers available on the Company's website or at the call center."*

The PDF has zero hospital network data for cashless. Any "can I get cashless at X hospital?" question is 100% a web fetch.

**Case 4: Grievance officer details (Clause 5.16)**

*"For updated details of grievance officer, kindly refer the link: https://www.careinsurance.com/customer-grievance-redressal.html"*

Lower stakes, but still a routing case. At claim time, users need to know who to escalate to.

**Case 5: Ombudsman details (Annexure IV)**

The document lists all Ombudsman office addresses and contact numbers, then: *"The updated details of Insurance Ombudsman are available on website of IRDAI: www.irda.gov.in"*

IRDAI updates these. If a user is disputing a rejected claim and needs the current Ombudsman contact for Karnataka, the agent should fetch from IRDAI's live site, not trust the PDF's printed details.

### 5.4 Required Agent Behavior for Live-Data Routing

At ingestion time:
- Detect when a policy clause defers to a URL
- Store those URLs as structured metadata, tagged by what they govern

At query time:
- Decide whether the question triggers a live fetch before answering
- Handle fetch failures gracefully with explicit uncertainty messaging

---

## 6. Failure Modes and Fallback Design

### 6.1 Primary Failure Mode to Design Against

Web fetches fail. URLs change. Insurer websites go down or block scrapers.

### 6.2 Fallback Logic

```
If web_fetch succeeds → answer with live data + fetch timestamp

If web_fetch fails → answer from policy document + warn user:
"The policy refers to the insurer's website for this list,
but I couldn't retrieve it. The policy document (dated X) says Y,
but this may be outdated. Verify at [URL]."
```

---

## 7. Evaluation

Evaluation runs across three layers: retrieval quality, generation quality, and agent trajectory. Ragas powers offline scoring. Langfuse captures runtime traces.

### 7.1 Evaluation Set

**Target:** ~70 test questions covering the categories the agent is expected to handle. Set is not yet built; the steps below define how it will be constructed.

**How questions will be sourced:**

- Real user queries scraped from r/IndiaInvestments and r/personalfinanceindia (health insurance threads)
- Twitter / X posts tagging insurers about claim denials
- IRDAI ombudsman award summaries (publicly available)
- Direct read of the 5 policy documents in scope

**Categories to cover:**

- Waiting periods
- Pre-existing condition exclusions
- Sub-limits and room rent caps
- Live-data dependent questions (hospital network, cashless eligibility)
- Partial coverage and co-payment triggers
- Optional covers and add-ons
- Adversarial / ambiguous questions designed to fail or trigger uncertainty

Final count per category to be decided during the build, based on how many natural-language questions actually exist in each category from the sources above. The 70-question target is approximate, not a hard cap.

**Raw data source:** The raw posts, comments, and questions that will be processed into the golden evaluation set are maintained in a separate document, `health-insurance-RawQuestions.md`. That document contains verbatim entries from X (sourced via Grok), Reddit (sourced via Perplexity), and Quora (sourced via web search), in a normalized table format. The golden dataset will be derived from this raw pool during the build phase.

**Format:** Each question will have a ground truth answer, supporting clause reference, and expected verdict. Adversarial questions will have expected behaviors instead, "agent should refuse to answer," "agent should ask clarifying question," etc.

The golden dataset format is below. The first row is a worked example.

| # | Question | Ground Truth Answer | Supporting Clause | Verdict | Category |
|---|---|---|---|---|---|
| 1 | What is the waiting period for maternity benefits under the policy? | Maternity benefits are covered after a 24-month waiting period from policy inception. | Clause 3.2, Page 11 | Yes | Waiting period |

### 7.2 Scoring: Three Layers

**Layer 1: Retrieval (Ragas)**

| Metric | Target |
|---|---|
| Context precision | ≥ 0.80 |
| Context recall | ≥ 0.80 |
| Recall@5 on golden set | ≥ 0.85 |

**Layer 2: Generation (Ragas)**

| Metric | Target |
|---|---|
| Faithfulness (no hallucinated claims) | ≥ 0.90 |
| Answer relevancy | ≥ 0.85 |
| Grounding rate (every claim cites a clause) | 100% |

**Layer 3: Agent trajectory (Langfuse traces, manual review)**

| Metric | Target |
|---|---|
| Tool selection accuracy (did the agent call the right tools?) | ≥ 0.85 |
| Redundant tool calls per query | < 0.5 |
| Correct fallback on failed web fetch | 100% |
| p95 end-to-end latency | Report, not target |
| Tokens per query | Report, not target |

Grounding rate is a hard gate. An answer that is correct but uncited is not acceptable in this product.

### 7.3 Human Review

For ambiguous calls (Ragas score borderline, agent uncertainty flagged, partial verdicts), one human grader reads the agent response alongside the policy clause and marks it correct / partially correct / wrong. Inter-rater reliability is skipped for v1 (solo project). Acknowledged limitation.

### 7.4 What "Good" Looks Like

| Metric | Target |
|---|---|
| Overall answer correctness (verdict match) | ≥ 80% |
| Faithfulness | ≥ 0.90 |
| Grounding rate | 100% |
| Correct fallback on failed web fetch | 100% |
| Tool selection accuracy | ≥ 0.85 |

---

## 8. Risks and Mitigations

### 8.1 Disclaimer Design

Every response carries a fixed disclaimer below the answer:

> *This response is generated by AI and may contain mistakes. Please double-check before acting on it.*

### 8.2 Feedback System

**UI: Two buttons below every response**

```
[ 👍 Helpful ]   [ 👎 Not helpful ]
```

On click of either button, a text box opens inline beneath the buttons.

- Placeholder text in the box: *"Optional: tell us more about this response."*
- A "Submit" button appears alongside it.
- Submitting (or dismissing) closes the input.
- The text entry is entirely optional. Clicking a button and doing nothing else still logs the thumbs signal.

**What gets logged to the feedback DB (Supabase / Postgres):**

| Field | Description |
|---|---|
| `session_id` | Anonymised session identifier |
| `conversation` | Full conversation log as a structured array (see below) |
| `retrieval_log` | Which tools were called, in what order, with what outputs |
| `web_fetch_status` | Succeeded / failed / not triggered |
| `feedback_signal` | Thumbs up / Thumbs down |
| `feedback_text` | Free text, if provided (nullable) |
| `policy_document_id` | Which policy document was queried |
| `timestamp` | UTC |

**Conversation log format:**

```json
"conversation": [
  { "turn": 1, "role": "user", "content": "..." },
  { "turn": 1, "role": "agent", "content": "..." },
  { "turn": 2, "role": "user", "content": "..." },
  { "turn": 2, "role": "agent", "content": "..." }
]
```

Every turn is captured in sequence. When the user submits feedback, the entire conversation up to that point is logged. No question or answer is dropped.

**Why the retrieval log matters here.** A thumbs-down on an answer the system was certain about is the most valuable signal. It means either the retrieval was wrong, the clause was misinterpreted, or the system failed quietly. That combination is what you use to improve over time.

### 8.3 Compliance and Data Handling

This tool processes health-related queries, which fall under personal data per India's DPDP Act 2023 (Rules notified November 2025, full enforcement May 2027). The product is built with these boundaries:

- **Tool is informational, not advisory.** No IRDAI license claimed. Disclaimer below every response makes this explicit.
- **No PII collected at signup.** Sessions are anonymised at the identifier level. Health queries are not linked to identity.
- **Uploaded documents** (treatment estimates, policy PDFs) are processed in-memory during the session. Not persisted beyond the session window.
- **Logged data** (conversation, retrieval traces) retained for evaluation only. Deleted after 30 days.
- **Out of scope:** long-term policy storage, claims filing, anything that crosses into regulated advisory.

This is a v1 stance. A production deployment would need a formal DPIA, a privacy policy, and a consent flow before the May 2027 enforcement date.

### 8.4 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Agent gives wrong coverage verdict; user acts on it | Medium | High | Disclaimer on every response; explicit "do not act on this" language when agent admits uncertainty |
| Web fetch fails silently; stale PDF data returned as current | High | High | Fallback messaging is mandatory (Section 6.2); fetch timestamp always shown when live data is used |
| Policy document ingested incorrectly; clauses misread | Medium | High | Evaluation set (Section 7) catches this at ingestion; human spot-check on first parse of each new document |
| IRDAI regulations change; agent uses outdated rules | Medium | Medium | `irdai_regulation_lookup` tool with periodic refresh |
| Numeric questions (sub-limits, waiting periods) retrieved imprecisely from prose chunks | Medium | Medium | Hybrid retrieval + reranker mitigates; structured extraction is the v2 fix (see Section 10) |
| Health data handling not yet DPDP-compliant for production | High (post-enforcement) | High | v1 is a portfolio prototype; production rollout needs formal compliance work before May 2027 |

---

## 9. User Journey

The end-to-end flow from a user landing on the page to ending the session.

### 9.1 Landing

Single-page UI. No login, no signup, no email gate.

What's on the page:

- A clear headline. Something like "Understand your health insurance policy in plain English."
- Two paths the user can take:
  - **Upload your policy PDF** (file picker)
  - **Try a sample policy** (one pre-loaded, pre-evaluated policy, labeled by insurer name)
- Below the upload, a one-liner on data handling: "Your document is processed in memory and deleted after the session. We don't store it."

### 9.2 Processing

After upload or sample selection, a processing screen.

- Stages visible to the user: Parsing document → Extracting clauses → Indexing → Ready
- Each stage flips from grey to green as it completes
- Estimated time: 30-60 seconds for a fresh upload, near-instant for the sample policy (pre-indexed)
- If parsing fails (corrupted PDF, password-protected, scanned image only): clean error message. "We couldn't read this PDF. Try a text-based PDF or use the sample policy."

If the user uploaded their own document, a one-line banner persists into the chat: "This policy hasn't been evaluated for accuracy. Answers are unverified."

### 9.3 Ready State

Brief transition screen. "Your policy is ready. Ask me anything about it."

Show 4 suggested starter questions tailored to common queries:

- "What's my waiting period for pre-existing conditions?"
- "Is maternity covered?"
- "Can I get cashless treatment at [hospital name]?"
- "What's not covered by this policy?"

Suggested questions remove the blank-input problem and shape first-message quality.

### 9.4 Chat Mode

Two-pane layout.

**Left pane: chat.** Standard chat UI. Each agent response contains:

- The answer (Yes / No / Partial with reasoning)
- Cited clauses with page numbers
- For live-data answers: a "Live data" tag, the URL that was fetched, and the timestamp
- The standard disclaimer at the bottom
- Thumbs up / thumbs down buttons (feedback flow per Section 8.2)

**Right pane: policy PDF view.** When a clause is cited in the chat, the right pane scrolls to the cited page. The user can see the actual source document, not just a chunk of text. This makes the system feel grounded rather than generative.

Bounding-box highlighting of the exact clause within the page is a v2 enhancement (see Section 10).

Conversation persists in-session. No saved history across sessions, by design.

### 9.5 Session End

Two ways out:

- User closes the tab. Session data is wiped automatically.
- "Start over" button at the top. Resets the state and lets them pick a different policy.

No "save your chat" or "email me this answer" features in v1. Both create data-handling burden without adding demo value.

---

## 10. Future Scope

Things explicitly out of scope for v1 but called out as the natural next steps.

### 10.1 Structured Extraction with LlamaExtract

**The problem this solves.** Sub-limits, waiting periods, room rent caps, exclusion lists, and co-payment triggers are tabular data. In v1 they live as embedded chunks alongside prose. Hybrid retrieval and the reranker handle most of this well, but precise numeric questions ("what's my exact room rent cap for plan B?") are where the architecture is weakest.

**The fix.** LlamaExtract is LlamaIndex's schema-driven extraction service. Define a Pydantic schema for the structured fields, run it once at ingestion, get JSON back, expose a `policy_facts_lookup` tool that the agent calls for numeric and categorical questions. Prose questions continue to use the existing retrieval pipeline.

**Why not in v1.** LlamaExtract is currently in public beta. For a one-day build with a live demo, a beta dependency introduces schema iteration risk and API stability risk that outweighs the benefit at this scale.

**When to add it.** When numeric question accuracy clusters as the dominant failure mode in eval results, or when the product scales beyond a single user / single demo, or when LlamaExtract leaves beta.

### 10.2 Multi-Policy Support

v1 assumes one or a handful of policy documents are pre-ingested. Production needs:

- User uploads their own PDF
- Per-user policy store with versioning (policies get reissued at renewal)
- Caching of parsed + extracted output to avoid re-running LlamaParse per query
- Cost model per user

### 10.3 Multilingual Support

India has 22 official languages. Many policyholders read Hindi or regional languages, not English. v1 is English-only. Multilingual support would mean:

- Query translation layer
- Response translation
- Evaluation set translated and re-graded

This is a heavy scope expansion and only makes sense after v1 retrieval quality is validated.

### 10.4 PDF Clause Highlighting

v1 scrolls the right-pane PDF to the cited page when a clause is referenced. v2 would draw a bounding box around the exact clause text on the page. Requires storing bounding-box coordinates from LlamaParse's `extract_layout` output at ingestion time, then overlaying them on the rendered PDF at query time. Visually striking, but fiddly to wire up.

### 10.5 User Accounts and Session Persistence

v1 has no sign-in. Sessions are ephemeral by design (data-handling stance, demo simplicity). A production version would offer optional accounts for:

- Saving past chats and re-opening them
- Storing multiple policies per user (family coverage)
- Tracking claim disputes over time

This is a separate product surface and adds compliance load (consent, retention policy, account deletion flow). Push to a future phase.

### 10.6 MCP Server Exposure

The agent's tools (`policy_query_engine`, `hospital_network_checker`, etc.) could be exposed as a Model Context Protocol server. This would let other AI agents (Claude, ChatGPT, internal enterprise agents) consume the policy interpreter as a tool. Useful for B2B distribution (TPAs, brokers, hospital billing desks).

### 10.7 Formal Compliance Work

For any production deployment serving real users at scale:

- Data Protection Impact Assessment (DPIA) per DPDP Rules 2025
- Consent flow with explicit health-data acknowledgment
- Privacy policy and terms of service
- Data breach response playbook (72-hour notification per DPDP)
- Possible registration as Significant Data Fiduciary depending on user volume
