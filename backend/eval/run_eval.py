"""
Eval pipeline — production-ready.

Features:
- All 4 retrieval modes run in parallel
- No sleep between questions (Anthropic Tier 2: 1K RPM)
- Full context saved in checkpoint (not 200-char preview)
- Context validation before Ragas scoring
- Retry with exponential backoff per question
- Token/cost tracking per question
- Per-question structured JSONL log
- Pinecone connection shared across all modes
- Ragas: Claude Haiku judge + Cohere production embeddings

Usage:
    uv run python eval/run_eval.py --policy_id care-insurance-sample --output eval/results/
"""
import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy, answer_correctness  # noqa: F401 — used via copy.deepcopy below
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_anthropic import ChatAnthropic
from langchain_cohere import CohereEmbeddings

from app.config import get_settings
from app.ingestion.indexer import get_pinecone_index
from app.agents.orchestrator import OrchestratorAgent

RETRIEVAL_MODES = ["dense", "hybrid", "dense_rerank", "hybrid_rerank"]
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubles on each retry


def clean_stale_files(output_dir: str):
    """Delete old checkpoint and log files before a fresh eval run."""
    output_path = Path(output_dir)
    removed = []
    for pattern in ["checkpoint_*.jsonl", "log_*.jsonl"]:
        for f in output_path.glob(pattern):
            f.unlink()
            removed.append(f.name)
    if removed:
        print(f"Cleaned {len(removed)} stale file(s): {removed}")


def load_golden_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth_answer", row.get("ground_truth", "")),
                "clause": row.get("supporting_clause", ""),
                "verdict": row.get("expected_verdict", row.get("verdict", "")),
                "category": row.get("category", ""),
                "live_data_dependent": row.get("live_data_dependent", "No"),
            })
    return rows


def checkpoint_path(output_dir: str, mode: str) -> Path:
    return Path(output_dir) / f"checkpoint_{mode}.jsonl"


def log_path(output_dir: str, mode: str, timestamp: str) -> Path:
    return Path(output_dir) / f"log_{mode}_{timestamp}.jsonl"


def load_checkpoint(output_dir: str, mode: str) -> dict[str, dict]:
    path = checkpoint_path(output_dir, mode)
    completed = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                if not entry.get("response", "").startswith("ERROR:"):
                    completed[entry["question"]] = entry
    return completed


def save_checkpoint(output_dir: str, mode: str, entry: dict):
    with open(checkpoint_path(output_dir, mode), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def save_log(log_file: Path, entry: dict):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def run_single_with_retry(
    settings,
    pinecone_index,
    policy_id: str,
    mode: str,
    question: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[str, str, list[dict], str, int, int, float]:
    """
    Run one question through the agent with exponential backoff retry.
    Returns: (response, context, retrieval_log, web_fetch_status, input_tokens, output_tokens, latency_s)
    """
    last_error = ""
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            orchestrator = OrchestratorAgent(
                policy_id=policy_id,
                pinecone_index=pinecone_index,
                cohere_api_key=settings.cohere_api_key,
                anthropic_api_key=settings.anthropic_api_key,
                data_dir=settings.data_dir,
                retrieval_mode=mode,
                top_k=5,
                candidate_k=20,
                # No model override — uses default Sonnet, same as production
            )

            response = ""
            retrieval_log = []
            web_fetch_status = "not_triggered"
            input_tokens = 0
            output_tokens = 0

            async for event in orchestrator.stream_chat(question, conversation_history=[]):
                if event.get("type") == "chunk":
                    response += event.get("content", "")
                elif event.get("type") == "done":
                    retrieval_log = event.get("retrieval_log", [])
                    web_fetch_status = event.get("web_fetch_status", "not_triggered")
                    input_tokens = event.get("input_tokens", 0)
                    output_tokens = event.get("output_tokens", 0)
                elif event.get("type") == "error":
                    raise RuntimeError(event.get("content", "Unknown error"))

            # Extract full context from retrieval_log
            context_parts = []
            for entry in retrieval_log:
                if entry.get("tool") == "retrieve_from_policy":
                    full = entry.get("result_full", entry.get("result_preview", ""))
                    if full:
                        context_parts.append(full)
            context = " | ".join(context_parts)

            latency = round(time.perf_counter() - t0, 2)
            return response, context, retrieval_log, web_fetch_status, input_tokens, output_tokens, latency

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    Attempt {attempt + 1} failed: {last_error[:80]}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

    latency = round(time.perf_counter() - t0, 2)
    return f"ERROR: {last_error}", "", [], "not_triggered", 0, 0, latency


async def evaluate_mode(
    mode: str,
    golden: list[dict],
    policy_id: str,
    pinecone_index,
    settings,
    output_dir: str,
    timestamp: str,
) -> dict:
    print(f"\n=== Mode: {mode} ===")

    completed = load_checkpoint(output_dir, mode)
    skipped = sum(1 for r in golden if r["question"] in completed)
    if skipped:
        print(f"  Resuming — {skipped} cached, {len(golden) - skipped} remaining.")

    log_file = log_path(output_dir, mode, timestamp)
    questions, answers, ground_truths, contexts = [], [], [], []
    total_input_tokens = 0
    total_output_tokens = 0
    errors = 0

    for i, row in enumerate(golden):
        question = row["question"]
        ground_truth = row["ground_truth"]

        if question in completed:
            entry = completed[question]
            print(f"  [{i+1}/{len(golden)}] SKIP: {question[:55]}...")
            questions.append(question)
            answers.append(entry["response"])
            ground_truths.append(ground_truth)
            contexts.append([entry["context"]] if entry.get("context") else [""])
            continue

        print(f"  [{i+1}/{len(golden)}] {question[:60]}...")

        response, context, retrieval_log, web_fetch_status, in_tok, out_tok, latency = \
            await run_single_with_retry(settings, pinecone_index, policy_id, mode, question)

        is_error = response.startswith("ERROR:")
        if is_error:
            errors += 1
            print(f"    ERROR after {MAX_RETRIES} retries: {response[:80]}")

        total_input_tokens += in_tok
        total_output_tokens += out_tok

        # Validate context before saving
        context_valid = len(context) > 50
        if not context_valid and not is_error:
            print(f"    WARNING: Empty/short context for Q{i+1} — will be excluded from Ragas scoring")

        entry = {
            "id": row["id"],
            "question": question,
            "response": response,
            "context": context,
            "context_valid": context_valid,
            "ground_truth": ground_truth,
            "category": row["category"],
            "web_fetch_status": web_fetch_status,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_s": latency,
            "tools_called": [e.get("tool", e.get("agent", "")) for e in retrieval_log],
            "mode": mode,
            "is_error": is_error,
        }

        save_checkpoint(output_dir, mode, entry)
        save_log(log_file, entry)

        questions.append(question)
        answers.append(response)
        ground_truths.append(ground_truth)
        contexts.append([context] if context_valid else [""])

    # Filter to valid scoreable rows
    valid = [
        (q, a, g, c) for q, a, g, c in zip(questions, answers, ground_truths, contexts)
        if not a.startswith("ERROR:") and a.strip() and c != [""]
    ]

    print(f"\n  Collection done — {len(valid)}/{len(questions)} valid for scoring ({errors} errors, {len(questions)-len(valid)-errors} empty context).")
    print(f"  Tokens: {total_input_tokens} in / {total_output_tokens} out")
    print(f"  Est. cost: ${(total_input_tokens * 0.0008 + total_output_tokens * 0.004) / 1000:.4f} (Haiku pricing)")

    if not valid:
        return {"mode": mode, "error": "No valid responses", "scores": {}, "total": len(questions), "scored": 0}

    qs, ans, gts, ctxs = zip(*valid)
    dataset = Dataset.from_dict({
        "question": list(qs),
        "answer": list(ans),
        "ground_truth": list(gts),
        "contexts": list(ctxs),
    })

    print(f"  Scoring with Ragas...")
    ragas_llm = LangchainLLMWrapper(ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=settings.anthropic_api_key,
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=settings.cohere_api_key,
    ))

    # Ragas 0.4.x does not fully propagate llm/embeddings from evaluate() to all metrics.
    # Must set them explicitly per metric instance to avoid "LLM is not set" errors.
    from ragas.metrics import faithfulness as f_m, context_precision as cp_m, context_recall as cr_m
    from ragas.metrics import answer_relevancy as ar_m, answer_correctness as ac_m
    import copy

    metrics = []
    for m in [f_m, cp_m, cr_m, ar_m, ac_m]:
        m_copy = copy.deepcopy(m)
        m_copy.llm = ragas_llm
        if hasattr(m_copy, "embeddings"):
            m_copy.embeddings = ragas_embeddings
        metrics.append(m_copy)

    scores = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        batch_size=5,
        run_config=RunConfig(max_workers=2, max_wait=60, timeout=120),
    )

    df = scores.to_pandas()
    numeric_df = df.select_dtypes(include="number")
    mean_scores = numeric_df.mean().to_dict()

    return {
        "mode": mode,
        "total_questions": len(questions),
        "scored_questions": len(valid),
        "errors": errors,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "scores": mean_scores,
        "raw": numeric_df.to_dict(orient="records"),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_id", default="care-insurance-sample")
    parser.add_argument("--dataset", default="eval/golden_dataset.csv")
    parser.add_argument("--output", default="eval/results")
    parser.add_argument("--modes", nargs="+", default=RETRIEVAL_MODES)
    parser.add_argument("--fresh", action="store_true", help="Delete stale checkpoints before running")
    args = parser.parse_args()

    settings = get_settings()
    golden = load_golden_dataset(args.dataset)
    print(f"Loaded {len(golden)} questions. Modes: {args.modes}")
    print(f"Answer model: claude-sonnet-4-6 (production). Ragas judge: {settings.eval_model}")

    # Single Pinecone connection shared across all modes
    pinecone_index = get_pinecone_index(settings.pinecone_api_key, settings.pinecone_index_name)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean stale files only on explicit --fresh flag
    if getattr(args, "fresh", False):
        clean_stale_files(args.output)

    # Run all modes in parallel
    print(f"\nRunning {len(args.modes)} modes in parallel...")
    tasks = [
        evaluate_mode(mode, golden, args.policy_id, pinecone_index, settings, args.output, timestamp)
        for mode in args.modes
    ]
    all_results = await asyncio.gather(*tasks)

    # Save summary
    summary_path = Path(args.output) / f"eval_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(list(all_results), indent=2), encoding="utf-8")
    print(f"\nResults saved to {summary_path}")

    # Print comparison table
    metrics = ["faithfulness", "context_precision", "context_recall", "answer_relevancy", "answer_correctness"]
    print("\n=== Results ===")
    header = f"{'Mode':<18}" + "".join(f"{m[:14]:<16}" for m in metrics) + "  scored/total"
    print(header)
    print("-" * len(header))
    for r in all_results:
        row = f"{r['mode']:<18}"
        for m in metrics:
            val = r["scores"].get(m, float("nan"))
            row += f"{val:.3f}{'':11}"
        row += f"  {r.get('scored_questions','?')}/{r.get('total_questions','?')}"
        print(row)

    # Total cost summary
    total_in = sum(r.get("total_input_tokens", 0) for r in all_results)
    total_out = sum(r.get("total_output_tokens", 0) for r in all_results)
    cost = (total_in * 0.0008 + total_out * 0.004) / 1000
    print(f"\nTotal tokens: {total_in} in / {total_out} out")
    print(f"Estimated agent cost: ${cost:.4f} (Haiku pricing)")


if __name__ == "__main__":
    asyncio.run(main())
