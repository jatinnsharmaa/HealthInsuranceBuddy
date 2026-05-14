"""
LLM Baseline Eval — full-document single-call comparison.

Sends the entire policy PDF directly to Claude Sonnet (no RAG, no chunking).
Native web_search_20260209 tool for live data. Single API call per question.

Scores only answer_relevancy and answer_correctness (no contexts field needed).

Usage:
    uv run python eval/run_baseline_eval.py --policy_id care-insurance-sample --output eval/results/

Flags:
    --policy_id   Pinecone namespace / PDF filename stem (default: care-insurance-sample)
    --dataset     Path to golden dataset CSV (default: eval/golden_dataset.csv)
    --output      Output directory for checkpoints and summary (default: eval/results)
    --fresh       Delete stale checkpoint before running
"""
import argparse
import asyncio
import base64
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy as ar_m, answer_correctness as ac_m
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_anthropic import ChatAnthropic
from langchain_cohere import CohereEmbeddings

from app.config import get_settings
from app.prompts.baseline_prompt import BASELINE_SYSTEM_PROMPT

MODEL = "claude-sonnet-4-5-20250929"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0


def load_golden_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth_answer", row.get("ground_truth", "")),
                "category": row.get("category", ""),
            })
    return rows


def load_pdf_bytes(policy_id: str, data_dir: str) -> bytes:
    pdf_path = Path(data_dir) / "pdfs" / f"{policy_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. "
            f"Ensure the policy PDF is at data/pdfs/{policy_id}.pdf before running."
        )
    return pdf_path.read_bytes()


def checkpoint_path(output_dir: str) -> Path:
    return Path(output_dir) / "checkpoint_llm_baseline.jsonl"


def log_path(output_dir: str, timestamp: str) -> Path:
    return Path(output_dir) / f"log_llm_baseline_{timestamp}.jsonl"


def load_checkpoint(output_dir: str) -> dict[str, dict]:
    path = checkpoint_path(output_dir)
    completed = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                if not entry.get("response", "").startswith("ERROR:"):
                    completed[entry["question"]] = entry
    return completed


def save_checkpoint(output_dir: str, entry: dict):
    with open(checkpoint_path(output_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def save_log(log_file: Path, entry: dict):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def clean_stale_checkpoint(output_dir: str):
    path = checkpoint_path(output_dir)
    if path.exists():
        path.unlink()
        print(f"Cleaned stale checkpoint: {path.name}")


async def run_single(
    question: str,
    pdf_b64: str,
    anthropic_api_key: str,
) -> tuple[str, int, int, float, int, list[str], str]:
    """
    Single Sonnet call with full PDF attached and native web search.
    Returns: (response_text, input_tokens, output_tokens, latency_s, web_search_count, queries, web_fetch_status)
    """
    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
    t0 = time.perf_counter()

    response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": BASELINE_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": question},
            ],
        }],
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
    )

    latency = round(time.perf_counter() - t0, 2)

    text_parts = []
    search_queries = []
    web_fetch_status = "not_triggered"

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "server_tool_use" and block.name == "web_search":
            query = getattr(block.input, "query", None) or block.input.get("query", "")
            if query:
                search_queries.append(query)
            web_fetch_status = "success"
        elif block.type == "web_search_tool_result":
            content = getattr(block, "content", None)
            if isinstance(content, dict) and content.get("type") == "web_search_tool_result_error":
                web_fetch_status = f"error:{content.get('error_code', 'unknown')}"

    response_text = "".join(text_parts).strip()
    if not response_text:
        response_text = "ERROR: no text blocks in response"

    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    server_tool_use = getattr(usage, "server_tool_use", None)
    web_search_count = getattr(server_tool_use, "web_search_requests", 0) if server_tool_use else 0

    return response_text, input_tokens, output_tokens, latency, web_search_count, search_queries, web_fetch_status


async def run_single_with_retry(
    question: str,
    pdf_b64: str,
    anthropic_api_key: str,
) -> tuple[str, int, int, float, int, list[str], str]:
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            return await run_single(question, pdf_b64, anthropic_api_key)
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    Attempt {attempt + 1} failed: {last_error[:80]}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

    return f"ERROR: {last_error}", 0, 0, 0.0, 0, [], "not_triggered"


async def evaluate_baseline(
    golden: list[dict],
    policy_id: str,
    settings,
    output_dir: str,
    timestamp: str,
) -> dict:
    print("\n=== Mode: llm_baseline ===")
    print(f"  Model: {MODEL} | PDF: data/pdfs/{policy_id}.pdf")

    pdf_bytes = load_pdf_bytes(policy_id, settings.data_dir)
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    print(f"  PDF loaded: {len(pdf_bytes) / 1024:.0f} KB")

    completed = load_checkpoint(output_dir)
    skipped = sum(1 for r in golden if r["question"] in completed)
    if skipped:
        print(f"  Resuming — {skipped} cached, {len(golden) - skipped} remaining.")

    log_file = log_path(output_dir, timestamp)
    questions, answers, ground_truths = [], [], []
    total_input_tokens = 0
    total_output_tokens = 0
    total_web_searches = 0
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
            continue

        print(f"  [{i+1}/{len(golden)}] {question[:60]}...")

        response, in_tok, out_tok, latency, web_searches, search_queries, web_fetch_status = \
            await run_single_with_retry(question, pdf_b64, settings.anthropic_api_key)

        is_error = response.startswith("ERROR:")
        if is_error:
            errors += 1
            print(f"    ERROR after {MAX_RETRIES} retries: {response[:80]}")

        total_input_tokens += in_tok
        total_output_tokens += out_tok
        total_web_searches += web_searches

        if web_searches:
            print(f"    Web searches: {web_searches} | queries: {search_queries}")

        entry = {
            "id": row["id"],
            "question": question,
            "response": response,
            "context": "full_pdf",
            "context_valid": not is_error,
            "ground_truth": ground_truth,
            "category": row["category"],
            "web_fetch_status": web_fetch_status,
            "web_search_count": web_searches,
            "web_search_queries": search_queries,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_s": latency,
            "tools_called": ["web_search"] * web_searches if web_searches else [],
            "mode": "llm_baseline",
            "is_error": is_error,
        }

        save_checkpoint(output_dir, entry)
        save_log(log_file, entry)

        questions.append(question)
        answers.append(response)
        ground_truths.append(ground_truth)

    valid = [
        (q, a, g) for q, a, g in zip(questions, answers, ground_truths)
        if not a.startswith("ERROR:") and a.strip()
    ]

    print(f"\n  Collection done — {len(valid)}/{len(questions)} valid for scoring ({errors} errors).")
    print(f"  Tokens: {total_input_tokens} in / {total_output_tokens} out")
    print(f"  Web searches triggered: {total_web_searches} total across {len(golden)} questions")

    cache_cost = (total_input_tokens * 0.003) / 1000
    output_cost = (total_output_tokens * 0.015) / 1000
    web_cost = total_web_searches * 0.01
    print(f"  Est. cost: ${cache_cost + output_cost + web_cost:.4f} (Sonnet pricing + web search)")

    if not valid:
        return {
            "mode": "llm_baseline",
            "error": "No valid responses",
            "scores": {},
            "total_questions": len(questions),
            "scored_questions": 0,
            "errors": errors,
        }

    qs, ans, gts = zip(*valid)
    dataset = Dataset.from_dict({
        "question": list(qs),
        "answer": list(ans),
        "ground_truth": list(gts),
    })

    print("  Scoring with Ragas (answer_relevancy, answer_correctness only)...")
    ragas_llm = LangchainLLMWrapper(ChatAnthropic(
        model=settings.eval_model,
        api_key=settings.anthropic_api_key,
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=settings.cohere_api_key,
    ))

    metrics = []
    for m in [ar_m, ac_m]:
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
    mean_scores = df.select_dtypes(include="number").mean().to_dict()

    return {
        "mode": "llm_baseline",
        "model": MODEL,
        "total_questions": len(questions),
        "scored_questions": len(valid),
        "errors": errors,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_web_searches": total_web_searches,
        "scores": mean_scores,
        "raw": df.select_dtypes(include="number").to_dict(orient="records"),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_id", default="care-insurance-sample")
    parser.add_argument("--dataset", default="eval/golden_dataset.csv")
    parser.add_argument("--output", default="eval/results")
    parser.add_argument("--fresh", action="store_true", help="Delete stale checkpoint before running")
    args = parser.parse_args()

    settings = get_settings()
    golden = load_golden_dataset(args.dataset)
    print(f"Loaded {len(golden)} questions.")
    print(f"Answer model: {MODEL} | Ragas judge: {settings.eval_model}")
    print("\nNOTE: Web search requires org-level enablement at platform.anthropic.com → Settings → Privacy.")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.fresh:
        clean_stale_checkpoint(args.output)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = await evaluate_baseline(golden, args.policy_id, settings, args.output, timestamp)

    summary_path = Path(args.output) / f"eval_summary_baseline_{timestamp}.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults saved to {summary_path}")

    print("\n=== Results: llm_baseline ===")
    metrics = ["answer_relevancy", "answer_correctness"]
    header = f"{'Mode':<18}" + "".join(f"{m[:20]:<22}" for m in metrics) + "  scored/total"
    print(header)
    print("-" * len(header))
    row = f"{'llm_baseline':<18}"
    for m in metrics:
        val = result["scores"].get(m, float("nan"))
        row += f"{val:.3f}{'':18}"
    row += f"  {result.get('scored_questions','?')}/{result.get('total_questions','?')}"
    print(row)

    print("\nFor comparison with RAG modes, run: uv run python eval/run_eval.py")


if __name__ == "__main__":
    asyncio.run(main())
