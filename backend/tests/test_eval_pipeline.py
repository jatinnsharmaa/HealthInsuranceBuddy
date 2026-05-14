"""
Eval pipeline integrity tests: checkpoint schema, Ragas config, stale file detection,
pandas crash prevention, context validation.
"""
import json
import pytest
from pathlib import Path
from datasets import Dataset


# ── Ragas config ───────────────────────────────────────────────────────────────

def test_ragas_llm_is_not_openai():
    """
    Regression: Ragas defaults to OpenAI if LLM not set explicitly.
    This caused an immediate crash with 'Missing OPENAI_API_KEY'.
    """
    import os
    # Ensure OPENAI_API_KEY is NOT set
    assert "OPENAI_API_KEY" not in os.environ, \
        "OPENAI_API_KEY is set — Ragas might use OpenAI silently. Remove it."


def test_ragas_evaluate_uses_claude(settings):
    """Verify we can configure Ragas with Claude without crashing."""
    from ragas.llms import LangchainLLMWrapper
    from langchain_anthropic import ChatAnthropic
    llm = LangchainLLMWrapper(ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=settings.anthropic_api_key,
    ))
    assert llm is not None


def test_ragas_embeddings_uses_cohere(settings):
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_cohere import CohereEmbeddings
    emb = LangchainEmbeddingsWrapper(CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=settings.cohere_api_key,
    ))
    assert emb is not None


def test_ragas_pandas_numeric_only():
    """
    Regression: scores.to_pandas().mean() crashed on string columns.
    Fixed by select_dtypes(include='number').
    Verify the pattern works on a realistic mock DataFrame.
    """
    import pandas as pd
    import numpy as np

    # Simulate what Ragas returns — mix of string and numeric columns
    df = pd.DataFrame({
        "question": ["q1", "q2"],
        "answer": ["ans1", "ans2"],
        "faithfulness": [0.9, 0.8],
        "context_precision": [0.85, 0.75],
        "context_recall": [0.88, 0.82],
    })
    numeric_df = df.select_dtypes(include="number")
    means = numeric_df.mean()
    assert "faithfulness" in means.index
    assert abs(means["faithfulness"] - 0.85) < 0.01


def test_ragas_run_config_importable():
    from ragas.run_config import RunConfig
    cfg = RunConfig(max_workers=2, max_wait=60, timeout=120)
    assert cfg is not None


# ── Checkpoint schema ──────────────────────────────────────────────────────────

REQUIRED_CHECKPOINT_FIELDS = {
    "id", "question", "response", "context", "context_valid",
    "ground_truth", "category", "web_fetch_status",
    "input_tokens", "output_tokens", "latency_s", "tools_called",
    "mode", "is_error",
}


def _load_checkpoints(results_dir: Path, mode: str) -> list[dict]:
    path = results_dir / f"checkpoint_{mode}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize("mode", ["dense", "hybrid", "dense_rerank", "hybrid_rerank"])
def test_checkpoint_schema_if_exists(mode):
    """If a checkpoint file exists, verify every entry has all required fields."""
    results_dir = Path("eval/results")
    rows = _load_checkpoints(results_dir, mode)
    if not rows:
        pytest.skip(f"No checkpoint for mode '{mode}' — run eval first")

    for i, row in enumerate(rows):
        missing = REQUIRED_CHECKPOINT_FIELDS - set(row.keys())
        assert not missing, \
            f"Checkpoint '{mode}' row {i} missing fields: {missing}"


@pytest.mark.parametrize("mode", ["dense", "hybrid", "dense_rerank", "hybrid_rerank"])
def test_checkpoint_context_not_empty_for_non_error_rows(mode):
    """
    Critical: every non-error row must have context > 50 chars.
    Empty context → garbage Ragas scores.
    """
    results_dir = Path("eval/results")
    rows = _load_checkpoints(results_dir, mode)
    if not rows:
        pytest.skip(f"No checkpoint for mode '{mode}'")

    bad = [r for r in rows if not r.get("is_error") and len(r.get("context", "")) < 50]
    assert len(bad) == 0, (
        f"Mode '{mode}' has {len(bad)} rows with empty/short context. "
        f"Sample questions: {[r['question'][:60] for r in bad[:3]]}"
    )


@pytest.mark.parametrize("mode", ["dense", "hybrid", "dense_rerank", "hybrid_rerank"])
def test_checkpoint_error_rate_acceptable(mode):
    """Fail if more than 10% of questions errored — indicates a systemic issue."""
    results_dir = Path("eval/results")
    rows = _load_checkpoints(results_dir, mode)
    if not rows:
        pytest.skip(f"No checkpoint for mode '{mode}'")

    error_count = sum(1 for r in rows if r.get("is_error"))
    error_rate = error_count / len(rows)
    assert error_rate <= 0.10, \
        f"Mode '{mode}' error rate {error_rate:.0%} exceeds 10% ({error_count}/{len(rows)} errors)"


@pytest.mark.parametrize("mode", ["dense", "hybrid", "dense_rerank", "hybrid_rerank"])
def test_checkpoint_token_counts_present(mode):
    """Verify token tracking is working — needed for cost analysis."""
    results_dir = Path("eval/results")
    rows = _load_checkpoints(results_dir, mode)
    if not rows:
        pytest.skip(f"No checkpoint for mode '{mode}'")

    non_error = [r for r in rows if not r.get("is_error")]
    if not non_error:
        pytest.skip("All rows are errors")

    zero_token_rows = [r for r in non_error if r.get("input_tokens", 0) == 0]
    # Some rows may have 0 tokens if streaming doesn't expose usage metadata
    # warn but don't fail hard — token tracking is best-effort
    if len(zero_token_rows) > len(non_error) * 0.5:
        pytest.xfail(
            f"More than 50% of rows have 0 input_tokens in mode '{mode}'. "
            "Token tracking via stream_events may not be working."
        )


# ── Eval script integrity ──────────────────────────────────────────────────────

def test_eval_script_has_no_unconditional_sleep():
    """
    Regression: 7s sleep between questions was left from Cohere trial throttling.
    With production keys, no sleep is needed — verify there's no sleep outside of retry logic.
    Sleep is only allowed inside retry backoff (where 'delay' variable is used).
    """
    script = Path("eval/run_eval.py").read_text()
    bad_sleep_lines = []
    for line in script.split("\n"):
        stripped = line.strip()
        if "asyncio.sleep" in stripped and not stripped.startswith("#"):
            # Only allow sleep(delay) — that's the retry backoff pattern
            if "delay" not in stripped and "RETRY_BASE_DELAY" not in stripped:
                bad_sleep_lines.append(stripped)
    assert len(bad_sleep_lines) == 0, \
        f"Found unconditional sleep in eval script (not retry backoff): {bad_sleep_lines}"


def test_eval_script_uses_haiku_not_sonnet():
    script = Path("eval/run_eval.py").read_text()
    assert "claude-sonnet" not in script or "eval_model" in script, \
        "eval/run_eval.py hardcodes Sonnet. Switch to settings.eval_model (Haiku)."


def test_eval_script_has_retry_logic():
    script = Path("eval/run_eval.py").read_text()
    assert "MAX_RETRIES" in script or "max_retries" in script, \
        "eval/run_eval.py missing retry logic"


def test_eval_script_validates_context_before_ragas():
    script = Path("eval/run_eval.py").read_text()
    assert "context_valid" in script or "len(context)" in script, \
        "eval/run_eval.py missing context validation before Ragas scoring"


def test_eval_script_uses_select_dtypes():
    script = Path("eval/run_eval.py").read_text()
    assert "select_dtypes" in script, \
        "eval/run_eval.py missing select_dtypes fix — .mean() will crash on string columns"


def test_eval_script_runs_modes_in_parallel():
    script = Path("eval/run_eval.py").read_text()
    assert "asyncio.gather" in script, \
        "eval/run_eval.py runs modes sequentially. Use asyncio.gather for parallel execution."


def test_eval_script_uses_pinecone_once():
    """Pinecone connection should be created once, shared across all modes."""
    script = Path("eval/run_eval.py").read_text()
    pinecone_calls = script.count("get_pinecone_index(")
    assert pinecone_calls == 1, \
        f"get_pinecone_index() called {pinecone_calls} times — should be called once and shared"
