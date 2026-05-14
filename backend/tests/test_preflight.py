"""
Pre-flight checks: env vars, file system state, stale files.
Run first before any eval. All must pass.
"""
import json
import os
from pathlib import Path
import pytest


# ── Env vars ──────────────────────────────────────────────────────────────────

def test_anthropic_api_key_present(settings):
    assert settings.anthropic_api_key, "ANTHROPIC_API_KEY is not set"
    assert settings.anthropic_api_key.startswith("sk-ant-"), \
        f"ANTHROPIC_API_KEY looks wrong: {settings.anthropic_api_key[:10]}..."


def test_cohere_api_key_present(settings):
    assert settings.cohere_api_key, "COHERE_API_KEY is not set"


def test_pinecone_api_key_present(settings):
    assert settings.pinecone_api_key, "PINECONE_API_KEY is not set"


def test_llama_cloud_api_key_present(settings):
    assert settings.llama_cloud_api_key, "LLAMA_CLOUD_API_KEY is not set"


def test_eval_model_is_haiku(settings):
    assert "haiku" in settings.eval_model.lower(), \
        f"eval_model should be Haiku for cost efficiency, got: {settings.eval_model}"


def test_pinecone_index_name_set(settings):
    assert settings.pinecone_index_name == "health-insurance-policies", \
        f"Unexpected index name: {settings.pinecone_index_name}"


# ── File system ────────────────────────────────────────────────────────────────

def test_policy_pdf_exists(settings, policy_id):
    pdf_path = Path(settings.data_dir) / "pdfs" / f"{policy_id}.pdf"
    assert pdf_path.exists(), f"Policy PDF not found: {pdf_path}"
    size_mb = pdf_path.stat().st_size / 1024 / 1024
    assert size_mb > 0.5, f"PDF too small ({size_mb:.1f}MB) — may be corrupted"


def test_bm25_index_exists_for_policy(settings, policy_id):
    bm25_path = Path(settings.data_dir) / "bm25_indexes" / f"{policy_id}.json"
    assert bm25_path.exists(), \
        f"BM25 index missing: {bm25_path}. Run ingestion first."


def test_bm25_index_not_empty(settings, policy_id):
    bm25_path = Path(settings.data_dir) / "bm25_indexes" / f"{policy_id}.json"
    if not bm25_path.exists():
        pytest.skip("BM25 index not found — covered by test_bm25_index_exists_for_policy")
    data = json.loads(bm25_path.read_text())
    assert len(data.get("corpus", [])) > 100, \
        f"BM25 index has only {len(data.get('corpus', []))} entries — re-run ingestion"


def test_no_stale_checkpoint_files():
    """Fail if checkpoint files exist — they will corrupt context in the next eval run."""
    results_dir = Path("eval/results")
    if not results_dir.exists():
        return
    stale = list(results_dir.glob("checkpoint_*.jsonl")) + list(results_dir.glob("log_*.jsonl"))
    assert len(stale) == 0, (
        f"Found {len(stale)} stale file(s): {[f.name for f in stale]}. "
        "Run: from eval.run_eval import clean_stale_files; clean_stale_files('eval/results') "
        "OR just run eval/run_eval.py — it cleans automatically at startup."
    )


def test_eval_script_importable():
    """Catch wrong-directory errors before running eval."""
    try:
        import eval.run_eval  # noqa: F401
    except ModuleNotFoundError as e:
        pytest.fail(
            f"eval/run_eval.py cannot be imported: {e}. "
            "Run pytest from the backend/ directory."
        )


def test_golden_dataset_exists():
    path = Path("eval/golden_dataset.csv")
    assert path.exists(), "eval/golden_dataset.csv not found"


def test_golden_dataset_has_required_columns():
    import csv
    path = Path("eval/golden_dataset.csv")
    if not path.exists():
        pytest.skip("golden_dataset.csv not found")
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    required = {"question", "ground_truth_answer", "expected_verdict", "category"}
    missing = required - set(fieldnames)
    assert not missing, f"golden_dataset.csv missing columns: {missing}"


def test_golden_dataset_row_count():
    import csv
    path = Path("eval/golden_dataset.csv")
    if not path.exists():
        pytest.skip("golden_dataset.csv not found")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 60, f"Expected ≥60 eval questions, got {len(rows)}"
