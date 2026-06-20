"""
Security tests for policy_id validation — issues #5 and #6.

Layer 1: validate_policy_id rejects traversal strings (already fixed).
Layer 2: path assertion in ingest rejects any policy_id that resolves outside data_dir.
Layer 3: path assertion in retriever/sparse rejects escapes from bm25_indexes dir.
"""
import pytest
from fastapi import HTTPException
from pathlib import Path

from app.api.chat import validate_policy_id
from app.api.ingest import assert_safe_pdf_path
from app.retrieval.sparse import assert_safe_bm25_path


# ── Layer 1: validate_policy_id (already working) ────────────────────────────

@pytest.mark.parametrize("bad_id", [
    "../../../etc/passwd",
    "../../tmp/evil",
    "foo/bar",
    "foo\x00bar",
    "",
    "a" * 129,
])
def test_validate_policy_id_rejects_traversal(bad_id):
    with pytest.raises((HTTPException, ValueError)):
        validate_policy_id(bad_id)


@pytest.mark.parametrize("good_id", [
    "care-insurance-sample",
    "policy-123",
    "abc_DEF",
])
def test_validate_policy_id_accepts_valid(good_id):
    assert validate_policy_id(good_id) == good_id


# ── Layer 2: ingest path assertion ───────────────────────────────────────────

def test_assert_safe_pdf_path_accepts_valid(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    # Should not raise
    assert_safe_pdf_path(pdf_dir / "my-policy.pdf", pdf_dir)


def test_assert_safe_pdf_path_rejects_escape(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    # Simulate a path that somehow escapes (defense-in-depth)
    evil_path = tmp_path / "pdfs" / ".." / "evil.pdf"
    with pytest.raises(ValueError, match="outside"):
        assert_safe_pdf_path(evil_path, pdf_dir)


# ── Layer 3: sparse / retriever path assertion ────────────────────────────────

def test_assert_safe_bm25_path_accepts_valid(tmp_path):
    bm25_dir = tmp_path / "bm25_indexes"
    bm25_dir.mkdir()
    assert_safe_bm25_path(bm25_dir / "policy-123.json", bm25_dir)


def test_assert_safe_bm25_path_rejects_escape(tmp_path):
    bm25_dir = tmp_path / "bm25_indexes"
    bm25_dir.mkdir()
    evil_path = tmp_path / "bm25_indexes" / ".." / "etc" / "passwd"
    with pytest.raises(ValueError, match="outside"):
        assert_safe_bm25_path(evil_path, bm25_dir)
