"""
Unit tests for chunker.py — no external API calls needed.

Problem 3: _extract_section regex misses multi-level clause numbers (4.2, 4.2.1)
Problem 2: all_nodes duplicates prose content from both md_parser and hier_parser
"""
import pytest
from llama_index.core.schema import TextNode

from app.ingestion.chunker import _extract_section, _merge_parser_nodes


# ── Problem 3: section regex ──────────────────────────────────────────────────

def test_extract_section_matches_top_level_clause():
    assert _extract_section("4. Coverage\nsome text") == "4. Coverage"


def test_extract_section_matches_second_level_clause():
    # Fails before fix: regex r"^(\d+\.\s+.+)$" requires space after dot,
    # so "4.2 Coverage Benefits" does not match
    assert _extract_section("4.2 Coverage Benefits\nsome text") == "4.2 Coverage Benefits"


def test_extract_section_matches_third_level_clause():
    assert _extract_section("4.2.1 Pre-existing Disease Waiting Period\nsome text") == "4.2.1 Pre-existing Disease Waiting Period"


def test_extract_section_returns_empty_for_plain_prose():
    assert _extract_section("This is just a paragraph with no section numbering.") == ""


# ── Problem 2: no prose duplication in merged node list ───────────────────────

def _prose_node(node_id: str, text: str) -> TextNode:
    return TextNode(id_=node_id, text=text)


def _table_node(node_id: str) -> TextNode:
    return TextNode(
        id_=node_id,
        text="| Column A | Column B |\n|----------|----------|\n| val1 | val2 |",
    )


def test_merge_excludes_prose_md_nodes():
    """Prose nodes from MarkdownElementNodeParser must not appear in merged output."""
    prose_md = _prose_node("p1", "3.1 Some clause text about coverage eligibility.")
    table_md = _table_node("t1")
    hier = _prose_node("h1", "3.1 Some clause text about coverage eligibility.")

    result = _merge_parser_nodes([prose_md, table_md], [hier])

    ids = {n.node_id for n in result}
    assert "p1" not in ids, "Prose md node should be excluded to avoid duplication"
    assert "t1" in ids, "Table md node should be included"
    assert "h1" in ids, "Hier node should always be included"


def test_merge_keeps_all_hier_nodes():
    """All hier_nodes must always be present in merged output."""
    md_nodes = [_prose_node("m1", "some prose")]
    hier_nodes = [_prose_node(f"h{i}", f"chunk {i}") for i in range(5)]

    result = _merge_parser_nodes(md_nodes, hier_nodes)

    ids = {n.node_id for n in result}
    for i in range(5):
        assert f"h{i}" in ids


def test_merge_prose_only_document_equals_hier_nodes():
    """With no tables, merged output should be identical to hier_nodes alone."""
    md_nodes = [
        _prose_node("m1", "3.1 Waiting period clause."),
        _prose_node("m2", "3.2 Pre-existing disease clause."),
    ]
    hier_nodes = [
        _prose_node("h1", "3.1 Waiting period clause."),
        _prose_node("h2", "3.2 Pre-existing disease clause."),
    ]

    result = _merge_parser_nodes(md_nodes, hier_nodes)

    assert len(result) == len(hier_nodes), (
        f"Prose-only doc should produce len(hier_nodes)={len(hier_nodes)} nodes, "
        f"got {len(result)}"
    )


def test_merge_with_tables_adds_table_nodes():
    """With one table, merged output = hier_nodes + that table node."""
    md_nodes = [_prose_node("m1", "prose"), _table_node("t1")]
    hier_nodes = [_prose_node("h1", "prose")]

    result = _merge_parser_nodes(md_nodes, hier_nodes)

    ids = {n.node_id for n in result}
    assert "t1" in ids
    assert "m1" not in ids
    assert len(result) == 2  # h1 + t1
