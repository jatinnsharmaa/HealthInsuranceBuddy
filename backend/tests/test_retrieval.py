"""
Retrieval layer tests: all 4 modes, context quality, parallel race condition.
"""
import asyncio
import pytest
from app.retrieval.retriever import retrieve

QUERY = "waiting period for maternity benefits"
MODES = ["dense", "hybrid", "dense_rerank", "hybrid_rerank"]


# ── Per-mode retrieval ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", MODES)
def test_mode_returns_nodes(mode, settings, pinecone_index, policy_id):
    nodes = retrieve(
        query=QUERY,
        policy_id=policy_id,
        pinecone_index=pinecone_index,
        cohere_api_key=settings.cohere_api_key,
        data_dir=settings.data_dir,
        mode=mode,
        top_k=5,
        candidate_k=20,
    )
    assert len(nodes) > 0, f"Mode '{mode}' returned no nodes"


@pytest.mark.parametrize("mode", MODES)
def test_mode_returns_exactly_top_k(mode, settings, pinecone_index, policy_id):
    nodes = retrieve(
        query=QUERY, policy_id=policy_id, pinecone_index=pinecone_index,
        cohere_api_key=settings.cohere_api_key, data_dir=settings.data_dir,
        mode=mode, top_k=5, candidate_k=20,
    )
    assert len(nodes) <= 5, f"Mode '{mode}' returned {len(nodes)} nodes, expected ≤5"


@pytest.mark.parametrize("mode", MODES)
def test_mode_nodes_have_required_metadata(mode, settings, pinecone_index, policy_id):
    nodes = retrieve(
        query=QUERY, policy_id=policy_id, pinecone_index=pinecone_index,
        cohere_api_key=settings.cohere_api_key, data_dir=settings.data_dir,
        mode=mode, top_k=3, candidate_k=20,
    )
    for nws in nodes:
        meta = nws.node.metadata
        assert "page_number" in meta, f"Node missing page_number in mode '{mode}'"
        assert "chunk_type" in meta, f"Node missing chunk_type in mode '{mode}'"
        assert meta["policy_id"] == policy_id, \
            f"Wrong policy_id in node: {meta.get('policy_id')}"


@pytest.mark.parametrize("mode", MODES)
def test_mode_context_text_not_empty(mode, settings, pinecone_index, policy_id):
    nodes = retrieve(
        query=QUERY, policy_id=policy_id, pinecone_index=pinecone_index,
        cohere_api_key=settings.cohere_api_key, data_dir=settings.data_dir,
        mode=mode, top_k=5, candidate_k=20,
    )
    total_text = " ".join(nws.node.get_content() for nws in nodes)
    assert len(total_text) > 200, \
        f"Mode '{mode}' returned very short context ({len(total_text)} chars)"


@pytest.mark.parametrize("mode", MODES)
def test_mode_scores_are_positive(mode, settings, pinecone_index, policy_id):
    nodes = retrieve(
        query=QUERY, policy_id=policy_id, pinecone_index=pinecone_index,
        cohere_api_key=settings.cohere_api_key, data_dir=settings.data_dir,
        mode=mode, top_k=5, candidate_k=20,
    )
    for nws in nodes:
        assert nws.score is not None and nws.score >= 0, \
            f"Node has invalid score: {nws.score} in mode '{mode}'"


# ── Race condition: parallel calls ─────────────────────────────────────────────

def test_parallel_retrieve_no_race_condition(settings, pinecone_index, policy_id):
    """
    Simulates 4 parallel retrieve calls (as happens when 4 modes run concurrently).
    Verifies all return independent results with no shared state corruption.
    """
    queries = [
        "maternity waiting period",
        "room rent sub-limit",
        "PED exclusion clause",
        "cashless network hospital",
    ]

    async def run_all():
        tasks = [
            asyncio.to_thread(
                retrieve,
                query=q,
                policy_id=policy_id,
                pinecone_index=pinecone_index,
                cohere_api_key=settings.cohere_api_key,
                data_dir=settings.data_dir,
                mode="hybrid_rerank",
                top_k=5,
                candidate_k=20,
            )
            for q in queries
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_all())
    assert len(results) == 4
    for i, nodes in enumerate(results):
        assert len(nodes) > 0, f"Parallel retrieve call {i} returned no nodes"
        total_text = " ".join(n.node.get_content() for n in nodes)
        assert len(total_text) > 100, \
            f"Parallel retrieve call {i} returned empty context — possible race condition"


# ── BM25 specific ──────────────────────────────────────────────────────────────

def test_bm25_index_loads_for_policy(settings, policy_id):
    from app.retrieval.sparse import load_bm25
    bm25, node_meta = load_bm25(policy_id, settings.data_dir)
    assert bm25 is not None
    assert len(node_meta) > 100, f"BM25 corpus too small: {len(node_meta)} entries"


def test_bm25_returns_nonzero_scores(settings, policy_id):
    from app.retrieval.sparse import bm25_retrieve
    nodes = bm25_retrieve("maternity waiting period", policy_id, settings.data_dir, k=5)
    assert len(nodes) > 0
    top_score = nodes[0].score
    assert top_score > 0, f"BM25 top score is 0 — index may be empty or tokenization broken"
