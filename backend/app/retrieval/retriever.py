"""
Main retrieval entry point with 4 configurable modes and AutoMerging logic.
"""
import json
from pathlib import Path
from llama_index.core.schema import NodeWithScore

from app.retrieval.dense import dense_retrieve
from app.retrieval.sparse import bm25_retrieve
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import cohere_rerank
from app import timing

AUTOMERGE_THRESHOLD = 0.5  # expand to parent if >=50% of children retrieved


def retrieve(
    query: str,
    policy_id: str,
    pinecone_index,
    cohere_api_key: str,
    data_dir: str,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[NodeWithScore]:
    """
    Retrieve top-k nodes using the specified mode.

    Modes:
        dense          - Cohere + Pinecone, top_k direct
        hybrid         - BM25 + dense, fused via RRF, top_k
        dense_rerank   - Dense top candidate_k -> Cohere Rerank -> top_k
        hybrid_rerank  - Hybrid top candidate_k -> Cohere Rerank -> top_k
    """
    if mode == "dense":
        timing.t("retrieval: dense embed+query START")
        nodes = dense_retrieve(query, policy_id, pinecone_index, cohere_api_key, k=top_k)
        timing.t("retrieval: dense DONE")

    elif mode == "hybrid":
        timing.t("retrieval: dense embed+query START")
        dense_nodes = dense_retrieve(query, policy_id, pinecone_index, cohere_api_key, k=candidate_k)
        timing.t("retrieval: dense DONE — BM25 START")
        bm25_nodes = bm25_retrieve(query, policy_id, data_dir, k=candidate_k)
        timing.t("retrieval: BM25 DONE — RRF START")
        nodes = reciprocal_rank_fusion([dense_nodes, bm25_nodes], k=top_k)
        timing.t("retrieval: RRF DONE")

    elif mode == "dense_rerank":
        timing.t("retrieval: dense embed+query START")
        candidates = dense_retrieve(query, policy_id, pinecone_index, cohere_api_key, k=candidate_k)
        timing.t("retrieval: dense DONE — Cohere rerank START")
        nodes = cohere_rerank(candidates, query, cohere_api_key, top_n=top_k)
        timing.t("retrieval: Cohere rerank DONE")

    elif mode == "hybrid_rerank":
        timing.t("retrieval: dense embed+query START")
        dense_nodes = dense_retrieve(query, policy_id, pinecone_index, cohere_api_key, k=candidate_k)
        timing.t("retrieval: dense DONE — BM25 START")
        bm25_nodes = bm25_retrieve(query, policy_id, data_dir, k=candidate_k)
        timing.t("retrieval: BM25 DONE — RRF START")
        fused = reciprocal_rank_fusion([dense_nodes, bm25_nodes], k=candidate_k)
        timing.t("retrieval: RRF DONE — Cohere rerank START")
        nodes = cohere_rerank(fused, query, cohere_api_key, top_n=top_k)
        timing.t("retrieval: Cohere rerank DONE")

    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    return auto_merge(nodes, policy_id, data_dir)


def auto_merge(
    leaf_nodes: list[NodeWithScore],
    policy_id: str,
    data_dir: str,
) -> list[NodeWithScore]:
    """
    If >=50% of a parent's children are in leaf_nodes, replace them with the parent.
    Falls back to leaf_nodes if parent store not available.
    """
    from app.retrieval.sparse import assert_safe_bm25_path
    bm25_dir = Path(data_dir) / "bm25_indexes"
    nodes_path = bm25_dir / f"{policy_id}_all_nodes.json"
    assert_safe_bm25_path(nodes_path, bm25_dir)
    if not nodes_path.exists():
        return leaf_nodes

    try:
        all_nodes_raw = json.loads(nodes_path.read_text(encoding="utf-8"))
    except Exception:
        return leaf_nodes

    # Build parent -> children map from stored metadata
    # (LlamaIndex stores parent_id in metadata during HierarchicalNodeParser)
    retrieved_ids = {nws.node.node_id for nws in leaf_nodes}
    parent_children: dict[str, list[str]] = {}
    node_lookup: dict[str, dict] = {n["node_id"]: n for n in all_nodes_raw}

    for raw in all_nodes_raw:
        parent_id = raw.get("metadata", {}).get("parent_id")
        if parent_id:
            parent_children.setdefault(parent_id, []).append(raw["node_id"])

    merged_ids: set[str] = set()
    result_nodes: list[NodeWithScore] = []

    for nws in leaf_nodes:
        node_id = nws.node.node_id
        parent_id = nws.node.metadata.get("parent_id")

        if parent_id and parent_id in parent_children:
            siblings = parent_children[parent_id]
            overlap = sum(1 for s in siblings if s in retrieved_ids)
            if overlap / len(siblings) >= AUTOMERGE_THRESHOLD:
                if parent_id not in merged_ids:
                    merged_ids.add(parent_id)
                    parent_raw = node_lookup.get(parent_id)
                    if parent_raw:
                        from llama_index.core.schema import TextNode
                        parent_node = TextNode(
                            node_id=parent_id,
                            text=parent_raw["text"],
                            metadata=parent_raw.get("metadata", {}),
                        )
                        result_nodes.append(NodeWithScore(node=parent_node, score=nws.score))
                continue

        if node_id not in merged_ids:
            result_nodes.append(nws)

    return result_nodes
