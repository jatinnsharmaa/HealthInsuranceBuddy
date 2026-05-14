"""Reciprocal Rank Fusion (RRF) over two ranked node lists."""
from llama_index.core.schema import NodeWithScore

RRF_K = 60  # standard RRF constant


def reciprocal_rank_fusion(
    ranked_lists: list[list[NodeWithScore]],
    k: int = 5,
) -> list[NodeWithScore]:
    """
    Fuse multiple ranked lists using RRF.
    Returns top-k nodes by fused score.
    """
    rrf_scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    for ranked in ranked_lists:
        for rank, nws in enumerate(ranked):
            node_id = nws.node.node_id
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            if node_id not in node_map:
                node_map[node_id] = nws

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
    return [
        NodeWithScore(node=node_map[nid].node, score=rrf_scores[nid])
        for nid in sorted_ids
    ]
