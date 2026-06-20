"""BM25 sparse retrieval from JSON-serialized index."""
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
from llama_index.core.schema import NodeWithScore, TextNode


def assert_safe_bm25_path(bm25_path: Path, bm25_dir: Path) -> None:
    """Raise ValueError if bm25_path resolves outside bm25_dir (defense-in-depth)."""
    if not bm25_path.resolve().is_relative_to(bm25_dir.resolve()):
        raise ValueError(f"Resolved path is outside {bm25_dir}: {bm25_path}")


def load_bm25(policy_id: str, data_dir: str) -> tuple[BM25Okapi, list[dict]]:
    bm25_dir = Path(data_dir) / "bm25_indexes"
    bm25_path = bm25_dir / f"{policy_id}.json"
    assert_safe_bm25_path(bm25_path, bm25_dir)
    data = json.loads(bm25_path.read_text(encoding="utf-8"))
    corpus = data["corpus"]
    node_meta = data["nodes"]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, node_meta


def bm25_retrieve(
    query: str,
    policy_id: str,
    data_dir: str,
    k: int = 5,
) -> list[NodeWithScore]:
    bm25, node_meta = load_bm25(policy_id, data_dir)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    nodes = []
    for idx in top_k_idx:
        meta = node_meta[idx]
        node = TextNode(
            node_id=meta["node_id"],
            text=meta["text"],
            metadata={
                "policy_id": meta.get("policy_id", policy_id),
                "page_number": meta.get("page_number", 0),
                "section": meta.get("section", ""),
                "sub_clause": meta.get("sub_clause", ""),
                "chunk_type": meta.get("chunk_type", "clause"),
            },
        )
        nodes.append(NodeWithScore(node=node, score=float(scores[idx])))

    return nodes
