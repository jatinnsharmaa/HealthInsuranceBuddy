"""Cohere cross-encoder reranking — uses ClientV2 (cohere v5.x)."""
import cohere
from llama_index.core.schema import NodeWithScore


def cohere_rerank(
    nodes: list[NodeWithScore],
    query: str,
    cohere_api_key: str,
    top_n: int = 5,
) -> list[NodeWithScore]:
    if not nodes:
        return []

    co = cohere.ClientV2(api_key=cohere_api_key)
    documents = [n.node.get_content() for n in nodes]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        original = nodes[result.index]
        reranked.append(
            NodeWithScore(node=original.node, score=result.relevance_score)
        )

    return reranked
