"""Dense retrieval using Cohere embeddings + Pinecone."""
import json
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core.schema import NodeWithScore, TextNode


def _extract_text(meta: dict) -> str:
    """
    LlamaIndex stores node content in Pinecone metadata under '_node_content' as a JSON string.
    Extract the text field from it. Falls back to 'text' key for compatibility.
    """
    raw = meta.get("_node_content", "")
    if raw:
        try:
            node_data = json.loads(raw)
            # LlamaIndex stores text in the 'text' field of the serialized node
            text = node_data.get("text", "")
            if text:
                return text
            # Older format may use 'content'
            return node_data.get("content", "")
        except (json.JSONDecodeError, TypeError):
            return raw
    return meta.get("text", "")


def dense_retrieve(
    query: str,
    policy_id: str,
    pinecone_index,
    cohere_api_key: str,
    k: int = 5,
) -> list[NodeWithScore]:
    embed_model = CohereEmbedding(
        api_key=cohere_api_key,
        model_name="embed-english-v3.0",
        input_type="search_query",
    )
    query_embedding = embed_model.get_query_embedding(query)

    result = pinecone_index.query(
        vector=query_embedding,
        top_k=k,
        namespace=policy_id,
        include_metadata=True,
    )

    nodes = []
    for match in result.matches:
        meta = match.metadata or {}
        text = _extract_text(meta)
        node = TextNode(
            node_id=match.id,
            text=text,
            metadata={
                "policy_id": meta.get("policy_id", policy_id),
                "page_number": meta.get("page_number", 0),
                "section": meta.get("section", ""),
                "sub_clause": meta.get("sub_clause", ""),
                "chunk_type": meta.get("chunk_type", "clause"),
            },
        )
        nodes.append(NodeWithScore(node=node, score=match.score))

    return nodes
