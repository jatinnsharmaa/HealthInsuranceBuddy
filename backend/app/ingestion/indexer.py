"""
Indexes leaf nodes to Pinecone (dense) and builds a JSON-serialized BM25 index.
Parent nodes are stored in-memory for AutoMergingRetriever.
"""
import json
import os
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from pinecone import Pinecone, ServerlessSpec


def get_pinecone_index(api_key: str, index_name: str, dimension: int = 1024):
    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


def build_and_store_indexes(
    leaf_nodes: list[BaseNode],
    all_nodes: list[BaseNode],
    policy_id: str,
    cohere_api_key: str,
    pinecone_api_key: str,
    pinecone_index_name: str,
    data_dir: str,
) -> dict:
    """
    1. Upsert leaf nodes into Pinecone (namespace = policy_id).
    2. Build BM25 index over leaf nodes and serialize to JSON.
    3. Persist all_nodes as JSON for AutoMergingRetriever.

    Returns metadata dict with counts.
    """
    embed_model = CohereEmbedding(
        api_key=cohere_api_key,
        model_name="embed-english-v3.0",
        input_type="search_document",
    )

    pinecone_idx = get_pinecone_index(pinecone_api_key, pinecone_index_name)
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_idx,
        namespace=policy_id,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes=leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    # BM25 — serialize as JSON (tokenized corpus + metadata)
    bm25_dir = Path(data_dir) / "bm25_indexes"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = bm25_dir / f"{policy_id}.json"

    corpus = []
    node_meta = []
    for node in leaf_nodes:
        corpus.append(node.get_content())
        node_meta.append({
            "node_id": node.node_id,
            "policy_id": node.metadata.get("policy_id", ""),
            "page_number": node.metadata.get("page_number", 0),
            "section": node.metadata.get("section", ""),
            "sub_clause": node.metadata.get("sub_clause", ""),
            "chunk_type": node.metadata.get("chunk_type", "clause"),
            "text": node.get_content(),
        })

    bm25_path.write_text(
        json.dumps({"corpus": corpus, "nodes": node_meta}),
        encoding="utf-8",
    )

    # Persist all nodes for AutoMergingRetriever
    nodes_path = bm25_dir / f"{policy_id}_all_nodes.json"
    nodes_path.write_text(
        json.dumps([
            {
                "node_id": n.node_id,
                "text": n.get_content(),
                "metadata": n.metadata,
            }
            for n in all_nodes
        ]),
        encoding="utf-8",
    )

    return {
        "leaf_count": len(leaf_nodes),
        "total_nodes": len(all_nodes),
        "bm25_path": str(bm25_path),
    }
