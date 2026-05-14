"""
Two-step chunking: MarkdownElementNodeParser for tables, then
HierarchicalNodeParser for prose. Tags every node with policy metadata.
"""
import re
from llama_index.core import Document
from llama_index.core.node_parser import (
    MarkdownElementNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.schema import BaseNode, NodeRelationship
from llama_index.llms.anthropic import Anthropic


SECTION_PATTERNS = [
    re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE),
    re.compile(r"^(\d+\.\s+.+)$", re.MULTILINE),
]

CLAUSE_PATTERN = re.compile(
    r"(?:Clause|Section|Annexure|Schedule)\s+([\d.IVXivx]+)", re.IGNORECASE
)

CHUNK_SIZES = [2048, 512, 128]


def _detect_chunk_type(text: str) -> str:
    lower = text.lower()
    if "|" in text and "---" in text:
        return "table"
    if any(k in lower for k in ["means", "refers to", '"defined"', "definition"]):
        return "definition"
    return "clause"


def _extract_section(text: str) -> str:
    for pattern in SECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()[:100]
    return ""


def _extract_sub_clause(text: str) -> str:
    m = CLAUSE_PATTERN.search(text)
    return m.group(1) if m else ""


def chunk_pages(
    pages: list[dict],
    policy_id: str,
    anthropic_api_key: str = "",
) -> tuple[list[BaseNode], list[BaseNode]]:
    """
    Convert parsed pages into hierarchical nodes.

    Returns:
        (leaf_nodes, all_nodes) - leaf_nodes for indexing, all_nodes for AutoMergingRetriever
    """
    # Build LlamaIndex Documents from pages
    documents = [
        Document(
            text=p["text"],
            metadata={
                "policy_id": policy_id,
                "page_number": p["page_number"],
                "page_label": p.get("page_label", str(p["page_number"])),
            },
        )
        for p in pages
    ]

    # Step 1: parse markdown elements (tables become atomic nodes)
    # Explicitly pass Claude so it doesn't fall back to OpenAI for table summaries
    llm = Anthropic(model="claude-haiku-4-5-20251001", api_key=anthropic_api_key) if anthropic_api_key else None
    md_parser = MarkdownElementNodeParser(num_workers=1, llm=llm)
    md_nodes = md_parser.get_nodes_from_documents(documents)

    # Step 2: hierarchical parse for prose
    hier_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)
    hier_nodes = hier_parser.get_nodes_from_documents(documents)

    all_nodes = md_nodes + hier_nodes

    # Tag every node with rich metadata
    for node in all_nodes:
        text = node.get_content()
        node.metadata.update(
            {
                "policy_id": policy_id,
                "section": _extract_section(text),
                "sub_clause": _extract_sub_clause(text),
                "chunk_type": _detect_chunk_type(text),
            }
        )
        # Ensure page_number propagates (may be missing on hier nodes)
        if "page_number" not in node.metadata:
            node.metadata["page_number"] = node.metadata.get("page_label", 0)

    leaf_nodes = get_leaf_nodes(hier_nodes) + [
        n for n in md_nodes
        if not any(
            r.node_id == n.node_id
            for r in n.relationships.get(NodeRelationship.CHILD, [])  # type: ignore[arg-type]
        )
    ]

    return leaf_nodes, all_nodes
