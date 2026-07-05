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
    re.compile(r"^(\d+(?:\.\d+)*\.?\s+.+)$", re.MULTILINE),
]

CLAUSE_PATTERN = re.compile(
    r"(?:Clause|Section|Annexure|Schedule)\s+([\d.IVXivx]+)\b", re.IGNORECASE
)

# Matches bare multi-level clause numbers: 2.1.1, 3.2.18, 4.1(a)(ii)
# Requires at least two dot-separated components to avoid matching list items
BARE_CLAUSE_RE = re.compile(
    r"(?:^|\*\*)(\d+(?:\.\d+)+(?:\([a-zA-Z0-9ivxIVX]+\))*)",
    re.MULTILINE,
)

# Splits on multi-level clause headings (2.1., 2.1.1., 2.1.4.) and markdown headers.
# Requires a digit after the first dot so single-level items (1. Asthma;, 9.  Note)
# are not treated as clause boundaries.
CLAUSE_BOUNDARY_RE = re.compile(
    r"(?m)^(?=\d+\.\d[\d.]*[\s.(]|#{1,3}\s)",
)

CHUNK_SIZES = [2048, 512, 128]


def _merge_parser_nodes(
    md_nodes: list,
    hier_nodes: list,
) -> list:
    """Combine parser outputs without duplicating prose.
    Only table nodes from md_nodes are kept; prose is covered by hier_nodes.
    """
    table_nodes = [n for n in md_nodes if _detect_chunk_type(n.get_content()) == "table"]
    return table_nodes + hier_nodes


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
    m = BARE_CLAUSE_RE.match(text)        # heading at chunk start wins
    if m:
        return m.group(1)
    m = CLAUSE_PATTERN.search(text)       # explicit prefix (cross-refs, table labels)
    if m:
        return m.group(1)
    m = BARE_CLAUSE_RE.search(text)       # bare number anywhere in text as last resort
    return m.group(1) if m else ""


def _split_into_clause_documents(doc: Document, min_chars: int = 80) -> list[Document]:
    """Pre-split a page document on multi-level clause headings and markdown headers."""
    text = doc.text
    positions = [m.start() for m in CLAUSE_BOUNDARY_RE.finditer(text)]
    if not positions:
        return [doc]
    boundaries = [0] + positions + [len(text)]
    result = []
    for i in range(len(boundaries) - 1):
        chunk = text[boundaries[i]:boundaries[i + 1]].strip()
        if len(chunk) >= min_chars:
            result.append(Document(text=chunk, metadata=doc.metadata.copy()))
    return result or [doc]


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

    # Pre-split each page on clause boundaries so HierarchicalNodeParser only
    # sub-splits oversized individual clauses, not across clause boundaries
    clause_docs: list[Document] = []
    for doc in documents:
        clause_docs.extend(_split_into_clause_documents(doc))

    # Step 2: hierarchical parse on clause-pre-split documents
    hier_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)
    hier_nodes = hier_parser.get_nodes_from_documents(clause_docs)

    all_nodes = _merge_parser_nodes(md_nodes, hier_nodes)

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

    table_md_nodes = [n for n in md_nodes if _detect_chunk_type(n.get_content()) == "table"]
    leaf_nodes = get_leaf_nodes(hier_nodes) + table_md_nodes

    return leaf_nodes, all_nodes
