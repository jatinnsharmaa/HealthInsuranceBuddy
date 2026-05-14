"""Parse policy PDFs using LlamaParse in agentic mode."""
import os
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader


def parse_policy_pdf(pdf_path: str, llama_cloud_api_key: str) -> tuple[list[dict], int]:
    """
    Parse a policy PDF using LlamaParse agentic mode.

    Returns:
        (pages, page_count) where pages is a list of {"page_number": int, "text": str}
    """
    parser = LlamaParse(
        api_key=llama_cloud_api_key,
        result_type="markdown",
        verbose=True,
        language="en",
        # Agentic mode: better handling of tables and complex layouts
        parsing_instruction=(
            "This is an Indian health insurance policy document. "
            "Preserve all tables intact, including Schedule of Benefits, "
            "sub-limit tables, exclusion lists, and waiting period tables. "
            "Keep clause numbers (e.g. 'Clause 3.2', 'Annexure II') exactly as they appear. "
            "Preserve page breaks between sections."
        ),
    )

    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader(
        input_files=[pdf_path],
        file_extractor=file_extractor,
    )
    documents = reader.load_data()

    pages = []
    for i, doc in enumerate(documents):
        page_num = i + 1
        metadata = doc.metadata or {}
        # LlamaParse may provide page_label in metadata
        actual_page = metadata.get("page_label", str(page_num))
        pages.append({
            "page_number": page_num,
            "page_label": actual_page,
            "text": doc.text,
        })

    return pages, len(pages)
