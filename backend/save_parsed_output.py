"""Run LlamaParse on the sample policy PDF and save the output to a .md file."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from app.ingestion.parser import parse_policy_pdf

PDF_PATH = Path(__file__).parent / "data/pdfs/care-insurance-sample.pdf"
OUT_PATH = Path(__file__).parent / "data/care-insurance-sample-parsed.md"

api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
if not api_key:
    sys.exit("LLAMA_CLOUD_API_KEY not set in .env")

print(f"Parsing {PDF_PATH} ...")
pages, page_count = parse_policy_pdf(str(PDF_PATH), api_key)
print(f"Got {page_count} pages from LlamaParse")

with OUT_PATH.open("w", encoding="utf-8") as f:
    for page in pages:
        f.write(f"<!-- page {page['page_number']} (label: {page['page_label']}) -->\n\n")
        f.write(page["text"])
        f.write("\n\n---\n\n")

print(f"Saved to {OUT_PATH}")
