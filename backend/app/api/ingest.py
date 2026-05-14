"""Ingestion API: accepts a PDF upload and runs the full ingestion pipeline."""
import asyncio
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.ingestion.parser import parse_policy_pdf
from app.ingestion.chunker import chunk_pages
from app.ingestion.indexer import build_and_store_indexes, get_pinecone_index
from app.ingestion.url_extractor import extract_deferred_urls

router = APIRouter()

# In-memory job status (replace with Redis or DB for production)
_jobs: dict[str, dict] = {}


def _run_ingestion(job_id: str, pdf_path: str, policy_id: str):
    settings = get_settings()
    try:
        _jobs[job_id] = {"status": "parsing", "progress": 10}

        pages, page_count = parse_policy_pdf(pdf_path, settings.llama_cloud_api_key)
        _jobs[job_id] = {"status": "chunking", "progress": 40}

        deferred_urls = extract_deferred_urls(pages)
        leaf_nodes, all_nodes = chunk_pages(pages, policy_id, anthropic_api_key=settings.anthropic_api_key)
        _jobs[job_id] = {"status": "indexing", "progress": 70}

        pinecone_idx = get_pinecone_index(
            settings.pinecone_api_key,
            settings.pinecone_index_name,
        )
        stats = build_and_store_indexes(
            leaf_nodes=leaf_nodes,
            all_nodes=all_nodes,
            policy_id=policy_id,
            cohere_api_key=settings.cohere_api_key,
            pinecone_api_key=settings.pinecone_api_key,
            pinecone_index_name=settings.pinecone_index_name,
            data_dir=settings.data_dir,
        )

        _jobs[job_id] = {
            "status": "ready",
            "progress": 100,
            "policy_id": policy_id,
            "page_count": page_count,
            "chunk_count": stats["leaf_count"],
            "deferred_urls": deferred_urls,
        }
    except Exception as e:
        _jobs[job_id] = {"status": "error", "error": str(e)}


@router.post("/ingest")
async def ingest_policy(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    policy_id: Annotated[str | None, Form()] = None,
):
    settings = get_settings()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    policy_id = policy_id or str(uuid.uuid4())
    pdf_dir = Path(settings.data_dir) / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = str(pdf_dir / f"{policy_id}.pdf")

    content = await file.read()
    Path(pdf_path).write_bytes(content)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "progress": 0}

    background_tasks.add_task(_run_ingestion, job_id, pdf_path, policy_id)

    return {"job_id": job_id, "policy_id": policy_id}


@router.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@router.get("/ingest/sample")
async def get_sample_policy():
    """Return metadata about the pre-indexed sample policy."""
    settings = get_settings()
    sample_id = "care-insurance-sample"
    pdf_path = Path(settings.data_dir) / "pdfs" / f"{sample_id}.pdf"

    if not pdf_path.exists():
        raise HTTPException(404, "Sample policy not yet indexed. Run ingestion first.")

    return {
        "policy_id": sample_id,
        "display_name": "Care Insurance Supreme – Sample Policy",
        "pdf_url": f"/pdfs/{sample_id}.pdf",
    }
