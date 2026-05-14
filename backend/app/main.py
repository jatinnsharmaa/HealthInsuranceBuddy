"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.feedback.store import init_db
from app.api import ingest as ingest_module
from app.api import chat as chat_module
from app.api.ingest import router as ingest_router
from app.api.chat import router as chat_router, set_pinecone_index
from app.api.feedback import router as feedback_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Ensure data directories exist
    for subdir in ["pdfs", "bm25_indexes"]:
        Path(settings.data_dir, subdir).mkdir(parents=True, exist_ok=True)

    # Init feedback DB
    db_path = str(Path(settings.data_dir) / "feedback.db")
    await init_db(db_path)

    # Init Pinecone index (if keys are available)
    if settings.pinecone_api_key and settings.cohere_api_key:
        try:
            from app.ingestion.indexer import get_pinecone_index
            idx = get_pinecone_index(settings.pinecone_api_key, settings.pinecone_index_name)
            set_pinecone_index(idx)
            print(f"Pinecone index '{settings.pinecone_index_name}' connected.")
        except Exception as e:
            print(f"Warning: Could not connect to Pinecone: {e}")
    else:
        print("Warning: PINECONE_API_KEY or COHERE_API_KEY not set. Chat will not function.")

    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Health Insurance Policy Interpreter",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingest_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")
    app.include_router(feedback_router, prefix="/api")

    # Serve PDFs statically so the frontend PDF viewer can load them
    pdfs_dir = Path(settings.data_dir) / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/pdfs", StaticFiles(directory=str(pdfs_dir)), name="pdfs")

    return app


app = create_app()
