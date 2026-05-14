"""Chat API: real SSE token streaming from the Orchestrator agent."""
import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.feedback.store import log_query

router = APIRouter()

_pinecone_index: Any = None


def set_pinecone_index(idx: Any):
    global _pinecone_index
    _pinecone_index = idx


class ChatRequest(BaseModel):
    session_id: str
    policy_id: str
    message: str
    conversation_history: list[dict] = []
    retrieval_mode: str = "hybrid_rerank"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_response(request: ChatRequest):
    settings = get_settings()
    db_path = str(Path(settings.data_dir) / "feedback.db")
    t0 = time.perf_counter()

    if _pinecone_index is None:
        yield _sse({"type": "error", "content": "Service not ready. Try again shortly."})
        return

    yield _sse({"type": "agent_step", "content": "Analysing your question..."})

    answer = None
    retrieval_log = []
    web_fetch_status = "not_triggered"
    input_tokens = 0
    output_tokens = 0
    is_error = False
    error_message = None

    try:
        from app.agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent(
            policy_id=request.policy_id,
            pinecone_index=_pinecone_index,
            cohere_api_key=settings.cohere_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            data_dir=settings.data_dir,
            retrieval_mode=request.retrieval_mode,
            top_k=settings.retrieval_top_k,
            candidate_k=settings.retrieval_candidate_k,
        )

        async for event in orchestrator.stream_chat(
            message=request.message,
            conversation_history=request.conversation_history,
        ):
            if event.get("type") == "done":
                retrieval_log = event.get("retrieval_log", [])
                web_fetch_status = event.get("web_fetch_status", "not_triggered")
                input_tokens = event.get("input_tokens", 0)
                output_tokens = event.get("output_tokens", 0)
                answer = event.get("answer") or answer
            elif event.get("type") == "chunk":
                if answer is None:
                    answer = ""
                answer += event.get("content", "")
            yield _sse(event)

    except Exception as e:
        is_error = True
        error_message = str(e)
        yield _sse({"type": "error", "content": f"An error occurred: {str(e)}"})
    finally:
        total_latency_ms = (time.perf_counter() - t0) * 1000

        # Extract retrieval latency from log
        retrieval_latency_ms = None
        for entry in retrieval_log:
            if entry.get("tool") == "retrieve_from_policy" and entry.get("latency_ms"):
                retrieval_latency_ms = entry["latency_ms"]
                break

        # Extract web latency and URLs
        web_latency_ms = None
        web_urls = []
        for entry in retrieval_log:
            if entry.get("tool") == "search_web":
                if entry.get("latency_ms"):
                    web_latency_ms = entry["latency_ms"]
                if entry.get("start_url"):
                    web_urls.append(entry["start_url"])

        # Extract retrieved chunks
        retrieved_chunks = [
            e for e in retrieval_log
            if e.get("tool") == "retrieve_from_policy" and e.get("result_full")
        ]

        tools_called = [e.get("tool", "") for e in retrieval_log]

        await log_query(
            db_path=db_path,
            session_id=request.session_id,
            policy_id=request.policy_id,
            question=request.message,
            answer=answer,
            conversation_history=request.conversation_history,
            retrieval_mode=request.retrieval_mode,
            retrieved_chunks=retrieved_chunks,
            retrieval_latency_ms=retrieval_latency_ms,
            web_triggered=web_fetch_status != "not_triggered",
            web_urls_attempted=web_urls,
            web_fetch_status=web_fetch_status,
            web_fetch_latency_ms=web_latency_ms,
            tools_called=tools_called,
            total_latency_ms=total_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            is_error=is_error,
            error_message=error_message,
        )


@router.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        _stream_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
