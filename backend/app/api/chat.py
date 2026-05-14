"""Chat API: real SSE token streaming from the Orchestrator agent."""
import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings

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

    if _pinecone_index is None:
        yield _sse({"type": "error", "content": "Service not ready. Try again shortly."})
        return

    yield _sse({"type": "agent_step", "content": "Analysing your question..."})

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
            yield _sse(event)

    except Exception as e:
        yield _sse({"type": "error", "content": f"An error occurred: {str(e)}"})


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
