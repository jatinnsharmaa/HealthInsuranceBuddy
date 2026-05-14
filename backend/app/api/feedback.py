"""Feedback API: logs thumbs up/down with conversation context."""
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings
from app.feedback.store import log_feedback

router = APIRouter()


class FeedbackRequest(BaseModel):
    session_id: str
    policy_id: str
    conversation: list[dict]
    retrieval_log: list[dict] = []
    web_fetch_status: str = "not_triggered"
    feedback_signal: str  # "up" or "down"
    feedback_text: str | None = None
    query_id: int | None = None


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    settings = get_settings()
    db_path = str(Path(settings.data_dir) / "feedback.db")

    await log_feedback(
        db_path=db_path,
        session_id=request.session_id,
        conversation=request.conversation,
        retrieval_log=request.retrieval_log,
        web_fetch_status=request.web_fetch_status,
        feedback_signal=request.feedback_signal,
        feedback_text=request.feedback_text,
        policy_document_id=request.policy_id,
        query_id=request.query_id,
    )

    return {"status": "logged"}
