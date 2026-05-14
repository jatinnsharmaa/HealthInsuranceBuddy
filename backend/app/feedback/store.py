"""SQLite-backed feedback store using aiosqlite."""
import json
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path


async def init_db(db_path: str):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                conversation TEXT,
                retrieval_log TEXT,
                web_fetch_status TEXT,
                feedback_signal TEXT,
                feedback_text TEXT,
                policy_document_id TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await db.commit()


async def log_feedback(
    db_path: str,
    session_id: str,
    conversation: list[dict],
    retrieval_log: list[dict],
    web_fetch_status: str,
    feedback_signal: str,
    feedback_text: str | None,
    policy_document_id: str,
):
    timestamp = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO feedback
              (session_id, conversation, retrieval_log, web_fetch_status,
               feedback_signal, feedback_text, policy_document_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                json.dumps(conversation),
                json.dumps(retrieval_log),
                web_fetch_status,
                feedback_signal,
                feedback_text,
                policy_document_id,
                timestamp,
            ),
        )
        await db.commit()
