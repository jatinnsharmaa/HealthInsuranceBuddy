"""SQLite-backed query log and feedback store using aiosqlite."""
import json
import aiosqlite
from datetime import datetime, timezone
from pathlib import Path


async def init_db(db_path: str):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                conversation_history TEXT,
                retrieval_mode TEXT,
                retrieved_chunks TEXT,
                retrieval_latency_ms REAL,
                web_triggered INTEGER DEFAULT 0,
                web_urls_attempted TEXT,
                web_fetch_status TEXT,
                web_fetch_latency_ms REAL,
                tools_called TEXT,
                total_latency_ms REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                estimated_cost_usd REAL,
                is_error INTEGER DEFAULT 0,
                error_message TEXT,
                timestamp TEXT NOT NULL
            )
        """)
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
                timestamp TEXT NOT NULL,
                query_id INTEGER REFERENCES queries(id)
            )
        """)
        # Migrate existing feedback table if query_id column is missing
        try:
            await db.execute("ALTER TABLE feedback ADD COLUMN query_id INTEGER REFERENCES queries(id)")
        except Exception:
            pass  # column already exists
        await db.commit()


async def log_query(
    db_path: str,
    session_id: str,
    policy_id: str,
    question: str,
    answer: str | None,
    conversation_history: list[dict],
    retrieval_mode: str,
    retrieved_chunks: list[dict],
    retrieval_latency_ms: float | None,
    web_triggered: bool,
    web_urls_attempted: list[str],
    web_fetch_status: str,
    web_fetch_latency_ms: float | None,
    tools_called: list[str],
    total_latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    is_error: bool = False,
    error_message: str | None = None,
) -> int:
    estimated_cost = (input_tokens * 1.0 + output_tokens * 5.0) / 1_000_000
    timestamp = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO queries (
                session_id, policy_id, question, answer, conversation_history,
                retrieval_mode, retrieved_chunks, retrieval_latency_ms,
                web_triggered, web_urls_attempted, web_fetch_status, web_fetch_latency_ms,
                tools_called, total_latency_ms, input_tokens, output_tokens,
                estimated_cost_usd, is_error, error_message, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                policy_id,
                question,
                answer,
                json.dumps(conversation_history),
                retrieval_mode,
                json.dumps(retrieved_chunks),
                retrieval_latency_ms,
                1 if web_triggered else 0,
                json.dumps(web_urls_attempted),
                web_fetch_status,
                web_fetch_latency_ms,
                json.dumps(tools_called),
                total_latency_ms,
                input_tokens,
                output_tokens,
                estimated_cost,
                1 if is_error else 0,
                error_message,
                timestamp,
            ),
        )
        await db.commit()
        return cursor.lastrowid


async def log_feedback(
    db_path: str,
    session_id: str,
    conversation: list[dict],
    retrieval_log: list[dict],
    web_fetch_status: str,
    feedback_signal: str,
    feedback_text: str | None,
    policy_document_id: str,
    query_id: int | None = None,
):
    timestamp = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO feedback
              (session_id, conversation, retrieval_log, web_fetch_status,
               feedback_signal, feedback_text, policy_document_id, timestamp, query_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                query_id,
            ),
        )
        await db.commit()
