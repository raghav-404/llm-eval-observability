import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _database_path() -> Path:
    return Path(os.getenv("DATABASE_PATH", "data/traces.db"))


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_database_path(), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def initialize_database() -> None:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                application TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                contexts_json TEXT NOT NULL,
                model TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                pii_detected INTEGER NOT NULL,
                secret_detected INTEGER NOT NULL,
                prompt_injection_detected INTEGER NOT NULL,
                safety_categories_json TEXT NOT NULL,
                matched_rules_json TEXT NOT NULL,
                faithfulness REAL,
                answer_relevance REAL,
                context_relevance REAL,
                evaluation_explanation TEXT,
                evaluation_status TEXT NOT NULL,
                evaluation_error TEXT,
                langsmith_run_id TEXT,
                created_at TEXT NOT NULL,
                evaluated_at TEXT
            )
            """
        )


def insert_trace(trace: dict[str, Any]) -> None:
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO traces (
                trace_id, application, query, answer, contexts_json, model,
                latency_ms, input_tokens, output_tokens, pii_detected,
                secret_detected, prompt_injection_detected,
                safety_categories_json, matched_rules_json, evaluation_status,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace["trace_id"],
                trace["application"],
                trace["query"],
                trace["answer"],
                json.dumps(trace["contexts"]),
                trace["model"],
                trace["latency_ms"],
                trace["input_tokens"],
                trace["output_tokens"],
                int(trace["pii_detected"]),
                int(trace["secret_detected"]),
                int(trace["prompt_injection_detected"]),
                json.dumps(trace["safety_categories"]),
                json.dumps(trace["matched_rules"]),
                trace["evaluation_status"],
                _utc_now(),
            ),
        )


def update_evaluation(
    trace_id: str,
    *,
    status: str,
    faithfulness: float | None = None,
    answer_relevance: float | None = None,
    context_relevance: float | None = None,
    explanation: str | None = None,
    error: str | None = None,
) -> None:
    evaluated_at = _utc_now() if status in {"completed", "failed"} else None
    with _connect() as connection:
        connection.execute(
            """
            UPDATE traces
            SET faithfulness = ?, answer_relevance = ?, context_relevance = ?,
                evaluation_explanation = ?, evaluation_status = ?,
                evaluation_error = ?, evaluated_at = ?
            WHERE trace_id = ?
            """,
            (
                faithfulness,
                answer_relevance,
                context_relevance,
                explanation,
                status,
                error,
                evaluated_at,
                trace_id,
            ),
        )


def update_langsmith_run_id(trace_id: str, run_id: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE traces SET langsmith_run_id = ? WHERE trace_id = ?",
            (run_id, trace_id),
        )


def _deserialize(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["contexts"] = json.loads(result.pop("contexts_json"))
    result["safety_categories"] = json.loads(
        result.pop("safety_categories_json")
    )
    result["matched_rules"] = json.loads(result.pop("matched_rules_json"))
    for field in ("pii_detected", "secret_detected", "prompt_injection_detected"):
        result[field] = bool(result[field])
    return result


def get_trace(trace_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
    return _deserialize(row) if row else None


def list_traces(limit: int, offset: int) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT trace_id, application, model, latency_ms, evaluation_status,
                   faithfulness, answer_relevance, context_relevance,
                   pii_detected, secret_detected, prompt_injection_detected,
                   created_at, evaluated_at
            FROM traces
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    results = [dict(row) for row in rows]
    for result in results:
        for field in ("pii_detected", "secret_detected", "prompt_injection_detected"):
            result[field] = bool(result[field])
    return results


def get_summary() -> dict[str, int | float | None]:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_traces,
                SUM(CASE WHEN evaluation_status = 'pending' THEN 1 ELSE 0 END)
                    AS pending_evaluations,
                SUM(CASE WHEN evaluation_status = 'completed' THEN 1 ELSE 0 END)
                    AS completed_evaluations,
                SUM(CASE WHEN evaluation_status = 'failed' THEN 1 ELSE 0 END)
                    AS failed_evaluations,
                SUM(CASE WHEN evaluation_status = 'skipped' THEN 1 ELSE 0 END)
                    AS skipped_evaluations,
                AVG(latency_ms) AS average_latency_ms,
                AVG(faithfulness) AS average_faithfulness,
                AVG(answer_relevance) AS average_answer_relevance,
                AVG(context_relevance) AS average_context_relevance,
                COALESCE(SUM(pii_detected), 0) AS pii_detections,
                COALESCE(SUM(secret_detected), 0) AS secret_detections,
                COALESCE(SUM(prompt_injection_detected), 0)
                    AS prompt_injection_detections
            FROM traces
            """
        ).fetchone()
    summary = dict(row)
    for key in (
        "pending_evaluations",
        "completed_evaluations",
        "failed_evaluations",
        "skipped_evaluations",
    ):
        summary[key] = summary[key] or 0
    return summary


def check_database() -> bool:
    try:
        with _connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1
    except sqlite3.Error:
        return False
