import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import database, observability
from app.evaluator import EvaluationFailure, evaluate_trace
from app.safety import inspect_and_sanitize

logger = logging.getLogger("llm_reliability")


def evaluation_is_enabled() -> bool:
    return os.getenv("EVALUATION_ENABLED", "true").lower() == "true"


class TraceSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=4_000)
    answer: str = Field(min_length=1, max_length=12_000)
    contexts: list[str] = Field(min_length=1, max_length=20)
    model: str = Field(min_length=1, max_length=200)
    latency_ms: int = Field(ge=0, le=3_600_000)
    input_tokens: int = Field(ge=0, le=10_000_000)
    output_tokens: int = Field(ge=0, le=10_000_000)

    @field_validator("application", "query", "answer", "model")
    @classmethod
    def required_strings_cannot_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("contexts")
    @classmethod
    def contexts_must_contain_text(cls, contexts: list[str]) -> list[str]:
        cleaned = [context.strip() for context in contexts]
        if any(not context for context in cleaned):
            raise ValueError("contexts must not contain blank strings")
        if any(len(context) > 12_000 for context in cleaned):
            raise ValueError("each context must be at most 12000 characters")
        return cleaned


class SafetyResponse(BaseModel):
    pii_detected: bool
    secret_detected: bool
    prompt_injection_detected: bool
    categories: list[str]


class TraceAcceptedResponse(BaseModel):
    trace_id: str
    status: str
    evaluation_status: str
    safety: SafetyResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.initialize_database()
    yield


app = FastAPI(
    title="LLM Reliability, Evaluation & Safety Platform",
    version="0.2.0",
    lifespan=lifespan,
)


def _run_evaluation(
    trace_id: str, sanitized_trace: dict[str, Any], langsmith_run_id: str | None
) -> None:
    try:
        result = evaluate_trace(sanitized_trace)
        database.update_evaluation(
            trace_id,
            status="completed",
            faithfulness=result.faithfulness,
            answer_relevance=result.answer_relevance,
            context_relevance=result.context_relevance,
            explanation=result.explanation,
        )
        observability.record_evaluation(langsmith_run_id, trace_id, result)
    except EvaluationFailure as exc:
        database.update_evaluation(trace_id, status="failed", error=exc.category)
        observability.record_error(langsmith_run_id, trace_id, exc.category)
        logger.warning("Evaluation failed for trace %s: %s", trace_id, exc.category)
    except Exception:
        database.update_evaluation(trace_id, status="failed", error="provider_error")
        observability.record_error(langsmith_run_id, trace_id, "provider_error")
        logger.exception("Unexpected evaluation failure for trace %s", trace_id)


@app.post(
    "/traces",
    response_model=TraceAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_trace(submission: TraceSubmission, background_tasks: BackgroundTasks):
    trace_id = str(uuid.uuid4())
    safety_result = inspect_and_sanitize(
        submission.query, submission.answer, submission.contexts
    )
    evaluation_status = "pending" if evaluation_is_enabled() else "skipped"

    sanitized_trace: dict[str, Any] = {
        "trace_id": trace_id,
        "application": submission.application,
        "query": safety_result["sanitized_query"],
        "answer": safety_result["sanitized_answer"],
        "contexts": safety_result["sanitized_contexts"],
        "model": submission.model,
        "latency_ms": submission.latency_ms,
        "input_tokens": submission.input_tokens,
        "output_tokens": submission.output_tokens,
    }

    database.insert_trace(
        {
            **sanitized_trace,
            "pii_detected": safety_result["pii_detected"],
            "secret_detected": safety_result["secret_detected"],
            "prompt_injection_detected": safety_result[
                "prompt_injection_detected"
            ],
            "safety_categories": safety_result["categories"],
            "matched_rules": safety_result["matched_rules"],
            "evaluation_status": evaluation_status,
        }
    )

    langsmith_run_id = observability.start_trace(sanitized_trace, evaluation_status)
    if langsmith_run_id:
        database.update_langsmith_run_id(trace_id, langsmith_run_id)
    observability.record_safety_result(langsmith_run_id, trace_id, safety_result)

    if evaluation_status == "pending":
        background_tasks.add_task(
            _run_evaluation, trace_id, sanitized_trace, langsmith_run_id
        )
    else:
        observability.record_evaluation(langsmith_run_id, trace_id, None, "skipped")

    return {
        "trace_id": trace_id,
        "status": "accepted",
        "evaluation_status": evaluation_status,
        "safety": {
            "pii_detected": safety_result["pii_detected"],
            "secret_detected": safety_result["secret_detected"],
            "prompt_injection_detected": safety_result[
                "prompt_injection_detected"
            ],
            "categories": safety_result["categories"],
        },
    }


@app.get("/traces/{trace_id}")
def retrieve_trace(trace_id: str):
    trace = database.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@app.get("/traces")
def retrieve_traces(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return {
        "items": database.list_traces(limit, offset),
        "limit": limit,
        "offset": offset,
    }


@app.get("/summary")
def summary():
    return database.get_summary()


@app.get("/health")
def health():
    if not database.check_database():
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"status": "healthy", "database": "connected"}
