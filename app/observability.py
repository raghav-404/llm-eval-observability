import os
import uuid
from typing import Any

from app.evaluator import EvaluationResult


def _enabled() -> bool:
    return (
        os.getenv("LANGSMITH_ENABLED", "false").lower() == "true"
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


def _client():
    from langsmith import Client

    return Client(api_key=os.environ["LANGSMITH_API_KEY"])


def _project() -> str:
    return os.getenv("LANGSMITH_PROJECT", "llm-reliability-platform")


def _create_child(
    client: Any,
    root_run_id: str,
    name: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> None:
    child_id = uuid.uuid4()
    client.create_run(
        name=name,
        run_type="chain",
        inputs=inputs,
        id=child_id,
        parent_run_id=root_run_id,
        trace_id=root_run_id,
        project_name=_project(),
    )
    client.update_run(child_id, outputs=outputs)


def start_trace(trace: dict[str, Any], evaluation_status: str) -> str | None:
    if not _enabled():
        return None
    try:
        run_id = uuid.uuid4()
        _client().create_run(
            name="process_trace",
            run_type="chain",
            inputs={
                "query": trace["query"],
                "answer": trace["answer"],
                "contexts": trace["contexts"],
            },
            id=run_id,
            project_name=_project(),
            extra={
                "metadata": {
                    "trace_id": trace["trace_id"],
                    "application": trace["application"],
                    "model": trace["model"],
                    "latency_ms": trace["latency_ms"],
                    "input_tokens": trace["input_tokens"],
                    "output_tokens": trace["output_tokens"],
                    "evaluation_status": evaluation_status,
                }
            },
        )
        return str(run_id)
    except Exception:
        return None


def record_safety_result(
    run_id: str | None, trace_id: str, result: dict[str, Any]
) -> None:
    if not run_id or not _enabled():
        return
    try:
        client = _client()
        safe_output = {
            "pii_detected": result["pii_detected"],
            "secret_detected": result["secret_detected"],
            "prompt_injection_detected": result["prompt_injection_detected"],
            "categories": result["categories"],
            "matched_rules": result["matched_rules"],
        }
        _create_child(
            client, run_id, "safety_check", {"trace_id": trace_id}, safe_output
        )
        _create_child(
            client,
            run_id,
            "store_trace",
            {"trace_id": trace_id},
            {"stored": True},
        )
    except Exception:
        pass


def record_evaluation(
    run_id: str | None,
    trace_id: str,
    result: EvaluationResult | None,
    status: str = "completed",
) -> None:
    if not run_id or not _enabled():
        return
    try:
        client = _client()
        output: dict[str, Any] = {"evaluation_status": status}
        if result:
            output.update(result.model_dump())
        _create_child(
            client,
            run_id,
            "evaluate_response",
            {"trace_id": trace_id},
            output,
        )
        client.update_run(run_id, outputs=output)
        if result:
            for key in ("faithfulness", "answer_relevance", "context_relevance"):
                client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=getattr(result, key),
                )
    except Exception:
        pass


def record_error(run_id: str | None, trace_id: str, category: str) -> None:
    if not run_id or not _enabled():
        return
    try:
        client = _client()
        output = {"evaluation_status": "failed", "error_category": category}
        _create_child(
            client,
            run_id,
            "evaluate_response",
            {"trace_id": trace_id},
            output,
        )
        client.update_run(run_id, outputs=output)
    except Exception:
        pass
