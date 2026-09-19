import json
import os
from typing import Any, Literal

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

EVALUATION_PROMPT = """You are an impartial evaluator for a retrieval-augmented answer.
Score all metrics from 0.0 to 1.0.
- faithfulness: how fully the answer is supported by the contexts
- answer_relevance: how directly the answer addresses the query
- context_relevance: how useful the contexts are for answering the query
Give one short explanation. Judge only the supplied content."""


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    context_relevance: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1, max_length=500)


class EvaluationFailure(Exception):
    def __init__(
        self,
        category: Literal[
            "authentication_error",
            "rate_limit_error",
            "timeout_error",
            "invalid_output",
            "provider_error",
        ],
    ) -> None:
        super().__init__(category)
        self.category = category


def _context_limit() -> int:
    try:
        return max(1, int(os.getenv("MAX_CONTEXT_CHARACTERS", "6000")))
    except ValueError:
        return 6000


def _truncate_contexts(contexts: list[str]) -> list[str]:
    remaining = _context_limit()
    truncated: list[str] = []
    for context in contexts:
        if remaining <= 0:
            break
        piece = context[:remaining]
        truncated.append(piece)
        remaining -= len(piece)
    return truncated


def _request_payload(trace: dict[str, Any]) -> str:
    return json.dumps(
        {
            "query": trace["query"],
            "answer": trace["answer"],
            "contexts": _truncate_contexts(trace["contexts"]),
        },
        ensure_ascii=False,
    )


def evaluate_trace(trace: dict[str, Any]) -> EvaluationResult:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EvaluationFailure("authentication_error")

    try:
        timeout = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
    except ValueError:
        timeout = 30.0

    try:
        client = Groq(api_key=api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b"),
            messages=[
                {"role": "system", "content": EVALUATION_PROMPT},
                {"role": "user", "content": _request_payload(trace)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation_scores",
                    "strict": True,
                    "schema": EvaluationResult.model_json_schema(),
                },
            },
            max_completion_tokens=300,
        )
        content = response.choices[0].message.content
        if not content:
            raise EvaluationFailure("invalid_output")
        return EvaluationResult.model_validate_json(content)
    except EvaluationFailure:
        raise
    except AuthenticationError as exc:
        raise EvaluationFailure("authentication_error") from exc
    except RateLimitError as exc:
        raise EvaluationFailure("rate_limit_error") from exc
    except APITimeoutError as exc:
        raise EvaluationFailure("timeout_error") from exc
    except (json.JSONDecodeError, ValidationError, IndexError, AttributeError) as exc:
        raise EvaluationFailure("invalid_output") from exc
    except (APIConnectionError, APIStatusError) as exc:
        raise EvaluationFailure("provider_error") from exc
    except Exception as exc:
        raise EvaluationFailure("provider_error") from exc
