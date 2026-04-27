import os
import json
import re

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
import requests

from app.llm import MODEL, OLLAMA_URL

def evaluate_response(question, answer, contexts, reference=None):
    record = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }

    metrics = [faithfulness, answer_relevancy]

    if reference is not None:
        record["reference"] = reference
        metrics.append(context_precision)

    dataset = Dataset.from_list([record])

    if os.getenv("OPENAI_API_KEY"):
        try:
            result = evaluate(dataset, metrics=metrics)
            scores = {
                "faithfulness": float(result["faithfulness"]),
                "relevancy": float(result["answer_relevancy"]),
                "context_precision": None,
                "evaluation_mode": "ragas",
            }

            if reference is not None:
                scores["context_precision"] = float(result["context_precision"])

            return scores
        except Exception:
            pass

    return _ollama_evaluate(question, answer, contexts, reference=reference)


def _extract_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _ollama_evaluate(question, answer, contexts, reference=None):
    prompt = (
        "You are scoring an LLM answer.\n"
        "Return only JSON with numeric values between 0 and 1.\n"
        "Use this schema: "
        '{"faithfulness": 0.0, "relevancy": 0.0, "context_precision": null}\n\n'
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Contexts: {contexts}\n"
    )

    if reference is not None:
        prompt += f"Reference: {reference}\n"
        prompt += "Set context_precision to a number between 0 and 1.\n"
    else:
        prompt += "Set context_precision to null.\n"

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    parsed = _extract_json(data.get("response", ""))

    return {
        "faithfulness": float(parsed.get("faithfulness", 0.0)),
        "relevancy": float(parsed.get("relevancy", 0.0)),
        "context_precision": (
            float(parsed["context_precision"])
            if parsed.get("context_precision") is not None
            else None
        ),
        "evaluation_mode": "ollama",
    }
