from fastapi import FastAPI
import time
import uuid

from prometheus_client import Counter, Histogram, Gauge, start_http_server

from app.llm import generate_answer
from app.eval import evaluate_response
from app.logger import get_logger

app = FastAPI()
logger = get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter("llm_requests_total", "Total requests")
LATENCY = Histogram("llm_latency_seconds", "Request latency")

FAITHFULNESS = Gauge("rag_faithfulness", "Faithfulness score")
RELEVANCY = Gauge("rag_relevancy", "Relevancy score")
CONTEXT_PRECISION = Gauge("rag_context_precision", "Context precision score")

# Start metrics server
start_http_server(8001)


@app.post("/ask")
def ask(query: str, reference: str | None = None):
    request_id = str(uuid.uuid4())
    start = time.time()

    try:
        REQUEST_COUNT.inc()

        # ---- LLM ----
        answer = generate_answer(query)

        # ---- Mock context (replace later with real RAG) ----
        contexts = [answer]  # simple placeholder

        # ---- Evaluation ----
        scores = evaluate_response(query, answer, contexts, reference=reference)

        # ---- Metrics ----
        FAITHFULNESS.set(scores["faithfulness"])
        RELEVANCY.set(scores["relevancy"])
        if scores["context_precision"] is not None:
            CONTEXT_PRECISION.set(scores["context_precision"])

        latency = time.time() - start
        LATENCY.observe(latency)

        # ---- Logging ----
        logger.info({
            "request_id": request_id,
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "reference": reference,
            "latency_ms": latency * 1000,
            "faithfulness": scores["faithfulness"],
            "relevancy": scores["relevancy"],
            "context_precision": scores["context_precision"],
            "model": "qwen2.5:7b",
            "status": "success"
        })

        return {
            "answer": answer,
            "metrics": scores
        }

    except Exception as e:
        logger.error({
            "request_id": request_id,
            "query": query,
            "error": str(e),
            "status": "failure"
        })
        return {"error": str(e)}
