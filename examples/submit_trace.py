import json
import time
import urllib.error
import urllib.request

BASE_URL = "http://localhost:8000"
TRACE = {
    "application": "demo-rag",
    "query": "What is the battery warranty?",
    "answer": "The battery warranty is eight years.",
    "contexts": ["The vehicle battery is covered for eight years or 160000 km."],
    "model": "example-model",
    "latency_ms": 1250,
    "input_tokens": 320,
    "output_tokens": 45,
}


def request_json(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def main() -> None:
    try:
        accepted = request_json(f"{BASE_URL}/traces", TRACE)
        trace_id = accepted["trace_id"]
        print(f"Accepted trace: {trace_id}")

        while True:
            trace = request_json(f"{BASE_URL}/traces/{trace_id}")
            current_status = trace["evaluation_status"]
            print(f"Evaluation status: {current_status}")
            if current_status in {"completed", "failed", "skipped"}:
                print(json.dumps(trace, indent=2))
                return
            time.sleep(1)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Could not reach the platform at {BASE_URL}: {exc}")
    except (KeyError, json.JSONDecodeError) as exc:
        print(f"The platform returned an unexpected response: {exc}")


if __name__ == "__main__":
    main()
