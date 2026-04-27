import requests
import time
from requests.exceptions import Timeout, HTTPError

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

def generate_answer(query: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": query,
        "stream": False
    }

    last_error = None

    for attempt in range(3):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except HTTPError as exc:
            last_error = exc
            status_code = getattr(exc.response, "status_code", None)
            if status_code not in {500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
        except Timeout:
            raise

    raise last_error
