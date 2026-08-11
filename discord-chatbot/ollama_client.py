import requests
from config import OLLAMA_URL, OLLAMA_MODEL

def ollama_generate(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 500,
            },
        },
        timeout=180,
    )
    r.raise_for_status()
    return (r.json().get("response", "") or "").strip()