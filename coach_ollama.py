from __future__ import annotations
import requests, json
from typing import Dict, Any

OLLAMA_URL = "http://localhost:11434" # change if remote
MODEL = "mistral:7b"

SYSTEM = (
"You are a concise speaking coach. Use the provided metrics and transcript to give actionable feedback. "
"Focus on clarity, confidence, tone, pacing, engagement, cadence, and flow. Provide 3–5 concrete tips."
)

class Coach:
    def __init__(self, base_url: str = OLLAMA_URL, model: str = MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model

    def coach(self, transcript: str, metrics: Dict[str,Any], rubric: Dict[str,Any]) -> str:
        prompt = (
        "Metrics:\n" + json.dumps(metrics, indent=2) +
        "\n\nRubric Scores:\n" + json.dumps(rubric, indent=2) +
        "\n\nTranscript (excerpt):\n" + transcript[:1200] +
        "\n\nWrite feedback: bullet points, prioritized, with small examples to try."
        )
        data = {"model": self.model, "prompt": f"[SYSTEM]\n{SYSTEM}\n\n[USER]\n{prompt}", "stream": False}
        r = requests.post(f"{self.base_url}/api/generate", json=data, timeout=60)
        r.raise_for_status()
        out = r.json().get("response", "")
        return out.strip()