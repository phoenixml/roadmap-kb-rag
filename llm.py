from __future__ import annotations
import os, json, requests
from typing import Any, Dict, List, Optional
from kb.config import load_yaml

import json
import re

class LLM:
    def __init__(self, model_cfg_path: str = "configs/model.yaml"):
        cfg = load_yaml(model_cfg_path)
        self.backend = (cfg.get("backend") or "ollama").strip().lower()
        self.cfg = cfg

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 1200) -> str:
        if self.backend == "openai":
            return self._openai_chat(messages, temperature=temperature, max_tokens=max_tokens)
        return self._ollama_chat(messages, temperature=temperature, max_tokens=max_tokens)

    def _openai_chat(self, messages, temperature: float, max_tokens: int) -> str:
        # Uses OpenAI-compatible REST endpoint from env via `OPENAI_BASE_URL` if desired
        # Default expects official OpenAI. You can also point to a proxy.
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set, but backend=openai")
        model = self.cfg.get("openai", {}).get("model", "gpt-4.1")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _ollama_chat(self, messages, temperature: float, max_tokens: int) -> str:
        base_url = (self.cfg.get("ollama", {}) or {}).get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = (self.cfg.get("ollama", {}) or {}).get("model") or os.getenv("OLLAMA_MODEL", "llama3.2")
        url = base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False
        }
        r = requests.post(url, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")



def json_from_llm(text: str):

    if not text:
        raise ValueError("Empty LLM output")

    # Remove code fences
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text)
    text = text.replace("```", "")

    # Extract first JSON block
    match = re.search(r"\{.*", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM output:\n{text}")

    json_text = match.group(0)

    # Try progressively trimming until valid JSON
    for i in range(len(json_text), 0, -1):
        try:
            return json.loads(json_text[:i])
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Failed to parse JSON from model output:\n{text}")

def safe_extract_json(text: str, retries: int = 2):
    """
    Attempts to extract valid JSON from LLM output.
    Handles truncation, markdown, and partial generations.
    """
    if not text:
        raise ValueError("Empty LLM response")

    for _ in range(retries):
        cleaned = text.strip()

        # Remove code fences
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = cleaned.replace("```", "")

        # Extract first JSON object
        match = re.search(r"\{.*", cleaned, re.DOTALL)
        if not match:
            continue

        candidate = match.group(0)

        # Try trimming progressively
        for i in range(len(candidate), 0, -1):
            try:
                return json.loads(candidate[:i])
            except json.JSONDecodeError:
                continue

    raise ValueError("Failed to parse JSON from LLM output")
