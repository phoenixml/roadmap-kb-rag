# kb/llm_extractor.py

import os
import json
from openai import OpenAI

MODEL = "gpt-4.1"
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def safe_json_loads(text: str) -> dict:
    text = (text or "").strip()

    # strip markdown fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return JSON:\n{text}")

    return json.loads(text[start:end + 1])

def extract_attack_from_text(text: str) -> dict:
    """
    Extract a single adversarial / poisoning attack from raw text.
    Returns structured JSON.
    """

    prompt = f"""
You are an adversarial ML researcher.

Extract exactly ONE attack.

Return ONLY valid JSON with fields:
- attack_name
- input
- output
- formula
- explanation
- graph_node: {{ "type": "<model family>" }}

Text:
{text}
"""

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return safe_json_loads(resp.choices[0].message.content)

