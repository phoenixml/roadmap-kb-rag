from __future__ import annotations
from typing import Dict, List
from llm import LLM, json_from_llm

SYSTEM = """
You are an information extraction system.
Return ONLY valid JSON.
Do NOT include explanations, markdown, or extra text.
"""

PROMPT = """Extract entities from the given text chunk for an adversarial-ML knowledge base.
Return ONLY valid JSON:
{
  "entities":[
    {"name":"...", "type":"Attack|Defense|Technique|Model|Dataset|Metric|Assumption|Constraint|Threat|Unknown"}
  ]
}

Chunk:
"""

def split_into_paragraphs(text: str, max_len: int = 1200):
    paragraphs = []
    buf = ""

    for line in text.split("\n"):
        if len(buf) + len(line) > max_len:
            paragraphs.append(buf.strip())
            buf = line
        else:
            buf += " " + line

    if buf.strip():
        paragraphs.append(buf.strip())

    return paragraphs


def extract_entities(chunk_text: str, llm: LLM) -> List[Dict]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT + chunk_text},
    ]
    out = llm.chat(messages, temperature=0.0, max_tokens=800)
    from llm import safe_extract_json
    data = safe_extract_json(out)
    ents = data.get("entities", [])
    # basic cleanup
    clean = []
    for e in ents:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        clean.append({"name": name, "type": (e.get("type") or "Unknown").strip()})
    return clean
