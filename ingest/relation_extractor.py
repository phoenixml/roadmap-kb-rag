from __future__ import annotations
from typing import Dict, List
from llm import LLM, json_from_llm

SYSTEM = "You are an information extraction engine. Extract relations for a knowledge graph."

PROMPT = """Extract relations from the given text chunk.
Return ONLY valid JSON:
{
  "relations":[
    {"src":"...", "rel":"RELATED_TO|MITIGATES|CAUSES|USES|EVALUATED_ON|TARGETS|IMPACTS|REQUIRES|IMPLEMENTS|DETECTS", "dst":"..."}
  ]
}

Rules:
- Use concise entity names present in the chunk.
- Prefer existing entities if possible.

Chunk:
"""


def extract_relations(chunk_text: str, llm: LLM) -> List[Dict]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT + chunk_text},
    ]
    out = llm.chat(messages, temperature=0.0, max_tokens=900)
    data = json_from_llm(out)
    rels = data.get("relations", [])
    clean = []
    for r in rels:
        src = (r.get("src") or "").strip()
        dst = (r.get("dst") or "").strip()
        rel = (r.get("rel") or "RELATED_TO").strip()
        if src and dst:
            clean.append({"src": src, "rel": rel, "dst": dst})
    return clean
