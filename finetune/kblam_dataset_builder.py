from __future__ import annotations
import json
from typing import Dict, Iterable, List

def build_kv_memory_examples(qa_rows: List[Dict]) -> List[Dict]:
    """
    Prototype: convert QA rows to KV-memory style examples.
    In real KBLAM, you construct key-value memory from KB nodes/edges
    and train the model to reconstruct or query it.
    """
    out = []
    for r in qa_rows:
        out.append({
            "query": r["question"],
            "kv_memory": r.get("kv_memory", []),
            "answer": r.get("answer",""),
            "evidence": r.get("evidence_chunks", [])
        })
    return out

def save_jsonl(rows: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
