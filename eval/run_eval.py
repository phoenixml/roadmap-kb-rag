from __future__ import annotations
import json
from typing import Dict, List
from eval.metrics.answer_metrics import exact_match, f1_token
from eval.metrics.evidence_metrics import precision_at_k, recall_at_k
from eval.metrics.graph_metrics import path_validity_rate

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def evaluate(pred_rows: List[Dict], gold_rows: List[Dict]) -> Dict:
    gold_map = {r["id"]: r for r in gold_rows}
    agg = {"em":0.0,"f1":0.0,"p@5":0.0,"r@5":0.0,"path_valid":0.0,"n":0}
    for pr in pred_rows:
        gid = pr["id"]
        gr = gold_map.get(gid)
        if not gr:
            continue
        pred_answer = pr.get("answer","")
        gold_answer = gr.get("answer","")
        pred_chunks = pr.get("evidence_chunks", [])
        gold_chunks = gr.get("evidence_chunks", [])
        paths = pr.get("paths", [])
        agg["em"] += exact_match(pred_answer, gold_answer)
        agg["f1"] += f1_token(pred_answer, gold_answer)
        agg["p@5"] += precision_at_k(pred_chunks, gold_chunks, k=5)
        agg["r@5"] += recall_at_k(pred_chunks, gold_chunks, k=5)
        agg["path_valid"] += path_validity_rate(paths)
        agg["n"] += 1
    if agg["n"] == 0:
        return agg
    for k in ["em","f1","p@5","r@5","path_valid"]:
        agg[k] /= agg["n"]
    return agg
