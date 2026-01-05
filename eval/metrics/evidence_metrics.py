from __future__ import annotations
from typing import Dict, List, Set

def precision_at_k(pred_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
    pred_top = [p for p in pred_ids[:k] if p]
    gold = set([g for g in gold_ids if g])
    if not pred_top:
        return 0.0
    hit = sum(1 for p in pred_top if p in gold)
    return hit / len(pred_top)

def recall_at_k(pred_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
    pred_top = set([p for p in pred_ids[:k] if p])
    gold = set([g for g in gold_ids if g])
    if not gold:
        return 1.0
    hit = len(pred_top & gold)
    return hit / len(gold)
