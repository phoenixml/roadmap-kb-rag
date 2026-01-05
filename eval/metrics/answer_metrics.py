from __future__ import annotations
import re

def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0

def f1_token(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_set = set(p); g_set = set(g)
    inter = len(p_set & g_set)
    prec = inter / max(len(p_set), 1)
    rec = inter / max(len(g_set), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)
