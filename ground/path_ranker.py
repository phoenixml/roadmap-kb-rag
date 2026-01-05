from __future__ import annotations
from typing import List, Tuple
# Prototype: score by path length (shorter better). Replace with semantic scoring.
def rank_paths(paths: List[str], top_k: int = 12) -> List[str]:
    return sorted(paths, key=lambda p: len(p))[:top_k]
