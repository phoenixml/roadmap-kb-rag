from __future__ import annotations
from typing import List

def path_validity_rate(paths: List[str]) -> float:
    # Prototype: if we have any paths, consider valid. In production:
    # parse and verify against Neo4j.
    if paths is None:
        return 0.0
    return 1.0 if len(paths) > 0 else 0.0
