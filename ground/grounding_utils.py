from __future__ import annotations
from typing import List
def stringify_neo4j_paths(rows) -> List[str]:
    out = []
    for r in rows:
        out.append(str(r["p"]))
    return out
