from __future__ import annotations
from typing import Dict, List, Tuple
from rapidfuzz import process, fuzz

def canonicalize_entities(names: List[str], existing: List[str], score_cutoff: int = 92) -> Dict[str, str]:
    """
    Map extracted entity names to existing entity names if a close match exists.
    Returns mapping original->canonical.
    """
    mapping = {}
    for n in names:
        if not existing:
            mapping[n] = n
            continue
        match = process.extractOne(n, existing, scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff)
        mapping[n] = match[0] if match else n
    return mapping

def normalize_relation(rel: str) -> str:
    rel = (rel or "").strip()
    rel = rel.replace(" ", "_").replace("-", "_")
    return "".join([c for c in rel.upper() if c.isalnum() or c == "_"]) or "RELATED_TO"

from pathlib import Path

from pathlib import Path

def update_attack_graph(
    attack: dict,
    graph_path,
):
    """
    Update (or create) TrackA DOT graph with a newly ingested attack.
    The KB must already be updated.
    """

    graph_path = Path(graph_path)

    attack_name = attack.get("attack_name") or "UnknownAttack"
    attack_node = attack_name.replace(" ", "_").replace("-", "_")

    # --- CREATE GRAPH IF IT DOES NOT EXIST ---
    if not graph_path.exists():
        graph_path.write_text(
            "digraph TrackA {\n"
            "    rankdir=LR;\n"
            "}\n",
            encoding="utf-8",
        )

    dot = graph_path.read_text(encoding="utf-8")

    # --- ADD NODE IF NOT PRESENT ---
    node_stmt = f'    "{attack_node}" [label="{attack_name}"];\n'

    if node_stmt in dot:
        # Node already exists → do nothing
        return

    # --- INSERT NODE BEFORE CLOSING BRACE ---
    updated = dot.rstrip().rstrip("}") + "\n" + node_stmt + "}\n"
    graph_path.write_text(updated, encoding="utf-8")
