import json
import re
from pathlib import Path
from difflib import get_close_matches


# -------------------------------
# Utility helpers
# -------------------------------

def normalize(text: str) -> str:
    """Normalize text for matching."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------
# Build ontology index
# -------------------------------

def build_ontology_index(graph_json):
    """
    Builds lookup from graph nodes for fuzzy matching.
    """
    index = {}

    for node in graph_json.get("nodes", []):
        label = node.get("label") or node.get("name")
        if not label:
            continue

        key = normalize(label)
        index[key] = {
            "id": node.get("id"),
            "label": label,
            "type": node.get("type", "Unknown"),
            "relations": node.get("relations", [])
        }

    return index


def match_attack_to_graph(attack_name, ontology_index):
    """
    Fuzzy match attack name to ontology node.
    """
    norm = normalize(attack_name)
    keys = list(ontology_index.keys())

    matches = get_close_matches(norm, keys, n=1, cutoff=0.6)
    if matches:
        return ontology_index[matches[0]]

    return None


# -------------------------------
# Merge datasets
# -------------------------------

def merge_datasets(dpathex_path, graph_path, output_path):
    dpatex = load_json(dpathex_path)
    graph = load_json(graph_path)

    ontology_index = build_ontology_index(graph)

    merged = []

    for entry in dpatex:
        attack_name = entry.get("Attack") or entry.get("attack") or entry.get("Filename", "")

        graph_node = match_attack_to_graph(attack_name, ontology_index)

        merged.append({
            "attack_name": attack_name,
            "input": entry.get("Input"),
            "output": entry.get("Output"),
            "formula": entry.get("Formula"),
            "explanation": entry.get("Explanation"),
            "graph_node": graph_node,
            "source_file": entry.get("Filename")
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"✅ Merged dataset written to {output_path}")
    print(f"✔ Matched {sum(1 for x in merged if x['graph_node'])} / {len(merged)} entries")


# -------------------------------
# ENTRY POINT
# -------------------------------

if __name__ == "__main__":
    merge_datasets(
        dpathex_path="DPATex-1.0.json",
        graph_path="TrackA_with_defences.json",
        output_path="Unified_Attack_Knowledge_Base.json"
    )
