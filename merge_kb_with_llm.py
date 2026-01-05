import json
import os
import re
from typing import Dict
from difflib import get_close_matches

from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

OPENAI_MODEL = "gpt-4o"
client = OpenAI()   # uses OPENAI_API_KEY from environment


# ============================================================
# Utilities
# ============================================================

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# LLM MATCHING
# ============================================================

def infer_graph_node(attack_name: str, description: str, candidates: list):
    """
    Uses OpenAI to semantically match an attack to a known graph node.
    """

    prompt = f"""
You are a security research assistant.

Given the attack name and description, choose the BEST matching attack
from the list of known attacks.

Attack name:
{attack_name}

Description:
{description}

Candidate attacks:
{json.dumps(candidates, indent=2)}

Return ONLY valid JSON in this format:
{{
  "match": "<best matching name>",
  "confidence": 0.0-1.0,
  "reason": "short justification"
}}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    text = response.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except Exception:
        return {"match": None, "confidence": 0.0, "reason": "parse_failed"}


# ============================================================
# MAIN MERGE LOGIC
# ============================================================

def merge_kb(dpathex_path, graph_path, output_path):
    dpatex = load_json(dpathex_path)
    graph = load_json(graph_path)

    graph_nodes = graph.get("nodes", [])
    node_index = {normalize(n.get("label", "")): n for n in graph_nodes}

    merged = []

    for entry in dpatex:
        attack_name = entry.get("attack_name") or entry.get("Filename")
        normalized = normalize(attack_name)

        graph_node = node_index.get(normalized)

        # Use LLM if direct match fails
        if not graph_node:
            result = infer_graph_node(
                attack_name=attack_name,
                description=entry.get("Explanation", ""),
                candidates=[n["label"] for n in graph_nodes],
            )
            match = result.get("match")
            if match:
                graph_node = next(
                    (n for n in graph_nodes if n["label"] == match),
                    None
                )

        merged.append({
            "attack_name": attack_name,
            "input": entry.get("Input"),
            "output": entry.get("Output"),
            "formula": entry.get("Formula"),
            "explanation": entry.get("Explanation"),
            "graph_node": graph_node,
            "citation": {
                "source": entry.get("Filename"),
                "confidence": "auto"
            }
        })

    save_json(merged, output_path)
    print(f"✅ Unified knowledge base written to {output_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    merge_kb(
        dpathex_path="DPATex-1.0.json",
        graph_path="TrackA_with_defences.json",
        output_path="Unified_Attack_Knowledge_Base.json"
    )
