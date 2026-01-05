import json
from pathlib import Path

from scripts.ingest_kb_v2 import ingest_one_attack
from scripts.dpa_agent_m_v2 import append_to_kb_json, write_attack_latex

KB_JSON_OUT = "Unified_Attack_Knowledge_Base.json"
ATTACKS_TEX_DIR = "attacks_tex"

NEO_CFG = "configs/neo4j.yaml"
QDRANT_CFG = "configs/qdrant.yaml"

def write_attack_latex(
    attack: dict,
    out_dir: str = "attacks_tex",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    name = attack.get("attack_name", "Unknown_Attack")
    tex_path = Path(out_dir) / f"{name}.tex"

    tex = rf"""
%AttackName: {attack.get("attack_name","")}

%Input:
{attack.get("input","")}

%Output:
{attack.get("output","")}

%Formula:
{attack.get("formula","")}

%Explanation:
{attack.get("explanation","")}
"""

    tex_path.write_text(tex.strip(), encoding="utf-8")

def normalize_attack(entry: dict, idx: int) -> dict:
    """
    Normalize DPATex entry to unified internal schema.
    """

    filename = entry.get("Filename", "").strip()
    attack_name = (
        filename.replace(".tex", "")
        if filename
        else entry.get("AttackName")
        or entry.get("attack_name")
        or f"DPATex_Attack_{idx}"
    )

    return {
        "attack_name": attack_name,

        # 🔑 map CAPITALIZED dataset keys → lowercase internal keys
        "input": entry.get("Input", ""),
        "output": entry.get("Output", ""),
        "formula": entry.get("Formula", ""),
        "explanation": entry.get("Explanation", ""),

        "source_file": filename,
        "source": "DPATex-1.0",
    }



def init_from_json(json_path: str):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    print(f"[+] Initializing KB from {json_path}")
    print(f"[+] Found {len(data)} attacks")

    for i, entry in enumerate(data, 1):
        attack = normalize_attack(entry, i)
        write_attack_latex(attack)       
        append_to_kb_json(attack)
        ingest_one_attack(
            attack=attack,
            neo_cfg_path=NEO_CFG,
            qd_cfg_path=QDRANT_CFG,
        )

        print(f"[{i}/{len(data)}] Ingesting {attack['attack_name']}")

        # 1. Persist canonical artifacts
        append_to_kb_json(attack, KB_JSON_OUT)
        write_attack_latex(attack, ATTACKS_TEX_DIR)

        # 2. Ingest into KB systems
        ingest_one_attack(
            attack=attack,
            neo_cfg_path=NEO_CFG,
            qd_cfg_path=QDRANT_CFG,
        )

    print("[✓] KB initialization complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize KB from JSON dataset")
    parser.add_argument("--input", required=True, help="Path to DPATex JSON file")

    args = parser.parse_args()
    init_from_json(args.input)
