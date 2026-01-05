"""
DPA Agent v2
------------
Pipeline:
1. Extract attack from source (PDF / text)
2. Ingest attack into KB (Neo4j + Qdrant)
3. Perform reasoning and update attack graph (DOT)

This file does NOT write directly to Neo4j or Qdrant.
That is delegated to ingest_kb_v2.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

# ====== External modules ======
from scripts.ingest_kb_v2 import ingest_one_attack
from kb.graph_utils import update_attack_graph
#from llm_extractor import extract_attack_from_text  # your existing LLM logic
from kb.llm_extractor import extract_attack_from_text
from pypdf import PdfReader


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

KB_CONFIG = "configs/neo4j.yaml"
QDRANT_CONFIG = "configs/qdrant.yaml"

TRACK_GRAPH_PATH = "TrackA_with_defences.dot"

def append_to_kb_json(
    attack: dict,
    kb_path: str = "Unified_Attack_Knowledge_Base.json",
):
    """
    Append extracted attack to the canonical KB JSON.
    """
    from pathlib import Path
    import json

    kb_file = Path(kb_path)

    if kb_file.exists():
        data = json.loads(kb_file.read_text(encoding="utf-8"))
    else:
        data = []

    data.append(attack)

    kb_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_attack_latex(
    attack: dict,
    out_dir: str = "attacks_tex",
):
    """
    Persist the extracted attack as a LaTeX file.
    """
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    name = attack.get("attack_name", "unknown").replace(" ", "_")
    tex_path = Path(out_dir) / f"{name}.tex"

    tex = rf"""
%AttackName: {attack.get("attack_name", "Unknown Attack")}

%Input:{attack.get("input","")}
%Output:{attack.get("output","")}
%Formula:{attack.get("formula","")}
%Explanation:{attack.get("explanation","")}
"""

    tex_path.write_text(tex.strip(), encoding="utf-8")


def process_single_document(doc_path: Path):
    """
    Process one PDF/text document end-to-end.
    """

    # 1. Extract
    attack = extract_attack_from_document(str(doc_path))

    # stamp provenance
    attack["source_file"] = doc_path.name

    # 2. Persist canonical artifacts
    append_to_kb_json(attack)
    write_attack_latex(attack)

    # 3. Ingest into KB
    ingest_attack(attack)




# ------------------------------------------------------------------
# STEP 1 — Extract Attack
# ------------------------------------------------------------------

def extract_attack_from_document(document_path: str) -> dict:
    """
    Runs the LLM-based extractor.
    Returns structured attack JSON.
    """
    print(f"[+] Extracting attack from {document_path}")
    

    reader = PdfReader(document_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    attack = extract_attack_from_text(text)

    if not isinstance(attack, dict):
            raise ValueError("LLM did not return structured attack JSON")

    return attack


# ------------------------------------------------------------------
# STEP 2 — Ingest into Knowledge Base
# ------------------------------------------------------------------

def ingest_attack(attack: dict):
    """
    Adds attack to KB (Neo4j + Qdrant).
    Does NOT update graph.
    """
    print(f"[+] Ingesting attack: {attack.get('attack_name')}")

    ingest_one_attack(
        attack=attack,
        neo_cfg_path=KB_CONFIG,
        qd_cfg_path=QDRANT_CONFIG,
    )


# ------------------------------------------------------------------
# STEP 3 — Graph Reasoning + DOT Update
# ------------------------------------------------------------------

def update_graph(attack: dict):
    """
    Uses existing KB to infer relationships and update DOT graph.
    """
    print("[+] Updating attack graph...")

    update_attack_graph(
        attack=attack,
        graph_path=TRACK_GRAPH_PATH,
    )


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------

from pathlib import Path

def run_pipeline(input_path: str):
    """
    Full DPA pipeline:
      - If input is a file → process once
      - If input is a directory → process all PDFs inside
    """

    input_path = Path(input_path)

    print("\n=== DPA PIPELINE START ===")

    if input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))

        if not pdfs:
            raise RuntimeError(f"No PDFs found in directory: {input_path}")

        print(f"[+] Found {len(pdfs)} PDFs to process")

        for pdf in pdfs:
            print(f"\n--- Processing {pdf.name} ---")
            process_single_document(pdf)

    elif input_path.is_file():
        process_single_document(input_path)

    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print("\n=== DPA PIPELINE COMPLETE ===")



# ------------------------------------------------------------------
# CLI ENTRY
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DPA Agent Pipeline")
    parser.add_argument("--input", required=True, help="Path to PDF or text file")

    args = parser.parse_args()
    run_pipeline(args.input)
