#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from pypdf import PdfReader
from openai import OpenAI
from scripts.ingest_kb_v2 import ingest_one_attack


# =========================================================
# CONFIG
# =========================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1"

PDF_INPUT_DIR = "pdfs_to_process"
DOT_INPUT = "TrackA.dot"
DOT_OUTPUT = "TrackA_with_defences.dot"
LATEX_DIR = "attacks_tex"

os.makedirs(LATEX_DIR, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================================
# MODEL FAMILIES (CANONICAL)
# =========================================================

NODE_ID_MAP = {
    "LLM": "n15",
    "CNN": "n13",
    "RNN": "n14",
    "GNN": "n17",
    "SVM": "n16",
    "RF": "n19",
    "DT": "n705",
    "UNKNOWN": "n2000"
}

ALLOWED_FAMILIES = [
    "LLM", "CNN", "RNN", "GNN",
    "SVM", "RF", "DT",
    "Federated Learning",
    "Graph-NNs",
    "Generative Models",
    "Recommender Systems",
    "Semi-Supervised",
    "Multimodal Models",
    "Unknown"
]

# =========================================================
# UTILS
# =========================================================

def normalize(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def normalize_perturbation_name(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["gradient", "pgd", "fgsm", "bfgs", "adam"]):
        return "Gradient-based"
    if any(k in t for k in ["saliency", "jacobian"]):
        return "Saliency-based"
    if any(k in t for k in ["poison", "poisoning"]):
        return "Data Poisoning"
    if any(k in t for k in ["prompt", "instruction"]):
        return "Prompt Injection"
    if any(k in t for k in ["optimization", "search"]):
        return "Optimization-based"
    return "Adversarial Perturbation"


# =========================================================
# LLM CLASSIFIER (CORE FIX)
# =========================================================

def llm_classify_victim_family(attack: dict, paper_hint: str = "") -> str:
    """
    Use LLM to classify the victim model family.
    Returns one canonical family name.
    """
    prompt = f"""
You are classifying the target model family of an adversarial ML attack.

Choose exactly ONE label from:
{", ".join(ALLOWED_FAMILIES)}

Return ONLY the label.

Attack name: {attack.get("attack_name","")}
Explanation: {attack.get("explanation","")}
Paper hint: {paper_hint[:1200]}
"""

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    label = resp.choices[0].message.content.strip()
    return label if label in ALLOWED_FAMILIES else "Unknown"


def family_to_node(family: str) -> str:
    return NODE_ID_MAP.get(family, NODE_ID_MAP["UNKNOWN"])


# =========================================================
# DOT GRAPH INJECTION
# =========================================================

def append_attack(dot_text, attack, paper_text=""):
    attack_name = normalize(attack["attack_name"])
    perturb = normalize_perturbation_name(attack.get("perturbation_search", ""))

    # Infer parent via LLM classification
    parent = family_to_node(
        llm_classify_victim_family(attack, paper_text)
    )

    attack_id = f"A_{abs(hash(attack_name)) % 100000}"
    pert_id = f"P_{abs(hash(attack_name + perturb)) % 100000}"

    # Prevent duplicates
    if attack_id in dot_text:
        return dot_text

    block = f"""
// --- Auto-added attack: {attack_name} ---
"{parent}" -> "{attack_id}" [label="PC={attack_name}"];
"{attack_id}" [label="Visibility=High", shape=plaintext];
"{attack_id}" -> "{pert_id}" [label="Perturbation Search"];
"{pert_id}" [label="{perturb}", shape=plaintext];
"""

    # Remove final closing brace safely
    dot_text = dot_text.rstrip()

    if dot_text.endswith("}"):
        dot_text = dot_text[:-1].rstrip()

    # Append block and re-close graph
    dot_text = f"{dot_text}\n{block}\n}}"

    return dot_text


# =========================================================
# EXTRACT ATTACK FROM PAPER
# =========================================================

def extract_attack(text: str) -> dict:
    prompt = f"""
Return ONLY JSON:

{{
  "attack_name": "",
  "input": "",
  "output": "",
  "formula": "",
  "explanation": "",
  "perturbation_search": ""
}}

Paper:
{text}
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = resp.choices[0].message.content
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    if not os.path.isdir(PDF_INPUT_DIR):
        print(f"[ERROR] Folder '{PDF_INPUT_DIR}' not found.")
        return

    pdfs = [p for p in os.listdir(PDF_INPUT_DIR) if p.endswith(".pdf")]
    print(f"[+] Found {len(pdfs)} PDF(s)")

    # 🔥 IMPORTANT FIX: load existing graph if already created
    if os.path.exists(DOT_OUTPUT):
        with open(DOT_OUTPUT, "r") as f:
            dot_text = f.read()
        print("[+] Loaded existing graph for appending")
    else:
        with open(DOT_INPUT, "r") as f:
            dot_text = f.read()
        print("[+] Loaded base graph")

    for pdf in pdfs:
        print(f"[+] Processing: {pdf}")
        pdf_path = os.path.join(PDF_INPUT_DIR, pdf)

        text = "\n".join(p.extract_text() or "" for p in PdfReader(pdf_path).pages)

        attack = extract_attack(text)
        # 1) stamp provenance
        attack["source_file"] = pdf  # or the generated .tex name
        attack["graph_node"] = {"type": llm_classify_victim_family(attack, text)}  # you already have this

        # 2) upsert into Neo4j + Qdrant (single-attack ingest)
        ingest_one_attack(
            attack,
            neo_cfg_path="configs/neo4j.yaml",
            qd_cfg_path="configs/qdrant.yaml",
            chunk_mode="section",
        )

        # Save LaTeX
        tex_path = os.path.join(
            LATEX_DIR, f"{attack['attack_name'].replace(' ', '_')}.tex"
        )
        with open(tex_path, "w") as f:
            f.write(
                f"%Input\n{attack['input']}\n\n"
                f"%Output\n{attack['output']}\n\n"
                f"%Formula\n{attack['formula']}\n\n"
                f"%Explanation\n{attack.get('explanation','')}\n"
            )

        # Append to graph (does NOT overwrite)
        dot_text = append_attack(dot_text, attack, text)

    # 🔥 Persist updated graph
    with open(DOT_OUTPUT, "w") as f:
        f.write(dot_text)

    print("✅ All attacks appended successfully.")

if __name__ == "__main__":
    main()
