"""
Defence Agent (v1)
------------------
Goal:
- Given an attack name, retrieve evidence from Qdrant (RAG)
- (Optional) Ground with Neo4j k-hop subgraph
- Run Loop B: Retrieve -> Generate -> Critique -> Refine
- Output:
  - defences/<attack>.defence.json (includes full trace)
  - optionally update TrackA_with_defences.dot (derived artifact)

Does NOT modify your ingestion pipeline or dpa_agent_m_v2.py.
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from qdrant_client import models
from scripts.defence_confidence import compute_defence_confidence



# --- Import your existing clients (adjust module paths if needed) ---
# Expect these to exist in your repo as kb/qdrant_client.py and kb/neo4j_client.py
from kb.qdrant_client import QdrantConfig, VectorStore
from kb.neo4j_client import Neo4jConfig, Neo4jClient


# =========================
# Config
# =========================

DEFAULT_QDRANT_YAML = "configs/qdrant.yaml"
DEFAULT_NEO4J_YAML = "configs/neo4j.yaml"

DEFAULT_DOT_PATH = "TrackA_with_defences.dot"
DEFAULT_OUT_DIR = "defences"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NODE_ID_MAP = {
    "LLM": "n15",
    "CNN": "n13",
    "RNN": "n14",
    "GNN": "n17",
    "SVM": "n16",
    "RF": "n19",
    "DT": "n705",
    "UNKNOWN": "n2000",
}


# =========================
# DOT helper functions (local, authoritative for defence_agent)
# =========================
def canonical_family(fam: str) -> str:
    f = (fam or "").strip().upper()

    # normalize common variants
    if "LLM" in f or "TRANSFORM" in f:
        return "LLM"
    if f == "CNN":
        return "CNN"
    if f == "RNN":
        return "RNN"
    if f == "GNN" or "GRAPH" in f:
        return "GNN"
    if f == "SVM":
        return "SVM"
    if f == "RF" or "RANDOMFOREST" in f or "RANDOM_FOREST" in f:
        return "RF"
    if f == "DT" or "DECISIONTREE" in f or "DECISION_TREE" in f:
        return "DT"

    return "UNKNOWN"

def normalize(text: str) -> str:
    if not text:
        return "Unknown"
    return re.sub(r"\s+", "_", text.strip())


def normalize_perturbation_name(text: str) -> str:
    return normalize(text) if text else "Unknown"


def family_to_node(family: str) -> str:
    """
    Map model family to canonical DOT node ID.
    Reuses NODE_ID_MAP defined in this file.
    """
    family = (family or "").strip()
    return NODE_ID_MAP.get(family, NODE_ID_MAP["UNKNOWN"])


def llm_classify_victim_family(attack: dict, paper_text: str = "") -> str:
    return canonical_family(attack.get("family"))
# =========================
# DOT helpers (LOCAL, authoritative)
# =========================

import re

def normalize(text: str) -> str:
    if not text:
        return "Unknown"
    return re.sub(r"\s+", "_", text.strip())


def extract_attack_semantics(attack_obj: dict):
    """
    Normalize attack semantic fields across DPATeX / Qdrant / Defence Agent.
    """
    # Default empty
    input_text = ""
    output_text = ""
    explanation_text = ""
    formula_latex = ""

    # --- Case 1: nested latex / content block ---
    for key in ("latex", "content", "fields"):
        if key in attack_obj and isinstance(attack_obj[key], dict):
            block = attack_obj[key]
            input_text = block.get("input", input_text)
            output_text = block.get("output", output_text)
            explanation_text = block.get("explanation", block.get("rationale", explanation_text))
            formula_latex = block.get("formula", block.get("equation", formula_latex))

    # --- Case 2: flat keys ---
    input_text = attack_obj.get("input_text", input_text)
    output_text = attack_obj.get("output_text", output_text)
    explanation_text = attack_obj.get("explanation", attack_obj.get("rationale", explanation_text))
    formula_latex = attack_obj.get("formula", attack_obj.get("equation", formula_latex))

    return input_text, output_text, explanation_text, formula_latex

def extract_semantics_from_text(text: str) -> dict:
    """
    Best-effort extraction of Input / Output / Formula / Explanation
    from retrieved paper text or attack description.
    """
    if not text:
        return {
            "input": "",
            "output": "",
            "formula": "",
            "explanation": "",
        }

    # VERY conservative heuristics (non-destructive)
    input_text = ""
    output_text = ""
    formula_text = ""
    explanation_text = text.strip()

    # Try to isolate formulas (LaTeX-like)
    lines = text.splitlines()
    formula_lines = [l for l in lines if "$" in l or "\\(" in l or "\\[" in l]

    if formula_lines:
        formula_text = "\n".join(formula_lines[:5])

    # Heuristic input/output hints
    for l in lines:
        if "input" in l.lower() and not input_text:
            input_text = l.strip()
        if "output" in l.lower() and not output_text:
            output_text = l.strip()

    return {
        "input": input_text,
        "output": output_text,
        "formula": formula_text,
        "explanation": explanation_text,
    }


def normalize_perturbation_name(text: str) -> str:
    t = (text or "").lower()
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


def family_to_node(family: str) -> str:
    return NODE_ID_MAP.get(family, NODE_ID_MAP["UNKNOWN"])


def llm_classify_victim_family(attack: dict, paper_text: str = "") -> str:
    """
    Defence agent MUST NOT re-classify.
    It trusts ingestion-time classification.
    """
    return attack.get("family", "UNKNOWN")


def append_attack(
    dot_text: str,
    attack: dict,
    paper_text: str = "",
    defence_name: str | None = None,
):
    attack_name = normalize(attack["attack_name"])

    # Parent family node (unchanged semantics)
    parent = family_to_node(
        llm_classify_victim_family(attack, paper_text)
    )

    attack_id = f"A_{abs(hash(attack_name)) % 100000}"
    pert_id = f"P_{abs(hash(attack_name)) % 100000}"

    # Do not duplicate attacks
    if attack_id in dot_text:
        return dot_text

    # ---- Perturbation search label (THIS WAS MISSING) ----
    pert_search = attack.get("perturbation_search", "").strip()
    edge_label = (
        f"Perturbation Search={pert_search}"
        if pert_search
        else "Perturbation Search"
    )

    # ---- Leaf label ----
    # If defence exists → defence IS the perturbation leaf
    # Else → fallback to normalized perturbation category
    if defence_name:
        leaf_label = defence_name
    else:
        leaf_label = normalize_perturbation_name(pert_search)

    block = f"""
// --- Auto-added attack: {attack_name} ---
"{parent}" -> "{attack_id}" [label="PC={attack_name}"];
"{attack_id}" [label="Visibility=High", shape=plaintext];
"{attack_id}" -> "{pert_id}" [label="{edge_label}"];
"{pert_id}" [label="{leaf_label}", shape=plaintext];
"""

    # Remove final closing brace safely
    dot_text = dot_text.rstrip()
    if dot_text.endswith("}"):
        dot_text = dot_text[:-1].rstrip()

    # Append and re-close
    dot_text = f"{dot_text}\n{block}\n}}"

    return dot_text

# =========================
# Utils
# =========================

def safe_json_loads(text: str) -> dict:
    """
    Extract JSON object from an LLM output safely.
    """
    text = (text or "").strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    # Extract first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return JSON.\nRaw:\n{text}")

    return json.loads(text[start:end + 1])


def slugify(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s or "Unknown"


def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def qdrant_search_compat(vs, query_text: str, limit: int):
    """
    Vector-only Qdrant retrieval for >=1.9.
    FastEmbed is fully bypassed.
    """
    # unwrap real client
    client = vs.client.client if hasattr(vs.client, "client") else vs.client

    # manual embedding (you already have this working)
    vector = vs.embed(query_text)

    # ---- MODERN QDRANT (>=1.9) ----
    if hasattr(client, "query_points"):
        return client.query_points(
            collection_name=vs.cfg.collection,
            query=vector,   # ✅ THIS IS THE KEY FIX
            limit=limit,
        )

    # ---- MID VERSION ----
    if hasattr(client, "search_points"):
        return client.search_points(
            collection_name=vs.cfg.collection,
            vector=vector,
            limit=limit,
        )

    # ---- LEGACY ----
    if hasattr(client, "search"):
        return client.search(
            collection_name=vs.cfg.collection,
            query_vector=vector,
            limit=limit,
        )

    raise RuntimeError(
        "Unsupported Qdrant client API.\n"
        f"Client type: {type(client)}"
    )


# =========================
# DOT updater (self-contained)
# =========================

def ensure_dot_exists(dot_path: Path):
    if not dot_path.exists():
        dot_path.write_text(
            "digraph TrackA_with_defences {\n"
            "    rankdir=LR;\n"
            "}\n",
            encoding="utf-8",
        )


def dot_has_node(dot: str, node_id: str) -> bool:
    return re.search(rf'^\s*"{re.escape(node_id)}"\s*\[', dot, flags=re.MULTILINE) is not None


def dot_has_edge(dot: str, src: str, dst: str, label: Optional[str] = None) -> bool:
    if label:
        return re.search(
            rf'^\s*"{re.escape(src)}"\s*->\s*"{re.escape(dst)}"\s*\[.*label="{re.escape(label)}".*\]\s*;',
            dot,
            flags=re.MULTILINE,
        ) is not None
    return re.search(
        rf'^\s*"{re.escape(src)}"\s*->\s*"{re.escape(dst)}"\s*\[',
        dot,
        flags=re.MULTILINE,
    ) is not None


def dot_insert_before_close(dot: str, insertion: str) -> str:
    dot = dot.rstrip()
    if not dot.endswith("}"):
        # fallback: if file is corrupted, wrap it
        dot = "digraph TrackA_with_defences {\n" + dot + "\n}\n"
    return dot[:-1].rstrip() + "\n" + insertion.rstrip() + "\n}\n"





# =========================
# Retrieval (Qdrant)
# =========================

def make_vectorstore_from_yaml(qdrant_yaml_path: str) -> VectorStore:
    cfg_raw = load_yaml(qdrant_yaml_path)
    raw_url = cfg_raw["url"]
    if isinstance(raw_url, str) and raw_url.startswith("${") and raw_url.endswith("}"):
        env_name = raw_url[2:-1]
        url = os.getenv(env_name)
        if not url:
            raise RuntimeError(f"Environment variable {env_name} is not set")
    else:
        url = raw_url
    cfg = QdrantConfig(
    url=url,   
    api_key=cfg_raw.get("api_key"),
    collection=cfg_raw.get("collection", "docs"),
    vector_size=int(cfg_raw.get("vector_size", 384)),
    distance=cfg_raw.get("distance", "Cosine"),
)
    return VectorStore(cfg)

def qdrant_retrieve_evidence(vs, query: str, attack_hint=None, k: int = 8):
    q = query if not attack_hint else f"{query}. {attack_hint}"

    res = qdrant_search_compat(
        vs=vs,
        query_text=q,
        limit=k,
    )

    evidence = []
    for r in res:
        evidence.append({
            "score": getattr(r, "score", None),
            "payload": getattr(r, "payload", {}) or {},
        })

    return evidence

def pick_attack_candidate_from_hits(attack_name: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Try to reconstruct an 'attack object' from the best hit payload.
    Falls back to attack_name only.
    """
    for h in hits:
        p = h.get("payload") or {}
        # If your payload contains these fields (you enabled them), use them
        if any(p.get(k) for k in ("formula", "explanation", "input", "output")):
            return {
                "attack_name": p.get("attack_name") or attack_name,
                "input": p.get("input", ""),
                "output": p.get("output", ""),
                "formula": p.get("formula", ""),
                "explanation": p.get("explanation", ""),
                "family": p.get("family", "Unknown"),
                "attack_id": p.get("attack_id", ""),
                "source_file": p.get("source_file", ""),
            }
    return {"attack_name": attack_name, "family": "Unknown"}


# =========================
# Neo4j grounding (optional Loop C)
# =========================

def _resolve_env(val: str) -> str:
    if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
        env_name = val[2:-1]
        resolved = os.getenv(env_name)
        if not resolved:
            raise RuntimeError(f"Environment variable {env_name} is not set")
        return resolved
    return val


def make_neo4j_client_from_yaml(neo4j_yaml_path: str) -> Neo4jClient:
    cfg_raw = load_yaml(neo4j_yaml_path)

    uri = _resolve_env(cfg_raw["uri"])
    user = _resolve_env(cfg_raw["user"])
    password = _resolve_env(cfg_raw["password"])

    cfg = Neo4jConfig(
        uri=uri,
        user=user,
        password=password,
        database=cfg_raw.get("database", "neo4j"),
    )
    return Neo4jClient(cfg)



def neo4j_khop_grounding(
    neo: Neo4jClient,
    seed_names: List[str],
    hops: int = 2,
    limit: int = 200,
) -> Dict[str, Any]:
    """
    Uses your Neo4jClient.get_k_hop_subgraph which returns paths.
    We'll convert it into a lightweight "triples-like" summary for the LLM.
    """
    try:
        records = neo.get_k_hop_subgraph(seed_entities=seed_names, hops=hops, limit=limit)
    except Exception as e:
        return {"enabled": True, "error": str(e), "summary": ""}

    # Best-effort summarization: stringify records safely
    # (Exact structure depends on neo4j driver record types)
    lines = []
    for rec in records[: min(len(records), 50)]:
        lines.append(str(rec))

    summary = "\n".join(lines)
    return {"enabled": True, "seed_names": seed_names, "hops": hops, "limit": limit, "summary": summary}


# =========================
# Loop B: Generate / Critique / Refine
# =========================

def llm_generate_defences(attack: Dict[str, Any], evidence: List[Dict[str, Any]], neo_summary: Optional[str]) -> Dict[str, Any]:
    evidence_lines = []
    for i, h in enumerate(evidence, 1):
        p = h.get("payload") or {}
        # Short snippets: prefer explanation/formula if present
        snippet = (p.get("explanation") or p.get("formula") or p.get("input") or "")[:500]
        evidence_lines.append(f"[E{i}] attack_name={p.get('attack_name','')} family={p.get('family','')}\n{snippet}")

    neo_block = f"\nNeo4j subgraph evidence:\n{neo_summary}\n" if neo_summary else ""

    prompt = f"""
You are an adversarial ML defender.

Task:
Given an attack, propose 1-3 concrete defence methods.

STRICT OUTPUT:
Return ONLY valid JSON.

Attack:
{json.dumps(attack, ensure_ascii=False, indent=2)}

Retrieved evidence (cite as E1, E2, ...):
{chr(10).join(evidence_lines)}
{neo_block}

JSON schema:
{{
  "defences": [
    {{
      "name": "...",
      "category": "training|data_sanitization|detection|certification|inference_time|other",
      "mechanism": "how it mitigates the attack mechanism",
      "applicability": "high|medium|low",
      "evidence": ["E1", "E3"],
      "limitations": ["..."]
    }}
  ],
  "recommended": "name of best defence",
  "rationale": "1-3 sentences grounded in evidence IDs only"
}}

Rules:
- Every defence MUST cite at least one evidence ID.
- Do NOT invent papers or citations; only cite E1..En.
- If evidence is insufficient, return a defence with applicability="low" and limitations explaining missing evidence.
"""

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return safe_json_loads(resp.choices[0].message.content)


def llm_critic(attack: Dict[str, Any], draft: Dict[str, Any], evidence: List[Dict[str, Any]], neo_summary: Optional[str]) -> Dict[str, Any]:
    evidence_ids = [f"E{i}" for i in range(1, len(evidence) + 1)]
    neo_block = f"\nNeo4j subgraph evidence:\n{neo_summary}\n" if neo_summary else ""

    prompt = f"""
You are a strict security reviewer ("Critic") for RAG-grounded reasoning.

Check the draft defence proposal for:
- Unsupported claims (no evidence ID)
- Evidence mismatch (defence cites IDs that don't exist)
- Missing key mitigations (obvious gaps)
- Conflicts with Neo4j subgraph evidence (if provided)
- Overconfidence

Return ONLY JSON with schema:
{{
  "issues": [
    {{
      "type": "unsupported|bad_citation|missing_evidence|conflict|overconfident|other",
      "detail": "...",
      "fix": "concrete instruction to fix"
    }}
  ],
  "overall": "pass|revise",
  "confidence_adjustment": "increase|decrease|no_change"
}}

Attack:
{json.dumps(attack, ensure_ascii=False, indent=2)}

Valid evidence IDs: {evidence_ids}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}
{neo_block}
"""

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return safe_json_loads(resp.choices[0].message.content)


def llm_refine(attack: Dict[str, Any], draft: Dict[str, Any], critic: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are refining a defence proposal based on Critic feedback.

Return ONLY valid JSON with the same schema as the draft:
{{
  "defences": [...],
  "recommended": "...",
  "rationale": "..."
}}

Attack:
{json.dumps(attack, ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Critic feedback:
{json.dumps(critic, ensure_ascii=False, indent=2)}

Rules:
- Fix every Critic issue.
- Ensure every defence cites at least one evidence ID.
- Do not add external citations.
"""

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return safe_json_loads(resp.choices[0].message.content)


# =========================
# Main pipeline
# =========================
from kb.defence_kb import DefenceKB
from scripts.defence_confidence import compute_defence_confidence
from scripts.dot_bootstrap import ensure_defence_dot_inherits_attack_dot

def update_dot_with_defence(
    dot_path: str,
    attack_obj: dict,
    defence_name: str,
    applicability: str,      # kept for Neo4j, ignored for DOT
    evidence_count: int,     # kept for Neo4j, ignored for DOT
    paper_text: str = "",
):
    dot_path = Path(dot_path)
    dot_text = dot_path.read_text(encoding="utf-8")

    # Rebuild / append attack with defence folded into perturbation leaf
    dot_text = append_attack(
        dot_text=dot_text,
        attack=attack_obj,
        paper_text=paper_text,
        defence_name=defence_name,
    )

    dot_path.write_text(dot_text, encoding="utf-8")


def run_defence_pipeline(
    attack_name: str,
    qdrant_yaml: str,
    neo4j_yaml: Optional[str],
    enable_neo4j: bool,
    hops: int,
    out_dir: str,
    update_dot: bool,
    dot_path: str,
    
):
    # --------------------------------------------------
    # Bootstrap DOT: TrackA → TrackA_with_defences
    # --------------------------------------------------
    if update_dot:
        ensure_defence_dot_inherits_attack_dot(
            attack_dot="TrackA.dot",
            defence_dot=dot_path,
        )

    trace: Dict[str, Any] = {
        "attack_name_input": attack_name,
        "started_at": time.time(),
        "model": MODEL,
        "steps": {},
    }

    # --- Retrieve (Qdrant) ---
    vs = make_vectorstore_from_yaml(qdrant_yaml)
    evidence = qdrant_retrieve_evidence(vs, attack_name, attack_hint=None, k=8)
    trace["steps"]["retrieve_qdrant"] = {
        "k": 8,
        "hits": [
            {"score": h.get("score"), "payload_keys": sorted(list((h.get("payload") or {}).keys()))}
            for h in evidence
        ],
    }

    # Build an attack object from best evidence payload if possible
    attack_obj = pick_attack_candidate_from_hits(attack_name, evidence)
    trace["attack_object"] = attack_obj

    # --- Optional Neo4j grounding (Loop C) ---
    neo_summary = None
    if enable_neo4j:
        if not neo4j_yaml:
            raise ValueError("Neo4j enabled but neo4j_yaml not provided")
        neo = make_neo4j_client_from_yaml(neo4j_yaml)
        seeds = [attack_obj.get("attack_name") or attack_name]
        # optionally add similar seed strings by splitting words
        seeds = list({s for s in seeds if s})
        grounding = neo4j_khop_grounding(neo, seed_names=seeds, hops=hops, limit=200)
        neo_summary = grounding.get("summary") or ""
        trace["steps"]["neo4j_grounding"] = grounding
        try:
            neo.close()
        except Exception:
            pass

    # --- Generate ---
    draft = llm_generate_defences(attack_obj, evidence, neo_summary)
    trace["steps"]["generate"] = draft

    # --- Critique ---
    # --- Critique ---
    critic = llm_critic(attack_obj, draft, evidence, neo_summary)
    trace["steps"]["critic"] = critic

    # --- Refine if needed ---
    if critic.get("overall") == "revise":
        refined = llm_refine(attack_obj, draft, critic)
    else:
        refined = draft
    trace["steps"]["refine"] = refined

    # ==========================================================
    # NEW: Neo4j Defence Memory (Attack -> Defence ONLY)
    # ==========================================================
    if enable_neo4j:
        from kb.defence_kb import DefenceKB
        from scripts.defence_confidence import compute_defence_confidence

        neo_cfg = load_yaml(neo4j_yaml)
        dkb = DefenceKB(
            uri=neo_cfg["uri"],
            user=neo_cfg["user"],
            password=neo_cfg["password"],
            database=neo_cfg.get("database", "neo4j"),
        )
        

        attack_key = attack_obj.get("attack_name") or attack_name

        # Prior defences (Loop-C memory, NOT taxonomy)
        prior_defences = dkb.get_defences(attack_key)

        # Resolve recommended defence object
        recommended = refined.get("recommended")
        defence_obj = None
        for d in refined.get("defences", []) or []:
            if d.get("name") == recommended:
                defence_obj = d
                break

        # Fallback: first defence
        if not defence_obj and refined.get("defences"):
            defence_obj = refined["defences"][0]
            recommended = defence_obj.get("name")

        if defence_obj and recommended:
            confidence = compute_defence_confidence(
                evidence_count=len(defence_obj.get("evidence", []) or []),
                applicability=defence_obj.get("applicability", "medium"),
                critic_adjustment=critic.get("confidence_adjustment", "no_change"),
                has_prior=bool(prior_defences),
            )

            neo_client = make_neo4j_client_from_yaml(neo4j_yaml)
            attack_id = (
                attack_obj.get("attack_id")
                or attack_obj.get("id")
                or f"A_{abs(hash(attack_obj['attack_name'])) % 100000}"
            )

            semantic_source_text = (
                attack_obj.get("text")
                or attack_obj.get("content")
                or attack_obj.get("description")
            )

            semantics = extract_semantics_from_text(semantic_source_text)
            neo = make_neo4j_client_from_yaml(neo4j_yaml)

            # ================= DEBUG: attack semantics availability =================
            print("\n========== DEBUG: defence_agent semantic inputs ==========")
            print("DEBUG attack_obj keys:", list(attack_obj.keys()))
            #print("DEBUG paper_text length:", len(paper_text) if paper_text else "None")

            for k in ["input", "output", "formula", "explanation", "text", "content", "description"]:
                print(f"DEBUG attack_obj[{k!r}]:", repr(attack_obj.get(k)))

            print("=========================================================\n")
            # =======================================================================




            neo.upsert_attack_semantics(
                attack_id=attack_id,
                name=attack_obj.get("attack_name"),
                family=attack_obj.get("family"),
                input_text=semantics["input"],
                output_text=semantics["output"],
                explanation_text=semantics["explanation"],
                formula_latex=semantics["formula"],
                source_file="defence_agent:derived",
            )

            neo.close()
            dkb.upsert_attack_and_defence(
                attack=attack_obj,
                defence_name=recommended,
                defence_category=defence_obj.get("category", "other"),
                confidence=confidence,
                source="defence_agent:v1",
            )

            conflict = dkb.detect_conflict(attack_key)
            if conflict:
                trace["steps"]["neo4j_conflict"] = conflict

        dkb.close()

    # --- Final output object ---
    final = {
        "attack": attack_obj,
        "defence_plan": refined,
        "trace": trace,
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_name = slugify(attack_obj.get("attack_name") or attack_name)
    out_path = Path(out_dir) / f"{out_name}.defence.json"
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- DOT update (optional derived artifact) ---
    if update_dot:
        rec_name = refined.get("recommended") or ""
        # find the recommended defence object to estimate evidence_count/applicability
        defence_obj = None
        for d in refined.get("defences", []) or []:
            if d.get("name") == rec_name:
                defence_obj = d
                break
        if not defence_obj and (refined.get("defences") or []):
            defence_obj = refined["defences"][0]
            rec_name = defence_obj.get("name") or rec_name

        if rec_name:
            ev_count = len(defence_obj.get("evidence", []) or []) if defence_obj else 0
            app = defence_obj.get("applicability", "medium") if defence_obj else "medium"
            update_dot_with_defence(
                dot_path=dot_path,
                attack_obj=attack_obj,
                defence_name=rec_name,
                applicability=app,
                evidence_count=ev_count,
            )

    trace["finished_at"] = time.time()

    print(f"[✓] Defence plan saved: {out_path}")
    if update_dot:
        print(f"[✓] DOT updated: {dot_path}")
    else:
        print("[i] DOT not updated (use --update_dot to write derived artifact)")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Defence Agent (Self-RAG + optional Neo4j grounding)")
    parser.add_argument("--attack_name", required=True, help="Attack name (string)")
    parser.add_argument("--qdrant_yaml", default=DEFAULT_QDRANT_YAML, help="Path to Qdrant YAML config")
    parser.add_argument("--neo4j_yaml", default=DEFAULT_NEO4J_YAML, help="Path to Neo4j YAML config")
    parser.add_argument("--enable_neo4j", action="store_true", help="Enable Neo4j k-hop grounding (Loop C)")
    parser.add_argument("--hops", type=int, default=2, help="Neo4j k-hop depth")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory for defence JSON")
    parser.add_argument("--update_dot", action="store_true", help="Update DOT graph with recommended defence")
    parser.add_argument("--dot_path", default=DEFAULT_DOT_PATH, help="Path to TrackA_with_defences.dot")

    args = parser.parse_args()

    run_defence_pipeline(
        attack_name=args.attack_name,
        qdrant_yaml=args.qdrant_yaml,
        neo4j_yaml=args.neo4j_yaml,
        enable_neo4j=args.enable_neo4j,
        hops=args.hops,
        out_dir=args.out_dir,
        update_dot=args.update_dot,
        dot_path=args.dot_path,
    )
