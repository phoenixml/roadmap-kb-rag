import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kb.config import load_yaml
from kb.neo4j_client import Neo4jClient, Neo4jConfig
from kb.qdrant_client import VectorStore, QdrantConfig
from ingest.build_vector_index import upsert_chunk_vectors

LOG = logging.getLogger("ingest_kb")

# ---------------------------
# Helpers
# ---------------------------

def to_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)

def load_kb(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"KB JSON must be a list[dict], got: {type(data)}")
    return data

def attack_id_from_name(name: str) -> str:
    # stable ID for cross-system joins
    n = (name or "unknown").strip().lower()
    n = re.sub(r"\s+", "_", n)
    n = n.replace("/", "_")
    n = re.sub(r"[^a-z0-9_\-\.]", "", n)
    return n[:200] or "unknown"

def parse_family(row: dict) -> Optional[str]:
    gn = row.get("graph_node")
    if isinstance(gn, dict):
        return gn.get("type") or gn.get("family")
    if isinstance(gn, str):
        return gn
    return None

def parse_name(row: dict) -> str:
    return (
        row.get("attack_name")
        or row.get("attack")
        or row.get("Filename")
        or row.get("source_file")
        or "unknown"
    )

def build_attack_fields(row: dict) -> Dict[str, str]:
    return {
        "input": to_str(row.get("input") or row.get("Input")),
        "output": to_str(row.get("output") or row.get("Output")),
        "explanation": to_str(row.get("explanation") or row.get("Explanation")),
        "formula": to_str(row.get("formula") or row.get("Formula")),
    }

# ---------------------------
# Chunking for RAG
# ---------------------------

@dataclass(frozen=True)
class Chunk:
    chunk_type: str
    text: str

def make_chunks(name: str, fields: Dict[str, str], chunk_mode: str = "section") -> List[Chunk]:
    """
    chunk_mode:
      - "single": one big chunk (legacy)
      - "section": multiple chunks by section (recommended)
    """
    title = f"Attack: {name}".strip()

    if chunk_mode == "single":
        kb_text = (
            f"{title}\n\n"
            f"Input:\n{fields['input']}\n\n"
            f"Output:\n{fields['output']}\n\n"
            f"Formula:\n{fields['formula']}\n\n"
            f"Explanation:\n{fields['explanation']}"
        )
        return [Chunk("full", kb_text)]

    # section mode
    chunks: List[Chunk] = [Chunk("title", title)]
    if fields["input"] or fields["output"]:
        io = f"{title}\n\nInput:\n{fields['input']}\n\nOutput:\n{fields['output']}"
        chunks.append(Chunk("io", io))
    if fields["formula"]:
        chunks.append(Chunk("formula", f"{title}\n\nFormula:\n{fields['formula']}"))
    if fields["explanation"]:
        chunks.append(Chunk("explanation", f"{title}\n\nExplanation:\n{fields['explanation']}"))

    # remove empties
    return [c for c in chunks if c.text.strip()]

def build_payload_base(
    attack_id: str,
    name: str,
    family: Optional[str],
    source_file: Optional[str],
    fields: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "type": "kb_attack",
        "attack_id": attack_id,
        "attack_name": name,
        "family": family or "Unknown",
        "source_file": source_file or "",
        "input": fields.get("input", ""),
        "output": fields.get("output", ""),
        "formula": fields.get("formula", ""),
        "explanation": fields.get("explanation", ""),
        "has_formula": bool(fields.get("formula")),
        "has_explanation": bool(fields.get("explanation")),
        "has_io": bool(fields.get("input") or fields.get("output")),
        
        "domain": "adversarial_ml",
        "source": "kb_ingest",
    }

# ---------------------------
# Ingest (bulk)
# ---------------------------

def ingest_kb_json(
    kb_path: str,
    neo_cfg_path: str = "configs/neo4j.yaml",
    qd_cfg_path: str = "configs/qdrant.yaml",
    *,
    chunk_mode: str = "section",
    limit: Optional[int] = None,
    dry_run: bool = False,
    only_graph: bool = False,
    only_vectors: bool = False,
) -> Dict[str, int]:
    kb = load_kb(kb_path)
    if limit is not None:
        kb = kb[: max(0, limit)]

    neo_cfg = load_yaml(neo_cfg_path)
    qd_cfg = load_yaml(qd_cfg_path)

    neo = Neo4jClient(Neo4jConfig(**neo_cfg))
    neo.ensure_constraints()

    vs = VectorStore(QdrantConfig(**qd_cfg))

    vec_ids: List[str] = []
    vec_texts: List[str] = []
    vec_payloads: List[dict] = []

    ingested_attacks = 0
    vector_count = 0

    for row in kb:
        name = parse_name(row)
        attack_id = attack_id_from_name(name)
        family = parse_family(row)
        source_file = row.get("source_file") or row.get("Filename")

        fields = build_attack_fields(row)

        if not only_vectors:
            if not dry_run:
                neo.upsert_attack_semantics(
                    attack_id=attack_id,
                    name=name,
                    family=family,
                    input_text=fields["input"],
                    output_text=fields["output"],
                    explanation_text=fields["explanation"],
                    formula_latex=fields["formula"],
                    source_file=source_file,
                )
            ingested_attacks += 1

        if not only_graph:
            payload_base = build_payload_base(attack_id, name, family, source_file, fields)
            chunks = make_chunks(name, fields, chunk_mode=chunk_mode)

            for c in chunks:
                vec_id = f"kb::{attack_id}::{c.chunk_type}"
                vec_ids.append(vec_id)
                vec_texts.append(c.text)
                p = dict(payload_base)
                p["chunk_type"] = c.chunk_type
                vec_payloads.append(p)
                vector_count += 1

    if not only_graph and not dry_run and vec_ids:
        upsert_chunk_vectors(vs, vec_ids, vec_texts, vec_payloads)

    neo.close()
    return {"attacks": ingested_attacks, "qdrant_vectors": vector_count}

# ---------------------------
# Ingest (single attack) — for DPA agent integration
# ---------------------------

def ingest_one_attack(
    attack: Dict[str, Any],
    neo_cfg_path: str = "configs/neo4j.yaml",
    qd_cfg_path: str = "configs/qdrant.yaml",
    *,
    chunk_mode: str = "section",
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Attack schema expected (matches your dpa_agent output):
      {
        "attack_name": "...",
        "input": "...",
        "output": "...",
        "formula": "...",
        "explanation": "...",
        "graph_node": {"type": "..."}  # optional
        "source_file": "..."          # optional
      }
    """
    name = parse_name(attack)
    attack_id = attack_id_from_name(name)
    family = parse_family(attack)
    source_file = attack.get("source_file") or attack.get("Filename")

    fields = build_attack_fields(attack)

    neo_cfg = load_yaml(neo_cfg_path)
    qd_cfg = load_yaml(qd_cfg_path)

    neo = Neo4jClient(Neo4jConfig(**neo_cfg))
    neo.ensure_constraints()

    vs = VectorStore(QdrantConfig(**qd_cfg))

    if not dry_run:
        neo.upsert_attack_semantics(
            attack_id=attack_id,
            name=name,
            family=family,
            input_text=fields["input"],
            output_text=fields["output"],
            explanation_text=fields["explanation"],
            formula_latex=fields["formula"],
            source_file=source_file,
        )

    payload_base = build_payload_base(attack_id, name, family, source_file, fields)
    chunks = make_chunks(name, fields, chunk_mode=chunk_mode)

    vec_ids, vec_texts, vec_payloads = [], [], []
    for c in chunks:
        vec_ids.append(f"kb::{attack_id}::{c.chunk_type}")
        vec_texts.append(c.text)
        p = dict(payload_base)
        p["chunk_type"] = c.chunk_type
        vec_payloads.append(p)

    if not dry_run and vec_ids:
        upsert_chunk_vectors(vs, vec_ids, vec_texts, vec_payloads)

    neo.close()
    return {"attacks": 1, "qdrant_vectors": len(vec_ids)}

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb_path", required=True)
    ap.add_argument("--neo_cfg", default="configs/neo4j.yaml")
    ap.add_argument("--qdrant_cfg", default="configs/qdrant.yaml")

    ap.add_argument("--chunk_mode", choices=["single", "section"], default="section")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--only_graph", action="store_true")
    ap.add_argument("--only_vectors", action="store_true")

    ap.add_argument("--log_level", default="INFO")

    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.only_graph and args.only_vectors:
        raise SystemExit("Choose at most one of --only_graph / --only_vectors")

    res = ingest_kb_json(
        args.kb_path,
        args.neo_cfg,
        args.qdrant_cfg,
        chunk_mode=args.chunk_mode,
        limit=args.limit,
        dry_run=args.dry_run,
        only_graph=args.only_graph,
        only_vectors=args.only_vectors,
    )
    print(res)

if __name__ == "__main__":
    main()
