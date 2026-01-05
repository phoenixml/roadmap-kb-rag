import argparse
import json
from kb.config import load_yaml
from kb.neo4j_client import Neo4jClient, Neo4jConfig
from kb.qdrant_client import VectorStore, QdrantConfig
from ingest.build_vector_index import upsert_chunk_vectors

def to_str(val):
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)

def safe_str(x):
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return "" if x is None else str(x)

def load_kb(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def attack_id_from_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_")

def ingest_kb_json(kb_path: str, neo_cfg_path="configs/neo4j.yaml", qd_cfg_path="configs/qdrant.yaml"):
    kb = load_kb(kb_path)

    neo_cfg = load_yaml(neo_cfg_path)
    neo = Neo4jClient(Neo4jConfig(**neo_cfg))
    neo.ensure_constraints()

    qd_cfg = load_yaml(qd_cfg_path)
    vs = VectorStore(QdrantConfig(**qd_cfg))

    # We’ll also embed each attack as a “KB chunk” into Qdrant
    vec_ids, vec_texts, vec_payloads = [], [], []

    for row in kb:
        name = row.get("attack_name") or row.get("attack") or row.get("Filename") or "unknown"
        attack_id = attack_id_from_name(name)

        input_t = to_str(row.get("input") or row.get("Input"))
        output_t = to_str(row.get("output") or row.get("Output"))
        expl_t = to_str(row.get("explanation") or row.get("Explanation"))
        formula_t = to_str(row.get("formula") or row.get("Formula"))

        # --- Neo4j upsert Attack semantics ---
        neo.upsert_attack_semantics(
            attack_id=attack_id,
            name=name,
            family=(row.get("graph_node") or {}).get("type") if isinstance(row.get("graph_node"), dict) else None,
            input_text=input_t,
            output_text=output_t,
            explanation_text=expl_t,
            formula_latex=formula_t,
            source_file=row.get("source_file") or row.get("Filename")
        )

        # --- Qdrant embedding for KB retrieval ---
        kb_text = f"Attack: {name}\n\nInput:\n{input_t}\n\nOutput:\n{output_t}\n\nFormula:\n{formula_t}\n\nExplanation:\n{expl_t}"
        vec_id = f"kb::{attack_id}"

        vec_ids.append(vec_id)
        vec_texts.append(kb_text)
        vec_payloads.append({
            "type": "kb_attack",
            "attack_id": attack_id,
            "attack_name": name,
            "source_file": row.get("source_file") or row.get("Filename")
        })

    upsert_chunk_vectors(vs, vec_ids, vec_texts, vec_payloads)
    neo.close()

    return {"attacks": len(kb), "qdrant_vectors": len(vec_ids)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb_path", required=True)
    ap.add_argument("--neo_cfg", default="configs/neo4j.yaml")
    ap.add_argument("--qdrant_cfg", default="configs/qdrant.yaml")
    args = ap.parse_args()

    res = ingest_kb_json(args.kb_path, args.neo_cfg, args.qdrant_cfg)
    print(res)

if __name__ == "__main__":
    main()
