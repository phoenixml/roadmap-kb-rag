from __future__ import annotations
from typing import Dict, List, Tuple
from kb.config import load_yaml
from kb.neo4j_client import Neo4jClient, Neo4jConfig
from kb.qdrant_client import VectorStore, QdrantConfig

def _connect():
    neo_cfg = load_yaml("configs/neo4j.yaml")
    qd_cfg = load_yaml("configs/qdrant.yaml")
    neo = Neo4jClient(Neo4jConfig(**neo_cfg))
    vs = VectorStore(QdrantConfig(**qd_cfg))
    return neo, vs

def retrieve_text(query: str, k: int = 6) -> List[Dict]:
    _, vs = _connect()
    hits = vs.search(query, k=k)
    results = []
    for h in hits:
        payload = h.payload or {}
        results.append({
            "chunk_id": payload.get("chunk_id"),
            "doc_id": payload.get("doc_id"),
            "score": float(h.score),
            "payload": payload,
        })
    return results

def retrieve_graph_paths(query: str, k: int = 50, hops: int = 2):
    neo, _ = _connect()
    # naive seed entity detection: find entities whose name appears in query
    seeds = neo.fetch_entity_names_like(query.lower()[:32])
    seeds = seeds[:5]
    if not seeds:
        return []
    rows = neo.get_k_hop_subgraph(seeds, hops=hops, limit=k)
    # stringify paths
    out = []
    for r in rows:
        p = r["p"]
        # neo4j Path object stringifies nicely
        out.append(str(p))
    return out

def retrieve_hybrid(query: str, k_text: int = 6, hops: int = 2):
    text_hits = retrieve_text(query, k=k_text)
    paths = retrieve_graph_paths(query, hops=hops)
    return text_hits, paths
