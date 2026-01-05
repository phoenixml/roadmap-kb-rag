from __future__ import annotations
from typing import Dict, List
from kb.neo4j_client import Neo4jClient
from kb.graph_utils import canonicalize_entities, normalize_relation

def upsert_chunk_kg(
    neo4j: Neo4jClient,
    doc_id: str,
    chunk_id: str,
    chunk_text: str,
    entities: List[Dict],
    relations: List[Dict],
    provenance: Dict
):
    neo4j.upsert_chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text, props=provenance)

    existing = neo4j.fetch_entity_names_like("")  # cheap: fetch small? We'll just use empty -> limited list
    # Better: fetch all entity names; for large graphs, implement a better strategy.
    # Keep it small for prototype:
    existing = existing[:2000] if existing else []

    extracted_names = [e["name"] for e in entities]
    mapping = canonicalize_entities(extracted_names, existing, score_cutoff=92)

    for e in entities:
        cname = mapping.get(e["name"], e["name"])
        neo4j.upsert_entity(name=cname, etype=e.get("type","Unknown"), props={"source_doc": doc_id})
        neo4j.link_entity_to_chunk(cname, chunk_id, rel="MENTIONED_IN")

    for r in relations:
        src = mapping.get(r["src"], r["src"])
        dst = mapping.get(r["dst"], r["dst"])
        rel = normalize_relation(r.get("rel","RELATED_TO"))
        neo4j.upsert_entity(src, "Unknown")
        neo4j.upsert_entity(dst, "Unknown")
        neo4j.upsert_relation(src, rel, dst, props={"source_doc": doc_id})
