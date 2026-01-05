from __future__ import annotations
import os, hashlib
from typing import Dict, List
from tqdm import tqdm

from ingest.pdf_loader import extract_text_from_pdf
from ingest.chunker import simple_chunk
from ingest.entity_extractor import extract_entities
from ingest.relation_extractor import extract_relations
from ingest.build_graph import upsert_chunk_kg
from ingest.build_vector_index import upsert_chunk_vectors
from llm import LLM
from kb.config import load_yaml
from kb.neo4j_client import Neo4jClient, Neo4jConfig
from kb.qdrant_client import VectorStore, QdrantConfig

def _doc_id_from_path(path: str) -> str:
    name = os.path.basename(path)
    h = hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]
    return f"{name}:{h}"

def ingest_pdf(pdf_path: str, neo_cfg_path="configs/neo4j.yaml", qd_cfg_path="configs/qdrant.yaml", model_cfg_path="configs/model.yaml"):
    llm = LLM(model_cfg_path)

    neo_cfg_raw = load_yaml(neo_cfg_path)
    neo = Neo4jClient(Neo4jConfig(**neo_cfg_raw))
    neo.ensure_constraints()

    qd_cfg_raw = load_yaml(qd_cfg_path)
    vs = VectorStore(QdrantConfig(**qd_cfg_raw))

    doc_id = _doc_id_from_path(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    chunks = simple_chunk(text)

    chunk_ids, chunk_texts, payloads = [], [], []

    for idx, ch in enumerate(tqdm(chunks, desc=f"Chunks {os.path.basename(pdf_path)}")):
        chunk_id = f"{doc_id}::chunk{idx:04d}"
        provenance = {
            "source_pdf": os.path.basename(pdf_path),
            "chunk_index": idx,
        }
        ents = extract_entities(ch, llm)
        rels = extract_relations(ch, llm)

        upsert_chunk_kg(
            neo4j=neo,
            doc_id=doc_id,
            chunk_id=chunk_id,
            chunk_text=ch,
            entities=ents,
            relations=rels,
            provenance=provenance
        )

        chunk_ids.append(chunk_id)
        chunk_texts.append(ch)
        payloads.append({"doc_id": doc_id, **provenance})

    upsert_chunk_vectors(vs, chunk_ids, chunk_texts, payloads)
    neo.close()
    return {"doc_id": doc_id, "chunks": len(chunks)}

def ingest_folder(pdf_dir: str, **kwargs):
    results = []
    for fn in sorted(os.listdir(pdf_dir)):
        if fn.lower().endswith(".pdf"):
            results.append(ingest_pdf(os.path.join(pdf_dir, fn), **kwargs))
    return results
