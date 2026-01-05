from __future__ import annotations
import os, hashlib
from typing import List
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
from kb.qdrant_client import QdrantConfig, VectorStore


# -----------------------------
# Helpers
# -----------------------------

def split_into_paragraphs(text: str, max_len: int = 800):
    """Split long text into manageable chunks for LLM extraction."""
    paragraphs = []
    buf = ""

    for line in text.split("\n"):
        if len(buf) + len(line) > max_len:
            if buf.strip():
                paragraphs.append(buf.strip())
            buf = line
        else:
            buf += " " + line

    if buf.strip():
        paragraphs.append(buf.strip())

    return paragraphs


def _doc_id_from_path(path: str) -> str:
    name = os.path.basename(path)
    h = hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]
    return f"{name}:{h}"


# -----------------------------
# Main ingestion function
# -----------------------------

def ingest_pdf(pdf_path: str,
               neo_cfg_path="configs/neo4j.yaml",
               qd_cfg_path="configs/qdrant.yaml",
               model_cfg_path="configs/model.yaml"):

    llm = LLM(model_cfg_path)

    neo_cfg = load_yaml(neo_cfg_path)
    neo = Neo4jClient(Neo4jConfig(**neo_cfg))
    neo.ensure_constraints()

    qd_cfg = load_yaml(qd_cfg_path)
    vs = VectorStore(QdrantConfig(**qd_cfg))

    doc_id = _doc_id_from_path(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    chunks = simple_chunk(text)

    chunk_ids, chunk_texts, payloads = [], [], []

    for idx, chunk in enumerate(tqdm(chunks, desc=f"Chunks {os.path.basename(pdf_path)}")):
        chunk_id = f"{doc_id}::chunk{idx:04d}"
        provenance = {
            "source_pdf": os.path.basename(pdf_path),
            "chunk_index": idx,
        }

        # --- NEW: paragraph-level processing ---
        paragraphs = split_into_paragraphs(chunk)

        for para in paragraphs:
            try:
                entities = extract_entities(para, llm)
                relations = extract_relations(para, llm)

                upsert_chunk_kg(
                    neo4j=neo,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    chunk_text=para,
                    entities=entities,
                    relations=relations
                )

            except Exception as e:
                print(f"[WARN] Skipping paragraph due to error: {e}")

        chunk_ids.append(chunk_id)
        chunk_texts.append(chunk)
        payloads.append({"doc_id": doc_id, "chunk_index": idx})

    # Vector index
    upsert_chunk_vectors(vs, chunk_ids, chunk_texts, payloads)

    neo.close()
    return {"doc_id": doc_id, "chunks": len(chunks)}


def ingest_folder(pdf_dir: str):
    results = []
    for fn in sorted(os.listdir(pdf_dir)):
        if fn.lower().endswith(".pdf"):
            results.append(ingest_pdf(os.path.join(pdf_dir, fn)))
    return results
