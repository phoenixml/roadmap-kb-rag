from __future__ import annotations
from typing import Dict, List
from kb.qdrant_client import VectorStore

def upsert_chunk_vectors(vs: VectorStore, chunk_ids: List[str], chunk_texts: List[str], payloads: List[Dict]):
    vs.upsert_chunks(chunk_ids=chunk_ids, texts=chunk_texts, payloads=payloads)
