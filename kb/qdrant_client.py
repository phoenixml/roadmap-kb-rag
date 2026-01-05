from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

@dataclass
class QdrantConfig:
    url: str
    api_key: Optional[str] = None
    collection: str = "docs"
    vector_size: int = 384
    distance: str = "Cosine"

class VectorStore:
    def __init__(self, cfg: QdrantConfig, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.cfg = cfg
        self.client = QdrantClient(url=cfg.url, api_key=cfg.api_key)
        self.embedder = SentenceTransformer(embed_model)

    def ensure_collection(self):
        dist = Distance.COSINE if self.cfg.distance.lower() == "cosine" else Distance.DOT
        if not self.client.collection_exists(self.cfg.collection):
            self.client.create_collection(
                collection_name=self.cfg.collection,
                vectors_config=VectorParams(size=self.cfg.vector_size, distance=dist),
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def upsert_chunks(self, chunk_ids: List[str], texts: List[str], payloads: List[Dict[str, Any]]):
        self.ensure_collection()
        vectors = self.embed(texts)
        points = []
        for i, cid in enumerate(chunk_ids):
            # Qdrant point id can be int or uuid; we use deterministic hash int
            pid = abs(hash(cid)) % (2**31)
            points.append(PointStruct(id=pid, vector=vectors[i], payload={**payloads[i], "chunk_id": cid}))
        self.client.upsert(collection_name=self.cfg.collection, points=points)

    def search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None):
        self.ensure_collection()
        vec = self.embed([query])[0]
        # minimal: no advanced filter composition
        return self.client.search(collection_name=self.cfg.collection, query_vector=vec, limit=k)
