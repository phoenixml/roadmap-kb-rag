from __future__ import annotations
from typing import List
from kb.neo4j_client import Neo4jClient

def extract_subgraph(neo4j: Neo4jClient, seed_entities: List[str], hops: int = 2, limit: int = 100):
    return neo4j.get_k_hop_subgraph(seed_entities, hops=hops, limit=limit)
