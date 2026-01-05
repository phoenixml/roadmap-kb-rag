from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from neo4j import GraphDatabase

@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"

class Neo4jClient:
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

    def close(self):
        self.driver.close()
    '''
    def run(self, query: str, **params):
        with self.driver.session(database=self.cfg.database) as s:
            return list(s.run(query, **params))
    '''
    def run(self, query: str, params: dict = None):
        with self.driver.session(database=self.cfg.database) as session:
            if params:
                return list(session.run(query, params))
            else:
                return list(session.run(query))
    # --- schema helpers ---
    def ensure_constraints(self):
        queries = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        ]
        for q in queries:
            self.run(q)

        def upsert_attack_semantic_subgraph(self, attack: dict):
            query = """
            MERGE (a:Attack {name: $attack_name})

            FOREACH (_ IN CASE WHEN $input <> "" THEN [1] ELSE [] END |
                MERGE (i:Input {text: $input})
                MERGE (a)-[:HAS_INPUT]->(i)
            )

            FOREACH (_ IN CASE WHEN $output <> "" THEN [1] ELSE [] END |
                MERGE (o:Output {text: $output})
                MERGE (a)-[:HAS_OUTPUT]->(o)
            )

            FOREACH (_ IN CASE WHEN $formula <> "" THEN [1] ELSE [] END |
                MERGE (f:Formula {latex: $formula})
                MERGE (a)-[:HAS_FORMULA]->(f)
            )

            FOREACH (_ IN CASE WHEN $explanation <> "" THEN [1] ELSE [] END |
                MERGE (e:Explanation {text: $explanation})
                MERGE (a)-[:HAS_EXPLANATION]->(e)
            )
            """

            params = {
                "attack_name": attack.get("attack_name"),
                "input": attack.get("input", ""),
                "output": attack.get("output", ""),
                "formula": attack.get("formula", ""),
                "explanation": attack.get("explanation", ""),
            }

            with self.driver.session(database=self.database) as session:
                session.run(query, params)
    
    def ensure_constraints(self):
        queries = [
            """
            CREATE CONSTRAINT attack_id_unique IF NOT EXISTS
            FOR (a:Attack)
            REQUIRE a.attack_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT kbfield_unique IF NOT EXISTS
            FOR (f:KBField)
            REQUIRE (f.attack_id, f.field) IS UNIQUE
            """,
            """
            CREATE CONSTRAINT formula_unique IF NOT EXISTS
            FOR (f:Formula)
            REQUIRE f.attack_id IS UNIQUE
            """
        ]
        for q in queries:
            self.run(q)

    
    
    def upsert_entity(self, name: str, etype: str = "Unknown", props: Optional[Dict[str, Any]] = None):
        props = props or {}
        q = '''
        MERGE (e:Entity {name:$name})
        ON CREATE SET e.type=$etype, e.created_at=timestamp()
        SET e.type=coalesce(e.type, $etype)
        SET e += $props
        RETURN e
        '''
        self.run(q, name=name, etype=etype, props=props)

    def upsert_chunk(self, chunk_id: str, doc_id: str, text: str, props: Optional[Dict[str, Any]] = None):
        props = props or {}
        q = '''
        MERGE (c:Chunk {chunk_id:$chunk_id})
        ON CREATE SET c.created_at=timestamp()
        SET c.doc_id=$doc_id, c.text=$text
        SET c += $props
        RETURN c
        '''
        self.run(q, chunk_id=chunk_id, doc_id=doc_id, text=text, props=props)

    def link_entity_to_chunk(self, entity_name: str, chunk_id: str, rel: str = "MENTIONED_IN"):
        # relationship type cannot be parameterized in Cypher; we validate to safe uppercase/underscore
        safe_rel = "".join([ch for ch in rel.upper() if ch.isalnum() or ch == "_"])
        q = f'''
        MATCH (e:Entity {{name:$entity}}), (c:Chunk {{chunk_id:$chunk_id}})
        MERGE (e)-[r:{safe_rel}]->(c)
        ON CREATE SET r.created_at=timestamp()
        RETURN r
        '''
        self.run(q, entity=entity_name, chunk_id=chunk_id)

    def upsert_relation(self, src: str, rel: str, dst: str, props: Optional[Dict[str, Any]] = None):
        props = props or {}
        safe_rel = "".join([ch for ch in rel.upper() if ch.isalnum() or ch == "_"])
        q = f'''
        MATCH (a:Entity {{name:$src}}), (b:Entity {{name:$dst}})
        MERGE (a)-[r:{safe_rel}]->(b)
        ON CREATE SET r.created_at=timestamp()
        SET r += $props
        RETURN r
        '''
        self.run(q, src=src, dst=dst, props=props)

    def get_k_hop_subgraph(self, seed_entities: List[str], hops: int = 2, limit: int = 200):
        q = f'''
        MATCH p=(n:Entity)-[*1..{hops}]-(m:Entity)
        WHERE n.name IN $seeds
        RETURN p
        LIMIT $limit
        '''
        return self.run(q, seeds=seed_entities, limit=limit)

    def fetch_entity_names_like(self, query: str, limit: int = 20) -> List[str]:
        q = '''
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($q)
        RETURN e.name AS name
        LIMIT $limit
        '''
        rows = self.run(q, q=query, limit=limit)
        return [r["name"] for r in rows]

    def upsert_attack_semantics(
        self,
        attack_id: str,
        name: str,
        family: str,
        input_text: str,
        output_text: str,
        explanation_text: str,
        formula_latex: str,
        source_file: str
    ):
        query = """
        MERGE (a:Attack {attack_id: $attack_id})
        SET a.name = $name,
            a.family = $family,
            a.source_file = $source_file
        WITH a
        MERGE (i:KBField {attack_id: $attack_id, field: "Input"})
        SET i.value = $input_text
        MERGE (a)-[:HAS_INPUT]->(i)
        WITH a
        MERGE (o:KBField {attack_id: $attack_id, field: "Output"})
        SET o.value = $output_text
        MERGE (a)-[:HAS_OUTPUT]->(o)
        WITH a
        MERGE (e:KBField {attack_id: $attack_id, field: "Explanation"})
        SET e.value = $explanation_text
        MERGE (a)-[:HAS_EXPLANATION]->(e)
        WITH a
        MERGE (f:Formula {attack_id: $attack_id})
        SET f.latex = $formula_latex
        MERGE (a)-[:HAS_FORMULA]->(f)

        """

        self.run(query, {
            "attack_id": attack_id,
            "name": name,
            "family": family,
            "source_file": source_file,
            "input_text": input_text,
            "output_text": output_text,
            "explanation_text": explanation_text,
            "formula_latex": formula_latex,
        })

