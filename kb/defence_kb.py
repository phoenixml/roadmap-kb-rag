# kb/defence_kb.py

from typing import Dict, List, Optional
from neo4j import GraphDatabase


class DefenceKB:
    """
    Minimal Neo4j memory for defence reasoning.

    Stores ONLY:
      (:Attack)-[:MITIGATED_BY]->(:Defence)

    This is NOT a taxonomy.
    """

    # kb/defence_kb.py

import os
from neo4j import GraphDatabase
from typing import Dict, List, Optional


def _resolve_env(val: str) -> str:
    if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
        env = val[2:-1]
        resolved = os.getenv(env)
        if not resolved:
            raise RuntimeError(f"Environment variable {env} is not set")
        return resolved
    return val


class DefenceKB:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        uri = _resolve_env(uri)
        user = _resolve_env(user)
        password = _resolve_env(password)

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
        )
        self.database = database


    def close(self):
        self.driver.close()

    # -------------------------
    # Write
    # -------------------------
    def upsert_attack_and_defence(
        self,
        attack: dict,
        defence_name: str,
        defence_category: str,
        confidence: float,
        source: str,
    ):
        """
        attack dict MUST include:
        - attack_name
        - formula
        - input
        - output
        - explanation
        """

        query = """
        MERGE (a:Attack {name: $attack_name})
        SET
            a.formula = $formula,
            a.input = $input,
            a.output = $output,
            a.explanation = $explanation,
            a.updated_at = timestamp()

        MERGE (d:Defence {name: $defence_name, category: $defence_category})

        MERGE (a)-[r:MITIGATED_BY]->(d)
        SET
            r.confidence = $confidence,
            r.source = $source,
            r.updated_at = timestamp()
        """

        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                attack_name=attack.get("attack_name"),
                formula=attack.get("formula", ""),
                input=attack.get("input", ""),
                output=attack.get("output", ""),
                explanation=attack.get("explanation", ""),
                defence_name=defence_name,
                defence_category=defence_category,
                confidence=confidence,
                source=source,
            )

    # -------------------------
    # Read (Loop-C memory)
    # -------------------------
    def get_defences(self, attack_name: str) -> List[Dict]:
        query = """
        MATCH (a:Attack {name: $attack})-[r:MITIGATED_BY]->(d:Defence)
        RETURN d.name AS name,
               d.category AS category,
               r.confidence AS confidence,
               r.source AS source
        """
        with self.driver.session(database=self.database) as session:
            res = session.run(query, attack=attack_name)
            return [dict(r) for r in res]

    # -------------------------
    # Conflict detection
    # -------------------------
    def detect_conflict(self, attack_name: str) -> Optional[str]:
        """
        Flags multiple defence categories for same attack.
        """
        query = """
        MATCH (a:Attack {name: $attack})-[:MITIGATED_BY]->(d:Defence)
        RETURN collect(DISTINCT d.category) AS cats
        """
        with self.driver.session(database=self.database) as session:
            rec = session.run(query, attack=attack_name).single()
            if not rec:
                return None

            cats = rec["cats"]
            if len(cats) > 1:
                return f"conflicting defence categories: {cats}"
        return None
