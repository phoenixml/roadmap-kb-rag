# neo4j_defense_kb.py

from neo4j import GraphDatabase
from typing import List, Dict, Optional


class DefenseKB:
    """
    Minimal Neo4j helper.
    ONLY stores:
      (:Attack)-[:MITIGATED_BY]->(:Defense)

    No VM, no TA, no PC ingestion.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # -------------------------------
    # Core write operation
    # -------------------------------
    def add_defense(
        self,
        attack_name: str,
        defense_name: str,
        defense_category: str,
        confidence: float,
        source: str
    ):
        query = """
        MERGE (a:Attack {name: $attack})
        MERGE (d:Defense {name: $defense, category: $category})
        MERGE (a)-[r:MITIGATED_BY]->(d)
        SET r.confidence = $confidence,
            r.source = $source,
            r.updated_at = timestamp()
        """
        with self.driver.session() as session:
            session.run(
                query,
                attack=attack_name,
                defense=defense_name,
                category=defense_category,
                confidence=confidence,
                source=source
            )

    # -------------------------------
    # Read for Loop-C grounding
    # -------------------------------
    def get_defenses(self, attack_name: str) -> List[Dict]:
        query = """
        MATCH (a:Attack {name: $attack})-[r:MITIGATED_BY]->(d:Defense)
        RETURN d.name AS name,
               d.category AS category,
               r.confidence AS confidence,
               r.source AS source
        """
        with self.driver.session() as session:
            result = session.run(query, attack=attack_name)
            return [dict(record) for record in result]

    # -------------------------------
    # Conflict detection
    # -------------------------------
    def detect_conflicts(self, attack_name: str) -> Optional[str]:
        """
        Flags conflicting defense categories for the same attack.
        Example conflict:
          Gradient Masking vs Certified Robustness
        """
        query = """
        MATCH (a:Attack {name: $attack})-[r:MITIGATED_BY]->(d:Defense)
        RETURN collect(DISTINCT d.category) AS categories
        """
        with self.driver.session() as session:
            record = session.run(query, attack=attack_name).single()
            if not record:
                return None

            categories = record["categories"]
            if len(categories) > 1:
                return f"CONFLICT: multiple defense categories found: {categories}"

        return None
