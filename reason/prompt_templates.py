SYSTEM_REASONER = "You are a careful research assistant. Use only provided evidence. Cite evidence IDs."

USER_TEMPLATE = """Question:
{question}

Evidence (text chunks):
{chunks}

Evidence (graph paths):
{paths}

Return JSON:
{{
  "answer": "...",
  "citations": {{
     "chunks": ["chunk_id", ...],
     "paths": [0, 1, ...]
  }},
  "confidence": 0.0
}}
"""

SYSTEM_CRITIC = "You are a strict verifier. Check if the answer is supported by evidence; identify missing/contradicting evidence."

CRITIC_TEMPLATE = """Question:
{question}

Proposed answer:
{answer}

Evidence:
{text_evidence}

Graph paths:
{paths}

Return JSON:
{{
  "status": "accept"|"reject",
  "reasons": ["..."],
  "missing_queries": ["..."],
  "contradictions": ["..."]
}}
"""
