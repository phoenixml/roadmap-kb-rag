from __future__ import annotations
from typing import Dict, List
from llm import LLM, json_from_llm
from reason.prompt_templates import SYSTEM_CRITIC, CRITIC_TEMPLATE

def critique(question: str, answer: str, text_evidence: str, paths: str, llm: LLM) -> Dict:
    messages = [
        {"role": "system", "content": SYSTEM_CRITIC},
        {"role": "user", "content": CRITIC_TEMPLATE.format(
            question=question, answer=answer, text_evidence=text_evidence, paths=paths
        )}
    ]
    out = llm.chat(messages, temperature=0.0, max_tokens=600)
    return json_from_llm(out)
