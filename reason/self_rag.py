from __future__ import annotations
from typing import Dict, List
from llm import LLM, json_from_llm
from reason.prompt_templates import SYSTEM_REASONER, USER_TEMPLATE
from reason.critic import critique
from reason.retriever import retrieve_hybrid

def _format_text_hits(text_hits: List[Dict]) -> str:
    lines = []
    for h in text_hits:
        cid = h.get("chunk_id")
        score = h.get("score")
        # we didn't store text in qdrant payload; in production you should.
        # For prototype, you can fetch chunk text from Neo4j by chunk_id.
        lines.append(f"- {cid} (score={score:.4f})")
    return "\n".join(lines)

def _format_paths(paths: List[str]) -> str:
    if not paths:
        return ""
    return "\n".join([f"[{i}] {p}" for i,p in enumerate(paths[:12])])

def run_self_rag(question: str, model_cfg_path="configs/model.yaml") -> Dict:
    llm = LLM(model_cfg_path)

    text_hits, paths = retrieve_hybrid(question, k_text=6, hops=2)
    chunks_str = _format_text_hits(text_hits)
    paths_str = _format_paths(paths)

    messages = [
        {"role": "system", "content": SYSTEM_REASONER},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, chunks=chunks_str, paths=paths_str)}
    ]
    draft = llm.chat(messages, temperature=0.2, max_tokens=900)

    verdict = critique(
        question=question,
        answer=draft,
        text_evidence=chunks_str,
        paths=paths_str,
        llm=llm
    )

    refined = None
    if verdict.get("status") == "reject":
        # refine: broaden retrieval
        text_hits2, paths2 = retrieve_hybrid(question + " " + " ".join(verdict.get("missing_queries", [])[:2]), k_text=10, hops=3)
        chunks_str2 = _format_text_hits(text_hits2)
        paths_str2 = _format_paths(paths2)
        messages2 = [
            {"role": "system", "content": SYSTEM_REASONER},
            {"role": "user", "content": USER_TEMPLATE.format(question=question, chunks=chunks_str2, paths=paths_str2)}
        ]
        refined = llm.chat(messages2, temperature=0.2, max_tokens=900)
        return {
            "question": question,
            "draft": draft,
            "answer": refined,
            "text_hits": text_hits2,
            "paths": paths2,
            "trace": {"critic": verdict}
        }

    # accept
    return {
        "question": question,
        "draft": draft,
        "answer": draft,
        "text_hits": text_hits,
        "paths": paths,
        "trace": {"critic": verdict}
    }
