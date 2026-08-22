"""
_rerun_fixed.py  (local, untracked)
Corrected ATLAS-QA RAG evaluation with REAL semantic retrieval.

Fixes the broken pipeline (which never matched the KB, so base==plus):
  - For each question, embed it (OpenAI text-embedding-3-small) and retrieve
    the top-k most similar entries from the BASE KB (roadmap_qa_data.json) vs
    the enriched ROADMAP+ KB (roadmap_qa_data_plus.json).
  - base_roadmap  -> context from BASE KB
  - roadmap_plus  -> context from PLUS KB (richer per-entry fields)
  - no_context    -> no retrieval
Judge = GPT-4.1 (the judge that actually reproduces the paper's No-Context column).
Metrics: LLM Judge (0-3), Exact Match (%), ROUGE-L F1.

Outputs (fresh files):
  outputs/qa_eval_results_atlas_fixed.json
  outputs/qa_eval_summary_atlas_fixed.csv
Resumable: saves every 20 rows; rerun to continue.
"""
import os, json, time, re, csv, sys, urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np

OLLAMA_URL = "https://ollama-llama-349218458434.us-west1.run.app"

# ── load .env ────────────────────────────────────────────────────────────────
_env = Path(".env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI
oc = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import anthropic
ac = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
from rouge_score import rouge_scorer
_rl = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# ── config ───────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"
TOP_K       = 3
QA_FILE     = "outputs/atlas_qa.json"
BASE_KB     = "outputs/roadmap_qa_data.json"
PLUS_KB     = "outputs/roadmap_qa_data_plus.json"
OUT_JSON    = "outputs/qa_eval_results_atlas_fixed.json"
OUT_CSV     = "outputs/qa_eval_summary_atlas_fixed.csv"
JUDGE_MODEL = "gpt-4.1"

MODELS = {
    "gpt-4.1":           {"provider": "openai",    "model": "gpt-4.1"},
    "gpt-4.1-mini":      {"provider": "openai",    "model": "gpt-4.1-mini"},
    "gpt-4o-mini":       {"provider": "openai",    "model": "gpt-4o-mini"},
    "claude-sonnet-4-6": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "claude-opus-4-8":   {"provider": "anthropic", "model": "claude-opus-4-8"},
    "gemma3":            {"provider": "ollama",    "model": "gemma3:latest"},
    "llama3.1:8b":       {"provider": "ollama",    "model": "llama3.1:8b"},
    "gemma3:1b":         {"provider": "ollama",    "model": "gemma3:1b"},
    "llama3.2:1b":       {"provider": "ollama",    "model": "llama3.2:1b"},
}
CONDITIONS = ["no_context", "base_roadmap", "roadmap_plus"]

SYSTEM_BASE = (
    "You are an expert in adversarial machine learning, AI security, and the "
    "MITRE ATLAS framework. Answer concisely and precisely. "
    "If you are not sure, say 'Unknown' rather than guessing."
)

# ── KB doc formatting + embeddings ───────────────────────────────────────────
FIELD_ORDER = ["attack", "family", "visibility", "perturbation_search", "defence",
               "description", "mechanism", "defence_mechanism", "variants",
               "threat_model", "target_vulnerability"]

def kb_doc(e):
    return "\n".join(f"{k.replace('_',' ').title()}: {e[k]}"
                     for k in FIELD_ORDER if e.get(k))

def embed(texts):
    out = []
    for i in range(0, len(texts), 100):
        chunk = texts[i:i+100]
        resp = oc.embeddings.create(model=EMBED_MODEL, input=chunk)
        out.extend(d.embedding for d in resp.data)
    return np.array(out, dtype=np.float32)

def topk_context(q_vec, kb_mat, kb_docs, k=TOP_K):
    sims = kb_mat @ q_vec            # both L2-normalised -> cosine
    idx = np.argsort(-sims)[:k]
    return "\n\n---\n\n".join(kb_docs[i] for i in idx)

def norm(m):
    n = np.linalg.norm(m, axis=1, keepdims=True); n[n == 0] = 1.0
    return m / n

# ── model callers ────────────────────────────────────────────────────────────
def call_openai(model, system, user):
    r = oc.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0, max_tokens=300)
    return r.choices[0].message.content.strip()

def call_anthropic(model, system, user):
    for attempt in range(5):
        try:
            r = ac.messages.create(model=model, max_tokens=300, system=system,
                                   messages=[{"role": "user", "content": user}])
            return r.content[0].text.strip()
        except Exception as e:
            if attempt == 4:
                return f"[ERROR: {e}]"
            time.sleep(2 * (attempt + 1))

def call_ollama(model, system, user):
    payload = json.dumps({
        "model": model, "prompt": f"{system}\n\n{user}", "stream": False,
        "options": {"temperature": 0.0, "num_predict": 300},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            txt = json.loads(r.read()).get("response", "").strip()
        # strip DeepSeek-R1 reasoning tags, keep the final answer
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        return txt
    except Exception as e:
        return f"[ERROR: {e}]"

def call_model(key, system, user):
    cfg = MODELS[key]
    if cfg["provider"] == "openai":
        try:    return call_openai(cfg["model"], system, user)
        except Exception as e: return f"[ERROR: {e}]"
    if cfg["provider"] == "ollama":
        return call_ollama(cfg["model"], system, user)
    return call_anthropic(cfg["model"], system, user)

def build_prompt(question, context, condition):
    if condition == "no_context":
        return SYSTEM_BASE, f"Question: {question}\n\nAnswer:"
    system = SYSTEM_BASE + ("\n\nYou have access to the following retrieved "
                            "knowledge-base entries. Use them to answer accurately.")
    return system, f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

def llm_judge(question, reference, predicted):
    prompt = (
        f"You are evaluating an AI answer about adversarial machine learning and "
        f"the MITRE ATLAS framework.\n\n"
        f"Question  : {question}\nReference : {reference}\nPredicted : {predicted}\n\n"
        "Rate the predicted answer:\n3 = Correct and complete\n"
        "2 = Mostly correct, minor gaps\n1 = Partially correct\n0 = Wrong or missing\n\n"
        "Respond with a single digit only: 0, 1, 2, or 3.")
    try:
        resp = call_openai(JUDGE_MODEL, "You are a strict academic evaluator.", prompt)
        m = re.search(r"[0-3]", resp)
        return int(m.group()) if m else 0
    except Exception:
        return 0

def rouge_l(pred, ref):
    try:    return round(_rl.score(ref, pred)["rougeL"].fmeasure, 3)
    except Exception: return 0.0

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    qa   = json.loads(Path(QA_FILE).read_text(encoding="utf-8"))
    base = json.loads(Path(BASE_KB).read_text(encoding="utf-8"))
    plus = json.loads(Path(PLUS_KB).read_text(encoding="utf-8"))

    print(f"[setup] QA={len(qa)}  baseKB={len(base)}  plusKB={len(plus)}  k={TOP_K}")
    print("[setup] embedding KB + questions ...")
    base_docs = [kb_doc(e) for e in base]
    plus_docs = [kb_doc(e) for e in plus]
    base_mat  = norm(embed(base_docs))
    plus_mat  = norm(embed(plus_docs))
    q_mat     = norm(embed([q["question"] for q in qa]))

    # precompute retrieved context per question
    ctx_base = [topk_context(q_mat[i], base_mat, base_docs) for i in range(len(qa))]
    ctx_plus = [topk_context(q_mat[i], plus_mat, plus_docs) for i in range(len(qa))]
    print("[setup] retrieval ready.")

    out = Path(OUT_JSON)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["model"], r["condition"], r["id"]) for r in results}
    total = len(qa) * len(MODELS) * len(CONDITIONS)
    print(f"[run] total={total}  done={len(done)}  remaining={total-len(done)}")

    for mkey in MODELS:
        for cond in CONDITIONS:
            print(f"  {mkey} / {cond}")
            for i, q in enumerate(qa):
                if (mkey, cond, q["id"]) in done:
                    continue
                ctx = "" if cond == "no_context" else (ctx_base[i] if cond == "base_roadmap" else ctx_plus[i])
                system, user = build_prompt(q["question"], ctx, cond)
                pred = call_model(mkey, system, user)
                j    = llm_judge(q["question"], q["answer"], pred)
                results.append({
                    "id": q["id"], "type": q.get("type"), "technique_id": q.get("technique_id"),
                    "question": q["question"], "reference": q["answer"], "predicted": pred,
                    "model": mkey, "condition": cond,
                    "llm_judge": j, "exact": 1 if j == 3 else 0, "rouge_l": rouge_l(pred, q["answer"]),
                })
                done.add((mkey, cond, q["id"]))
                if len(results) % 20 == 0:
                    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"    saved {len(results)}/{total}")
                time.sleep(0.2)

    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summarize(results)

def summarize(results):
    agg = defaultdict(lambda: {"j": [], "e": 0, "r": [], "n": 0, "empty": 0})
    for r in results:
        a = agg[(r["model"], r["condition"])]
        a["j"].append(r["llm_judge"]); a["r"].append(r["rouge_l"])
        a["e"] += r["exact"]; a["n"] += 1
        p = str(r.get("predicted", ""))
        if not p.strip() or p.startswith("[ERROR"):
            a["empty"] += 1
    rows = []
    for (m, c), a in sorted(agg.items()):
        n = a["n"]
        rows.append({"model": m, "condition": c, "n": n,
                     "avg_rouge_l": round(sum(a["r"])/n, 3),
                     "avg_llm_judge": round(sum(a["j"])/n, 3),
                     "pct_exact": round(a["e"]/n*100, 1),
                     "pct_empty": round(a["empty"]/n*100, 1)})
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"\n[OK] results -> {OUT_JSON}\n[OK] summary -> {OUT_CSV}")

if __name__ == "__main__":
    main()
