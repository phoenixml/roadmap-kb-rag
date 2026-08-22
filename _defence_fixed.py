"""
_defence_fixed.py  (local, untracked)
Corrected defence-recommendation benchmark.

Fixes the original eval/retrieval_defence_benchmark.py:
  - label swap ("No KB" was wired to a KB; "Human KB" to None)
  - inconsistent KBs (graph KBs carried no defences)
Task: given a recent unseen attack (benchmarks/unseen_recent.csv, 20 attacks,
2024-2025, each with ground-truth defences), predict its defence(s).
Conditions: no_context | base_roadmap | roadmap_plus, both roadmap KBs carrying
attack->defence pairs (roadmap_qa_data.json vs roadmap_qa_data_plus.json).
Retrieval: top-5 semantic (OpenAI text-embedding-3-small). Model: gpt-4.1.
Metric: lenient Top-1 / Top-3 defence-hit against ground truth.
"""
import os, json, re
from pathlib import Path
import numpy as np, pandas as pd

_env = Path(".env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
from openai import OpenAI
oc = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED = "text-embedding-3-small"; MODEL = "gpt-4.1"; TOPK = 5
CSV = "benchmarks/unseen_recent.csv"
BASE = "outputs/roadmap_qa_data.json"; PLUS = "outputs/roadmap_qa_data_plus.json"

def norm(t):
    t = str(t).strip().lower()
    return re.sub(r"\s+", " ", t.replace(" defence", "").replace(" defense", "").replace("_", " ")).strip()

def toks(t): return set(norm(t).split())

def match(pred, gt_list):
    """lenient: a predicted defence hits if it equals / is substring of / shares >=50% tokens with any GT."""
    p = norm(pred); pt = toks(pred)
    for g in gt_list:
        gn = norm(g); gt_ = toks(g)
        if not p or not gn: continue
        if p == gn or p in gn or gn in p: return True
        if pt and gt_ and len(pt & gt_) / len(pt | gt_) >= 0.5: return True
    return False

def kb_doc(e, rich):
    parts = [f"Attack: {e.get('attack','')}", f"Family: {e.get('family','')}",
             f"Visibility: {e.get('visibility','')}",
             f"Perturbation Search: {e.get('perturbation_search','')}",
             f"Defence: {e.get('defence','')}"]
    if rich:
        for k in ("description", "mechanism", "defence_mechanism", "threat_model"):
            if e.get(k): parts.append(f"{k.replace('_',' ').title()}: {e[k]}")
    return "\n".join(parts)

def kb_line(e, rich):
    line = f"Attack={e.get('attack','')} | Family={e.get('family','')} | Perturbation={e.get('perturbation_search','')} | Defence={e.get('defence','')}"
    if rich and e.get("defence_mechanism"):
        line += f" | Defence detail={e['defence_mechanism'][:160]}"
    return line

def embed(texts):
    out = []
    for i in range(0, len(texts), 100):
        r = oc.embeddings.create(model=EMBED, input=texts[i:i+100])
        out.extend(d.embedding for d in r.data)
    m = np.array(out, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True); n[n == 0] = 1
    return m / n

def query_text(row):
    return (f"{row['attack_name']}. {row['description']} "
            f"Type: {row['attack_type']}. Family: {row['family']}. "
            f"Threat model: {row['threat_model']}. Visibility: {row['visibility']}. "
            f"Search: {row['perturbation_search']}.")

def predict(row, context):
    p = (f"You are an adversarial machine learning expert. Predict the best "
         f"defence(s) for the following attack.\n\n"
         f"Attack Name: {row['attack_name']}\nDescription: {row['description']}\n"
         f"Attack Type: {row['attack_type']}\nFamily: {row['family']}\n"
         f"Threat Model: {row['threat_model']}\nVisibility: {row['visibility']}\n"
         f"Perturbation Search: {row['perturbation_search']}\n")
    if context:
        p += "\nRoadmap context (similar attacks and their defences):\n" + context + "\n"
    p += ('\nReturn ONLY JSON, no prose: {"defences": ["...", "...", "..."]} '
          'ordered best-first (up to 5).')
    try:
        r = oc.chat.completions.create(model=MODEL, temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": "You are an adversarial ML expert. Return only JSON."},
                          {"role": "user", "content": p}])
        d = json.loads(r.choices[0].message.content)
        return [str(x) for x in d.get("defences", [])][:5]
    except Exception as e:
        return []

def main():
    df = pd.read_csv(CSV)
    df["gt"] = df["ground_truth_defence"].apply(lambda x: [s.strip() for s in str(x).split("|") if s.strip()])
    base = json.loads(Path(BASE).read_text(encoding="utf-8"))
    plus = json.loads(Path(PLUS).read_text(encoding="utf-8"))

    base_docs = [kb_doc(e, False) for e in base]
    plus_docs = [kb_doc(e, True)  for e in plus]
    bm = embed(base_docs); pm = embed(plus_docs)
    qm = embed([query_text(r) for _, r in df.iterrows()])

    def ctx(qv, mat, kb, rich):
        idx = np.argsort(-(mat @ qv))[:TOPK]
        return "\n".join(kb_line(kb[i], rich) for i in idx)

    conds = ["no_context", "base_roadmap", "roadmap_plus"]
    hits = {c: {"top1": [], "top3": []} for c in conds}
    rows_out = []
    for i, (_, row) in enumerate(df.iterrows()):
        for c in conds:
            context = "" if c == "no_context" else (
                ctx(qm[i], bm, base, False) if c == "base_roadmap" else ctx(qm[i], pm, plus, True))
            preds = predict(row, context)
            gt = row["gt"]
            t1 = int(bool(preds) and match(preds[0], gt))
            t3 = int(any(match(p, gt) for p in preds[:3]))
            hits[c]["top1"].append(t1); hits[c]["top3"].append(t3)
            rows_out.append({"attack": row["attack_name"], "condition": c,
                             "gt": "|".join(gt), "pred": "|".join(preds), "top1": t1, "top3": t3})
        print(f"  [{i+1}/{len(df)}] {row['attack_name'][:40]}")

    Path("outputs/defence_bench_fixed.json").write_text(
        json.dumps(rows_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Corrected defence-recommendation benchmark (N=%d attacks, model=%s) ===" % (len(df), MODEL))
    print(f"{'Condition':16}{'Top-1':>8}{'Top-3':>8}")
    for c in conds:
        print(f"{c:16}{np.mean(hits[c]['top1'])*100:>7.1f}%{np.mean(hits[c]['top3'])*100:>7.1f}%")

if __name__ == "__main__":
    main()
