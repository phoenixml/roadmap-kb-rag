"""
Lever #1: IMPROVED retrieval for ATLAS-QA.
  - retrieval source = full Unified KB (317 unique attacks, rich explanation/formula/defence)
  - embeddings = text-embedding-3-large
  - top-5, context de-duplicated by attack
Pilot models: gpt-4.1 (context hurt it most before) + gpt-4o-mini.
Conditions: no_context vs full_kb. Judge = gpt-4.1.
Reports honestly whether better retrieval moves the numbers.
"""
import os, json, re, time, collections
from pathlib import Path
import numpy as np

_env = Path(".env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
from openai import OpenAI
oc = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from rouge_score import rouge_scorer
_rl = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

EMBED = "text-embedding-3-large"; TOPK = 5
QA = "outputs/atlas_qa.json"; KB = "Unified_Attack_Knowledge_Base.json"
OUT = "outputs/qa_eval_atlas_improved.json"
MODELS = ["gpt-4.1", "gpt-4o-mini"]
CONDS = ["no_context", "full_kb"]

SYS = ("You are an expert in adversarial machine learning, AI security, and the "
       "MITRE ATLAS framework. Answer concisely and precisely. If unsure, say 'Unknown'.")

def defence_str(d):
    if isinstance(d, dict):
        s = d.get("defence_name", "")
        if d.get("mechanism"): s += f" -- {str(d['mechanism'])[:180]}"
        return s
    return str(d) if d else ""

def build_docs(kb):
    by = collections.OrderedDict()
    for e in kb:
        n = e.get("attack_name", "").strip()
        if not n: continue
        cur = by.get(n, {"expl": "", "form": "", "def": "", "fam": ""})
        ex = str(e.get("explanation", "") or "")
        if len(ex) > len(cur["expl"]): cur["expl"] = ex
        if not cur["form"] and e.get("formula"): cur["form"] = str(e["formula"])
        if not cur["def"]:
            ds = defence_str(e.get("defence"))
            if ds: cur["def"] = ds
        gn = e.get("graph_node")
        if not cur["fam"] and isinstance(gn, dict) and gn.get("type"): cur["fam"] = gn["type"]
        by[n] = cur
    names, docs = [], []
    for n, c in by.items():
        parts = [f"Attack: {n}"]
        if c["fam"]:  parts.append(f"Family: {c['fam']}")
        if c["expl"]: parts.append(f"Explanation: {c['expl'][:450]}")
        if c["form"]: parts.append(f"Formula: {c['form'][:200]}")
        if c["def"]:  parts.append(f"Defence: {c['def']}")
        names.append(n); docs.append("\n".join(parts))
    return names, docs

def embed(texts):
    out = []
    for i in range(0, len(texts), 100):
        r = oc.embeddings.create(model=EMBED, input=texts[i:i+100])
        out.extend(d.embedding for d in r.data)
    m = np.array(out, dtype=np.float32); nn = np.linalg.norm(m, axis=1, keepdims=True); nn[nn==0]=1
    return m / nn

def call(model, system, user):
    try:
        r = oc.chat.completions.create(model=model, temperature=0.0, max_tokens=300,
            messages=[{"role":"system","content":system},{"role":"user","content":user}])
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {e}]"

def judge(q, ref, pred):
    p=(f"You are evaluating an AI answer about adversarial ML and MITRE ATLAS.\n\n"
       f"Question  : {q}\nReference : {ref}\nPredicted : {pred}\n\n"
       "Rate 0-3 (3=correct&complete, 0=wrong). Respond with a single digit 0,1,2,3.")
    r = call("gpt-4.1", "You are a strict academic evaluator.", p)
    m = re.search(r"[0-3]", r); return int(m.group()) if m else 0

def rl(p, r):
    try: return round(_rl.score(r, p)["rougeL"].fmeasure, 3)
    except: return 0.0

def main():
    qa = json.loads(Path(QA).read_text(encoding="utf-8"))
    kb = json.loads(Path(KB).read_text(encoding="utf-8"))
    names, docs = build_docs(kb)
    print(f"[improved] KB docs (unique attacks): {len(docs)}  embed={EMBED}  k={TOPK}")
    kbm = embed(docs)
    qm  = embed([q["question"] for q in qa])
    ctx = []
    for i in range(len(qa)):
        idx = np.argsort(-(kbm @ qm[i]))[:TOPK]
        ctx.append("\n\n---\n\n".join(docs[j] for j in idx))
    print("[improved] retrieval ready.")

    results = []
    for m in MODELS:
        for c in CONDS:
            print(f"  {m} / {c}")
            for i, q in enumerate(qa):
                if c == "no_context":
                    system, user = SYS, f"Question: {q['question']}\n\nAnswer:"
                else:
                    system = SYS + "\n\nUse the retrieved knowledge-base entries below to answer."
                    user = f"Context:\n{ctx[i]}\n\nQuestion: {q['question']}\n\nAnswer:"
                pred = call(m, system, user)
                j = judge(q["question"], q["answer"], pred)
                results.append({"id": q["id"], "model": m, "condition": c,
                    "predicted": pred, "reference": q["answer"],
                    "llm_judge": j, "exact": 1 if j==3 else 0, "rouge_l": rl(pred, q["answer"])})
                time.sleep(0.15)
            Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    agg = collections.defaultdict(lambda: {"j":[], "e":0, "n":0})
    for r in results:
        a=agg[(r["model"], r["condition"])]; a["j"].append(r["llm_judge"]); a["e"]+=r["exact"]; a["n"]+=1
    print("\n=== IMPROVED retrieval (full KB, embed-3-large, top-5) ===")
    prev = {"gpt-4.1": (46.7, 28.3), "gpt-4o-mini": (15.8, 18.3)}  # (no_ctx, old base) for reference
    for m in MODELS:
        for c in CONDS:
            a=agg[(m,c)]; n=a["n"]
            print(f"  {m:14}{c:12} EM {a['e']/n*100:5.1f}  Judge {sum(a['j'])/n:.2f}")
        print(f"     (prior: {m} no_ctx {prev[m][0]}, OLD base_roadmap {prev[m][1]})")

if __name__ == "__main__":
    main()
