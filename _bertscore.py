"""Compute real BERTScore (roberta-large F1) on the CORRECTED predictions,
for all 11 models x 3 conditions, so every metric comes from the same run."""
import json, collections
from pathlib import Path
from bert_score import score

FILES = ["outputs/qa_eval_results_atlas_fixed.json",
         "outputs/qa_eval_results_atlas_extra.json"]
rows = []
for f in FILES:
    rows.extend(json.loads(Path(f).read_text(encoding="utf-8")))

def clean(p):
    p = str(p).strip()
    if not p or p.startswith("[ERROR"):
        return "N/A"
    return p

cands = [clean(r.get("predicted", "")) for r in rows]
refs  = [str(r.get("reference", "")) or "N/A" for r in rows]
print(f"[bertscore] pairs: {len(cands)}  (11 models x 3 conditions x 120)")

P, R, F1 = score(cands, refs, model_type="roberta-large", lang="en",
                 verbose=True, batch_size=64)
f1 = F1.tolist()

agg = collections.defaultdict(list)
for r, v in zip(rows, f1):
    agg[(r["model"], r["condition"])].append(v)

order_m = ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5",
           "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini", "gpt-3.5-turbo",
           "gemma3", "llama3.1:8b", "gemma3:1b", "llama3.2:1b"]
order_c = ["no_context", "base_roadmap", "roadmap_plus"]

out = {}
print("\n=== BERTScore F1 (roberta-large) on corrected predictions ===")
print(f"{'Model':20}{'no_ctx':>9}{'base':>9}{'plus':>9}")
for m in order_m:
    vals = []
    for c in order_c:
        v = sum(agg[(m, c)]) / len(agg[(m, c)]) if agg[(m, c)] else float("nan")
        vals.append(v); out[f"{m}|{c}"] = round(v, 4)
    print(f"{m:20}" + "".join(f"{v:>9.4f}" for v in vals))

# means across the 11 models
print("\n--- mean across 11 models ---")
for c in order_c:
    allv = [x for m in order_m for x in agg[(m, c)]]
    print(f"  {c:14} {sum(allv)/len(allv):.4f}")

Path("outputs/bertscore_corrected.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("\n[OK] saved -> outputs/bertscore_corrected.json")
