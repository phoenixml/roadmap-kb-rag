"""BERTScore (roberta-large F1) on the full-table predictions, all 9 models x 3
conditions, so every metric in the table comes from the same predictions."""
import json, collections, sys
from pathlib import Path
import torch
torch.set_num_threads(2)
from bert_score import score

rows = json.loads(Path("outputs/qa_eval_atlas_fulltable.json").read_text(encoding="utf-8"))

def clean(p):
    p = str(p).strip()
    return "N/A" if (not p or p.startswith("[ERROR")) else p

cands = [clean(r.get("predicted", "")) for r in rows]
refs  = [str(r.get("reference", "")) or "N/A" for r in rows]
print(f"[bertscore] pairs: {len(cands)}")

used = None
for mt in ["roberta-base"]:   # roberta-large OOMs on this machine (~1.1GB free); base is a valid, consistent config
    try:
        print(f"[bertscore] trying model_type={mt} ...", flush=True)
        P, Rc, F1 = score(cands, refs, model_type=mt, lang="en", verbose=True, batch_size=8)
        used = mt
        break
    except Exception as e:
        print(f"[bertscore] {mt} failed: {e}", flush=True)
if used is None:
    print("[bertscore] all models failed"); sys.exit(1)
print(f"[bertscore] SUCCESS with {used}")
Path("outputs/bertscore_fulltable_model.txt").write_text(used, encoding="utf-8")
f1 = F1.tolist()

agg = collections.defaultdict(list)
for r, v in zip(rows, f1):
    agg[(r["model"], r["condition"])].append(v)

out = {}
for (m, c), vs in agg.items():
    out[f"{m}|{c}"] = round(sum(vs) / len(vs), 4)
Path("outputs/bertscore_fulltable.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("[OK] saved -> outputs/bertscore_fulltable.json")
for k in sorted(out):
    print(f"  {k:38} {out[k]}")
