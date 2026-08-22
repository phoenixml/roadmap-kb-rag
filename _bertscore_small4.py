import json, collections
from pathlib import Path
import torch; torch.set_num_threads(2)
from bert_score import score
rows = json.loads(Path("outputs/qa_eval_small4.json").read_text(encoding="utf-8"))
def clean(p):
    p=str(p).strip(); return "N/A" if (not p or p.startswith("[ERROR")) else p
cands=[clean(r.get("predicted","")) for r in rows]; refs=[str(r.get("reference",""))or"N/A" for r in rows]
print("[bertscore-small4] pairs:", len(cands), flush=True)
P,R,F1=score(cands,refs,model_type="roberta-base",lang="en",verbose=True,batch_size=8)
agg=collections.defaultdict(list)
for r,v in zip(rows,F1.tolist()): agg[(r["model"],r["condition"])].append(v)
out={f"{m}|{c}":round(sum(v)/len(v),4) for (m,c),v in agg.items()}
Path("outputs/bertscore_small4.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print("[OK] saved outputs/bertscore_small4.json", flush=True)
for k in sorted(out): print(" ",k,out[k])
