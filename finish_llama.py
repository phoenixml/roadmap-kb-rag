"""Finish the one remaining cell: llama3.1:8b / base_roadmap (120 rows).
Keeps the model warm (keep_alive) so only the first call pays the cold-load cost.
Saves every 5 rows; retries a hung/failed call a few times."""
import json, time, socket, re, threading, urllib.request
from pathlib import Path
import numpy as np
import _rerun_improved as R

socket.setdefaulttimeout(300)
OLLAMA = "https://ollama-llama-349218458434.us-west1.run.app"
OUT = "outputs/qa_eval_atlas_fulltable.json"
MODEL = "llama3.1:8b"

def _once(payload, box):
    try:
        req = urllib.request.Request(f"{OLLAMA}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=150) as r:
            box["r"] = json.loads(r.read()).get("response", "").strip()
    except Exception as e:
        box["e"] = str(e)

def call(system, user):
    payload = json.dumps({"model": MODEL, "prompt": f"{system}\n\n{user}", "stream": False,
                          "keep_alive": "20m", "options": {"temperature": 0.0, "num_predict": 300}}).encode("utf-8")
    for attempt in range(5):
        box = {}
        th = threading.Thread(target=_once, args=(payload, box), daemon=True)
        th.start(); th.join(280)                      # hard wall-clock cap per try
        if not th.is_alive() and "r" in box:
            return re.sub(r"<think>.*?</think>", "", box["r"], flags=re.DOTALL).strip()
        # wedged socket or error -> abandon this try, retry with a fresh socket
        if attempt == 4:
            return f"[ERROR: {box.get('e', 'timeout/wedged')}]"
        time.sleep(2)

qa = json.loads(Path("outputs/atlas_qa.json").read_text(encoding="utf-8"))
_, base_docs = R.build_docs(json.loads(Path("Init_Unified_Attack_Knowledge_Base.json").read_text(encoding="utf-8")))
print(f"[finish] base docs {len(base_docs)}  embedding...")
bm = R.embed(base_docs); qm = R.embed([q["question"] for q in qa])
ctx = ["\n\n---\n\n".join(base_docs[j] for j in np.argsort(-(bm @ qm[i]))[:R.TOPK]) for i in range(len(qa))]
print("[finish] retrieval ready.")

results = json.loads(Path(OUT).read_text(encoding="utf-8"))
done = {(r["model"], r["condition"], r["id"]) for r in results}
todo = [i for i, q in enumerate(qa) if (MODEL, "base_roadmap", q["id"]) not in done]
print(f"[finish] rows to do: {len(todo)}")

sys_ctx = R.SYS + "\n\nUse the retrieved knowledge-base entries below to answer."
print("[finish] pre-warming model ...")
print("[finish] warm:", call("You are terse.", "Reply with the single word OK.")[:20])
t0 = time.time()
for n, i in enumerate(todo, 1):
    q = qa[i]
    pred = call(sys_ctx, f"Context:\n{ctx[i]}\n\nQuestion: {q['question']}\n\nAnswer:")
    j = R.judge(q["question"], q["answer"], pred)
    results.append({"id": q["id"], "model": MODEL, "condition": "base_roadmap", "predicted": pred,
                    "reference": q["answer"], "llm_judge": j, "exact": 1 if j == 3 else 0,
                    "rouge_l": R.rl(pred, q["answer"])})
    if n % 5 == 0 or n == len(todo):
        Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        rate = (time.time() - t0) / n
        print(f"  {n}/{len(todo)} done  ({rate:.1f}s/row, ETA {rate*(len(todo)-n)/60:.1f} min)")
print("[finish] DONE.")
