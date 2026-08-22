"""Extend the IMPROVED-retrieval run to the open/small models (Ollama),
appending to outputs/qa_eval_atlas_improved.json. Reuses the pilot's retrieval."""
import _rerun_improved as R
import urllib.request, json, re, time
import numpy as np
from pathlib import Path

OLLAMA = "https://ollama-llama-349218458434.us-west1.run.app"
OPEN = {"gemma3:1b": "gemma3:1b", "llama3.2:1b": "llama3.2:1b",
        "qwen2.5:0.5b": "qwen2.5:0.5b", "smollm2:360m": "smollm2:360m",
        "gemma3": "gemma3:latest", "llama3.1:8b": "llama3.1:8b"}

def call_ollama(mid, system, user):
    payload = json.dumps({"model": mid, "prompt": f"{system}\n\n{user}", "stream": False,
                          "options": {"temperature": 0.0, "num_predict": 300}}).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            t = json.loads(r.read()).get("response", "").strip()
        return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    except Exception as e:
        return f"[ERROR: {e}]"

qa = json.loads(Path(R.QA).read_text(encoding="utf-8"))
kb = json.loads(Path(R.KB).read_text(encoding="utf-8"))
names, docs = R.build_docs(kb)
print(f"[improved-open] KB docs: {len(docs)}  embedding...")
kbm = R.embed(docs); qm = R.embed([q["question"] for q in qa])
ctx = ["\n\n---\n\n".join(docs[j] for j in np.argsort(-(kbm @ qm[i]))[:R.TOPK]) for i in range(len(qa))]
print("[improved-open] retrieval ready.")

OUT = R.OUT
results = json.loads(Path(OUT).read_text(encoding="utf-8")) if Path(OUT).exists() else []
done = {(r["model"], r["condition"], r["id"]) for r in results}

for key, mid in OPEN.items():
    for c in ["no_context", "full_kb"]:
        print(f"  {key} / {c}")
        for i, q in enumerate(qa):
            if (key, c, q["id"]) in done:
                continue
            if c == "no_context":
                system, user = R.SYS, f"Question: {q['question']}\n\nAnswer:"
            else:
                system = R.SYS + "\n\nUse the retrieved knowledge-base entries below to answer."
                user = f"Context:\n{ctx[i]}\n\nQuestion: {q['question']}\n\nAnswer:"
            pred = call_ollama(mid, system, user)
            j = R.judge(q["question"], q["answer"], pred)
            results.append({"id": q["id"], "model": key, "condition": c, "predicted": pred,
                            "reference": q["answer"], "llm_judge": j,
                            "exact": 1 if j == 3 else 0, "rouge_l": R.rl(pred, q["answer"])})
            done.add((key, c, q["id"]))
            if len(results) % 20 == 0:
                Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"    saved {len(results)}")
            time.sleep(0.15)
        Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("[improved-open] done.")
