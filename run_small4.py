"""Run the ATLAS-QA RAG experiment on 4 NEW small models, same methodology as the
main table: 3 conditions (no_context / base_roadmap / roadmap_plus), full rich KB,
text-embedding-3-large top-5, GPT-4.1 judge. Reports ALL 4 models honestly.
Output: outputs/qa_eval_small4.json  (resumable, saves every 10 rows).
"""
import json, time, socket, re, threading, urllib.request
from pathlib import Path
from collections import defaultdict
import numpy as np
import _rerun_improved as R   # build_docs, embed, judge, rl, SYS, TOPK, EMBED

socket.setdefaulttimeout(200)
OLLAMA = "https://ollama-llama-349218458434.us-west1.run.app"
QA      = "outputs/atlas_qa.json"
BASE_KB = "Init_Unified_Attack_Knowledge_Base.json"   # pre-expansion (base)
PLUS_KB = "Unified_Attack_Knowledge_Base.json"        # Roadmap+
OUT     = "outputs/qa_eval_small4.json"
MODELS  = ["qwen2.5:1.5b", "gemma2:2b", "llama3.2:3b", "smollm2:1.7b"]
CONDS   = ["no_context", "base_roadmap", "roadmap_plus"]


def _once(mid, prompt, box):
    payload = json.dumps({"model": mid, "prompt": prompt, "stream": False,
                          "keep_alive": "20m",
                          "options": {"temperature": 0.0, "num_predict": 300}}).encode("utf-8")
    try:
        req = urllib.request.Request(f"{OLLAMA}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=200) as r:
            box["r"] = json.loads(r.read()).get("response", "").strip()
    except Exception as e:
        box["e"] = str(e)


def call_ollama(mid, system, user):
    prompt = f"{system}\n\n{user}"
    for attempt in range(5):
        box = {}
        th = threading.Thread(target=_once, args=(mid, prompt, box), daemon=True)
        th.start(); th.join(210)
        if not th.is_alive() and "r" in box:
            return re.sub(r"<think>.*?</think>", "", box["r"], flags=re.DOTALL).strip()
        if attempt == 4:
            return f"[ERROR: {box.get('e', 'timeout/wedged')}]"
        time.sleep(2)


def prompt(cond, q, ctx):
    if cond == "no_context":
        return R.SYS, f"Question: {q}\n\nAnswer:"
    system = R.SYS + "\n\nUse the retrieved knowledge-base entries below to answer."
    return system, f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"


def main():
    qa = json.loads(Path(QA).read_text(encoding="utf-8"))
    _, base_docs = R.build_docs(json.loads(Path(BASE_KB).read_text(encoding="utf-8")))
    _, plus_docs = R.build_docs(json.loads(Path(PLUS_KB).read_text(encoding="utf-8")))
    print(f"[setup] base {len(base_docs)} / plus {len(plus_docs)} docs; embed={R.EMBED} k={R.TOPK}", flush=True)
    bm = R.embed(base_docs); pm = R.embed(plus_docs); qm = R.embed([q["question"] for q in qa])
    ctx_base = ["\n\n---\n\n".join(base_docs[j] for j in np.argsort(-(bm @ qm[i]))[:R.TOPK]) for i in range(len(qa))]
    ctx_plus = ["\n\n---\n\n".join(plus_docs[j] for j in np.argsort(-(pm @ qm[i]))[:R.TOPK]) for i in range(len(qa))]
    print("[setup] retrieval ready.", flush=True)

    results = json.loads(Path(OUT).read_text(encoding="utf-8")) if Path(OUT).exists() else []
    done = {(r["model"], r["condition"], r["id"]) for r in results}
    total = len(qa) * len(MODELS) * len(CONDS)
    print(f"[run] total {total}  done {len(done)}", flush=True)

    for key in MODELS:
        # warm the model once
        _ = call_ollama(key, "You are terse.", "Reply with the single word OK.")
        for c in CONDS:
            todo = [q for q in qa if (key, c, q["id"]) not in done]
            if not todo:
                continue
            print(f"  {key} / {c}  ({len(todo)} to do)", flush=True)
            t0 = time.time(); n = 0
            for i, q in enumerate(qa):
                if (key, c, q["id"]) in done:
                    continue
                ctx = "" if c == "no_context" else (ctx_base[i] if c == "base_roadmap" else ctx_plus[i])
                system, user = prompt(c, q["question"], ctx)
                pred = call_ollama(key, system, user)
                j = R.judge(q["question"], q["answer"], pred)
                results.append({"id": q["id"], "model": key, "condition": c, "predicted": pred,
                                "reference": q["answer"], "llm_judge": j,
                                "exact": 1 if j == 3 else 0, "rouge_l": R.rl(pred, q["answer"])})
                done.add((key, c, q["id"])); n += 1
                if len(results) % 10 == 0:
                    Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"    {key}/{c} {n}/{len(todo)}  ({(time.time()-t0)/n:.1f}s/row)", flush=True)
            Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    agg = defaultdict(lambda: {"e": 0, "n": 0, "j": []})
    for r in results:
        a = agg[(r["model"], r["condition"])]; a["e"] += r["exact"]; a["n"] += 1; a["j"].append(r["llm_judge"])
    print("\n=== EM / Judge (No-Ctx / Base / Roadmap+) ===", flush=True)
    for key in MODELS:
        row = f"  {key:16}"
        for c in CONDS:
            a = agg[(key, c)]
            row += f"  {c[:4]}: EM {a['e']/a['n']*100:4.1f} J {sum(a['j'])/a['n']:.2f}" if a["n"] else f"  {c[:4]}: --"
        print(row, flush=True)
    print("[small4] done.", flush=True)


if __name__ == "__main__":
    main()
