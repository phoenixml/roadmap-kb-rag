"""
Fill the full 3-condition x 4-metric table under the IMPROVED retrieval methodology
(full rich KB, text-embedding-3-large, top-5, GPT-4.1 judge), for the 9 models in the
paper's Table 8.

  base_roadmap  = retrieval over Init_Unified KB (256 attacks, pre-expansion, no defence)
  roadmap_plus  = retrieval over Unified KB      (317 attacks, +defence, richer)  [== old full_kb]
  no_context    = no retrieval

Reuses the already-computed rows from outputs/qa_eval_atlas_improved.json
(6 models x {no_context, full_kb}) so we only compute what's missing:
  - base_roadmap for all 9 models
  - no_context + roadmap_plus for claude-sonnet-4-6, claude-opus-4-8, gpt-4.1-mini
Judge = gpt-4.1 (same as the reused rows). Resumable; saves every 20 rows.
Output: outputs/qa_eval_atlas_fulltable.json
"""
import os, json, time, re, socket, urllib.request
from pathlib import Path

socket.setdefaulttimeout(240)  # tolerate 8B cold-loads (~88s) but kill true multi-minute hangs
from collections import defaultdict
import numpy as np
import _rerun_improved as R   # build_docs, embed, judge, rl, call (openai), SYS, TOPK, EMBED

import anthropic
ac = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
OLLAMA = "https://ollama-llama-349218458434.us-west1.run.app"

QA        = "outputs/atlas_qa.json"
BASE_KB   = "Init_Unified_Attack_Knowledge_Base.json"   # pre-expansion
PLUS_KB   = "Unified_Attack_Knowledge_Base.json"        # Roadmap+
IMPROVED  = "outputs/qa_eval_atlas_improved.json"        # reuse source
OUT       = "outputs/qa_eval_atlas_fulltable.json"

# provider routing. Order: fully-missing API models first, then base_roadmap fills,
# Ollama (slow) last with the smallest models first.
MODELS = {
    "gpt-4.1-mini":      ("openai",    "gpt-4.1-mini"),
    "claude-sonnet-4-6": ("anthropic", "claude-sonnet-4-6"),
    "claude-opus-4-8":   ("anthropic", "claude-opus-4-8"),
    "gpt-4.1":           ("openai",    "gpt-4.1"),
    "gpt-4o-mini":       ("openai",    "gpt-4o-mini"),
    "gemma3:1b":         ("ollama",    "gemma3:1b"),
    "llama3.2:1b":       ("ollama",    "llama3.2:1b"),
    "qwen2.5:0.5b":      ("ollama",    "qwen2.5:0.5b"),
    "smollm2:360m":      ("ollama",    "smollm2:360m"),
    "gemma3":            ("ollama",    "gemma3:latest"),
    "llama3.1:8b":       ("ollama",    "llama3.1:8b"),
}
CONDS = ["no_context", "base_roadmap", "roadmap_plus"]


def call_ollama(mid, system, user):
    payload = json.dumps({"model": mid, "prompt": f"{system}\n\n{user}", "stream": False,
                          "options": {"temperature": 0.0, "num_predict": 300}}).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            t = json.loads(r.read()).get("response", "").strip()
        return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def call_anthropic(mid, system, user):
    for a in range(5):
        try:
            r = ac.messages.create(model=mid, max_tokens=300, system=system,
                                   messages=[{"role": "user", "content": user}])
            return r.content[0].text.strip()
        except Exception as e:
            if a == 4:
                return f"[ERROR: {e}]"
            time.sleep(2 * (a + 1))


def generate(key, system, user):
    prov, mid = MODELS[key]
    if prov == "openai":
        return R.call(mid, system, user)
    if prov == "anthropic":
        return call_anthropic(mid, system, user)
    return call_ollama(mid, system, user)


def prompt(cond, q, ctx):
    if cond == "no_context":
        return R.SYS, f"Question: {q}\n\nAnswer:"
    system = R.SYS + "\n\nUse the retrieved knowledge-base entries below to answer."
    return system, f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"


def main():
    qa = json.loads(Path(QA).read_text(encoding="utf-8"))

    # --- build retrieval for base and plus (same improved methodology) ---
    _, base_docs = R.build_docs(json.loads(Path(BASE_KB).read_text(encoding="utf-8")))
    _, plus_docs = R.build_docs(json.loads(Path(PLUS_KB).read_text(encoding="utf-8")))
    print(f"[setup] base docs {len(base_docs)}  plus docs {len(plus_docs)}  embed={R.EMBED} k={R.TOPK}")
    base_m = R.embed(base_docs); plus_m = R.embed(plus_docs)
    qm = R.embed([q["question"] for q in qa])
    ctx_base = ["\n\n---\n\n".join(base_docs[j] for j in np.argsort(-(base_m @ qm[i]))[:R.TOPK]) for i in range(len(qa))]
    ctx_plus = ["\n\n---\n\n".join(plus_docs[j] for j in np.argsort(-(plus_m @ qm[i]))[:R.TOPK]) for i in range(len(qa))]
    print("[setup] retrieval ready.")

    # --- seed output: merge in reusable improved rows (full_kb -> roadmap_plus) for
    #     every model in MODELS, idempotently (safe to re-run after adding models) ---
    results = json.loads(Path(OUT).read_text(encoding="utf-8")) if Path(OUT).exists() else []
    done = {(r["model"], r["condition"], r["id"]) for r in results}
    keep = set(MODELS)
    imp = json.loads(Path(IMPROVED).read_text(encoding="utf-8"))
    added = 0
    for r in imp:
        if r["model"] in keep and r["condition"] in ("no_context", "full_kb"):
            cond = "roadmap_plus" if r["condition"] == "full_kb" else r["condition"]
            if (r["model"], cond, r["id"]) not in done:
                r = dict(r); r["condition"] = cond
                results.append(r); done.add((r["model"], cond, r["id"])); added += 1
    if added:
        Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[seed] merged {added} reusable rows from improved run")
    total = len(qa) * len(MODELS) * len(CONDS)
    print(f"[run] total {total}  done {len(done)}  remaining {total - len(done)}")

    for key in MODELS:
        for c in CONDS:
            todo = [q for q in qa if (key, c, q["id"]) not in done]
            if not todo:
                continue
            print(f"  {key} / {c}  ({len(todo)} to do)")
            for i, q in enumerate(qa):
                if (key, c, q["id"]) in done:
                    continue
                ctx = "" if c == "no_context" else (ctx_base[i] if c == "base_roadmap" else ctx_plus[i])
                system, user = prompt(c, q["question"], ctx)
                pred = generate(key, system, user)
                j = R.judge(q["question"], q["answer"], pred)
                results.append({"id": q["id"], "model": key, "condition": c, "predicted": pred,
                                "reference": q["answer"], "llm_judge": j,
                                "exact": 1 if j == 3 else 0, "rouge_l": R.rl(pred, q["answer"])})
                done.add((key, c, q["id"]))
                if len(results) % 20 == 0:
                    Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"    saved {len(results)}/{total}")
                time.sleep(0.15)
            Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    Path(OUT).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    # quick summary
    agg = defaultdict(lambda: {"e": 0, "n": 0, "j": []})
    for r in results:
        a = agg[(r["model"], r["condition"])]; a["e"] += r["exact"]; a["n"] += 1; a["j"].append(r["llm_judge"])
    print("\n=== EM / Judge by model x condition ===")
    for key in MODELS:
        for c in CONDS:
            a = agg[(key, c)]
            if a["n"]:
                print(f"  {key:20}{c:14} EM {a['e']/a['n']*100:5.1f}  Judge {sum(a['j'])/a['n']:.2f}  (n={a['n']})")
    print("[fulltable] done.")


if __name__ == "__main__":
    main()
