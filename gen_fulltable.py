"""Generate Table 8 in Paper13's exact structure (scriptsize, multirow, 3 groups,
3 conditions, 4 metrics), populated with the honest full-table run
(embed-3-large, top-5, GPT-4.1 judge) + BERTScore. 11 models.
Bold marks the best value per model in each metric column (across all 3 conditions).
Writes Latex-Paper/table8_fulltable.tex and prints it."""
import json, collections
from pathlib import Path

rows = json.loads(Path("outputs/qa_eval_atlas_fulltable.json").read_text(encoding="utf-8"))
bs = json.loads(Path("outputs/bertscore_fulltable.json").read_text(encoding="utf-8"))
BS_MODEL = Path("outputs/bertscore_fulltable_model.txt").read_text(encoding="utf-8").strip() \
    if Path("outputs/bertscore_fulltable_model.txt").exists() else "roberta-base"

agg = collections.defaultdict(lambda: {"r": [], "j": [], "e": 0, "n": 0})
for r in rows:
    a = agg[(r["model"], r["condition"])]
    a["r"].append(r["rouge_l"]); a["j"].append(r["llm_judge"]); a["e"] += r["exact"]; a["n"] += 1

def ready(m, c):
    return agg[(m, c)]["n"] >= 120

def M(m, c):
    a = agg[(m, c)]; n = a["n"] or 1
    return (sum(a["r"]) / n, bs.get(f"{m}|{c}", float("nan")), sum(a["j"]) / n, a["e"] / n * 100)

groups = [
    ("Proprietary Models (API)",
     [("Claude Sonnet 4.6", "claude-sonnet-4-6"), ("Claude Opus 4.8", "claude-opus-4-8"),
      ("GPT-4.1", "gpt-4.1"), ("GPT-4.1-mini", "gpt-4.1-mini"), ("GPT-4o-mini", "gpt-4o-mini")]),
    ("Open-Source Models --- Large (Ollama Cloud)",
     [("Gemma3 (4.3B)", "gemma3"), ("Llama 3.1 (8B)", "llama3.1:8b")]),
    ("Open-Source Models --- Small / Edge (Ollama Cloud)",
     [("Gemma3 (1B)", "gemma3:1b"), ("Llama 3.2 (1B)", "llama3.2:1b"),
      ("Qwen2.5 (0.5B)", "qwen2.5:0.5b"), ("SmolLM2 (360M)", "smollm2:360m")]),
]
all_models = [(d, k) for _, ms in groups for d, k in ms]
conds = [("No Context \\textit{(Baseline)}", "no_context"),
         ("Base Roadmap \\textit{(RAG)}", "base_roadmap"),
         ("Roadmap$^+$ \\textit{(RAG)}", "roadmap_plus")]
specs = ["%.3f", "%.4f", "%.2f", "%.1f"]
digits = [3, 4, 2, 1]

L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{RAG evaluation on the independent ATLAS-QA benchmark (MITRE ATLAS v5.6.0),",
     r"using high-quality semantic retrieval (full 317-attack knowledge base,",
     r"\texttt{text-embedding-3-large}, top-5) and a GPT-4.1 judge. \textit{Base Roadmap}",
     r"retrieves over the pre-expansion catalogue (256 attacks); \textit{Roadmap}$^+$ over",
     r"the LLM-expanded catalogue (317 attacks, with added defences). Bold marks the best",
     r"value per model in each metric column.}",
     r"\label{tab:atlas_evaluation}", r"\scriptsize",
     r"\setlength{\tabcolsep}{4.5pt}", r"\renewcommand{\arraystretch}{1.2}",
     r"\begin{threeparttable}", r"\begin{tabular}{l l c c c c}", r"\toprule",
     r"\textbf{Model} & \textbf{Condition}", r"  & \textbf{ROUGE-L}",
     r"  & \textbf{BERTScore}", r"  & \textbf{LLM Judge (0--3)}",
     r"  & \textbf{Exact Match (\%)} \\", r"\midrule"]

for gi, (gname, models) in enumerate(groups):
    if gi > 0:
        L.append(r"\midrule")
    L.append(r"\multicolumn{6}{l}{\textit{%s}} \\" % gname)
    L.append(r"\addlinespace[2pt]")
    L.append("")
    for mi, (disp, key) in enumerate(models):
        cells, avail = [], []
        for _, ck in conds:
            if ready(key, ck):
                v = M(key, ck)
                cells.append([round(v[i], digits[i]) for i in range(4)]); avail.append(True)
            else:
                cells.append(None); avail.append(False)
        # per-column max only over available conditions
        maxes = [max(cells[ci][i] for ci in range(len(conds)) if avail[ci]) for i in range(4)]
        L.append(r"\multirow{3}{*}{\textbf{%s}}" % disp)
        for ci, (clabel, ck) in enumerate(conds):
            if not avail[ci]:
                L.append(r"  & %s & \multicolumn{4}{c}{\textit{--- pending ---}} \\" % clabel)
                continue
            parts = []
            for i in range(4):
                s = specs[i] % cells[ci][i]
                if abs(cells[ci][i] - maxes[i]) < 1e-9:
                    s = r"\textbf{%s}" % s
                parts.append(s)
            L.append(r"  & %s & %s & %s & %s & %s \\" % (clabel, parts[0], parts[1], parts[2], parts[3]))
        if mi < len(models) - 1:
            L.append(r"\addlinespace[3pt]")
        L.append("")

def complete(key):
    return all(ready(key, c) for _, c in conds)

def mean(mi, ck):
    # average only over models complete in ALL conditions -> fair base/plus/no-ctx comparison
    vals = [M(key, ck)[mi] for _, key in all_models if complete(key)]
    return sum(vals) / len(vals)

N_MEAN = sum(1 for _, key in all_models if complete(key))

em = [mean(3, c) for _, c in conds]
ju = [mean(2, c) for _, c in conds]
be = [mean(1, c) for _, c in conds]

L += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}", r"\small",
      r"\item \textit{ROUGE-L}: longest-common-subsequence F1 (stemmed).",
      r"      \textit{BERTScore}: contextual-embedding F1 (%s)." % BS_MODEL,
      r"      \textit{LLM Judge}: GPT-4.1 rates each answer 0 (wrong) to 3 (exact).",
      r"      \textit{Exact Match}: \% of answers scoring 3. Retrieval uses top-5 semantic",
      r"      search (OpenAI \texttt{text-embedding-3-large}) over the pre-expansion (base)",
      r"      vs.\ expanded (Roadmap$^+$) catalogue. All models evaluated on the same",
      r"      120-question benchmark (24 each: visibility, defence, family, perturbation, math).",
      r"      Bold marks the best value per model in each metric column.",
      r"\item Averaged across the %d fully-evaluated models, Exact Match is %.1f\%% (no context)," % (N_MEAN, em[0]),
      r"      %.1f\%% (base roadmap) and %.1f\%% (Roadmap$^+$); mean LLM Judge is %.2f, %.2f and %.2f;" % (em[1], em[2], ju[0], ju[1], ju[2]),
      r"      mean BERTScore is %.4f, %.4f and %.4f. Retrieval helps mid-capability models" % (be[0], be[1], be[2]),
      r"      (e.g.\ GPT-4o-mini and Gemma3) but not the strongest models, which are already",
      r"      at ceiling. On average the base roadmap performs on par with or slightly above",
      r"      Roadmap$^+$ (Exact Match %.1f\%% vs.\ %.1f\%%): the LLM-driven expansion does not" % (em[1], em[2]),
      r"      improve---and modestly dilutes---retrieval accuracy on this external benchmark.",
      r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table*}"]

txt = "\n".join(L)
Path("Latex-Paper/table8_fulltable.tex").write_text(txt, encoding="utf-8")
print(txt)
print("\n[OK] -> Latex-Paper/table8_fulltable.tex")
print("means EM:", [round(x,1) for x in em], "Judge:", [round(x,2) for x in ju], "BERT:", [round(x,4) for x in be])
