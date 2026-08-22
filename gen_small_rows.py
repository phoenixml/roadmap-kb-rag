import json, collections
from pathlib import Path

d = json.load(open("outputs/qa_eval_small2.json", encoding="utf-8"))
bs = json.load(open("outputs/bertscore_small2.json"))
agg = collections.defaultdict(lambda: {"r": [], "j": [], "e": 0, "n": 0})
for r in d:
    a = agg[(r["model"], r["condition"])]
    a["r"].append(r["rouge_l"]); a["j"].append(r["llm_judge"]); a["e"] += r["exact"]; a["n"] += 1

def M(m, c):
    a = agg[(m, c)]; n = a["n"]
    return (sum(a["r"]) / n, bs[m + "|" + c], sum(a["j"]) / n, a["e"] / n * 100)

conds = [(r"No Context \textit{(Baseline)}", "no_context"),
         (r"Base Roadmap \textit{(RAG)}", "base_roadmap"),
         (r"Roadmap$^+$ \textit{(RAG)}", "roadmap_plus")]
specs = ["%.3f", "%.4f", "%.2f", "%.1f"]; dig = [3, 4, 2, 1]
L = []
for disp, k in [("Qwen2.5 (3B)", "qwen2.5:3b"), ("Gemma (2B)", "gemma:2b")]:
    cells = [[round(M(k, c)[i], dig[i]) for i in range(4)] for _, c in conds]
    mx = [max(cells[j][i] for j in range(3)) for i in range(4)]
    L.append(r"\multirow{3}{*}{\textbf{%s}}" % disp)
    for j, (cl, _) in enumerate(conds):
        parts = [(r"\textbf{%s}" % (specs[i] % cells[j][i]) if abs(cells[j][i] - mx[i]) < 1e-9
                  else specs[i] % cells[j][i]) for i in range(4)]
        L.append(r"  & %s & %s & %s & %s & %s \\" % (cl, parts[0], parts[1], parts[2], parts[3]))
    L.append(r"\addlinespace[3pt]")
    L.append("")
out = "\n".join(L)
Path("outputs/table8_small_new_rows.tex").write_text(out, encoding="utf-8")
print(out)
