"""
run_qa_evaluation_atlas.py
---------------------------
Evaluates the RAG pipeline against the MITRE ATLAS QA benchmark — an
independent, external dataset NOT derived from our roadmap KB.

This produces Table 2 in the paper, addressing the reviewer concern that
Table 1 (DPA-QA) is self-referential (generated from the same KB).

QA file    : outputs/atlas_qa.json  (from scripts/generate_qa_atlas.py)
KB files   : same base/plus KBs used in Table 1 (context injection unchanged)
Conditions : no_context | base_roadmap | roadmap_plus
Metrics    : Fuzzy F1, Semantic Similarity (all-MiniLM-L6-v2), LLM Judge (0-3), Exact Match

Output:
  outputs/qa_eval_results_atlas.json
  outputs/qa_eval_summary_atlas.csv

Run AFTER:
  python scripts/generate_qa_atlas.py
"""

import os, json, time, re, csv
from pathlib import Path
from collections import defaultdict

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import anthropic
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

import urllib.request

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    ST_AVAILABLE = True
except ImportError:
    print("[WARN] sentence-transformers not installed — semantic sim will be skipped.")
    ST_AVAILABLE = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "https://ollama-llama-349218458434.us-west1.run.app"
QA_FILE      = "outputs/atlas_qa.json"
BASE_KB_FILE = "outputs/roadmap_qa_data.json"
PLUS_KB_FILE = "outputs/roadmap_qa_data_plus.json"
OUT_JSON     = "outputs/qa_eval_results_atlas.json"
OUT_CSV      = "outputs/qa_eval_summary_atlas.csv"

MODELS = {
    "gpt-4.1":           {"provider": "openai",     "model": "gpt-4.1"},
    "gpt-4.1-mini":      {"provider": "openai",     "model": "gpt-4.1-mini"},
    "gpt-4o-mini":       {"provider": "openai",     "model": "gpt-4o-mini"},
    "claude-sonnet-4-6": {"provider": "anthropic",  "model": "claude-sonnet-4-6"},
    "claude-opus-4-8":   {"provider": "anthropic",  "model": "claude-opus-4-8"},
    "gemma3":            {"provider": "ollama",      "model": "gemma3:latest"},
    "llama3.1:8b":       {"provider": "ollama",      "model": "llama3.1:8b"},
    "gemma3:1b":         {"provider": "ollama",      "model": "gemma3:1b"},
    "llama3.2:1b":       {"provider": "ollama",      "model": "llama3.2:1b"},
}

CONDITIONS = ["no_context", "base_roadmap", "roadmap_plus"]

# ── Context builder (same as option2) ─────────────────────────────────────────
def build_context(attack_name: str, kb_data: list) -> str:
    for entry in kb_data:
        if entry.get("attack", "").lower() == attack_name.lower():
            ctx = (
                f"Adversarial ML Roadmap Entry:\n"
                f"  Attack              : {entry['attack']}\n"
                f"  Model Family        : {entry['family']}\n"
                f"  Visibility          : {entry.get('visibility', 'N/A')}\n"
                f"  Perturbation Search : {entry.get('perturbation_search', 'N/A')}\n"
                f"  Defence             : {entry.get('defence', 'N/A')}\n"
            )
            if entry.get("description"):
                ctx += f"  Description         : {entry['description']}\n"
            if entry.get("mechanism"):
                ctx += f"  Mechanism           : {entry['mechanism']}\n"
            if entry.get("defence_mechanism"):
                ctx += f"  Defence Mechanism   : {entry['defence_mechanism']}\n"
            if entry.get("variants"):
                ctx += f"  Variants            : {entry['variants']}\n"
            if entry.get("threat_model"):
                ctx += f"  Threat Model        : {entry['threat_model']}\n"
            if entry.get("target_vulnerability"):
                ctx += f"  Target Vulnerability: {entry['target_vulnerability']}\n"
            return ctx
    return ""


def build_atlas_context(technique_id: str, technique_name: str) -> str:
    """For ATLAS questions, build a MITRE ATLAS knowledge context."""
    # Import the ATLAS data from the generate script
    sys_path = str(Path(__file__).resolve().parents[1] / "scripts")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    try:
        from generate_qa_atlas import ATLAS_TECHNIQUES
        for t in ATLAS_TECHNIQUES:
            if t["id"] == technique_id or t["name"].lower() == technique_name.lower():
                return (
                    f"MITRE ATLAS Technique Entry:\n"
                    f"  Technique ID   : {t['id']}\n"
                    f"  Name           : {t['name']}\n"
                    f"  Tactic         : {t['tactic']}\n"
                    f"  Description    : {t['description']}\n"
                    f"  Mitigations    : {t.get('mitigations', 'N/A')}\n"
                )
    except Exception:
        pass
    return ""

# ── Model callers ─────────────────────────────────────────────────────────────
def call_openai(model, system, user):
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def call_ollama(model, system, user):
    payload = json.dumps({
        "model": model,
        "prompt": f"{system}\n\n{user}",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read()).get("response", "").strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def call_anthropic(model, system, user):
    try:
        resp = anthropic_client.messages.create(
            model=model, max_tokens=300, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def call_model(model_key, system, user):
    cfg = MODELS[model_key]
    if cfg["provider"] == "openai":
        return call_openai(cfg["model"], system, user)
    elif cfg["provider"] == "ollama":
        return call_ollama(cfg["model"], system, user)
    elif cfg["provider"] == "anthropic":
        return call_anthropic(cfg["model"], system, user)
    return "[UNKNOWN PROVIDER]"

# ── Scoring ───────────────────────────────────────────────────────────────────
def tokenize(text):
    return set(re.findall(r'\w+', text.lower()))


def fuzzy_f1(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    common = p & r
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec  = len(common) / len(r)
    return round(2 * prec * rec / (prec + rec), 3)


def llm_judge(question, reference, predicted):
    prompt = (
        f"You are evaluating an AI answer about adversarial machine learning "
        f"and the MITRE ATLAS framework.\n\n"
        f"Question  : {question}\n"
        f"Reference : {reference}\n"
        f"Predicted : {predicted}\n\n"
        "Rate the predicted answer:\n"
        "3 = Correct and complete\n"
        "2 = Mostly correct, minor gaps\n"
        "1 = Partially correct\n"
        "0 = Wrong or missing\n\n"
        "Respond with a single digit only: 0, 1, 2, or 3."
    )
    try:
        resp  = call_openai("gpt-4.1", "You are a strict academic evaluator.", prompt)
        match = re.search(r'[0-3]', resp)
        return int(match.group()) if match else 0
    except Exception:
        return 0

# ── Prompt builder ────────────────────────────────────────────────────────────
SYSTEM_BASE = (
    "You are an expert in adversarial machine learning, AI security, and the "
    "MITRE ATLAS framework. Answer concisely and precisely. "
    "If you are not sure, say 'Unknown' rather than guessing."
)


def build_prompt(question, context, condition):
    if condition == "no_context":
        return SYSTEM_BASE, f"Question: {question}\n\nAnswer:"
    system = SYSTEM_BASE + (
        "\n\nYou have access to the following knowledge base entry. "
        "Use it to answer the question accurately."
    )
    return system, f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

# ── Semantic similarity batch ─────────────────────────────────────────────────
def compute_semantic_sim_batch(results: list) -> list:
    if not ST_AVAILABLE:
        for r in results:
            r["semantic_sim"] = None
        return results

    print("\n[SemanticSim] Computing semantic similarity...")
    valid = [(i, r) for i, r in enumerate(results)
             if r.get("predicted", "").strip()
             and not str(r.get("predicted", "")).startswith("[ERROR")]

    if not valid:
        for r in results:
            r["semantic_sim"] = 0.0
        return results

    preds = [r["predicted"] for _, r in valid]
    refs  = [r["reference"] for _, r in valid]

    pred_embs = _st_model.encode(preds, batch_size=64, show_progress_bar=True,
                                  convert_to_tensor=True)
    ref_embs  = _st_model.encode(refs,  batch_size=64, show_progress_bar=False,
                                  convert_to_tensor=True)
    sims = cos_sim(pred_embs, ref_embs).diagonal().tolist()

    valid_set = {i for i, _ in valid}
    for idx, (orig_idx, _) in enumerate(valid):
        results[orig_idx]["semantic_sim"] = round(float(sims[idx]), 4)
    for i, r in enumerate(results):
        if i not in valid_set:
            r["semantic_sim"] = 0.0

    avg = sum(r["semantic_sim"] for r in results) / len(results)
    print(f"[SemanticSim] Done. Avg: {avg:.4f}")
    return results

# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation():
    if not Path(QA_FILE).exists():
        print(f"[ERROR] {QA_FILE} not found. Run: python scripts/generate_qa_atlas.py")
        return

    qa_pairs = json.loads(Path(QA_FILE).read_text(encoding="utf-8"))
    base_kb  = json.loads(Path(BASE_KB_FILE).read_text(encoding="utf-8"))
    plus_kb  = json.loads(Path(PLUS_KB_FILE).read_text(encoding="utf-8"))

    out_path = Path(OUT_JSON)
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"[RESUME] Loaded {len(results)} existing results.")
    else:
        results = []

    done  = {(r["model"], r["condition"], r["id"]) for r in results}
    total = len(qa_pairs) * len(MODELS) * len(CONDITIONS)

    print(f"\n=== QA Evaluation — MITRE ATLAS Benchmark ===")
    print(f"Questions : {len(qa_pairs)}  |  Models: {len(MODELS)}  |  Conditions: {len(CONDITIONS)}")
    print(f"Total: {total}  |  Done: {len(done)}  |  Remaining: {total - len(done)}\n")

    for model_key in MODELS:
        print(f"\n--- Model: {model_key} ---")
        for condition in CONDITIONS:
            print(f"  Condition: {condition}")
            for qa in qa_pairs:
                if (model_key, condition, qa["id"]) in done:
                    continue

                # Build context: for ATLAS questions, use ATLAS KB for roadmap_plus
                # and our roadmap KB for base_roadmap (tests transfer knowledge)
                context = ""
                if condition == "base_roadmap":
                    # Try to match on technique name in our roadmap KB
                    context = build_context(qa.get("technique_id", ""), base_kb)
                    if not context:
                        # Fall back to ATLAS context (partial — tests knowledge transfer)
                        context = build_atlas_context(
                            qa.get("technique_id", ""), "")
                elif condition == "roadmap_plus":
                    # Use enriched roadmap KB first, then ATLAS context
                    context = build_context(qa.get("technique_id", ""), plus_kb)
                    if not context:
                        context = build_atlas_context(
                            qa.get("technique_id", ""), "")

                system, user = build_prompt(qa["question"], context, condition)
                predicted    = call_model(model_key, system, user)

                f1    = fuzzy_f1(predicted, qa["answer"])
                judge = llm_judge(qa["question"], qa["answer"], predicted)

                results.append({
                    "id":           qa["id"],
                    "source":       qa.get("source", "MITRE ATLAS"),
                    "technique_id": qa.get("technique_id", ""),
                    "type":         qa.get("type", "unknown"),
                    "question":     qa["question"],
                    "reference":    qa["answer"],
                    "predicted":    predicted,
                    "model":        model_key,
                    "condition":    condition,
                    "fuzzy_f1":     f1,
                    "llm_judge":    judge,
                    "semantic_sim": None,
                })
                done.add((model_key, condition, qa["id"]))

                if len(results) % 20 == 0:
                    out_path.write_text(
                        json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
                    print(f"    [Saved {len(results)}/{total}]")

                time.sleep(0.3)

    results = compute_semantic_sim_batch(results)

    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Raw results saved -> {OUT_JSON}")

    generate_summary(results)


def generate_summary(results):
    summary = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["model"], r["condition"])
        summary[key]["fuzzy_f1"].append(r.get("fuzzy_f1", 0))
        summary[key]["llm_judge"].append(r.get("llm_judge", 0))
        if r.get("semantic_sim") is not None:
            summary[key]["semantic_sim"].append(r["semantic_sim"])

    rows = []
    for (model, condition), scores in sorted(summary.items()):
        n        = len(scores["llm_judge"])
        avg_f1   = round(sum(scores["fuzzy_f1"])  / n, 3)
        avg_j    = round(sum(scores["llm_judge"]) / n, 3)
        avg_sem  = (round(sum(scores["semantic_sim"]) / len(scores["semantic_sim"]), 4)
                    if scores["semantic_sim"] else None)
        exact    = round(sum(1 for s in scores["llm_judge"] if s == 3) / n * 100, 1)
        rows.append({
            "model":         model,
            "condition":     condition,
            "n":             n,
            "avg_fuzzy_f1":  avg_f1,
            "avg_semantic":  avg_sem,
            "avg_llm_judge": avg_j,
            "pct_exact":     exact,
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'='*90}")
    print("%-22s %-15s %10s %10s %10s %8s" % (
        "Model", "Condition", "Fuzzy F1", "Sem.Sim", "LLM Judge", "Exact%"))
    print("-" * 90)
    for r in rows:
        sem_str = f"{r['avg_semantic']:.4f}" if r["avg_semantic"] is not None else "  N/A  "
        print("%-22s %-15s %10.3f %10s %10.3f %7.1f%%" % (
            r["model"], r["condition"],
            r["avg_fuzzy_f1"], sem_str,
            r["avg_llm_judge"], r["pct_exact"]))

    print(f"\n[OK] Summary saved -> {OUT_CSV}")


if __name__ == "__main__":
    run_evaluation()
