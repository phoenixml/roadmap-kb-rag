"""
generate_qa_atlas.py
---------------------
Generates a QA benchmark from MITRE ATLAS (v5.6.0) using the SAME 5 question
templates as DPA-QA, making Table 2 directly comparable to Table 1.

Pipeline:
  Step 1 — Enrich each ATLAS technique with 5 KB-aligned fields via GPT-4.1:
             family, visibility, perturbation_search, defence, math
  Step 2 — Generate QA using the same 5 template types as DPA-QA:
             visibility, defence, family, perturbation_search, math

This ensures:
  - RAG context directly answers the questions (same field mapping)
  - Models cannot answer from general knowledge alone
  - Tables 1 and 2 are evaluated on the same question taxonomy

Output : outputs/atlas_qa.json
Enrich : outputs/atlas_techniques_enriched.json  (cached enrichment)
Resume : skips questions/enrichments already done.

Run: python scripts/generate_qa_atlas.py
"""

import os, json, time
from pathlib import Path
from openai import OpenAI

_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ENRICH_FILE = "outputs/atlas_techniques_enriched.json"
OUT_FILE    = "outputs/atlas_qa.json"

# ── MITRE ATLAS Technique Dataset (v5.6.0, embedded) ──────────────────────────
# Coverage mapped to roadmap families: CNN, RNN, GAN, Transformer/LLM,
# Black-Box, CA (Classifier-Agnostic), Supply Chain, RAG
ATLAS_TECHNIQUES = [
    # ── CNN / Image Classifier attacks ───────────────────────────────────────
    {
        "id": "AML.T0043.000",
        "name": "White-Box Optimization (Craft Adversarial Data)",
        "tactic": "Execution",
        "roadmap_family": "CNN",
        "description": (
            "The adversary has full white-box access to the target CNN or image "
            "classifier and directly optimises adversarial examples using model "
            "gradients. Attacks such as FGSM, PGD, and C&W fall under this "
            "sub-technique. Examples are most effective against the target model "
            "because the perturbation is computed directly against its loss surface."
        ),
    },
    {
        "id": "AML.T0043.004",
        "name": "Insert Backdoor Trigger (Craft Adversarial Data)",
        "tactic": "Execution",
        "roadmap_family": "CNN",
        "description": (
            "The adversary inserts a perceptual trigger — imperceptible or "
            "non-obvious to humans — into inference data. Combined with poisoning "
            "techniques, the trigger causes the CNN to misclassify any input "
            "containing the trigger to an adversary-chosen label at inference time."
        ),
    },
    {
        "id": "AML.T0015",
        "name": "Evade AI Model",
        "tactic": "Defense Evasion / Impact",
        "roadmap_family": "CNN",
        "description": (
            "Adversaries craft adversarial inputs at inference time to prevent "
            "correct identification by image classifiers or other AI models. "
            "Covers generating deepfakes to fool biometric authentication and "
            "crafting perturbations that survive real-world image transformations "
            "such as JPEG compression and printing."
        ),
    },
    # ── Black-Box / Transfer attacks ─────────────────────────────────────────
    {
        "id": "AML.T0043.001",
        "name": "Black-Box Optimization (Craft Adversarial Data)",
        "tactic": "Execution",
        "roadmap_family": "Black-Box/Unknown",
        "description": (
            "The adversary has only API query access to the target model and "
            "optimises adversarial examples using estimated or finite-difference "
            "gradients. These attacks are less efficient than white-box methods "
            "and require more model queries but demand no knowledge of model "
            "architecture or weights."
        ),
    },
    {
        "id": "AML.T0043.002",
        "name": "Black-Box Transfer (Craft Adversarial Data)",
        "tactic": "Execution",
        "roadmap_family": "Black-Box/Unknown",
        "description": (
            "The adversary trains or uses a surrogate proxy model and generates "
            "adversarial examples on it using white-box optimisation. These "
            "examples transfer to the black-box target due to the transferability "
            "property of adversarial perturbations across model architectures."
        ),
    },
    {
        "id": "AML.T0005.001",
        "name": "Train Proxy via Replication (Model Stealing)",
        "tactic": "Execution",
        "roadmap_family": "Black-Box/Unknown",
        "description": (
            "The adversary replicates a private target model by repeatedly "
            "querying its inference API and collecting input-output pairs as "
            "labelled training data. A surrogate model is trained on these "
            "pairs to mimic the target's decision boundary, giving the adversary "
            "effective white-box access for downstream attacks."
        ),
    },
    # ── RNN / Sequential model attacks ───────────────────────────────────────
    {
        "id": "AML.T0043.003",
        "name": "Manual Modification (Craft Adversarial Data)",
        "tactic": "Execution",
        "roadmap_family": "RNN",
        "description": (
            "The adversary manually modifies sequential input (text, time-series) "
            "using domain knowledge of the target RNN or sequence model. "
            "Suspected components aiding model performance are altered through "
            "trial and error — common in NLP adversarial attacks where gradient "
            "access is limited and discrete token substitutions are used."
        ),
    },
    {
        "id": "AML.T0031",
        "name": "Erode AI Model Integrity",
        "tactic": "Impact",
        "roadmap_family": "RNN",
        "description": (
            "Adversaries degrade a deployed RNN or sequence model's performance "
            "over time by feeding adversarial inputs at inference, exploiting "
            "distribution shift. Repeated erroneous predictions reduce "
            "organisational confidence and force manual task completion "
            "alongside failed automation, with no immediate visible indicator."
        ),
    },
    # ── GAN / Generative model attacks ───────────────────────────────────────
    {
        "id": "AML.T0005.000",
        "name": "Train Proxy via Gathered AI Artifacts",
        "tactic": "Execution",
        "roadmap_family": "GAN",
        "description": (
            "The adversary trains a proxy generative or discriminator model "
            "from gathered AI artefacts — datasets, architectures, pre-trained "
            "weights — to approximate the target GAN. This enables crafting "
            "adversarial examples or performing model inversion attacks that "
            "reconstruct training data, without directly accessing the victim model."
        ),
    },
    {
        "id": "AML.T0024",
        "name": "Exfiltration via AI Inference API",
        "tactic": "Exfiltration",
        "roadmap_family": "GAN",
        "description": (
            "Adversaries exploit the inference API of a generative model or "
            "classifier to reconstruct private training data through model "
            "inversion or membership inference attacks. Carefully crafted "
            "queries reveal sensitive training samples or confirm membership "
            "of specific records, exfiltrating information without direct "
            "access to weights."
        ),
    },
    # ── Backdoor / Trojan attacks (all families) ──────────────────────────────
    {
        "id": "AML.T0018.000",
        "name": "Poison AI Model (Manipulate AI Model)",
        "tactic": "Persistence / Execution",
        "roadmap_family": "CNN",
        "description": (
            "The adversary manipulates model weights directly to embed a backdoor "
            "or change behaviour. Poisoning may occur through direct weight "
            "manipulation, training on poisoned data, fine-tuning on malicious "
            "datasets, or interfering with the training process itself. Results "
            "in a trojanised model that behaves normally on clean inputs but "
            "misbehaves when a trigger is present."
        ),
    },
    {
        "id": "AML.T0018.001",
        "name": "Modify AI Model Architecture",
        "tactic": "Persistence / Execution",
        "roadmap_family": "CNN",
        "description": (
            "The adversary directly modifies the model architecture by adding "
            "or removing layers, neurons, or preprocessing operations. Effects "
            "include removing prediction capability for specific classes, adding "
            "erroneous computation paths, or degrading overall performance. "
            "Requires direct access to the model artefact."
        ),
    },
    # ── Data Poisoning attacks (all families) ────────────────────────────────
    {
        "id": "AML.T0020",
        "name": "Poison Training Data",
        "tactic": "ML Attack Staging / Persistence",
        "roadmap_family": "Any/All",
        "description": (
            "Adversaries modify underlying training data or labels to embed "
            "vulnerabilities in any ML model family — CNN, RNN, GAN, or "
            "Transformer. Poisoning may be label-consistent (clean-label) or "
            "label-flipping, and may use backdoor triggers for later activation. "
            "Introduced via supply chain compromise or post-initial access."
        ),
    },
    {
        "id": "AML.T0019",
        "name": "Publish Poisoned Datasets",
        "tactic": "ML Attack Staging",
        "roadmap_family": "Any/All",
        "description": (
            "Adversaries poison training data and publish it publicly in open "
            "data repositories. Victims who download and train on the poisoned "
            "dataset unknowingly embed the vulnerability. Affects any model "
            "family that trains on publicly sourced data."
        ),
    },
    {
        "id": "AML.T0059",
        "name": "Erode Dataset Integrity",
        "tactic": "Impact",
        "roadmap_family": "Any/All",
        "description": (
            "Adversaries poison or manipulate portions of a dataset, reducing "
            "its statistical quality and usefulness. Unlike targeted poisoning, "
            "this is an indiscriminate attack that erodes trust in the data "
            "pipeline and wastes resources on error correction and re-labelling "
            "across any model family that depends on the dataset."
        ),
    },
    # ── Supply Chain attacks ──────────────────────────────────────────────────
    {
        "id": "AML.T0010.001",
        "name": "AI Software Supply Chain Compromise",
        "tactic": "Initial Access",
        "roadmap_family": "Any/All",
        "description": (
            "Adversaries target AI software packages — deep learning frameworks, "
            "generative AI integration libraries, inference engines, or dependency "
            "chains — by injecting malicious code or trojaned versions. Any model "
            "trained or deployed using the compromised software inherits the "
            "vulnerability regardless of model architecture."
        ),
    },
    {
        "id": "AML.T0010.002",
        "name": "Data Supply Chain Compromise",
        "tactic": "Initial Access",
        "roadmap_family": "Any/All",
        "description": (
            "Adversaries target open-source datasets or private datasets during "
            "the labelling phase by compromising labelling services or data "
            "pipelines. Poisoned labels are introduced upstream, affecting any "
            "model family that trains on the compromised data source."
        ),
    },
    {
        "id": "AML.T0010.003",
        "name": "Model Supply Chain Compromise",
        "tactic": "Initial Access",
        "roadmap_family": "Any/All",
        "description": (
            "Adversaries compromise open-source pre-trained models used for "
            "fine-tuning by embedding traditional malware or adversarial AI "
            "backdoors before publication. Any organisation that downloads and "
            "deploys the model inherits the trojan, affecting CNN, RNN, "
            "Transformer, and other fine-tuned architectures."
        ),
    },
    # ── Transformer / LLM attacks (targeted, reduced) ────────────────────────
    {
        "id": "AML.T0051",
        "name": "LLM Prompt Injection",
        "tactic": "Resource Development",
        "roadmap_family": "Transformer/LLM",
        "description": (
            "Adversaries craft malicious prompts causing Transformer-based LLMs "
            "to override system instructions and ignore safety guardrails. "
            "Direct injection embeds malicious instructions in user input; "
            "indirect injection plants instructions in external content the "
            "model retrieves (e.g. RAG documents, web pages)."
        ),
    },
    {
        "id": "AML.T0054",
        "name": "LLM Jailbreak",
        "tactic": "Defense Evasion",
        "roadmap_family": "Transformer/LLM",
        "description": (
            "Adversaries induce Transformer LLMs to override RLHF safety "
            "alignment via adversarial prompting. Automated methods such as "
            "Greedy Coordinate Gradient (GCG) optimise adversarial suffixes "
            "that maximise the probability of harmful completions. Manual "
            "strategies include roleplay, fictionalization, and format constraints."
        ),
    },
    # ── RAG-specific attacks ──────────────────────────────────────────────────
    {
        "id": "AML.T0070",
        "name": "RAG Poisoning",
        "tactic": "ML Attack Staging",
        "roadmap_family": "Transformer/LLM (RAG)",
        "description": (
            "Adversaries inject malicious documents into the vector store indexed "
            "by a Retrieval-Augmented Generation system. Poisoned content is "
            "retrieved by the embedding similarity search and included in the "
            "LLM context, contaminating responses for any user whose query "
            "retrieves the poisoned chunk."
        ),
    },
    {
        "id": "AML.T0066",
        "name": "Retrieval Content Crafting",
        "tactic": "ML Attack Staging",
        "roadmap_family": "Transformer/LLM (RAG)",
        "description": (
            "Adversaries craft documents with content specifically optimised "
            "to score highly in the RAG system's embedding similarity search. "
            "The adversarial content is retrieved in preference to legitimate "
            "documents and combined with prompt injection to redirect model "
            "behaviour and deceive end users."
        ),
    },
    # ── Classifier-Agnostic (CA) attacks ─────────────────────────────────────
    {
        "id": "AML.T0058",
        "name": "Publish Poisoned Models",
        "tactic": "ML Attack Staging",
        "roadmap_family": "CA",
        "description": (
            "Adversaries publish backdoored models to public model registries "
            "such as Hugging Face or GitHub. The trojaned models behave normally "
            "on standard benchmarks but misclassify or produce adversary-chosen "
            "outputs when a trigger is present. Applicable to any model family "
            "distributed via public repositories."
        ),
    },
    {
        "id": "AML.T0029",
        "name": "Denial of AI Service",
        "tactic": "Impact",
        "roadmap_family": "CA",
        "description": (
            "Adversaries flood AI inference endpoints with high-volume or "
            "computationally expensive crafted inputs to exhaust GPU/CPU "
            "resources and degrade service availability. Applicable to any "
            "model family; sponge examples specifically target energy and "
            "latency costs of deep neural networks."
        ),
    },
]


# ── Step 1: Enrich techniques with KB-aligned fields ──────────────────────────
ENRICH_PROMPT = """You are an adversarial machine learning expert.

Given this MITRE ATLAS technique and its roadmap family, fill the 5 fields below.
Be SPECIFIC and UNIQUE to this exact technique. No generic answers.

Technique : {name} ({id})
Family     : {roadmap_family}
Description: {description}

Fields:
1. family       — ML model family primarily targeted. Match the roadmap family above.
                  Use exactly: CNN, RNN, GAN, Transformer/LLM, Black-Box/Unknown,
                  CA (Classifier-Agnostic), Any/All, or Transformer/LLM (RAG)

2. visibility   — Is this detectable by a defender at execution time?
                  Answer EXACTLY one word: Visible OR Invisible

3. perturbation_search — The specific optimisation/search method this technique uses.
                  Be precise and unique. Use named methods where possible.
                  Examples: Gradient-Based PGD/FGSM, C&W L2 Optimisation,
                  Greedy Coordinate Gradient (GCG), Finite-Difference Estimation,
                  Transfer-Based (Surrogate Gradient), Genetic/Evolutionary Search,
                  Clean-Label Convex Polytope Optimisation, Sponge Example Generation,
                  No Perturbation (direct data injection), Label-Flipping

4. defence      — The single most specific named defence for THIS technique only.
                  Use a real published algorithm name unique to this technique.
                  Do NOT use generic phrases like "monitoring" or "filtering".
                  Examples: Spectral Signatures (Tran et al.), Neural Cleanse,
                  Activation Clustering (Chen et al.), STRIP (run-time trojan detection),
                  PRADA (model stealing defence), Randomised Smoothing (Cohen et al.),
                  SmoothLLM (Robey et al.), Membership Inference Auditing,
                  Differentially Private SGD (Abadi et al.), Sponge Defence (load balancing),
                  RAG Retrieval Scoring + Threshold Filtering, Input Transformation Defence

5. math         — Core mathematical objective, norm, or algorithm. NEVER write N/A.
                  Every attack has a mathematical basis — find it.
                  CNN evasion   : L-inf / L2 norm minimisation (PGD objective)
                  Backdoor      : cross-entropy loss on trigger-label pairs
                  Model stealing: cross-entropy minimisation on stolen labels
                  GAN inversion : model inversion via gradient ascent on input space
                  LLM jailbreak : GCG greedy token optimisation (cross-entropy on target sequence)
                  Prompt inject : softmax probability redistribution over token vocabulary
                  RAG poisoning : cosine similarity maximisation in embedding space
                  Data poison   : bilevel optimisation (inner: model training, outer: poison craft)
                  DoS/Sponge    : maximise inference latency via adversarial input complexity

Respond ONLY in this exact JSON format:
{{
  "family": "...",
  "visibility": "Visible|Invisible",
  "perturbation_search": "...",
  "defence": "...",
  "math": "..."
}}"""


def enrich_technique(tech: dict) -> dict:
    prompt = ENRICH_PROMPT.format(
        id=tech["id"],
        name=tech["name"],
        roadmap_family=tech.get("roadmap_family", "Any/All"),
        description=tech["description"],
    )
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    fields = json.loads(resp.choices[0].message.content)
    return {**tech, **fields}


def run_enrichment() -> list:
    enrich_path = Path(ENRICH_FILE)
    if enrich_path.exists():
        enriched = json.loads(enrich_path.read_text(encoding="utf-8"))
        done_ids = {e["id"] for e in enriched}
        print(f"[RESUME] {len(done_ids)} techniques already enriched.")
    else:
        enriched = []
        done_ids = set()

    remaining = [t for t in ATLAS_TECHNIQUES if t["id"] not in done_ids]
    print(f"Enriching {len(remaining)} techniques...\n")

    for i, tech in enumerate(remaining, 1):
        try:
            result = enrich_technique(tech)
            enriched.append(result)
            done_ids.add(tech["id"])
            print(f"[{i}/{len(remaining)}] {tech['id']} {tech['name']}")
            print(f"  family={result.get('family')}  visibility={result.get('visibility')}")
            print(f"  perturbation={result.get('perturbation_search')}")
            print(f"  defence={result.get('defence')}")
            print(f"  math={result.get('math')}\n")
        except Exception as e:
            print(f"[ERROR] {tech['id']}: {e}")
        time.sleep(0.4)

    enrich_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] Saved {len(enriched)} enriched techniques -> {ENRICH_FILE}\n")
    return enriched


# ── Step 2: Generate QA using same 5 templates as DPA-QA ─────────────────────
QA_TEMPLATES = {
    "visibility": (
        "MITRE ATLAS Technique: {name} ({id})\n"
        "Description: {description}\n"
        "Visibility: {visibility}\n\n"
        "Generate ONE question asking whether the {name} technique is detectable "
        "by defenders at execution time (visible or invisible), and give the "
        "one-word answer.\n"
        "Format:\nQ: <question>\nA: <answer>"
    ),
    "defence": (
        "MITRE ATLAS Technique: {name} ({id})\n"
        "Description: {description}\n"
        "Named Defence: {defence}\n\n"
        "Generate ONE question asking what specific named defence or mitigation "
        "applies to {name}, and give the answer using the named defence above.\n"
        "Format:\nQ: <question>\nA: <answer>"
    ),
    "family": (
        "MITRE ATLAS Technique: {name} ({id})\n"
        "Description: {description}\n"
        "Target Model Family: {family}\n\n"
        "Generate ONE question asking what type or family of ML model is "
        "primarily targeted or exploited by {name}, and give the answer.\n"
        "Format:\nQ: <question>\nA: <answer>"
    ),
    "perturbation_search": (
        "MITRE ATLAS Technique: {name} ({id})\n"
        "Description: {description}\n"
        "Perturbation/Search Strategy: {perturbation_search}\n\n"
        "Generate ONE question asking what optimisation or search strategy "
        "{name} uses to achieve its goal, and give the answer.\n"
        "Format:\nQ: <question>\nA: <answer>"
    ),
    "math": (
        "MITRE ATLAS Technique: {name} ({id})\n"
        "Description: {description}\n"
        "Core Mathematical Concept: {math}\n\n"
        "Generate ONE question asking about the core mathematical objective, "
        "norm, or formula involved in {name}, and give the answer.\n"
        "If the math field is 'N/A', generate a question about why no "
        "specific mathematical formulation is required.\n"
        "Format:\nQ: <question>\nA: <answer>"
    ),
}


def parse_qa(raw: str) -> tuple[str, str]:
    q, a = "", ""
    for line in raw.strip().splitlines():
        if line.startswith("Q:"):
            q = line[2:].strip()
        elif line.startswith("A:"):
            a = line[2:].strip()
    return q, a


def generate_qa_for_technique(tech: dict, qa_type: str, idx: int) -> dict | None:
    prompt = QA_TEMPLATES[qa_type].format(
        id=tech["id"],
        name=tech["name"],
        description=tech["description"],
        visibility=tech.get("visibility", "N/A"),
        defence=tech.get("defence", "N/A"),
        family=tech.get("family", "N/A"),
        perturbation_search=tech.get("perturbation_search", "N/A"),
        math=tech.get("math", "N/A"),
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        q, a = parse_qa(raw)
        if not q or not a:
            return None
        return {
            "id": f"atlas_{tech['id']}_{qa_type}_{idx:03d}",
            "source": "MITRE ATLAS v5.6.0",
            "technique_id": tech["id"],
            "type": qa_type,
            "family": tech.get("family", ""),
            "attack": tech["name"],
            "question": q,
            "answer": a,
        }
    except Exception as e:
        print(f"  [ERROR] {tech['id']} {qa_type}: {e}")
        return None


def run_qa_generation(enriched: list):
    out_path = Path(OUT_FILE)
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        done = {r["id"] for r in results}
        print(f"[RESUME] {len(done)} QA pairs already done.")
    else:
        results = []
        done = set()

    idx = len(results)
    types = list(QA_TEMPLATES.keys())  # visibility, defence, family, perturbation_search, math

    for tech in enriched:
        for qa_type in types:
            qa_id = f"atlas_{tech['id']}_{qa_type}_{idx:03d}"
            if qa_id in done:
                idx += 1
                continue

            qa = generate_qa_for_technique(tech, qa_type, idx)
            if qa:
                results.append(qa)
                done.add(qa["id"])
                print(f"[{len(results):03d}] {tech['id']} {qa_type:20s} Q: {qa['question'][:70]}...")
            else:
                print(f"[WARN] {tech['id']} {qa_type} — skipped")
            idx += 1
            time.sleep(0.3)

        if len(results) % 25 == 0 and len(results) > 0:
            out_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  [Saved {len(results)}]")

    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[OK] Saved {len(results)} ATLAS QA pairs -> {OUT_FILE}")

    from collections import Counter
    type_dist = Counter(r["type"] for r in results)
    print("\nType distribution:")
    for t, n in sorted(type_dist.items()):
        print(f"  {t:25s}: {n}")


def run():
    print("=== Step 1: Enrich ATLAS techniques with KB-aligned fields ===\n")
    enriched = run_enrichment()

    print("\n=== Step 2: Generate QA using DPA-QA templates ===\n")
    run_qa_generation(enriched)


if __name__ == "__main__":
    run()
