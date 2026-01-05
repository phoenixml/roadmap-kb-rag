# RoadMap+ KB RAG



# RoadMap+ — Adversarial ML Knowledge Graph & RAG

## Overview
RoadMap+ is a research-grade pipeline for extracting, structuring, and reasoning over
Adversarial ML and Data Poisoning attacks using LaTeX parsing, Qdrant RAG, and Neo4j.

## Architecture
PDF → LaTeX → KB → Qdrant → Neo4j → RAG → Self-Critique → Graph Reasoning

## Components
- dpa_agent_m-V1.5.py
- ingest_kb.py
- Unified_Attack_Knowledge_Base.json
- TrackA_with_defences.dot

## Quick Start

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

sudo docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

python -m scripts.ingest_kb --kb_path ./data/kb/Unified_Attack_Knowledge_Base.json



## Run first:
export QDRANT_URL=http://localhost:6333
export OPENAI_API_KEY="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"

export NEO4J_URI="neo4j+s://139a711a.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="jkSydLlN3clvA7hw33SPy7cGLoQqomjcncfTKAF5Ynw"

A- To Initiate the KB run which will create the KB in Qadrant Store:
python -m scripts.init_kb_from_json --input DPATex-1.0.json

B- We can add the pdf's academic papers we need to import only paper flag:
python -m scripts.dpa_agent_m_v2 --input pdfs_to_process

C- It is important to update the KB rag system:
python -m scripts.ingest_kb --kb_path Unified_Attack_Knowledge_Base.json

D- Then build the dot graph append only one existing attack:
python -m scripts.defence_agent --attack_name "Bullseye Polytope"
python -m scripts.defence_agent --attack_name "Bullseye Polytope" --update_dot
python -m scripts.defence_agent --attack_name "Greedy Coordinate Gradient (GCG) Universal Adversarial Suffix Attack" --enable_neo4j --update_dot
python -m scripts.defence_agent --attack_name "Bullseye Polytope" --enable_neo4j --hops 2 --update_dot

E- If we need to import one attack and apped the Graph Roadmap and update the KB on Qadrant and Neo4j with discovering the appropriate defence:
python -m scripts.defence_agent   
--attack_name "Greedy_Coordinate_Gradient_(GCG)_Universal_Adversarial_Suffix_Attack"   
--enable_neo4j   
--update_dot

F- To Do:
⦁	Import multiple pdf's files and process them.

⦁	Create a new CLI command to process all the extracted latex files and enable neo4j and update dot graph.

⦁	On the initialisation we need to move the latex_processed into a new file so we clear always the latex_processed to only the new files that we export and after ingestion should be moved to a new folder.

⦁	Same for Pdf_to_Process we need to move them to a new folder to ensure it is always empty after processing.

⦁	To only to KB  new processed pdf files we need to have a new KB json t have only the new processed files this is important.

