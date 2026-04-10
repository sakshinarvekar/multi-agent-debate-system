# Multi-Agent Debate System with Probabilistic Memory Auditor

## Overview
Brief description of what this system does —
multi-agent reasoning with shared memory, contradiction 
detection, Bayesian trust scoring, and AGM memory revision.

## Architecture 
- Layer I   — Ingestion: agents save (Q, R, A) to shared memory
- Layer II  — Trigger: uncertainty scanner detects contradictions
- Layer III — Probing: Other agents evaluate contradictions
- Layer IV  — Calibration: Bayesian update per memory
- Layer V   — Maintenance: AGM contraction (delete low-trust memories)

## Setup
pip install -r requirements.txt

## Quick Start
# insert test dataset
python scripts/load_memories.py memory/test_dataset.py

# run full pipeline demo
CUDA_VISIBLE_DEVICES=0 python week4_test.py

## Running Tests
# retrieval quality
python tests/test_retrieval.py

# contradiction detection
CUDA_VISIBLE_DEVICES=0 python tests/test_contradiction.py

# full pipeline
CUDA_VISIBLE_DEVICES=0 python tests/test_pipeline.py

# edge cases
CUDA_VISIBLE_DEVICES=0 python tests/test_edge_cases.py

## Agents
- Alpha (Qwen2.5-1.5B)  
- Beta  (Phi-2)         
- Gamma (DeepSeek-R1)    

## Memory System
- ChromaDB vector store
- Bayesian trust score per memory (prior=0.5)
- Deletion threshold τ_low = 0.2
- Solidify threshold τ_high = 0.8



## Quick Start
# fresh start
rm -rf chroma_db/ memory/memory_trust_scores.json

# load test dataset
python scripts/load_memories.py memory/test_dataset.py

# inspect memory pool
python scripts/inspect_memory.py

# run week 4 demo
CUDA_VISIBLE_DEVICES=0 python week4_test.py

## Running Tests
python tests/test_retrieval.py
CUDA_VISIBLE_DEVICES=0 python tests/test_pipeline.py
CUDA_VISIBLE_DEVICES=0 python tests/test_contradiction.py
CUDA_VISIBLE_DEVICES=0 python tests/test_final_context.py
CUDA_VISIBLE_DEVICES=0 python tests/test_edge_cases.py