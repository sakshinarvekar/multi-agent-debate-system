# Multi-Agent Debate System with Probabilistic Memory Auditor (PMA)

> A multi-agent reasoning system where agents share a vector memory pool, detect contradictions, probe each other, and use Bayesian scoring to maintain a self-calibrating, trustworthy memory base.

---

## How It Works

When an agent receives a query it:

1. **Reasons independently** before checking memory
2. **Retrieves relevant memories** using combined query + reasoning embedding
3. **Detects contradictions** via cosine similarity
4. **If contradictions exist** — other agents probe to determine which claim is correct
5. **Memory scores are updated** via Bayes rule
6. **Low-trust memories are deleted** (AGM contraction)

---

## Architecture — PMA Framework

| Layer | Name | Description |
|---|---|---|
| I | Ingestion | Agent saves `(Q, R, A)` → memory pool, prior score `P(H) = 0.5` |
| II | Trigger | Uncertainty scanner detects contradictions via cosine similarity |
| III | Probing | Other agents evaluate each contradiction → `TRUST_REASONING` / `TRUST_MEMORY` |
| IV | Calibration | Bayesian engine: `S_new = P(E|H) · S_old / P(E)` |
| V | Maintenance | `S_new < τ_low` → AGM contraction (physical delete from ChromaDB) |

---

## Pipeline

```
query q
  ↓
Alpha reasons independently      →  r_q, a_q
  ↓
Search memory (q + r_q + a_q)   →  m_q
  ↓
Contradiction check              →  u = count(cos_sim < threshold)
  ↓
if u > 0:
    Beta + Gamma probe           →  votes per memory
    Bayesian update              →  posterior per memory
    Weighted calibration         →  TRUST_REASONING / TRUST_MEMORY
    AGM contraction              →  delete if score < 0.2
  ↓
Final answer with clean memory   →  a*, r*
  ↓
save(q, r*, a*) to memory pool
```

---

## Agents

Alpha  -  `Qwen2.5-1.5B-Instruct` 
Beta -  `microsoft/phi-2` 
Gamma - `DeepSeek-R1-Distill-Qwen-1.5B`

---

## Memory System

**Vector store:** ChromaDB  
**Retrieval:** cosine similarity ≥ `0.6` — irrelevant memories are filtered out  
**Bayesian prior:** `0.5` (new memory — unknown trust)  
**Delete threshold (τ_low):** `0.2`  
**Solidify threshold (τ_high):** `0.8`

### How Bayesian Scoring Works

```
TRUST_REASONING vote  →  evidence memory is wrong
TRUST_MEMORY vote     →  evidence memory is correct

Example — 2 TRUST_REASONING votes:
  prior = 0.500  →  posterior = 0.030
  0.030 < 0.2   →  deleted from ChromaDB
```

---

## Project Structure

```
multi-agent-debate-system/
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py              # core pipeline logic
│   └── specialized_agents.py     # Alpha, Beta, Gamma definitions
│
├── memory/
│   ├── __init__.py
│   ├── mem0_store.py              # ChromaDB interface + filtered retrieval
│   ├── trust_store.py             # Bayesian scoring per memory
│   ├── test_dataset.py            # 30-entry handcrafted dataset
│   └── wrong_dataset.py           # 50 wrong answers for contradiction testing
│
├── scripts/
│   ├── load_truthfulqa.py         # loads TruthfulQA correct answers
│   ├── load_memories.py           # loads any dataset into ChromaDB
│   └── inspect_memory.py          # shows memory pool with scores
│
├── tests/
│   ├── test_retrieval.py          # retrieval quality
│   ├── test_contradiction.py      # contradiction detection
│   ├── test_pipeline.py           # full end-to-end pipeline
│   └── test_final_context.py      # memory vs own reasoning
│   
│
├── requirements.txt
├── COMMANDS.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/sakshinarvekar/multi-agent-debate-system.git
cd multi-agent-debate-system
```

### 2. Create environment

```bash
conda create -n debate python=3.10
conda activate debate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install datasets
```

> **Requirements:** CUDA-enabled GPU · Python 3.10 · ~8 GB GPU memory minimum

---

## Quick Start

```bash
# fresh start — clear all memory
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json

# load correct answers
python scripts/load_truthfulqa.py

# load wrong answers (for contradiction testing)
python scripts/load_memories.py memory/wrong_dataset.py

# inspect memory pool
python scripts/inspect_memory.py

# run demo
CUDA_VISIBLE_DEVICES=0 python week4_test.py
```

---

## Running Tests

```bash
# retrieval quality — no GPU needed
python tests/test_retrieval.py

# contradiction detection
CUDA_VISIBLE_DEVICES=0 python tests/test_contradiction.py

# full pipeline
CUDA_VISIBLE_DEVICES=0 python tests/test_pipeline.py

# memory vs own reasoning
CUDA_VISIBLE_DEVICES=0 python tests/test_final_context.py

# edge cases
CUDA_VISIBLE_DEVICES=0 python tests/test_edge_cases.py
```

---

## Key Parameters

| Parameter | Value | File |
|---|---|---|
| Contradiction threshold | `0.85` | `base_agent.py` |
| Retrieval threshold | `0.6` | `mem0_store.py` |
| Bayesian prior | `0.5` | `trust_store.py` |
| Deletion threshold | `0.2` | `trust_store.py` |
| Top-K retrieval | `3` | `mem0_store.py` |

---

## Known Issues

ChromaDB may show the following warning — this does not affect functionality:
```
chromadb/types.py: PydanticDeprecatedSince211
```

---

## References

- **TruthfulQA** — Lin et al. (2022) · [arxiv.org/abs/2109.07958](https://arxiv.org/abs/2109.07958)
- **Mem0** — [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0)
- **ChromaDB** — [github.com/chroma-core/chroma](https://github.com/chroma-core/chroma)