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
Agents reasons sequentially      →  r_q, a_q
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

| Agent | Model | Parameters |
|---|---|---|
| Alpha (α) | `Qwen2.5-1.5B-Instruct` | 1.5B |
| Beta (β) | `microsoft/phi-2` | 2.7B |
| Gamma (γ) | `DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B |
| Delta (δ) | `Mistral-7B-Instruct-v0.2` | 7B |
| Epsilon (ε) | `Qwen2.5-7B-Instruct` | 7B |

---
## Three Evaluation Setups

| Setup | Memory Retrieval | Memory Saving | Contradiction Detection |
|---|---|---|---|
| No Memory | ✗ | ✗ | ✗ |
| Baseline | ✓ | ✓ | ✗ |
| Full PMA | ✓ | ✓ | ✓ |

---

## Datasets

### HotpotQA — Distractor Setting
- Multi-hop question answering dataset
- Link: https://huggingface.co/datasets/hotpot_qa
- Setting: distractor — each question paired with 10 context paragraphs (2 relevant + 8 distractors)
- Context truncated to 1,500 chars due to GPU memory constraints
- Metric: exact substring match accuracy

### ALFWorld
- Interactive text-based household environment
- Link: https://github.com/alfworld/alfworld
- Agents navigate rooms and manipulate objects to complete multi-step tasks
- Setting: eval_in_distribution (valid_seen), 140 total games
- Metric: task completion (done=True, score > 0)
- Task types: pick_and_place_simple, look_at_obj_in_light, pick_clean_then_place_in_recep, pick_heat_then_place_in_recep, pick_cool_then_place_in_recep, pick_two_obj_and_place

---

## Memory System

**Vector store:** ChromaDB  
**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Retrieval threshold:** `0.3` (HotpotQA), `0.7` (ALFWorld)  
**Top-K retrieval:** `5`  
**Bayesian prior:** `0.5` (new memory — unknown trust)  
**Contradiction threshold:** `0.60`  
**Delete threshold (τ_low):** `0.2`  

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
│   ├── base_agent.py              # core pipeline, _generate, probe, contradiction
│   └── specialized_agents.py     # Alpha, Beta, Gamma, Delta, Epsilon definitions
│
├── memory/
│   ├── __init__.py
│   ├── mem0_store.py              # ChromaDB interface, search, save, delete
│   ├── trust_store.py             # Bayesian scoring, bayesian_update, should_delete
│   └── memory_trust_scores.json  # Bayesian trust scores per memory ID
│
├── scripts/
│   ├── __init__.py
│   └── inspect_memory.py          # inspect ChromaDB memory pool
│
├── eval/
│   ├── __init__.py
│   ├── load_datasets.py           # load_hotpotqa_distractor(n)
│   ├── load_alfworld.py           # load_alfworld_env(n, split)
│   ├── debate_no_memory_dis.py    # HotpotQA distractor — no memory
│   ├── debate_baseline_dis.py     # HotpotQA distractor — baseline memory
│   ├── debate_full_pma_dis.py     # HotpotQA distractor — full PMA
│   ├── debate_no_memory_alf.py    # ALFWorld — no memory
│   ├── debate_baseline_alf.py     # ALFWorld — baseline memory
│   ├── debate_full_pma_alf.py     # ALFWorld — full PMA
│   ├── run_debate.sh              # shell script to run all setups
│   ├── no_memory_dis_log.txt      # HotpotQA no memory log
│   ├── baseline_dis_log.txt       # HotpotQA baseline log
│   ├── full_pma_dis_log.txt       # HotpotQA full PMA log
│   ├── no_memory_alf_log.txt      # ALFWorld no memory log
│   ├── baseline_alf_log.txt       # ALFWorld baseline log
│   └── full_pma_alf_log.txt       # ALFWorld full PMA log
│
├── requirements.txt
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
pip install alfworld
```

### 4. Download ALFWorld game files

```bash
export ALFWORLD_DATA=~/.cache/alfworld
alfworld-download
```

---

## Running Evaluation

### HotpotQA Distractor

```bash
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
CUDA_VISIBLE_DEVICES=3 nohup bash eval/run_debate.sh > eval/debate_log.txt 2>&1 &
tail -f eval/debate_log.txt | grep -E "QUERY|accuracy|Contradictions"
```

### ALFWorld

```bash
# no memory
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
CUDA_VISIBLE_DEVICES=3 python eval/debate_no_memory_alf.py

# baseline
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
CUDA_VISIBLE_DEVICES=3 python eval/debate_baseline_alf.py

# full PMA
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
CUDA_VISIBLE_DEVICES=3 python eval/debate_full_pma_alf.py
```

---

## Key Parameters

| Parameter | Value | File |
|---|---|---|
| Contradiction threshold | `0.60` | `agents/base_agent.py` |
| Retrieval threshold (HotpotQA) | `0.3` | `eval/debate_*_dis.py` |
| Retrieval threshold (ALFWorld) | `0.7` | `eval/debate_*_alf.py` |
| Deletion threshold | `0.20` | `memory/trust_store.py` |
| Top-K retrieval | `5` | `memory/mem0_store.py` |
| Max new tokens (HotpotQA) | `300` | `eval/debate_*_dis.py` |
| Max new tokens (ALFWorld) | `50` | `eval/debate_*_alf.py` |
| Max steps per agent (ALFWorld) | `15` | `eval/debate_*_alf.py` |

---

## Hardware

- **GPU:** NVIDIA RTX 6000 Ada (49GB VRAM)
- **Python:** 3.10
- **Framework:** HuggingFace Transformers, ChromaDB, Mem0

---


## Known Issues

ChromaDB may show the following warning — this does not affect functionality:
```
chromadb/types.py: PydanticDeprecatedSince211
```

---
## References

- **HotpotQA** — Yang et al. (2018) · [arxiv.org/abs/1809.09600](https://arxiv.org/abs/1809.09600)
- **ALFWorld** — Shridhar et al. (2021) · [arxiv.org/abs/2010.03768](https://arxiv.org/abs/2010.03768)
- **ReAct** — Yao et al. (2023) · [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- **G-Memory** — Zhang et al. (2025)
- **Mem0** — [github.com/mem0ai/mem0](https://github.com/mem0ai/mem0)
- **ChromaDB** — [github.com/chroma-core/chroma](https://github.com/chroma-core/chroma)