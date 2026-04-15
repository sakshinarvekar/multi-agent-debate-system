#!/bin/bash
# eval/run_all.sh
# Runs all 6 setups (3 setups x 2 datasets) cleanly.
# Each setup gets a fresh ChromaDB.
#
# Usage:
#   bash eval/run_all.sh              # uses CPU
#   CUDA_VISIBLE_DEVICES=1 bash eval/run_all.sh   # uses GPU 1
#   CUDA_VISIBLE_DEVICES=2 bash eval/run_all.sh   # uses GPU 2

N=50

echo "==============================="
echo "  FULL BENCHMARK — START"
echo "==============================="

# ── TriviaQA ─────────────────────────────────────────────────

echo "--- TriviaQA: no_memory ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/run_setup.py --setup no_memory --dataset triviaqa --n $N

echo "--- TriviaQA: baseline ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/run_setup.py --setup baseline --dataset triviaqa --n $N

echo "--- TriviaQA: full_pma ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/run_setup.py --setup full_pma --dataset triviaqa --n $N

# ── FEVER ─────────────────────────────────────────────────────

echo "--- FEVER: no_memory ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/run_setup.py --setup no_memory --dataset fever --n $N

echo "--- FEVER: baseline ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/run_setup.py --setup baseline --dataset fever --n $N

echo "--- FEVER: full_pma ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/run_setup.py --setup full_pma --dataset fever --n $N

# ── Combine results ───────────────────────────────────────────

echo "--- Combining results ---"
python eval/eval_benchmark.py

echo "==============================="
echo "  FULL BENCHMARK — DONE"
echo "==============================="