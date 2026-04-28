#!/bin/bash
echo "==============================="
echo "  DEBATE BENCHMARK — START"
echo "==============================="


# ── Distractor setups ─────────────────────────────────
echo "--- No Memory (Distractor) ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/debate_no_memory_dis.py 2>&1 | tee eval/no_memory_dis_log.txt

echo "--- Baseline (Distractor) ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/debate_baseline_dis.py 2>&1 | tee eval/baseline_dis_log.txt

echo "--- Full PMA (Distractor) ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/debate_full_pma_dis.py 2>&1 | tee eval/full_pma_dis_log.txt


# ── ALFWorld setups ───────────────────────────────────
echo "--- No Memory (ALFWorld) ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/debate_no_memory_alf.py 2>&1 | tee eval/no_memory_alf_log.txt

echo "--- Baseline (ALFWorld) ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/debate_baseline_alf.py 2>&1 | tee eval/baseline_alf_log.txt

echo "--- Full PMA (ALFWorld) ---"
rm -rf chroma_db/ memory/memory_trust_scores.json memory_snapshot.json
python eval/debate_full_pma_alf.py 2>&1 | tee eval/full_pma_alf_log.txt



echo ""
echo "==============================="
echo "  DEBATE BENCHMARK — DONE"
echo "==============================="