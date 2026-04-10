"""
scripts/load_truthfulqa.py
Loads TruthfulQA correct answers into ChromaDB.
Correct answers → agent_alpha
Run: python scripts/load_truthfulqa.py
"""

import os, sys, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
sys.path.insert(0, ".")

from datasets import load_dataset
from memory.mem0_store import save

print("Loading TruthfulQA...")
ds = load_dataset("truthful_qa", "generation", split="validation")
print(f"Dataset size: {len(ds)} entries\n")

count = 0

for row in ds:
    question        = row["question"]
    correct_answers = row["correct_answers"]

    if correct_answers:
        save(
            query     = question,
            reasoning = correct_answers[0],
            answer    = correct_answers[0],
            agent_id  = "agent_alpha"
        )
        count += 1

    if count >= 100:
        break

print(f"  agent_alpha (correct): {count} memories inserted.")