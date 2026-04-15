"""
eval/load_datasets.py
Loads 100 cases each from TriviaQA and FEVER.
"""

import json
import random
from datasets import load_dataset


def load_triviaqa(n=100, seed=42):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext")
    samples = ds["validation"].select(range(min(n, 1000)))
    result = []
    for ex in samples:
        result.append({
            "prompt": ex["question"],
            "answer": ex["answer"]["normalized_aliases"]
        })
    return result[:n]


def load_fever(path="eval/fever_dev.jsonl", n=100, seed=42):
    cases = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("verifiable") == "VERIFIABLE" and entry.get("label") in ("SUPPORTS", "REFUTES"):
                cases.append({
                    "prompt": entry["claim"],
                    "answer": [entry["label"]]
                })
    random.seed(seed)
    random.shuffle(cases)
    return cases[:n]


if __name__ == "__main__":
    trivia = load_triviaqa()
    print(f"TriviaQA loaded: {len(trivia)} cases")
    print(f"  Example: {trivia[0]}")

    fever = load_fever()
    print(f"FEVER loaded: {len(fever)} cases")
    print(f"  Example: {fever[0]}")