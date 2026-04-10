"""
scripts/inspect_memory.py
Shows current memory pool.
Run: python scripts/inspect_memory.py
"""

import os, sys, json, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, ".")

from memory.mem0_store import get_all_memories
from memory.trust_store import get_memory_score

mems     = get_all_memories()
snapshot = []

for m in mems:
    mid   = m["id"]
    text  = m["memory"]
    score = get_memory_score(mid)

    agent_id  = text.split("]")[0].replace("[", "").strip() if text.startswith("[") else "unknown"
    query     = text.split("Q:")[-1].split("| Reasoning:")[0].strip() if "Q:" in text else ""
    reasoning = text.split("| Reasoning:")[-1].split("| A:")[0].strip() if "| Reasoning:" in text else ""
    answer    = text.split("| A:")[-1].strip() if "| A:" in text else ""

    snapshot.append({
        "id":        mid,
        "agent_id":  agent_id,
        "score":     score,
        "query":     query,
        "reasoning": reasoning,
        "answer":    answer,
    })

# save
with open("memory_snapshot.json", "w") as f:
    json.dump(snapshot, f, indent=2)

# print
print(f"Total memories: {len(snapshot)}\n")
# for e in snapshot:
#     print(f"  [{e['agent_id']}]  score={e['score']:.3f}")
#     print(f"  Q : {e['query'][:70]}")
#     print(f"  R : {e['reasoning'][:70]}")
#     print(f"  A : {e['answer'][:70]}")
#     print()

print(f"Saved to memory_snapshot.json")