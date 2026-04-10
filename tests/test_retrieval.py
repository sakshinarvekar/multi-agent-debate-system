"""
test.py
Pick a query, see what comes back from memory.
Run: python test.py
"""

import os, sys
os.makedirs("chroma_db", exist_ok=True)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.mem0_store import save, search
from memory.test_dataset import TEST_MEMORIES

# ── Load memories ─────────────────────────────────────────────
print("Loading memories...")
for m in TEST_MEMORIES:
    save(m["query"], m["reasoning"], m["answer"], m["agent_id"])
print(f"{len(TEST_MEMORIES)} memories loaded.\n")

# ── Query ─────────────────────────────────────────────────────
query = "Why are veins blue in color?"

print(f"Query : {query}")
print("─" * 60)

results = search(query)

if not results:
    print("No results returned.")
else:
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r}")

print(f"Total retrieved: {len(results)}")