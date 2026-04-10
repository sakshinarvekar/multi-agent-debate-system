"""
test_retrieval2.py
Tests the full flow: reason first → retrieve → show contradictions
Run: CUDA_VISIBLE_DEVICES=3 python test_retrieval2.py
"""

import os, sys, warnings, logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

os.makedirs("chroma_db", exist_ok=True)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.mem0_store import save, search
from memory.test_dataset import TEST_MEMORIES
from agents.specialized_agents import AlphaAgent

W = 60
def line(): print("─" * W)

# ── Step 1: Load test memories ────────────────────────────────
print("Loading test memories...")
for m in TEST_MEMORIES:
    save(m["query"], m["reasoning"], m["answer"], m["agent_id"])
print(f"{len(TEST_MEMORIES)} memories loaded.\n")

# ── Step 2: Load agent ────────────────────────────────────────
print("Loading Alpha agent...")
alpha = AlphaAgent()
alpha._load()
print("Agent ready.\n")

# ── Step 3: Run the flow ──────────────────────────────────────
query = "How many sides does a triangle have?"

print(f"Query : {query}")
line()

result = alpha.run(query, other_agents=None)

# ── Step 1 output: agent's own reasoning ──────────────────────
print("\n[ STEP 1 ]  Agent reasons first (before memory)")
line()
print(f"  r_q : {result['initial_reasoning']}")
print(f"  a_q : {result['initial_answer']}")

# ── Step 2 output: retrieve using (q + r_q + a_q) ─────────────
print("\n[ STEP 2 ]  Retrieve memory using ( q + r_q + a_q )")
line()
memories = result["memories"]
if not memories:
    print("  No memories retrieved.")
else:
    for i, m in enumerate(memories):
        print(f"  [{i+1}] {m}")
