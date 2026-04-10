"""
test_contradiction.py
Pure pipeline test — no logic written here.
Run: CUDA_VISIBLE_DEVICES=3 python test_contradiction.py
"""

import os, sys, warnings, logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

os.makedirs("chroma_db", exist_ok=True)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.specialized_agents import AlphaAgent
from memory.mem0_store import save
from memory.test_dataset import TEST_MEMORIES

# ── Load all test memories ────────────────────────────────────
for m in TEST_MEMORIES:
    save(m["query"], m["reasoning"], m["answer"], m["agent_id"])

# ── Load agent ────────────────────────────────────────────────
alpha = AlphaAgent()
alpha._load()

# ── Query ─────────────────────────────────────────────────────
query = "What is the speed of light?"

# ── Run pipeline ──────────────────────────────────────────────
result = alpha.run(query, other_agents=None)

# ── Print results ─────────────────────────────────────────────
print(f"query             : {query}")
print(f"initial_reasoning : {result['initial_reasoning']}")
print(f"initial_answer    : {result['initial_answer']}")
print(f"memories          : {result['memories']}")
print(f"contradiction_scores : {result['contradiction_scores']}")
print(f"uncertainty          : {result['uncertainty']}")
print(f"uncertainty_triggered: {result['uncertainty_triggered']}")
print(f"final reasoning   : {result['reasoning']}")
print(f"final answer      : {result['answer']}")