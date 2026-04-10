"""
test_final_context.py
Checks whether final answer uses memory context or own reasoning.
Run: CUDA_VISIBLE_DEVICES=3 python test_final_context.py
"""

import os, sys, warnings, logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

os.makedirs("chroma_db", exist_ok=True)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.specialized_agents import AlphaAgent, BetaAgent, GammaAgent

W = 62
def line(): print("─" * W)

# ── Load agents ───────────────────────────────────────────────
alpha = AlphaAgent()
beta  = BetaAgent()
gamma = GammaAgent()
alpha._load()
beta._load()
gamma._load()

# ── Queries ───────────────────────────────────────────────────
TEST_QUERIES = [
    "Triangles have three sides, is that correct?",
    "Who wrote Romeo and Juliet?",
]

for query in TEST_QUERIES:
    print(f"\n{'═'*W}")
    print(f"  Q : {query}")
    print(f"{'═'*W}")

    result = alpha.run(query, other_agents=[beta, gamma])

    # what memories were available at final prompt time
    final_memories = result["memories"]
    used_memory    = len(final_memories) > 0

    print(f"\n  initial_reasoning : {result['initial_reasoning'][:100]}")
    print(f"  initial_answer    : {result['initial_answer'][:80]}")

    line()
    print(f"  contradiction_scores : {result['contradiction_scores']}")
    print(f"  uncertainty          : {result['uncertainty']}")
    print(f"  calibration_verdict  : {result['calibration_verdict']}")

    line()
    print(f"  [ CONTEXT CHECK ]")
    print(f"  context used : {'MEMORY' if used_memory else 'OWN REASONING ONLY'}")
    print(f"  memories in final prompt : {len(final_memories)}")
    for m in final_memories:
        print(f"    → {m[:80]}")

    line()
    print(f"  final_reasoning : {result['reasoning'][:100]}")
    print(f"  final_answer    : {result['answer'][:80]}")