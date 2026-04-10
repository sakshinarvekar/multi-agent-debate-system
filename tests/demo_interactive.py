"""
demo_interactive.py
Single query demo — runs one query then exits cleanly.
Run again for a fresh start from Step 1.

Run: CUDA_VISIBLE_DEVICES=1 python tests/demo_interactive.py
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
from memory.mem0_store import get_all_memories

W = 62
def line():     print("─" * W)
def box(t):     print(f"\n{'═'*W}\n  {t}\n{'═'*W}")
def section(t): print(f"\n  [ {t} ]"); line()


# ── Startup ───────────────────────────────────────────────────
box("MULTI-AGENT DEBATE SYSTEM — Interactive Demo")

mems = get_all_memories()
print(f"\n  Memory pool : {len(mems)} entries loaded")
if len(mems) == 0:
    print("\n  ⚠  No memories found in ChromaDB.")
    print("     Load a dataset first:")
    print("     python scripts/load_memories.py memory/test_dataset.py")
    print("     python scripts/load_truthfulqa.py")
    sys.exit(0)

# ── Load agents ───────────────────────────────────────────────
box("Loading agents — please wait...")
alpha = AlphaAgent()
beta  = BetaAgent()
gamma = GammaAgent()
alpha._load()
beta._load()
gamma._load()
print("\n  All agents ready.")

# ── Single query input ────────────────────────────────────────
box("ENTER YOUR QUERY")
print(f"  Memory pool : {len(mems)} entries\n")
query = input("  > ").strip()

if not query:
    print("\n  ⚠  Empty query — exiting.\n")
    sys.exit(0)

# ── Run pipeline ──────────────────────────────────────────────
print(f"\n  Running pipeline for: \"{query}\"")
line()
result = alpha.run(query, other_agents=[beta, gamma])

# ── Step 1: Initial reasoning ─────────────────────────────────
section("STEP 1 — Alpha reasons independently (before memory)")
print(f"  reasoning : {result['initial_reasoning'][:120]}")
print(f"  answer    : {result['initial_answer'][:120]}")

# ── Step 2: Memory retrieval ──────────────────────────────────
section("STEP 2 — Memory retrieved")
initial_mems = result.get("initial_memories", [])
if not initial_mems:
    print("  none retrieved — no relevant memories found")
else:
    print(f"  {len(initial_mems)} memory/memories retrieved:\n")
    for i, m in enumerate(initial_mems):
        print(f"  [{i+1}] {m[:100]}")

# ── Step 3: Contradiction detection ──────────────────────────
section("STEP 3 — Contradiction detection")
scores = result["contradiction_scores"]
u      = result["uncertainty"]
if not scores:
    print("  no memories to compare against")
else:
    for i, score in enumerate(scores):
        tag = "⚠  CONTRADICTION" if score < 0.80 else "✓  OK"
        print(f"  [{i+1}] cosine score={score:.3f}  {tag}")
print(f"\n  contradictions found : {u}")
print(f"  probing triggered    : {result['uncertainty_triggered']}")

# ── Step 4: Probing ───────────────────────────────────────────
section("STEP 4 — Beta + Gamma probe each contradiction")
if not result["probe_results"]:
    print("  no probing — zero contradictions detected")
else:
    seen = []
    for p in result["probe_results"]:
        if p["m_q"] not in seen:
            seen.append(p["m_q"])
    for i, mq in enumerate(seen):
        print(f"\n  memory [{i+1}]: {mq[:80]}")
        for p in result["probe_results"]:
            if p["m_q"] == mq:
                verdict_icon = "✗" if p["verdict"] == "TRUST_REASONING" else "✓"
                print(f"    {p['agent']:<15} → {verdict_icon}  {p['verdict']}")

# ── Step 5: Bayesian update ───────────────────────────────────
section("STEP 5 — Bayesian score update per memory")
posteriors = result.get("memory_posteriors", {})
if not posteriors:
    print("  no updates — no contradictions were probed")
else:
    for mid, posterior in posteriors.items():
        direction = "↓  DROPPING" if posterior < 0.5 else "↑  HOLDING"
        print(f"  {mid[:36]}  prior=0.500 → posterior={posterior:.3f}  {direction}")

# ── Step 6: Calibration ───────────────────────────────────────
section("STEP 6 — Calibration verdict")
print(f"  verdict : {result['calibration_verdict']}")

# ── Step 7: Maintenance ───────────────────────────────────────
section("STEP 7 — Memory maintenance (AGM contraction)")
deleted = result.get("deleted_memories", [])
if posteriors:
    for mid, score in posteriors.items():
        if mid in deleted:
            status = "✗  DELETED  (score below 0.2)"
        elif score < 0.2:
            status = "✗  LOW — queued for deletion"
        else:
            status = "✓  OK"
        print(f"  {mid[:36]}  score={score:.3f}  {status}")
if not deleted:
    print("  no memories deleted this round")
else:
    print(f"\n  total deleted : {len(deleted)}")
    for d in deleted:
        print(f"    ✗  {d[:80]}")

# ── Step 8: Final answer ──────────────────────────────────────
section("STEP 8 — Final answer (with clean memory)")
print(f"  reasoning : {result['reasoning'][:120]}")
print(f"  answer    : {result['answer'][:120]}")
print(f"\n  Saved to memory ✓")

# ── Memory pool status ────────────────────────────────────────
mems_after = get_all_memories()


section("MEMORY POOL STATUS")
print(f"  before : {len(mems)} entries")
print(f"  after  : {len(mems_after)} entries")
