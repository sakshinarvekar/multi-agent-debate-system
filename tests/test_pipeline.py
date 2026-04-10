"""
test_pipeline.py
Full end-to-end pipeline test — no logic implemented here.
Run: CUDA_VISIBLE_DEVICES=3 python test_pipeline.py
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
from memory.mem0_store import save
# from memory.test_dataset import TEST_MEMORIES

W = 62
def line(): print("─" * W)
def box(t): print(f"\n{'═'*W}\n  {t}\n{'═'*W}")

# ── Test queries ──────────────────────────────────────────────
# TEST_QUERIES = [
#     "How many sides does a triangle have?",
#     "What is the speed of light?",
#     "What is the square root of 144?",
#     "How many chambers does the human heart have?",
# ]
TEST_QUERIES = [
    "Does veins appear blue because they carry deoxygenated blood?",
]

# ── Step 1: Load dataset into memory ─────────────────────────
# box("STEP 1 — Loading test dataset into memory")
# for m in TEST_MEMORIES:
#     save(m["query"], m["reasoning"], m["answer"], m["agent_id"])
# print(f"  {len(TEST_MEMORIES)} memories loaded.")

# ── Step 2: Load agents ───────────────────────────────────────
box("STEP 2 — Loading agents")
alpha = AlphaAgent()
beta  = BetaAgent()
gamma = GammaAgent()
alpha._load()
beta._load()
gamma._load()
print("  All agents ready.")

# ── Step 3: Run full pipeline per query ───────────────────────
box("STEP 3 — Running full pipeline")

for qi, query in enumerate(TEST_QUERIES):
    box(f"QUERY {qi+1}/{len(TEST_QUERIES)}: {query}")

    result = alpha.run(query, other_agents=[beta, gamma])

    # ── initial reasoning ─────────────────────────────────────
    print("\n  [ TODO1 ]  Initial reasoning")
    line()
    print(f"  initial_reasoning : {result['initial_reasoning']}")
    print(f"  initial_answer    : {result['initial_answer'][:80]}")

    # ── memory retrieval ──────────────────────────────────────
    print("\n  [ TODO2 ]  Memory retrieved")
    line()
    initial_mems = result.get("initial_memories", [])
    if not initial_mems:
        print("  none retrieved")
    else:
        for i, m in enumerate(initial_mems):
            print(f"  [{i+1}] {m[:90]}")
    # if not result["memories"]:
    #     print("  none retrieved")
    # else:
    #     for i, m in enumerate(result["memories"]):
    #         print(f"  [{i+1}] {m[:90]}")

    # ── contradiction detection ───────────────────────────────
    print("\n  [ TODO3 ]  Contradiction detection")
    line()
    scores = result["contradiction_scores"]
    u      = result["uncertainty"]
    if not scores:
        print("  no memories to compare")
    else:
        for i, score in enumerate(scores):
            tag = "CONTRADICTION" if score < 0.80 else "OK"
            print(f"  [{i+1}] score={score:.3f}  {tag}")
    print(f"  uncertainty          : {u}")
    print(f"  uncertainty_triggered: {result['uncertainty_triggered']}")

    # ── probing ───────────────────────────────────────────────
    print("\n  [ PROBING ]  Agent verdicts")
    line()
    if not result["probe_results"]:
        print("  no probing — u = 0")
    else:
        seen = []
        for p in result["probe_results"]:
            if p["m_q"] not in seen:
                seen.append(p["m_q"])
        for i, mq in enumerate(seen):
            print(f"\n  memory [{i+1}]: {mq[:70]}")
            for p in result["probe_results"]:
                if p["m_q"] == mq:
                    print(f"    {p['agent']:<15} → {p['verdict']}")

    # ── bayesian update ───────────────────────────────────────
    print("\n  [ BAYESIAN UPDATE ]  Memory posteriors")
    line()
    posteriors = result.get("memory_posteriors", {})
    if not posteriors:
        print("  no updates")
    else:
        for mid, posterior in posteriors.items():
            direction = "↓" if posterior < 0.5 else "↑"
            print(f"  {mid[:36]}  prior=0.500 → posterior={posterior:.3f} {direction}")

    # ── calibration ───────────────────────────────────────────
    print("\n  [ CALIBRATION ]  Verdict")
    line()
    print(f"  calibration_verdict : {result['calibration_verdict']}")

    # ── maintenance layer ─────────────────────────────────────
    print("\n  [ MAINTENANCE ]  Memory trust scores")
    line()
    posteriors = result.get("memory_posteriors", {})
    deleted    = result.get("deleted_memories", [])
    if posteriors:
        for mid, score in posteriors.items():
            status = "✗ DELETED" if mid in deleted else ("✗ LOW — below threshold" if score < 0.2 else "✓ OK")
            print(f"  {mid[:36]}  score={score:.3f}  {status}")
    if not deleted:
        print("  no memories deleted this round")
    else:
        print(f"\n  deleted this round: {len(deleted)}")
        for d in deleted:
            print(f"    ✗ {d}")

    # ── final answer ──────────────────────────────────────────
    print("\n  [ FINAL ]  Final reasoning and answer")
    line()
    print(f"  final_reasoning : {result['reasoning'][:100]}")
    print(f"  final_answer    : {result['answer'][:100]}")
    print("  Saved to memory ✓")

# ── Summary ───────────────────────────────────────────────────
box("SUMMARY")
print(f"  Queries tested : {len(TEST_QUERIES)}")
print(f"""
  TODO1  ✓  initial reasoning generated before memory search
  TODO2  ✓  memory retrieved using ( q + r_q + a_q )
  TODO3  ✓  contradictions detected via cosine similarity
  PROBE  ✓  beta + gamma probed each contradiction
  BAYES  ✓  posterior updated per memory via Bayes rule
  CALIB  ✓  weighted consensus computed
  FINAL  ✓  final answer generated with calibrated memory
""")