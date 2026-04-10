"""week4_test.py — Bayesian Memory Trust Scoring Demo"""

import os, warnings, logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from agents.specialized_agents import AlphaAgent, BetaAgent, GammaAgent
from memory.mem0_store import save, get_all_memories
from memory.trust_store import bayesian_update, get_memory_score

W = 62
def line():  print("─" * W)
def box(t):  print(f"\n{'═'*W}\n  {t}\n{'═'*W}")


def print_all_memory_scores(label: str):
    box(label)
    all_mems = get_all_memories()
    if not all_mems:
        print("  (no memories yet)")
        return
    for m in all_mems:
        mid    = m["id"]
        score  = get_memory_score(mid)
        author = m["memory"].split("]")[0].replace("[","").strip() if "]" in m["memory"] else "?"
        bar    = "█" * int(score * 20)
        flag   = " ← below threshold" if score < 0.2 else ""
        print(f"  {mid[:30]}  author={author:<15}  score={score:.3f}  |{bar:<20}|{flag}")


def run_demo():
    box("WEEK 4 — BAYESIAN MEMORY TRUST SCORING DEMO")
    print("""
  Agents : Alpha (Qwen) · Beta (Phi-2) · Gamma (DeepSeek)
  Memory : Mem0 + ChromaDB + Bayesian trust scores

  Prior     = 0.5  (unknown — no evidence yet)
  Evidence  = votes from Beta + Gamma during probing
  Posterior = updated score via Bayes rule (keyed by memory ID)
    """)

    print("  Loading agents...")
    alpha = AlphaAgent()
    beta  = BetaAgent()
    gamma = GammaAgent()
    alpha._load()
    beta._load()
    gamma._load()
    print("  All agents ready.\n")

    # ── ROUND 1 ───────────────────────────────────────────────
    box("ROUND 1  —  Empty Memory")
    q1 = "How many sides does a triangle have?"
    print(f"\n  Q : {q1}\n")
    r1 = alpha.run(q1, other_agents=None)

    print("  [ TODO1 ]  Alpha reasons first")
    line()
    print(f"  r_q : {r1['initial_reasoning']}")
    print(f"  a_q : {r1['initial_answer'][:80]}")
    print("\n  [ TODO2 ]  Retrieve memory")
    line()
    print("  m_q : empty — no memories yet")
    print("\n  [ TODO3 ]  Contradiction check")
    line()
    print("  u = 0 — nothing to check")
    print("\n  [ FINAL ]")
    line()
    print(f"  a* : {r1['answer'][:80]}")
    print("  Saved to memory ✓")

    # ── INJECT ────────────────────────────────────────────────
    box("INJECT BAD MEMORY")
    wrong_r = "Triangles are quadrilaterals with four equal sides."
    wrong_a = "A triangle has four sides."
    save(q1, wrong_r, wrong_a, agent_id="bad_agent")
    print(f"""
  bad_agent writes into shared memory:
  r : "{wrong_r}"
  a : "{wrong_a}"

  WRONG — correct answer is THREE sides.
  bad_agent starts with prior score = 0.5
    """)

    # ── ROUND 2 ───────────────────────────────────────────────
    box("ROUND 2  —  Contradiction → Probing → Bayesian Calibration")
    q2 = "What is the number of sides in a triangle?"
    print(f"\n  Q : {q2}\n")
    r2 = alpha.run(q2, other_agents=[beta, gamma])

    print("  [ TODO1 ]  Alpha reasons first")
    line()
    print(f"  r_q : {r2['initial_reasoning']}")
    print(f"  a_q : {r2['initial_answer'][:80]}")

    print("\n  [ TODO2 ]  Retrieve memory using ( q + r_q + a_q )")
    line()
    scores = r2.get("contradiction_scores", [])
    print(f"  contradiction scores : {scores}")
    print(f"  u = {r2['uncertainty']} contradiction(s) detected")

    print("\n  [ TODO3 ]  Contradiction check")
    line()
    u = r2["uncertainty"]
    print(f"  u = {u} → {'PROBING triggered ⚠' if u > 0 else 'no contradiction'}")

    if r2.get("probe_results"):
        print("\n  [ PROBING ]  Beta + Gamma evaluate each contradiction")
        line()
        seen_mems = []
        for p in r2["probe_results"]:
            if p["m_q"] not in seen_mems:
                seen_mems.append(p["m_q"])
        for i, mq in enumerate(seen_mems):
            author = next(p["mem_author"] for p in r2["probe_results"] if p["m_q"] == mq)
            print(f"\n  Contradicting memory [{i+1}]:")
            print(f"    author : {author}")
            print(f"    m_q    : {mq[:80]}")
            for p in r2["probe_results"]:
                if p["m_q"] == mq:
                    print(f"    {p['agent']:<15} → {p['verdict']}")

        print("\n  [ BAYESIAN UPDATE ]  Memory trust scores after probing")
        line()
        for mid, posterior in r2["memory_posteriors"].items():
            author = next(
                (p["mem_author"] for p in r2["probe_results"] if p["mem_id"] == mid),
                "unknown"
            )
            direction = "↓" if posterior < 0.5 else "↑"
            print(f"  {mid[:30]}  author={author:<15}  prior=0.500  posterior={posterior:.3f} {direction}")

        print("\n  [ CALIBRATION ]  Weighted vote (by memory posterior)")
        line()
        count_r = sum(1 for p in r2["probe_results"] if p["verdict"] == "TRUST_REASONING")
        count_m = sum(1 for p in r2["probe_results"] if p["verdict"] == "TRUST_MEMORY")
        print(f"  Raw vote count   : TRUST_REASONING={count_r}  TRUST_MEMORY={count_m}")
        print(f"  Consensus        : {r2['calibration_verdict']}")
        print("  → Alpha re-retrieves with calibrated query")

    print("\n  [ FINAL ]  (a*, r*) = Alpha( q, r_q, a_q, calibrated m_q )")
    line()
    print(f"  a* : {r2['answer'][:100]}")
    print("  Saved to memory ✓")

    print_all_memory_scores("ALL MEMORY SCORES  —  After Round 2")


if __name__ == "__main__":
    run_demo()

