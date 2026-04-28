"""
eval/debate_full_pma_dis.py
Setup (i) — Full PMA.
HotpotQA DISTRACTOR setting — with context paragraphs.

Run: CUDA_VISIBLE_DEVICES=3 python eval/debate_full_pma_dis.py
"""

import os, sys, time, warnings, logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import _split, _count_contradictions
from memory.mem0_store  import search, search_with_ids, save, delete_memory
from memory.trust_store import bayesian_update, should_delete

W = 62
def line():  print("─" * W)
def box(t):  print(f"\n{'═'*W}\n  {t}\n{'═'*W}")


def check_correct(final_answer: str, ground_truth: str, reasoning: str = "") -> bool:
    text = (final_answer + " " + reasoning).lower()
    return ground_truth.lower().strip() in text


def run_debate_full_pma(agents: list, case: dict, query_idx: int = 0) -> dict:
    query        = case["prompt"]
    ground_truth = case.get("answer", "")
    context      = case.get("context", "")

    n         = len(agents)
    final_idx = query_idx % n
    rotated   = agents[final_idx+1:] + agents[:final_idx+1]

    responses         = []
    token_counts      = []
    round_times       = []
    memories_per_turn = []
    deleted_memories  = []
    contradictions    = 0

    end_to_end_start = time.time()

    for i, agent in enumerate(rotated):

        round_start = time.time()

        memories  = search(query, threshold=0.3)
        mem_block = "\n".join(f"  - {m}" for m in memories) if memories else ""
        memories_per_turn.append(len(memories))

        prev = ""
        if responses:
            prev = "\n\nPrevious agents' responses:\n"
            for j, r in enumerate(responses):
                prev += f"\nAgent {j+1} ({r['agent_id']}): {r['reasoning'][:100]} | Answer: {r['answer']}"

        prompt = (
            f"You are {agent.agent_id}.\n\n"
            + (f"Context:\n{context[:1500]}\n\n" if context else "")
            + (f"[Relevant memories]\n{mem_block}\n\n" if mem_block else "")
            + f"Question: {query}\n"
            f"{prev}\n\n"
            f"Based on the context, question"
            + (" and memories" if mem_block else "")
            + (" and previous agents' responses" if responses else "")
            + ", give your reasoning then your answer.\n"
            f"Reasoning:"
        )

        raw = agent._generate(prompt, max_new_tokens=300)
        round_time = time.time() - round_start

        reasoning, answer = _split(raw)
        output_tokens = len(agent._tok.encode(raw))

        save(query, reasoning, answer, agent.agent_id)

        responses.append({
            "agent_id":      agent.agent_id,
            "reasoning":     reasoning,
            "answer":        answer,
            "memories_used": len(memories),
        })
        token_counts.append(output_tokens)
        round_times.append(round(round_time, 3))

    end_to_end = round(time.time() - end_to_end_start, 3)

    final_answer    = responses[-1]["answer"]
    final_reasoning = responses[-1]["reasoning"]
    final_agent     = rotated[-1]
    probe_agents    = rotated[:-1]

    # ── PMA AUDIT ─────────────────────────────────────────────
    all_memories = search(query, threshold=0.3)
    if all_memories:
        u, contradiction_scores = _count_contradictions(
            final_reasoning, all_memories, final_answer
        )
        contradictions = u

        if u > 0:
            memories_with_ids = search_with_ids(query, threshold=0.3)
            mem_id_map        = {m["memory"]: m["id"] for m in memories_with_ids}

            contradicting_memories = [
                mem for mem, score in zip(all_memories, contradiction_scores)
                if score < 0.60
            ]

            memory_posteriors = {}

            for c_mem in contradicting_memories:
                if "| Reasoning:" in c_mem and "| A:" in c_mem:
                    mem_reasoning = c_mem.split("| Reasoning:")[-1].split("| A:")[0].strip()
                else:
                    mem_reasoning = c_mem

                mem_id = mem_id_map.get(c_mem, "")
                votes  = []

                for probe_agent in probe_agents:
                    verdict, _ = probe_agent.probe(
                        query = query,
                        r_q   = final_reasoning,
                        m_q   = mem_reasoning,
                    )
                    votes.append(verdict)

                if mem_id:
                    posterior = bayesian_update(mem_id, votes)
                    memory_posteriors[mem_id] = posterior
                    print(f"  [PMA] mem {mem_id[:20]} votes={votes} → score={posterior:.3f}")

            for mid, posterior in memory_posteriors.items():
                if mid and should_delete(mid):
                    if delete_memory(mid):
                        deleted_memories.append(mid)
                        print(f"  [AGM] deleted memory {mid[:30]} — score={posterior:.3f}")

    return {
        "setup":             "full_pma",
        "query":             query,
        "ground_truth":      ground_truth,
        "final_agent":       final_agent.agent_id,
        "memories_per_turn": memories_per_turn,
        "responses":         responses,
        "final_answer":      final_answer,
        "correct":           check_correct(final_answer, ground_truth, final_reasoning),
        "token_counts":      token_counts,
        "total_tokens":      sum(token_counts),
        "round_times":       round_times,
        "end_to_end":        end_to_end,
        "contradictions":    contradictions,
        "deleted_memories":  deleted_memories,
    }


if __name__ == "__main__":
    from agents.specialized_agents import AlphaAgent, BetaAgent, GammaAgent, DeltaAgent, EpsilonAgent
    from eval.load_datasets import load_hotpotqa_distractor

    box("DEBATE — FULL PMA | 5 Agents | HotpotQA Distractor")

    print("\n  Loading agents...")
    alpha   = AlphaAgent()
    beta    = BetaAgent()
    gamma   = GammaAgent()
    delta   = DeltaAgent()
    epsilon = EpsilonAgent()
    alpha._load()
    beta._load()
    gamma._load()
    delta._load()
    epsilon._load()
    agents = [alpha, beta, gamma, delta, epsilon]
    print("  Agents ready.")

    cases                = load_hotpotqa_distractor(n=100)
    correct_count        = 0
    total_tokens_list    = []
    end_to_end_list      = []
    round_times_list     = [[] for _ in range(len(agents))]
    total_deletions      = 0
    total_contradictions = 0
    N = len(cases)

    for i, case in enumerate(cases):
        box(f"QUERY {i+1}/{N}")
        print(f"  Q            : {case['prompt']}")
        print(f"  Ground truth : {case['answer']}")
        print(f"  Final agent  : {agents[i % len(agents)].agent_id}")
        line()

        result = run_debate_full_pma(agents, case, query_idx=i)

        print(f"\n  [ Memories per turn : {result['memories_per_turn']} ]")
        print(f"  [ Contradictions    : {result['contradictions']} ]")
        print(f"  [ Deletions         : {len(result['deleted_memories'])} ]")
        print(f"\n  [ Per Agent ]")
        line()
        for j, r in enumerate(result["responses"]):
            print(f"  Agent {j+1} ({r['agent_id']})")
            print(f"    memories  : {r['memories_used']}")
            print(f"    reasoning : {r['reasoning'][:100]}")
            print(f"    answer    : {r['answer'][:100]}")
            print(f"    tokens    : {result['token_counts'][j]}")
            print(f"    time      : {result['round_times'][j]}s")

        print(f"\n  [ Summary ]")
        line()
        print(f"  final agent  : {result['final_agent']}")
        print(f"  final answer : {result['final_answer'][:100]}")
        print(f"  ground truth : {result['ground_truth']}")
        print(f"  correct      : {'✓' if result['correct'] else '✗'}")
        print(f"  total tokens : {result['total_tokens']}")
        print(f"  end-to-end   : {result['end_to_end']}s")
        print(f"  round times  : {result['round_times']}")

        if result["correct"]:
            correct_count += 1

        total_tokens_list.append(result["total_tokens"])
        end_to_end_list.append(result["end_to_end"])
        total_deletions      += len(result["deleted_memories"])
        total_contradictions += result["contradictions"]
        for j, t in enumerate(result["round_times"]):
            round_times_list[j].append(t)

    avg_tokens    = round(sum(total_tokens_list) / N, 1)
    avg_e2e       = round(sum(end_to_end_list) / N, 3)
    avg_per_round = [round(sum(round_times_list[j]) / N, 3) for j in range(len(agents))]
    accuracy      = round(correct_count / N * 100, 1)

    box("FINAL RESULTS")
    print(f"\n  Setup    : full_pma | 5 agents | HotpotQA Distractor | N={N}")
    print(f"\n  1. Time per round (avg across {N} queries):")
    for j in range(len(agents)):
        print(f"     Round {j+1} : {avg_per_round[j]}s")
    print(f"\n     End-to-end (avg) : {avg_e2e}s")
    print(f"\n  2. Token consumption (avg per query) : {avg_tokens} tokens")
    print(f"\n  3. Overall accuracy : {correct_count}/{N} = {accuracy}%")
    print(f"\n  4. Total contradictions detected : {total_contradictions}")
    print(f"     Total memories deleted        : {total_deletions}")