"""
eval/debate_baseline_dis.py
Setup (ii) — Baseline memory.
HotpotQA DISTRACTOR setting — with context paragraphs.

Run: CUDA_VISIBLE_DEVICES=3 python eval/debate_baseline_dis.py
"""

import os, sys, time, warnings, logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import _split
from memory.mem0_store import search, save

W = 62
def line():  print("─" * W)
def box(t):  print(f"\n{'═'*W}\n  {t}\n{'═'*W}")


def check_correct(final_answer: str, ground_truth: str, reasoning: str = "") -> bool:
    text = (final_answer + " " + reasoning).lower()
    return ground_truth.lower().strip() in text


def run_debate_baseline(agents: list, case: dict, query_idx: int = 0) -> dict:
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

    return {
        "setup":             "baseline",
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
    }


if __name__ == "__main__":
    from agents.specialized_agents import AlphaAgent, BetaAgent, GammaAgent, DeltaAgent, EpsilonAgent
    from eval.load_datasets import load_hotpotqa_distractor

    box("DEBATE — BASELINE MEMORY | 5 Agents | HotpotQA Distractor")

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

    cases         = load_hotpotqa_distractor(n=100)
    correct_count = 0
    total_tokens_list = []
    end_to_end_list   = []
    round_times_list  = [[] for _ in range(len(agents))]
    N = len(cases)

    for i, case in enumerate(cases):
        box(f"QUERY {i+1}/{N}")
        print(f"  Q            : {case['prompt']}")
        print(f"  Ground truth : {case['answer']}")
        print(f"  Final agent  : {agents[i % len(agents)].agent_id}")
        line()

        result = run_debate_baseline(agents, case, query_idx=i)

        print(f"\n  [ Memories per turn: {result['memories_per_turn']} ]")
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
        for j, t in enumerate(result["round_times"]):
            round_times_list[j].append(t)

    avg_tokens    = round(sum(total_tokens_list) / N, 1)
    avg_e2e       = round(sum(end_to_end_list) / N, 3)
    avg_per_round = [round(sum(round_times_list[j]) / N, 3) for j in range(len(agents))]
    accuracy      = round(correct_count / N * 100, 1)

    box("FINAL RESULTS")
    print(f"\n  Setup    : baseline | 5 agents | HotpotQA Distractor | N={N}")
    print(f"\n  1. Time per round (avg across {N} queries):")
    for j in range(len(agents)):
        print(f"     Round {j+1} : {avg_per_round[j]}s")
    print(f"\n     End-to-end (avg) : {avg_e2e}s")
    print(f"\n  2. Token consumption (avg per query) : {avg_tokens} tokens")
    print(f"\n  3. Overall accuracy : {correct_count}/{N} = {accuracy}%")