"""
eval/setup_baseline_memory.py
Setup (ii) — Memory exists but NO audit.
Alpha retrieves and uses memories directly.
No contradiction detection, no probing, no Bayesian, no deletion.
"""

import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")

from memory.mem0_store import search, save


def run_baseline_memory(alpha, query: str) -> dict:
    """
    Run Alpha with basic memory — retrieve and use, no audit.
    """
    # step 1 — reason independently first
    prompt_initial = (
        f"You are {alpha.agent_id}: {alpha.persona}.\n\n"
        f"Question: {query}\n\n"
        f"Give a brief reasoning then your answer.\n"
        f"Reasoning:"
    )
    initial_raw = alpha._generate(prompt_initial)
    lines = [l.strip() for l in initial_raw.strip().splitlines() if l.strip()]
    if len(lines) >= 2:
        initial_reasoning = " ".join(lines[:-1])
        initial_answer    = lines[-1]
    else:
        initial_reasoning = initial_raw.strip()
        initial_answer    = initial_raw.strip()

    # step 2 — retrieve memories (no contradiction check)
    search_query = query + " " + initial_reasoning + " " + initial_answer
    memories     = search(search_query)

    # step 3 — build final prompt with raw memories (no filtering)
    mem_block = "\n".join(f"  - {m}" for m in memories)
    prompt_final = (
        f"You are {alpha.agent_id}: {alpha.persona}.\n"
        + (f"\n[Memory from other agents]\n{mem_block}\n" if memories else "")
        + f"\nQuestion: {query}\n\n"
        f"Give a brief reasoning then your answer.\n"
        f"Reasoning:"
    )

    # step 4 — generate final answer
    final_raw = alpha._generate(prompt_final)
    lines = [l.strip() for l in final_raw.strip().splitlines() if l.strip()]
    if len(lines) >= 2:
        reasoning = " ".join(lines[:-1])
        answer    = lines[-1]
    else:
        reasoning = final_raw.strip()
        answer    = final_raw.strip()

    # step 5 — save to memory (no audit)
    save(query, reasoning, answer, alpha.agent_id)

    return {
        "setup":     "baseline_memory",
        "query":     query,
        "memories":  memories,
        "reasoning": reasoning,
        "answer":    answer,
    }