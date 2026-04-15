"""
eval/setup_no_memory.py
Setup (iii) — No memory at all.
Alpha answers query purely from its own reasoning.
No ChromaDB, no retrieval, no saving.
"""

import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")


def run_no_memory(alpha, query: str) -> dict:
    """
    Run Alpha with no memory — pure reasoning only.
    Skips all memory operations.
    """
    prompt = (
        f"You are {alpha.agent_id}: {alpha.persona}.\n\n"
        f"Question: {query}\n\n"
        f"Give a brief reasoning then your answer.\n"
        f"Reasoning:"
    )
    raw = alpha._generate(prompt)

    # split reasoning and answer
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if len(lines) >= 2:
        reasoning = " ".join(lines[:-1])
        answer = lines[-1]
    else:
        reasoning = raw.strip()
        answer = raw.strip()

    return {
        "setup":     "no_memory",
        "query":     query,
        "reasoning": reasoning,
        "answer":    answer,
    }