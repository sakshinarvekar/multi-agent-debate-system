"""
eval/setup_full_pma.py
Setup (i) — Full PMA pipeline.
Just wraps alpha.run() which already does everything.
"""

import os
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")


def run_full_pma(alpha, query: str, other_agents: list) -> dict:
    """
    Run full PMA pipeline — contradiction detection,
    probing, Bayesian scoring, AGM deletion.
    """
    result = alpha.run(query, other_agents=other_agents, audit=True)
    result["setup"] = "full_pma"
    return result
