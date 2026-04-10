"""router.py — Sends a query through all agents sequentially."""

from agents.specialized_agents import AlphaAgent, BetaAgent, GammaAgent

AGENTS = [AlphaAgent(), BetaAgent(), GammaAgent()]


def debate(query: str) -> list[dict]:
    """
    Run all agents in order. Because they share one Mem0 pool,
    each agent automatically sees what the previous agents wrote.
    """
    return [agent.run(query) for agent in AGENTS]
