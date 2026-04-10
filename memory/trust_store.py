"""memory/trust_store.py
Bayesian trust scoring for memory entries.
Score lives on the MEMORY, not the agent.

Prior  = current memory score (starts at 0.5 — max uncertainty)
Evidence = agent votes during probing
Posterior = updated score after votes

Score < DELETION_THRESHOLD → memory deleted from ChromaDB (AGM contraction)
"""

import json, os

TRUST_FILE        = "memory/memory_trust_scores.json"
INITIAL_SCORE     = 0.5   # flat prior — we know nothing about this memory yet
DELETION_THRESHOLD = 0.2  # below this → delete from ChromaDB

# Likelihood parameters — how much to trust each vote type
# P(agent votes TRUST_MEMORY | memory is actually correct)
P_VOTE_MEM_GIVEN_CORRECT   = 0.85
# P(agent votes TRUST_MEMORY | memory is actually wrong)
P_VOTE_MEM_GIVEN_INCORRECT = 0.15


# ── persistence ──────────────────────────────────────────────────────────────

def _load() -> dict:
    if os.path.exists(TRUST_FILE):
        try:
            with open(TRUST_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(store: dict) -> None:
    os.makedirs(os.path.dirname(TRUST_FILE), exist_ok=True)
    with open(TRUST_FILE, "w") as f:
        json.dump(store, f, indent=2)


# ── public API ────────────────────────────────────────────────────────────────

def get_memory_score(memory_id: str) -> float:
    """Return current trust score for a memory. Default = 0.5 (unknown)."""
    return _load().get(memory_id, INITIAL_SCORE)


def bayesian_update(memory_id: str, votes: list[str]) -> float:
    store = _load()
    prior = store.get(memory_id, INITIAL_SCORE)

    # Accumulate likelihood product over all votes (conditionally independent)
    L_correct   = 1.0   # P(all votes | memory correct)
    L_incorrect = 1.0   # P(all votes | memory wrong)

    for vote in votes:
        if vote == "TRUST_MEMORY":
            L_correct   *= P_VOTE_MEM_GIVEN_CORRECT          # 0.85
            L_incorrect *= P_VOTE_MEM_GIVEN_INCORRECT         # 0.15
        else:  # TRUST_REASONING — evidence against this memory
            L_correct   *= (1.0 - P_VOTE_MEM_GIVEN_CORRECT)  # 0.15
            L_incorrect *= (1.0 - P_VOTE_MEM_GIVEN_INCORRECT) # 0.85

    numerator   = L_correct * prior
    denominator = numerator + L_incorrect * (1.0 - prior)

    posterior = numerator / denominator if denominator > 1e-12 else 0.5

    store[memory_id] = round(posterior, 4)
    _save(store)
    return posterior


def bayesian_update_all(
    memory_ids:       list[str],
    votes_per_memory: list[list[str]],
) -> list[float]:
    return [
        bayesian_update(mid, votes)
        for mid, votes in zip(memory_ids, votes_per_memory)
    ]


def should_delete(memory_id: str) -> bool:
    """True if this memory's posterior has fallen below the deletion threshold."""
    return get_memory_score(memory_id) < DELETION_THRESHOLD


def print_memory_scores() -> None:
    """Pretty-print all memory trust scores."""
    store = _load()
    if not store:
        print("  (no memory scores yet)")
        return
    for mid, score in store.items():
        bar  = "█" * int(score * 20)
        flag = " ← below deletion threshold" if score < DELETION_THRESHOLD else ""
        print(f"  {mid[:30]:<32} score={score:.3f}  |{bar:<20}|{flag}")