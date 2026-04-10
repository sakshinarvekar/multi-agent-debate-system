"""memory/mem0_store.py"""

import os
os.makedirs("chroma_db", exist_ok=True)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")

from mem0 import Memory

_mem = Memory.from_config({
    "embedder": {
        "provider": "huggingface",
        "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "shared_agent_memory",
            "path": "chroma_db"
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": "dummy-key-not-used",
        }
    },
    "version": "v1.1",
})

SHARED_USER = "shared"


# def search(query: str, top_k: int = 3) -> list[str]:
#     try:
#         results = _mem.search(query=query, user_id=SHARED_USER, limit=top_k)
#         entries = results.get("results", results) if isinstance(results, dict) else results
#         return [e["memory"] for e in entries if "memory" in e]
#     except Exception:
#         return []
def search(query: str, top_k: int = 3, reasoning: str = "", threshold: float = 0.6) -> list[str]:
    """
    Retrieval:
    1. Fetch ALL memories from ChromaDB
    2. Re-score each with cosine similarity against (query + reasoning)
    3. Filter below threshold
    4. Return top_k sorted by similarity
    """
    try:
        # fetch everything — let YOUR scoring decide relevance
        all_entries = get_all_memories()
        if not all_entries:
            return []

        # combine query + reasoning for richer embedding
        combined  = (query + " " + reasoning).strip()
        embedder  = _get_embedder()
        query_vec = embedder.encode(combined, convert_to_tensor=True)

        from sentence_transformers import util as st_util

        scored = []
        for entry in all_entries:
            mem      = entry["memory"]
            mem_text = mem.split("| A:")[-1].strip() if "| A:" in mem else mem
            mem_vec  = embedder.encode(mem_text, convert_to_tensor=True)
            score    = float(st_util.cos_sim(query_vec, mem_vec)[0][0])
            if score >= threshold:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    except Exception as e:
        print(f"  [memory] search failed: {e}")
        return []



def search_with_ids(query: str, top_k: int = 3, reasoning: str = "", threshold: float = 0.6) -> list[dict]:
    try:
        all_entries = get_all_memories()
        if not all_entries:
            return []

        combined  = (query + " " + reasoning).strip()
        embedder  = _get_embedder()
        query_vec = embedder.encode(combined, convert_to_tensor=True)

        from sentence_transformers import util as st_util

        scored = []
        for entry in all_entries:
            mem      = entry["memory"]
            mem_id   = entry["id"]
            mem_text = mem.split("| A:")[-1].strip() if "| A:" in mem else mem
            mem_vec  = embedder.encode(mem_text, convert_to_tensor=True)
            score    = float(st_util.cos_sim(query_vec, mem_vec)[0][0])
            if score >= threshold:
                scored.append((score, {"id": mem_id, "memory": mem}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    except Exception as e:
        print(f"  [memory] search_with_ids failed: {e}")
        return []


def save(query: str, reasoning: str, answer: str, agent_id: str) -> str:
    """Save memory and return its ID for trust score tracking."""
    content = (
        f"[{agent_id}] "
        f"Q: {query} | "
        f"Reasoning: {reasoning[:200]} | "
        f"A: {answer[:200]}"
    )
    try:
        result = _mem.add(
            [{"role": "assistant", "content": content}],
            user_id=SHARED_USER,
            agent_id=agent_id,
            infer=False,
        )
        # Extract memory ID from result
        if isinstance(result, dict) and "results" in result:
            entries = result["results"]
            if entries and "id" in entries[0]:
                return entries[0]["id"]
        return ""
    except Exception as e:
        print(f"  [memory] save failed: {e}")
        return ""


def delete_memory(memory_id: str) -> bool:
    """
    Physically delete a memory from ChromaDB.
    Called when Bayesian score drops below deletion threshold.
    AGM contraction — removes contaminated memory permanently.
    """
    try:
        _mem.delete(memory_id=memory_id)
        return True
    except Exception as e:
        print(f"  [memory] delete failed for {memory_id}: {e}")
        return False


# def get_all_memories() -> list[dict]:
#     """Get all memories with their IDs — used for trust score display."""
#     try:
#         results = _mem.get_all(user_id=SHARED_USER)
#         entries = results.get("results", results) if isinstance(results, dict) else results
#         return [{"id": e.get("id", ""), "memory": e.get("memory", "")} for e in entries]
#     except Exception:
#         return []
def get_all_memories() -> list[dict]:
    try:
        results = _mem.get_all(user_id=SHARED_USER, limit=1000)
        entries = results.get("results", results) if isinstance(results, dict) else results
        return [{"id": e.get("id", ""), "memory": e.get("memory", "")} for e in entries]
    except Exception:
        return []



_embedder_instance = None

def _get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        from sentence_transformers import SentenceTransformer
        _embedder_instance = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder_instance