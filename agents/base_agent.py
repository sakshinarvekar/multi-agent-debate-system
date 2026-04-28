"""agents/base_agent.py"""
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from memory.mem0_store import search, save, search_with_ids, delete_memory
from memory.trust_store import bayesian_update, should_delete
from sentence_transformers import SentenceTransformer, util as st_util

_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

CONTRADICTION_THRESHOLD = 0.60


class BaseAgent:
    agent_id: str
    model_id: str
    _tok   = None
    _model = None

    def _load(self):
        if self.__class__._model is None:
            cls = self.__class__
            cls._tok   = AutoTokenizer.from_pretrained(self.model_id)
            cls._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            cls._model.eval()

    def _generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        self._load()
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def probe(self, query: str, r_q: str, m_q: str) -> tuple[str, str]:
        prompt = (
            f"You are {self.agent_id}.\n\n"
            f"There is a conflict between two claims about the following question:\n"
            f"Question: {query}\n\n"
            f"Claim A — Current agent's reasoning:\n{r_q}\n\n"
            f"Claim B — Retrieved memory:\n{m_q}\n\n"
            f"Your task:\n"
            f"1. Carefully reason about which claim is correct.\n"
            f"2. Consider the question and both claims critically.\n"
            f"3. End your response with exactly one of:\n"
            f"   TRUST_REASONING  (if Claim A is correct)\n"
            f"   TRUST_MEMORY     (if Claim B is correct)\n\n"
            f"Reasoning:"
        )
        response = self._generate(prompt, max_new_tokens=200)

        if "TRUST_REASONING" in response.upper():
            verdict = "TRUST_REASONING"
        elif "TRUST_MEMORY" in response.upper():
            verdict = "TRUST_MEMORY"
        else:
            verdict = "TRUST_REASONING"

        return verdict, response

    # def run(self, query: str, other_agents: list = None) -> dict:
    def run(self, query: str, other_agents: list = None, audit: bool = True) -> dict:

        # ── TODO1 — reason first, before checking memory ──────
        prompt_initial = (
            f"You are {self.agent_id}.\n\n"
            f"Question: {query}\n\n"
            f"Give a brief reasoning then your answer.\n"
            f"Reasoning:"
        )
        initial_raw = self._generate(prompt_initial)
        initial_reasoning, initial_answer = _split(initial_raw)

        # ── TODO2 — search memory using (q + r_q + a_q) ───────
        search_query = query + " " + initial_reasoning + " " + initial_answer
        # memories     = search(search_query)
        memories         = search(search_query)
        initial_memories = memories[:]   # snapshot before calibration overwrites it

        # ── TODO3 — uncertainty check ─────────────────────────
        u, contradiction_scores = _count_contradictions(
            initial_reasoning, memories, initial_answer
        )

        uncertainty_triggered = False
        probe_results         = []
        calibration_verdict   = None
        memory_posteriors     = {}
        deleted_memories      = []

        # if u > 0 and other_agents:
        if audit and u > 0 and other_agents:
            uncertainty_triggered = True

            # ── PROBING ───────────────────────────────────────
            memories_with_ids = search_with_ids(search_query)
            mem_id_map = {m["memory"]: m["id"] for m in memories_with_ids}

            contradicting_memories = [
                mem for mem, score in zip(memories, contradiction_scores)
                if score < CONTRADICTION_THRESHOLD
            ]

            memory_ids_flagged = []

            for c_mem in contradicting_memories:
                if "| Reasoning:" in c_mem and "| A:" in c_mem:
                    mem_reasoning = c_mem.split("| Reasoning:")[-1].split("| A:")[0].strip()
                else:
                    mem_reasoning = c_mem

                mem_id = mem_id_map.get(c_mem, "")

                # Extract who wrote this memory
                # stored format: "[agent_id] Q: ... | Reasoning: ... | A: ..."
                if c_mem.startswith("[") and "]" in c_mem:
                    mem_author = c_mem.split("]")[0].replace("[", "").strip()
                else:
                    mem_author = "unknown"

                mem_votes = []

                for other_agent in other_agents:
                    verdict, response = other_agent.probe(
                        query = query,
                        r_q   = initial_reasoning,
                        m_q   = mem_reasoning,
                    )
                    mem_votes.append(verdict)
                    probe_results.append({
                        "agent":      other_agent.agent_id,
                        "verdict":    verdict,
                        "response":   response[:200],
                        "m_q":        mem_reasoning[:100],
                        "mem_id":     mem_id,
                        "mem_author": mem_author,
                    })

                memory_ids_flagged.append(mem_id)

            # ── BAYESIAN UPDATE ───────────────────────────────
            # Score is assigned to each MEMORY by its mem_id.
            # Prior  = stored score for this mem_id (0.5 if first time)
            # Evidence = votes from Beta + Gamma for this memory
            # Posterior = updated trust score for this memory
            seen_mem_ids = set()
            for probe in probe_results:
                mid = probe["mem_id"]
                if mid and mid not in seen_mem_ids:
                    # collect all votes cast about this specific memory
                    votes_for_mem = [
                        p["verdict"] for p in probe_results
                        if p["mem_id"] == mid
                    ]
                    # update this memory's score
                    posterior = bayesian_update(mid, votes_for_mem)
                    memory_posteriors[mid] = posterior
                    seen_mem_ids.add(mid)

            # ── CALIBRATION — weighted by memory posterior ────────────────
            weighted_trust_mem = 0.0
            weighted_trust_rsn = 0.0
            for probe in probe_results:
                weight = memory_posteriors.get(probe["mem_id"], 0.5)
                if probe["verdict"] == "TRUST_MEMORY":
                    weighted_trust_mem += weight
                else:
                    weighted_trust_rsn += weight

            calibration_verdict = (
                "TRUST_REASONING" if weighted_trust_rsn >= weighted_trust_mem else "TRUST_MEMORY"
            )
            calibrated_query = (
                query + " " + initial_reasoning + " correct answer verified"
                if calibration_verdict == "TRUST_REASONING"
                else query + " " + initial_answer + " memory verified correct"
            )

            # Re-retrieve with calibrated query
            memories = search(calibrated_query)

        elif u > 0 and not other_agents:
            uncertainty_triggered = True
            calibration_verdict   = "NO_AGENTS"
            calibrated_query = (
                query + " " + initial_reasoning
                + " verify correct answer resolve contradiction"
            )
            
        # ── MAINTENANCE LAYER — AGM Contraction ──────────
        # Per PMA framework Layer V:
        # S_new > τ_high → memory solidified (kept)
        # S_new < τ_low  → AGM contraction (physical delete)
        for mid, posterior in memory_posteriors.items():
            if mid and should_delete(mid):
                if delete_memory(mid):
                    deleted_memories.append(mid)
                    print(f"  [AGM] deleted memory {mid[:30]} — score={posterior:.3f}")

            # Re-retrieve with calibrated query
            memories = search(calibrated_query)

        # ── Build final prompt with memory context ─────────────
        mem_block = "\n".join(f"  - {m}" for m in memories)
        prompt_final = (
            f"You are {self.agent_id}\n"
            + (f"\n[Memory from other agents]\n{mem_block}\n" if memories else "")
            + f"\nQuestion: {query}\n\n"
            f"Give a brief reasoning then your answer.\n"
            f"Reasoning:"
        )

        # ── Generate final answer ──────────────────────────────
        final_raw = self._generate(prompt_final)
        reasoning, answer = _split(final_raw)

        # ── Save (Q, R, A) to shared memory ───────────────────
        save(query, reasoning, answer, self.agent_id)

        return {
            "agent":                 self.agent_id,
            "memories":              memories,
            "initial_reasoning":     initial_reasoning,
            "initial_answer":        initial_answer,
            "reasoning":             reasoning,
            "answer":                answer,
            "uncertainty":           u,
            "contradiction_scores":  contradiction_scores,
            "uncertainty_triggered": uncertainty_triggered,
            "probe_results":         probe_results,
            "calibration_verdict":   calibration_verdict,
            "memory_posteriors":     memory_posteriors,
            "deleted_memories":      deleted_memories,
            "initial_memories": initial_memories,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _count_contradictions(
    reasoning: str,
    memories: list[str],
    initial_answer: str = ""
) -> tuple[int, list[float]]:
    if not memories or not reasoning:
        return 0, []

    compare_text = reasoning
    ans_vec = _embedder.encode(compare_text, convert_to_tensor=True)

    u, scores = 0, []

    for mem in memories:
        if "| Reasoning:" in mem:
            mem_reasoning = mem.split("| Reasoning:")[-1].split("| A:")[0].strip()
        else:
            mem_reasoning = mem

        m_vec = _embedder.encode(mem_reasoning, convert_to_tensor=True)
        score  = round(float(st_util.cos_sim(ans_vec, m_vec)[0][0]), 3)
        scores.append(score)
        if score < CONTRADICTION_THRESHOLD:
            u += 1

    return u, scores


def _split(text: str) -> tuple[str, str]:
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    if text.startswith("[agent_"):
        text = text.split("A:")[-1].strip() if "A:" in text else text.split("|")[-1].strip()

    if "Answer:" in text:
        parts = text.split("Answer:", 1)
        return parts[0].strip(), parts[1].strip()

    if "\nA:" in text:
        parts = text.split("\nA:", 1)
        return parts[0].strip(), parts[1].strip()

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) >= 2:
        return " ".join(lines[:-1]), lines[-1]

    return text.strip(), text.strip()
