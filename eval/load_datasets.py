"""
eval/load_datasets.py
Loads datasets for multi-agent debate evaluation.
Currently: HotpotQA fullwiki setting
"""

from datasets import load_dataset


# def load_coqa(n=100, seed=42):
#     """CoQA dataset — commented out, using HotpotQA fullwiki instead"""
#     ds = load_dataset("stanfordnlp/coqa")
#     validation = ds["validation"]
#     cases = []
#     for ex in validation:
#         story     = ex["story"]
#         questions = ex["questions"]
#         answers   = ex["answers"]["input_text"]
#         for q, a in zip(questions, answers):
#             cases.append({"prompt": q, "answer": a, "context": story})
#         if len(cases) >= n:
#             break
#     return cases[:n]

def load_hotpotqa_distractor(n=50, seed=42):
    """
    Loads HotpotQA distractor setting.
    Each case has question, answer, and 10 context paragraphs.
    2 relevant + 8 distractors.
    """
    ds = load_dataset("hotpot_qa", "distractor")
    validation = ds["validation"]

    cases = []
    for ex in validation.select(range(min(n, len(validation)))):
        # flatten context paragraphs
        context = ""
        for title, sentences in zip(
            ex["context"]["title"],
            ex["context"]["sentences"]
        ):
            context += f"\n[{title}]\n"
            context += " ".join(sentences)

        cases.append({
            "prompt":  ex["question"],
            "answer":  ex["answer"],
            "context": context.strip(),
            "type":    ex["type"],
            "level":   ex["level"],
        })

    return cases[:n]
    
def load_hotpotqa_fullwiki(n=100, seed=42):
    """
    Loads HotpotQA fullwiki setting.
    No context provided — agents reason from knowledge and memory.
    Each case has: prompt (question), answer only.
    Memory is essential here — agents build knowledge across turns.
    """
    ds         = load_dataset("hotpot_qa", "fullwiki")
    validation = ds["validation"]

    cases = []
    for ex in validation.select(range(min(n, len(validation)))):
        cases.append({
            "prompt":  ex["question"],
            "answer":  ex["answer"],
            "context": "",           # no context — fullwiki setting
            "type":    ex["type"],
            "level":   ex["level"],
        })

    return cases[:n]


# ── Add more datasets here as needed ──────────────────────────
# def load_hotpotqa_distractor(n=100):
#     """HotpotQA distractor — 10 paragraphs provided"""
#     pass
#
# def load_alfworld(n=100):
#     pass


if __name__ == "__main__":
    cases = load_hotpotqa_fullwiki(n=5)
    print(f"HotpotQA fullwiki loaded: {len(cases)} cases")
    print(f"\nExample:")
    print(f"  Q     : {cases[0]['prompt']}")
    print(f"  A     : {cases[0]['answer']}")
    print(f"  Type  : {cases[0]['type']}")
    print(f"  Level : {cases[0]['level']}")