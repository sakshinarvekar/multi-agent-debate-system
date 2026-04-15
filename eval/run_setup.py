"""
eval/run_setup.py
Runs a single setup across all cases and saves results to JSON.

Usage:
  python eval/run_setup.py --setup no_memory --dataset triviaqa --n 100
  python eval/run_setup.py --setup baseline  --dataset triviaqa --n 100
  python eval/run_setup.py --setup full_pma  --dataset triviaqa --n 100
  python eval/run_setup.py --setup no_memory --dataset fever    --n 100
  python eval/run_setup.py --setup baseline  --dataset fever    --n 100
  python eval/run_setup.py --setup full_pma  --dataset fever    --n 100
"""

import os, sys, time, json, argparse, warnings, logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.specialized_agents import AlphaAgent, BetaAgent, GammaAgent
from eval.load_datasets import load_triviaqa, load_fever
from eval.setup_no_memory import run_no_memory
from eval.setup_baseline_memory import run_baseline_memory
from eval.setup_full_pma import run_full_pma

W = 62
def line():  print("─" * W)
def box(t):  print(f"\n{'═'*W}\n  {t}\n{'═'*W}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup",   required=True, choices=["no_memory", "baseline", "full_pma"])
    parser.add_argument("--dataset", required=True, choices=["triviaqa", "fever"])
    parser.add_argument("--n",       type=int, default=100)
    args = parser.parse_args()

    # ── Load agents ───────────────────────────────────────────
    box(f"Setup: {args.setup} | Dataset: {args.dataset} | N: {args.n}")
    print("\n  Loading agents...")
    alpha = AlphaAgent()
    beta  = BetaAgent()
    gamma = GammaAgent()
    alpha._load()
    if args.setup == "full_pma":
        beta._load()
        gamma._load()
    print("  Agents ready.")

    # ── Load dataset ──────────────────────────────────────────
    print(f"\n  Loading {args.dataset}...")
    if args.dataset == "triviaqa":
        cases = load_triviaqa(n=args.n)
    else:
        cases = load_fever(n=args.n)
    print(f"  {len(cases)} cases loaded.")

    # ── Run cases ─────────────────────────────────────────────
    box(f"Running {args.n} cases...")
    times = []

    for i, case in enumerate(cases):
        query = case["prompt"]
        print(f"  [{i+1}/{len(cases)}] {query[:60]}")

        t0 = time.time()

        if args.setup == "no_memory":
            run_no_memory(alpha, query)

        elif args.setup == "baseline":
            run_baseline_memory(alpha, query)

        else:  # full_pma
            run_full_pma(alpha, query, [beta, gamma])

        elapsed = time.time() - t0
        times.append(elapsed)

        print(f"    time={elapsed:.2f}s  avg so far={round(sum(times)/len(times), 2)}s")

    # ── Save results ──────────────────────────────────────────
    avg_time = round(sum(times) / len(times), 3)

    results = {
        "setup":    args.setup,
        "dataset":  args.dataset,
        "n":        args.n,
        "avg_time": avg_time,
        "times":    times,
    }

    out_path = f"eval/results_{args.dataset}_{args.setup}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    box(f"DONE — {args.setup} on {args.dataset}")
    print(f"  avg_time : {avg_time}s")
    print(f"  saved    : {out_path}\n")


if __name__ == "__main__":
    main()