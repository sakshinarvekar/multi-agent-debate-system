"""
eval/eval_benchmark.py
Reads all result JSON files and prints final comparison table.

Run: python eval/eval_benchmark.py
"""

import json
import os

W = 62
def line():  print("─" * W)
def box(t):  print(f"\n{'═'*W}\n  {t}\n{'═'*W}")


def load_result(dataset: str, setup: str) -> dict:
    path = f"eval/results_{dataset}_{setup}.json"
    if not os.path.exists(path):
        print(f"  ⚠  Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    datasets = ["triviaqa", "fever"]
    setups   = ["no_memory", "baseline", "full_pma"]

    # get n from first available result
    n = "?"
    for dataset in datasets:
        for setup in setups:
            result = load_result(dataset, setup)
            if result:
                n = result["n"]
                break

    box(f"BENCHMARK RESULTS — Average Runtime per Query (seconds) | N={n} queries each")
    print(f"\n  {'Dataset':<12} {'No Memory':>12} {'Baseline':>12} {'Full PMA':>12}")
    line()

    for dataset in datasets:
        row = f"  {dataset:<12}"
        for setup in setups:
            result = load_result(dataset, setup)
            if result:
                row += f"{result['avg_time']}s".rjust(12)
            else:
                row += f"{'N/A':>12}"
        print(row)

    line()



if __name__ == "__main__":
    main()