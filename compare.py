import json
from pathlib import Path

from tabulate import tabulate


def main():
    results_path = Path("checkpoints") / "eval_results.json"
    if not results_path.exists():
        print("No evaluation results found. Run `python evaluate.py --model all` first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    if not results:
        print("No results to compare.")
        return

    headers = ["Model", "Test Top-1 (%)", "Test Top-5 (%)", "Test F1 (%)", "Val Top-1 (%)", "Best Epoch"]
    rows = [
        [
            r["model"],
            f"{r['test_top1']:.2f}",
            f"{r['test_top5']:.2f}",
            f"{r['test_f1']:.2f}",
            f"{r['train_val_top1']:.2f}",
            r["best_epoch"] + 1,
        ]
        for r in results
    ]

    # Sort by test_top1 descending
    rows.sort(key=lambda x: float(x[1]), reverse=True)

    print("\n" + "=" * 70)
    print("Stanford Dogs Breed Classification - Model Comparison (from scratch)")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()

    best = max(results, key=lambda r: r["test_top1"])
    print(f"Best model: {best['model']} with Test Top-1 = {best['test_top1']:.2f}%")


if __name__ == "__main__":
    main()
