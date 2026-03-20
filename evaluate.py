import argparse
import json
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from configs.default import TrainConfig
from src.utils import set_seed, get_device
from src.dataset import create_dataloaders
from src.models import create_model
from src.metrics import topk_accuracy, compute_f1


def evaluate_model(model_name: str, device: torch.device) -> dict:
    cfg = TrainConfig(model_name=model_name)
    set_seed(cfg.seed)

    checkpoint_path = Path(cfg.checkpoint_dir) / f"best_{model_name}.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return {}

    _, _, test_loader = create_dataloaders(cfg)
    model = create_model(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    total_top1 = 0.0
    total_top5 = 0.0
    n_batches = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)

            top1, top5 = topk_accuracy(outputs, labels, topk=(1, 5))
            total_top1 += top1
            total_top5 += top5
            n_batches += 1

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    f1 = compute_f1(all_preds, all_targets)
    results = {
        "model": model_name,
        "test_top1": total_top1 / n_batches,
        "test_top5": total_top5 / n_batches,
        "test_f1": f1,
        "train_val_top1": checkpoint.get("val_top1", 0),
        "best_epoch": checkpoint.get("epoch", 0),
    }

    print(f"\n{model_name}: Top-1={results['test_top1']:.2f}% | Top-5={results['test_top5']:.2f}% | F1={results['test_f1']:.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test set")
    parser.add_argument("--model", type=str, default="all",
                        choices=["vit_small", "swin_tiny", "convnext_tiny", "all"],
                        help="Model to evaluate (or 'all')")
    args = parser.parse_args()

    device = get_device()
    models = ["vit_small", "swin_tiny", "convnext_tiny"] if args.model == "all" else [args.model]

    all_results = []
    for m in models:
        result = evaluate_model(m, device)
        if result:
            all_results.append(result)

    # Save results
    results_path = Path("checkpoints") / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
