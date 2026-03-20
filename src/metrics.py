import torch
from sklearn.metrics import f1_score


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1, 5)) -> list[float]:
    """Compute top-k accuracy for the given k values."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append(correct_k.item() / batch_size * 100.0)
        return results


def compute_f1(all_preds: list, all_targets: list) -> float:
    """Compute macro F1 score."""
    return f1_score(all_targets, all_preds, average="macro") * 100.0
