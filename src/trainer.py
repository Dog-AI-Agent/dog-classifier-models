from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.default import TrainConfig
from src.metrics import topk_accuracy, compute_f1


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, val_metric: float) -> bool:
        if self.best_score is None or val_metric > self.best_score:
            self.best_score = val_metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        self.scaler = GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")
        self.early_stopping = EarlyStopping(patience=cfg.early_stopping_patience)

        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log_path = Path(cfg.log_dir) / cfg.model_name
        self.writer = SummaryWriter(log_dir=str(log_path))

        self.best_val_acc = 0.0

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            top1, top5 = topk_accuracy(outputs, labels, topk=(1, 5))
            total_loss += loss.item()
            total_top1 += top1
            total_top5 += top5
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", top1=f"{top1:.1f}%")

        return {
            "loss": total_loss / n_batches,
            "top1": total_top1 / n_batches,
            "top5": total_top5 / n_batches,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        n_batches = 0
        all_preds, all_targets = [], []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs} [Val]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            top1, top5 = topk_accuracy(outputs, labels, topk=(1, 5))
            total_loss += loss.item()
            total_top1 += top1
            total_top5 += top5
            n_batches += 1

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

            pbar.set_postfix(top1=f"{top1:.1f}%")

        f1 = compute_f1(all_preds, all_targets)
        return {
            "loss": total_loss / n_batches,
            "top1": total_top1 / n_batches,
            "top5": total_top5 / n_batches,
            "f1": f1,
        }

    def save_checkpoint(self, epoch: int, val_metrics: dict):
        path = self.checkpoint_dir / f"best_{self.cfg.model_name}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "val_f1": val_metrics["f1"],
            "config": {
                "model_name": self.cfg.model_name,
                "timm_model_name": self.cfg.timm_model_name,
                "num_classes": self.cfg.num_classes,
            },
        }, path)
        print(f"  Saved checkpoint: {path}")

    def log_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        self.writer.add_scalars("loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
        self.writer.add_scalars("top1_acc", {"train": train_metrics["top1"], "val": val_metrics["top1"]}, epoch)
        self.writer.add_scalars("top5_acc", {"train": train_metrics["top5"], "val": val_metrics["top5"]}, epoch)
        self.writer.add_scalar("val/f1", val_metrics["f1"], epoch)
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

    def fit(self):
        print(f"\nTraining {self.cfg.model_name} for {self.cfg.epochs} epochs on {self.device}")
        print(f"  Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}\n")

        for epoch in range(self.cfg.epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)
            self.scheduler.step()

            self.log_metrics(epoch, train_metrics, val_metrics)

            print(
                f"  Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Top-1: {train_metrics['top1']:.2f}% | "
                f"Val Top-1: {val_metrics['top1']:.2f}% | "
                f"Val Top-5: {val_metrics['top5']:.2f}% | "
                f"Val F1: {val_metrics['f1']:.2f}%"
            )

            if val_metrics["top1"] > self.best_val_acc:
                self.best_val_acc = val_metrics["top1"]
                self.save_checkpoint(epoch, val_metrics)

            if self.early_stopping(val_metrics["top1"]):
                print(f"\nEarly stopping at epoch {epoch+1}. Best Val Top-1: {self.best_val_acc:.2f}%")
                break

        self.writer.close()
        print(f"\nTraining complete. Best Val Top-1: {self.best_val_acc:.2f}%")
        return self.best_val_acc
